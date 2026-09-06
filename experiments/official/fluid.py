"""Fluid Benchmarking (Hofmann et al., 2025) run through the OFFICIAL code,
/data2/jeongtae/official_baselines/fluid-benchmarking (allenai/fluid-benchmarking).

What is called (file:function)
  fit      irt/fit_irt_model.py:main            -- the trainer call replicated in-process:
           py_irt.training.IrtModelTrainer(config=IrtConfig(model_type=TwoParamLogistic,
           priors='hierarchical'), dataset=...).train(epochs=1000, device='cpu'),
           TwoParamLogistic imported from irt/two_param_logistic.py, best_params read
           as a = exp(disc), b = diff (fit_irt_model.py lines 59-76).
  select   fluid_benchmarking/engine.py:run_fluid_benchmarking -> select_mfi (maximum
           Fisher information at the current MAP ability; ties -> np.argmax).
  ability  fluid_benchmarking/estimators.py:ability_estimate (MAP, Newton with backtracking
           and bisection fallback, mu0 = 0, sigma0 = 1, theta_range (-4, 4), D = 1), as
           called inside run_fluid_benchmarking with method = config.ESTIMATION_METHOD_IRT.
  start    scripts/run_experiments.py lines 53-55: start_ability = mean ability of the models
           the IRT model was trained on (mirrored, see ADAPTATIONS 4).

Readouts
  est          p-IRT plug-in on the SR scale at the final MAP ability (ADAPTATION 5).
  variants     'ability'     their native output (abilities_fb[B-1]),
               'sample_mean' mean of the administered responses (their random_accuracy
                             analogue, evaluation.py:random_accuracy applied to items_fb[:B]).
  stop()       {} -- Fluid publishes no stopping rule (ADAPTATION 9).

ADAPTATIONS
1. Dataset plumbing (irt/fit_irt_model.py:59 `Dataset.from_jsonlines(str(input_path))`):
   the script reads a jsonlines file from --input_path; fit() writes one from R into
   workdir (subject = calibration planner, item = 'item_<bank index>') and omits NaN
   cells -- py-irt's format lists only recorded (subject, item) responses, it has no
   missing-value code, so omission is the only way to express "no record".
2. In-process trainer (irt/fit_irt_model.py:52-56, 78): main() is argparse-driven and
   writes irt/params/<stem>.csv inside the read-only repo; the wrapper issues the same
   IrtConfig / IrtModelTrainer / train(epochs=1000, device='cpu') calls itself
   (lines 59-70), applies the same seeding block (lines 43-49) and reads
   trainer.best_params exactly as lines 73-76, then reorders the rows to bank-index
   order through best_params['item_ids'] (py-irt numbers items by first appearance).
   The seed is THEIRS: fit_irt_model.py --seed defaults to 0 and the task says to keep
   priors/epochs/seed; the harness's cell seed is not used for the fit (recorded in
   info['fit_seed']). The MFI/MAP stage has no randomness, so `seed` is unused there.
3. Logging only (py_irt.training.IrtModelTrainer(verbose=...), which main() leaves at
   its default): verbose=False and the py_irt.dataset console is quieted so 192 cells do
   not print rich tables. No effect on the fit -- py_irt/training.py:79 only sets
   console.quiet and :191-197 only swaps the rich Live display for a nullcontext;
   best_params is assigned in the epoch loop either way (:203-205).
4. start_ability (scripts/run_experiments.py:53-55): the script averages the abilities
   of the LMs the IRT model was trained on, read from data/open_llm_leaderboard_results.json
   on the HF hub. There is no such file for planners, so the wrapper takes the same
   quantity from the fit itself: mean of trainer.best_params['ability'] over the
   calibration planners (the training subjects of this IRT model).
5. SR-scale readout: Fluid reports ability, not accuracy (config.METHODS: the
   'fluid_benchmarking' method returns abilities_fb / items_fb only). 'est' is the
   p-IRT plug-in on the SR scale: (sum of the administered y + sum over the
   unadministered items of P(theta_MAP; a, b)) / n_bank, computed with their 2PL
   parameters, their final MAP ability and their irt_utils.sigmoid_stable. Their
   native ability is kept as variants['ability'].
6. Budget prefixes (engine.py:run_fluid_benchmarking n_max): one run with
   n_max = max(budgets); the budget-B result is the prefix items_fb[:B] with ability
   abilities_fb[B-1]. MFI at the running MAP ability is deterministic given the
   responses, so the prefix equals a run with n_max = B (asserted in selftest_fluid.py).
7. Response vector (engine.py:132 `lm_responses.shape != (n_items,)`, 153 `r0 =
   lm_responses[idx0]`, 181 `r = lm_responses[idx]`): the engine reads responses one
   index at a time, so a LeakGuard subclass exposing `.shape` is passed in place of the
   ndarray; reads are logged and restricted to the selected set, values are unchanged. evaluation.fluid_benchmarking
   is bypassed only because it calls np.array(lm_responses), which would read every item.
8. Items without any calibration record get no parameters from py-irt (they never enter
   the dataset). None occur in the 192 protocol cells (checked over every draw x K_cal x
   slot); the wrapper raises instead of inventing parameters.
9. stop() returns {}: the released code has no stopping rule (run_fluid_benchmarking
   stops at n_max only; the paper's dynamic-stopping demonstration is not in the repo).
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

from .base import OFFICIAL_ROOT, LeakGuard, OfficialMethod

FLUID_ROOT = OFFICIAL_ROOT / 'fluid-benchmarking'
if str(FLUID_ROOT / 'irt') not in sys.path:
    sys.path.insert(0, str(FLUID_ROOT / 'irt'))

import pyro                                                   # noqa: E402
import torch                                                  # noqa: E402
import py_irt.dataset                                         # noqa: E402
from py_irt.config import IrtConfig                           # noqa: E402
from py_irt.dataset import Dataset                            # noqa: E402
from py_irt.training import IrtModelTrainer                   # noqa: E402
from two_param_logistic import TwoParamLogistic               # noqa: E402  (official irt/two_param_logistic.py)
from fluid_benchmarking import config as fluid_config         # noqa: E402
from fluid_benchmarking import engine, irt_utils              # noqa: E402

FIT_SEED = 0            # irt/fit_irt_model.py --seed default (ADAPTATION 2)
FIT_EPOCHS = 1000       # irt/fit_irt_model.py --epochs default
FIT_DEVICE = 'cpu'


class _Responses(LeakGuard):
    """LeakGuard with the `.shape` the engine checks (ADAPTATION 7)."""

    @property
    def shape(self):
        return (len(self._y),)


class Fluid(OfficialMethod):
    name = 'fluid'
    adaptive = True

    def fit(self, R, seed, workdir):
        R = np.asarray(R, float)
        K, n = R.shape
        if np.isnan(R).all(0).any():
            raise RuntimeError('fluid: bank item(s) without any calibration record (ADAPTATION 8)')
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        path = workdir / 'fluid_calibration.jsonlines'
        with open(path, 'w') as f:                                       # ADAPTATION 1
            for k in range(K):
                resp = {f'item_{j}': int(R[k, j]) for j in range(n) if not np.isnan(R[k, j])}
                f.write(json.dumps({'subject_id': f'planner_{k}', 'responses': resp}) + '\n')

        # fit_irt_model.py lines 43-49 (seed 0 = their default)
        random.seed(FIT_SEED)
        np.random.seed(FIT_SEED)
        torch.manual_seed(FIT_SEED)
        pyro.set_rng_seed(FIT_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(True)

        # fit_irt_model.py lines 58-70
        py_irt.dataset.console.quiet = True                              # ADAPTATION 3
        data = Dataset.from_jsonlines(str(path))
        cfg = IrtConfig(model_type=TwoParamLogistic, priors='hierarchical')
        trainer = IrtModelTrainer(config=cfg, data_path=None, dataset=data, verbose=False)
        trainer.train(epochs=FIT_EPOCHS, device=FIT_DEVICE)

        # fit_irt_model.py lines 73-76, rows put back into bank order
        bp = trainer.best_params
        disc = [np.exp(i) for i in bp['disc']]
        diff = list(bp['diff'])
        item_ids = list(bp['item_ids'].values())
        pos = {iid: ix for ix, iid in enumerate(item_ids)}
        if len(pos) != n or any(f'item_{j}' not in pos for j in range(n)):
            raise RuntimeError('fluid: py-irt item set does not cover the bank')
        irt_model = np.array([[disc[pos[f'item_{j}']], diff[pos[f'item_{j}']]] for j in range(n)], float)

        abilities = np.asarray(bp['ability'], float)
        start_ability = float(np.mean(abilities))                        # ADAPTATION 4
        return {'irt_model': irt_model, 'start_ability': start_ability,
                'info': {'fit_seed': FIT_SEED, 'epochs': FIT_EPOCHS, 'priors': 'hierarchical',
                         'estimation_method': fluid_config.ESTIMATION_METHOD_IRT,
                         'n_obs': len(data.observations), 'K_cal': K, 'n_bank': n,
                         'a_mean': float(irt_model[:, 0].mean()), 'a_min': float(irt_model[:, 0].min()),
                         'a_max': float(irt_model[:, 0].max()),
                         'b_mean': float(irt_model[:, 1].mean()), 'b_sd': float(irt_model[:, 1].std()),
                         'cal_abilities': abilities.tolist(), 'start_ability': start_ability}}

    def estimate(self, model, y, budgets, seed):
        irt_model = np.asarray(model['irt_model'], float)
        n = irt_model.shape[0]
        guard = _Responses(y)
        n_max = int(max(budgets))
        out = engine.run_fluid_benchmarking(                             # official selection + MAP ability
            lm_responses=guard, irt_model=irt_model, start_ability=model['start_ability'],
            n_max=n_max, method=fluid_config.ESTIMATION_METHOD_IRT)
        items_all, abil_all = out['items_fb'], out['abilities_fb']
        a, b = irt_model[:, 0], irt_model[:, 1]
        res = {}
        for B in budgets:
            B = int(B)
            items = [int(i) for i in items_all[:B]]
            theta = float(abil_all[B - 1])
            yS = guard.values_at(items)
            unsel = np.ones(n, bool)
            unsel[items] = False
            P = irt_utils.sigmoid_stable(1.0 * a[unsel] * (theta - b[unsel]))   # D = 1 as in the engine
            est = float((yS.sum() + P.sum()) / n)                        # ADAPTATION 5
            rec = {'est': est, 'items': items,
                   'variants': {'ability': theta, 'sample_mean': float(yS.mean())}}
            if len(items) < B:
                rec['note'] = f'bank has only {len(items)} items'
            res[B] = rec
        return res

    def stop(self, model, y, seed):
        return {}                                                         # ADAPTATION 9


METHOD = Fluid()
