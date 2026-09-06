"""DISCO -- Diversifying Sample Condensation (Rubinstein et al., 2025,
arXiv:2510.07959) run through the OFFICIAL code,
/data2/jeongtae/official_baselines/disco-public (arubique/disco-public).

The whole repository is importable in the official environment (after
`pip install jsonlines h5py`), so every algorithmic step below is a call into
the original module; only the data plumbing around it is ours.

What is called (file:function)
  fit / threshold  experiments.py:189-206 -- the response threshold c is searched over
                   `cs = np.linspace(0.01, 0.99, 1000)` exactly as written (with nanmean,
                   ADAPTATION 3).  Our scores are already 0/1, so responses == scores for
                   every c in that grid (asserted; info['responses_equal_scores']).
  fit / D          experiments.py:222-305 "opt D" block -- for D in Ds = [2, 5, 10, 15]
                   (run_experiment.py:357): irt.py:create_irt_dataset ->
                   irt.py:train_irt_model (the `py-irt train 'multidim_2pl' ... --priors
                   'hierarchical' --seed 42 --deterministic` CLI, epochs = 2000, lr = 0.1
                   from experiments.py:134-135, device = 'cpu') -> irt.py:load_irt_parameters,
                   then irt.py:estimate_ability_parameters on the validation planners
                   (seen = even items, unseen = odd items) and
                   `ind_D = np.argmax(np.array(errors) - np.min(errors) < 0.0025)`
                   (experiments.py:300-301).  The final model is refit on all K_cal
                   planners at that D (experiments.py:340-348).
  fit / lambdas    experiments.py:307-337 with utils.py:526 get_lambda -- the G-PIRT
                   lambda grid; our sampling name is not one of the four keys, so
                   acc.py:600-607 falls back to 'anchor-irt_gpirt', i.e.
                   get_lambda(b, v / (4 * number_item)).
  select           selection.py:get_disagreement_scores (-> selection.py:pds / selection.py:jsd)
                   and selection.py:sample_by_disagreement, called with the headline
                   configuration of the paper's IRT-comparison row (sheets/train_eval.csv
                   line 3): sampling_name 'high-disagreement@100+nonstratified',
                   disagreement_type 'pds' (run_experiment.py:260), i.e.
                   high_first = True (selection.py:sample_items line "high_first=('high' in
                   sampling_name)") and the non-stratified branch (selection.py:216-227).
                   n_guiding_models is None here (ADAPTATION 6).
  estimate/naive   acc.py:494-497 -- (item_weights * y[seen]).sum(), item_weights =
                   ones(B)/B, i.e. the mean of the administered outcomes.
  estimate/pirt    acc.py:566-586 -- irt.py:estimate_ability_parameters at the seen items,
                   then acc.py:compute_acc_pirt(thresh=None).
  estimate/cirt    acc.py:587-598 -- acc.py:compute_acc_pirt(thresh=0.5).
  estimate/gpirt   acc.py:600-619 -- lambd * naive + (1 - lambd) * pirt.
  estimate/KNN     acc.py:522-529 -- acc.py:compute_acc_knn on the official embeddings.
  estimate/fitted  experiments.py:compute_embedding (via experiments.py:
                   make_train_test_model_embeddings, pca = 256 as in the headline runs,
                   ADAPTATION 7) and experiments.py:make_fitted_model (the `model.fit(...)`
                   of experiments.py:817-819) for acc.py:BEST_FITTING_METHODS; the
                   prediction is acc.py:558-560,
                   `fitted_model.predict(test_model_embedding.numpy().reshape(1, -1))[0]`.

Headline readout ('est')
  Table 1 of the paper is built by scripts/plot_tables.py:make_table_1, whose DISCO row is
  `("DISCO (ours)", "High PDS", "fit", "highest", "fit")`.  "High PDS" is the
  high-disagreement selection with disagreement_type 'pds'; "fit" is produced by
  extract_data_for_table_1 (scripts/plot_tables.py:545-573), which first drops
  MLP3_e700_lr0.001, Ridge_10, Lasso_e-4 and GradientBoostingRegressor_200
  ("# keep only Random Forest", l.560) and then takes the row-wise min over what is left --
  so the shipped 'fit' column is exactly RandomForestRegressor_100.  'est' is therefore
  pds selection + RandomForestRegressor(n_estimators=100) fitted on the calibration
  planners' anchor embeddings.  ("highest" is a groupby-min over the eight high-* selection
  variants; only one of them -- non-stratified, all guiding models -- is defined for a
  single-scenario bank of K_cal <= 12 planners, so no oracle minimum is taken here.)

  This headline predictor is a *signature metamodel*: it regresses the source models'
  true success rates on their responses at the anchors, and predicts the evaluation
  planner's success rate from its own anchor responses.  With K_cal in {4, 8, 12} source
  planners it has 4-12 training points, and a random forest on 4-12 points is close to
  the constant `mean_train_score`; info['fit_train_rmse'] and the 'mean_train_score'
  variant make that visible per cell.  It is reported as 'est' because it is the paper's
  headline, not because it is the best readout available -- 'pds_pirt' is DISCO's
  selection with tinyBenchmarks' p-IRT plug-in and is the fair-comparison row.

VARIANTS
  '<sel>_<est>' for sel in {'pds', 'jsd'} and est in {'naive', 'pirt', 'cirt', 'gpirt',
  'KNN', 'Ridge_10', 'Lasso_e-4', 'RandomForestRegressor_100',
  'GradientBoostingRegressor_200', 'MLP3_e700_lr0.001'} (the four base IRT estimators of
  run_experiment.py:241 plus every estimator of acc.py:BEST_FITTING_METHODS), minus
  'pds_RandomForestRegressor_100' which is 'est'.  Plus 'mean_train_score' (acc.py:416,
  the mean of the calibration planners' success rates -- a constant estimator that reads
  no outcome at all, kept as the degeneracy reference for the fitted predictors).
  'perfect_knn' (acc.py:646) is NOT offered: it needs the evaluation planner's TRUE
  success rate and is an oracle.

  The 'jsd_*' variants read the jsd top-B routes, which are a DIFFERENT subset of the bank
  of the same size B -- selection.py:jsd on one-hot outcomes is the binary entropy of the
  route's pass rate, whereas selection.py:pds on one-hot outcomes takes only two values
  (see ADAPTATION 5).  A cell that reports a jsd variant therefore drove
  |pds_B union jsd_B| <= 2B routes; 'items' and 'est' are the pds ones, and
  diag['n_read_union'] records the union size.  selftest_disco.py checks that 'est' and
  every 'pds_*' variant are bit-identical when the jsd-only routes are flipped.

  stop() returns {} -- DISCO publishes no stopping rule (it is a fixed-subset method;
  run_experiment.py takes --number_items as a fixed grid).

ADAPTATIONS
1. run_experiment.py:121-187 (load_and_split_model_outputs) / utils.py:686-748
   (prepare_and_split_data): the data loader reads data/model_outputs.pickle and splits
   models into source (train) and target (test) rows.  Here the protocol cell fixes them:
   source models = the K_cal calibration planners (R), target model = the single
   evaluation planner (y).  Single scenario, single subscenario:
   chosen_scenarios = ['b2d'], scenarios = {'b2d': ['b2d']},
   scenarios_position = {'b2d': range(n_bank)}, subscenarios_position =
   {'b2d': {'b2d': range(n_bank)}} and balance_weights = ones(n_bank) -- which is exactly
   what utils.py:718-726 computes for one subscenario (N / (1 * N) = 1).  Unavoidable:
   the protocol fixes who is source and who is target, and a bank of driving routes has
   no subscenario structure.
2. irt.py:19-40 (create_irt_dataset) writes `int(responses[i, j])` for every (i, j) and
   has no missing-value code.  A NaN cell (no record for that planner x route) is
   therefore dropped from the jsonlines after create_irt_dataset has written it --
   py-irt's format lists only the (subject, item) pairs that exist, so omission is the
   only way to express "no record".  Because omission changes py-irt's item numbering
   (items are numbered by first appearance), the loaded A and B are permuted back into
   bank order through best_parameters.json's 'item_ids'; irt.py:44-60
   (load_irt_parameters) assumes a complete matrix and returns them in py-irt order.
   No bank route is unrecorded by every calibration planner in the 192 protocol cells
   (checked over every draw x K_cal x slot), so every item keeps its parameters.
3. experiments.py:189-206 (threshold search) and experiments.py:277-296 / 313-315
   (validation errors, `vs`) use `.mean()` / `np.var()` over the source matrix, which is
   complete in their data; NaN cells make those NaN here, so the same expressions are
   evaluated with np.nanmean / np.nanvar.  experiments.py:265-266 similarly passes
   `seen_items` (all even items) to estimate_ability_parameters for every validation
   planner; a validation planner's unrecorded even items are dropped from its seen list
   (the official likelihood would otherwise be NaN).
4. experiments.py:262-272 parallelises the validation-theta estimates with mp.Pool(cpu);
   with 1-3 validation planners per cell the wrapper calls
   irt.py:estimate_ability_parameters in a plain loop (identical arguments, identical
   results, no process pool for 192 cells).
5. Class probabilities.  utils.py:create_predictions builds predictions of shape
   (n_models, n_items, n_classes) from the leaderboard's per-answer logits.  A driving
   rollout emits pass/fail only, so predictions_train is the ONE-HOT of the binary
   outcome, [1 - r, r], with n_classes = 2 (unrecorded cells are filled with the route's
   pass rate over the planners that do have a record, [1 - pbar, pbar]; the fill never
   enters the disagreement scores, see ADAPTATION 6).  Consequence, reported in
   info['pds_n_disagreeing']: selection.py:pds on one-hot inputs is
   1 + 1{the route is not unanimous}, so the pds order is "every route on which the
   calibration planners disagree, in bank order, then the unanimous ones, in bank order"
   (Python's `sorted` is stable, so the tie order is deterministic).  selection.py:jsd on
   the same inputs is the binary entropy of the route's pass rate and is graded.  This is
   a property of binary outcomes, not a change to their code.
6. selection.py:get_disagreement_scores lines 279-283 subsample n_guiding_models source
   models with np.random.choice; the headline row uses @100, and we have at most 12
   calibration planners, so n_guiding_models is None (= use all source models, the
   `high-disagreement+nonstratified` row of sheets/train_eval.csv) and that sampling
   branch never runs.  Because a NaN cell has no one-hot to contribute,
   get_disagreement_scores is called once per distinct availability pattern, each time on
   the routes with that pattern and only the planners that recorded them (the task's
   "use the available models only"); the official function itself is unchanged.
7. experiments.py:622-658 (compute_embedding) uses PCA(n_components=pca, svd_solver='full')
   with pca = 256 in every headline run.  sklearn requires n_components <=
   min(n_samples, n_features) = min(K_cal, 2B) = K_cal here, so n_components is
   min(256, K_cal, 2B) = K_cal (reported in info['pca_n_components']).  apply_softmax is
   left at the official default True (compute_embedding softmaxes unconditionally, even
   though our inputs are already probabilities); it is a fixed monotone map of the two
   one-hot values, so it does not change which routes a tree splits on.
8. experiments.py:713-767 (make_fitted_weights) wraps every fit in
   stnd.utility.data_utils.make_or_load_from_cache, keyed by
   f"fitted_model_{model_name}_{sampling_name}_{number_item}_{it}" -- a key that carries
   nothing about the protocol cell, so with cache_path resolved from the STAI-tuned
   project root it would hand cell (s, K, slot) the regressor fitted for another cell.
   The wrapper calls experiments.py:make_fitted_model directly (which is exactly what a
   cache miss does) and experiments.py:make_train_test_model_embeddings directly (it has
   no internal cache).  The same caching wrapper is why the disagreement scores and the
   D choice are computed in fit() rather than through make_or_load_from_cache.
9. acc.py:387-622 (calculate_accuracies) cannot be called as written: lines 503-515 and
   535-552 index accs[...]['mean_train_score'] and accs[...]['perfect_knn']
   unconditionally, and compute_perfect_knn (acc.py:646-656) needs the evaluation
   planner's TRUE success rate -- reading it would leak the target of the protocol.  The
   per-estimator statements (naive l.494-497, KNN l.522-529, fitted l.555-563, pirt/cirt
   l.566-598, gpirt l.600-619) are therefore executed here one by one, each calling the
   same official function with the same arguments; only 'perfect_knn' is dropped.
10. MLPRegressor (models.py:32-90, acc.py:35-47) defaults to device='cuda'; the official
    environment here is torch-cpu, so 'MLP3_e700_lr0.001' is built with device='cpu'
    (same hidden_channels [128, 128, 1], n_epochs 700, lr 0.001, AdamW, MSELoss).  It is
    a variant only -- the shipped Table 1 'fit' column is RandomForestRegressor_100.
11. Seeding.  irt.py:78 hard-codes `--seed 42 --deterministic` inside the py-irt command
    string, so THEIR seed is kept for every IRT fit (info['pyirt_seed'] = 42).  Everything
    else in run_experiment.py is seeded once by
    stnd.utility.utils.apply_random_seed(RANDOM_SEED = 42) (run_experiment.py:323,
    experiments.py:51) -- the sklearn regressors are constructed without random_state
    (acc.py:259-271) and draw from the global numpy RNG.  The wrapper calls the same
    apply_random_seed at the top of fit() and of estimate(), with the harness's cell seed
    instead of the module constant 42, so that 192 cells are independent and each is
    reproducible.  Nothing else consumes randomness (the selection is a stable sort, the
    PCA uses svd_solver='full' with random_state=RANDOM_SEED inside compute_embedding).
12. Logging only: irt.py:train_irt_model shells out with os.system, and py-irt prints a
    rich progress table per epoch block; file descriptors 1 and 2 are redirected to
    /dev/null around the call (5 IRT fits x 192 cells).  The official SuppressPrints
    context (utils.py:530) only swaps sys.stdout and cannot silence a subprocess.
    experiments.py:make_fitted_model's training-RMSE print is captured the same way and
    the value is recomputed into info['fit_train_rmse'].
13. `py-irt` must be on PATH for irt.py:train_irt_model's os.system call; the official
    environment's bin directory (base.OFFICIAL_PY.parent) is prepended at import time.
14. Divergence of the multidim 2PL.  py-irt's hierarchical SVI does not always converge on
    a panel this small: on cell (0, 12, 0) the D = 10 fit on the 9 validation-split
    planners ends with `ValueError: Expected parameter concentration ... nan` and py-irt
    writes no best_parameters.json.  irt.py:train_irt_model swallows the exit code
    (os.system) and experiments.py:243-256 would then crash in load_irt_parameters.  A D
    whose fit diverges is dropped from the candidate list before
    `ind_D = np.argmax(errors - min(errors) < 0.0025)` is applied (it cannot be chosen, so
    the rule is unchanged on the Ds that do exist), and if the FINAL all-planner fit at the
    chosen D diverges the remaining candidates are tried in the same order.  Every dropped
    D is recorded in info['D_diverged'] / info['D_final_diverged']; the cell fails loudly
    if no D survives.  The G-PIRT `b` of experiments.py:315 stays `errors2[ind_D]`, the
    validation error of the D the official rule picked, exactly as written -- the fallback
    changes which model is refit, not which validation error the lambda is built from.
"""
import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

from .base import OFFICIAL_PY, OFFICIAL_ROOT, LeakGuard, OfficialMethod

DISCO_ROOT = OFFICIAL_ROOT / 'disco-public'
if str(DISCO_ROOT) not in sys.path:
    sys.path.insert(0, str(DISCO_ROOT))
# ADAPTATION 13: irt.train_irt_model shells out to the `py-irt` CLI.
if str(OFFICIAL_PY.parent) not in os.environ.get('PATH', '').split(os.pathsep):
    os.environ['PATH'] = f'{OFFICIAL_PY.parent}{os.pathsep}' + os.environ.get('PATH', '')

import acc as disco_acc                                          # noqa: E402  official acc.py
import irt as disco_irt                                          # noqa: E402  official irt.py
import selection as disco_selection                              # noqa: E402  official selection.py
from utils import get_lambda                                     # noqa: E402  official utils.py
from stnd.utility.utils import apply_random_seed                 # noqa: E402  run_experiment.py:30


def _load_official_experiments():
    """Official experiments.py, loaded by path so that the generic module name
    'experiments' is not claimed (the ATDrive repo has an experiments/ directory)."""
    if 'disco_experiments' in sys.modules:
        return sys.modules['disco_experiments']
    spec = importlib.util.spec_from_file_location('disco_experiments', DISCO_ROOT / 'experiments.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['disco_experiments'] = mod
    spec.loader.exec_module(mod)
    return mod


disco_experiments = _load_official_experiments()

SCENARIO = 'b2d'
DS = [2, 5, 10, 15]                 # run_experiment.py:357
EPOCHS = 2000                       # experiments.py:134
LR = 0.1                            # experiments.py:135
DEVICE = 'cpu'
PYIRT_SEED = 42                     # irt.py:78 (theirs, kept -- ADAPTATION 11)
PCA_DIM = 256                       # sheets/train_eval.csv delta:kwargs/pca
SAMPLING_NAME = 'high-disagreement+nonstratified'
HIGH_FIRST = True                   # selection.py:sample_items -> high_first=('high' in sampling_name)
DISAGREEMENT_KEY = SAMPLING_NAME.split('-')[1]      # selection.py:186 -> 'disagreement'
SELECTIONS = ('pds', 'jsd')         # run_experiment.py:260 default is 'pds' (the headline)
HEADLINE_EST = 'RandomForestRegressor_100'          # scripts/plot_tables.py: the 'fit' column
FITTING_METHODS = list(disco_acc.BEST_FITTING_METHODS)   # MLP3_e700 + BEST_LINEAR_METHODS


@contextlib.contextmanager
def _quiet():
    """ADAPTATION 12: silence the py-irt subprocess and the fitting prints."""
    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, 'w') as devnull:
        keep = os.dup(1), os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(keep[0], 1)
            os.dup2(keep[1], 2)
            os.close(keep[0])
            os.close(keep[1])


def _positions(n):
    """ADAPTATION 1: the single-scenario / single-subscenario structures."""
    pos = list(range(n))
    return ([SCENARIO], {SCENARIO: [SCENARIO]}, {SCENARIO: pos},
            {SCENARIO: {SCENARIO: pos}}, np.ones(n))


def _write_dataset(responses, path):
    """official irt.create_irt_dataset, then drop the unrecorded cells (ADAPTATION 2)."""
    disco_irt.create_irt_dataset(np.where(np.isnan(responses), 0, responses), str(path))
    rows = [json.loads(l) for l in open(path)]
    kept = 0
    for i, row in enumerate(rows):
        row['responses'] = {k: v for k, v in row['responses'].items()
                            if not np.isnan(responses[i, int(k[1:])])}
        kept += len(row['responses'])
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    return kept


def _train_and_load(responses, model_dir, D, n):
    """experiments.py:243-256 / 340-348: load if already fitted, else train, then load."""
    model_dir = str(model_dir) + os.sep
    ds = Path(model_dir).parent / (Path(model_dir).name.rstrip(os.sep) + '.jsonlines')
    try:
        return _load_params(model_dir, n)
    except Exception:
        pass
    _write_dataset(responses, ds)
    with _quiet():
        disco_irt.train_irt_model(str(ds), model_dir, D, LR, EPOCHS, DEVICE)
    return _load_params(model_dir, n)


def _load_params(model_dir, n):
    """official irt.load_irt_parameters + permutation back to bank order (ADAPTATION 2)."""
    A, B, Theta = disco_irt.load_irt_parameters(model_dir)
    p = json.load(open(model_dir + 'best_parameters.json'))
    order = [None] * n
    for idx, name in p['item_ids'].items():
        order[int(str(name)[1:])] = int(idx)
    if any(o is None for o in order):
        raise RuntimeError('disco: py-irt item set does not cover the bank (ADAPTATION 2)')
    sub = [None] * len(p['subject_ids'])
    for idx, name in p['subject_ids'].items():
        sub[int(name)] = int(idx)
    return A[:, :, order], B[:, :, order], Theta[sub]


def _disagreement(predictions_train, recorded, kind):
    """official selection.get_disagreement_scores, per availability pattern (ADAPTATION 6)."""
    n = recorded.shape[1]
    scores = np.full(n, np.nan)
    groups = {}
    for j in range(n):
        groups.setdefault(tuple(np.where(recorded[:, j])[0].tolist()), []).append(j)
    for models, cols in groups.items():
        if not models:
            raise RuntimeError('disco: bank route with no calibration record (ADAPTATION 6)')
        sub = predictions_train[list(models)][:, cols, :]
        scores[cols] = disco_selection.get_disagreement_scores(sub, None, disagreement_type=kind)
    return scores


class Disco(OfficialMethod):
    name = 'disco'
    adaptive = False            # fixed subset; the budget grid is nested (stable sort)

    # ---------------------------------------------------------------- fit ---
    def fit(self, R, seed, workdir):
        apply_random_seed(seed)                                   # ADAPTATION 11
        R = np.asarray(R, float)
        K, n = R.shape
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        chosen_scenarios, scenarios, scenarios_position, subscenarios_position, balance_weights = _positions(n)
        recorded = ~np.isnan(R)
        if not recorded.any(0).all():
            raise RuntimeError('disco: bank route with no calibration record')

        # --- experiments.py:189-206, threshold -> responses (ADAPTATION 3) ---
        scores_train = R
        ind = scenarios_position[SCENARIO]
        cs = np.linspace(0.01, 0.99, 1000)
        c = cs[np.argmin([np.mean(np.abs(np.nanmean(np.where(recorded[:, ind], scores_train[:, ind] > c, np.nan), axis=1)
                                         - np.nanmean(scores_train[:, ind], axis=1))) for c in cs])]
        responses_train = np.where(recorded, (scores_train > c).astype(float), np.nan)
        same = bool(np.array_equal(responses_train[recorded], scores_train[recorded]))

        # --- predictions_train: one-hot class probabilities (ADAPTATION 5) ---
        pbar = np.nanmean(R, axis=0)
        Rfill = np.where(recorded, R, pbar[None, :])
        predictions_train = np.stack([1.0 - Rfill, Rfill], axis=-1)

        # --- selection.py:get_disagreement_scores (ADAPTATION 6) ------------
        dis = {k: _disagreement(predictions_train, recorded, k) for k in SELECTIONS}

        # --- experiments.py:222-305, "opt D" -------------------------------
        val_ind = list(range(0, K, 5))
        train_ind = [i for i in range(K) if i not in val_ind]
        seen_items = list(range(0, n, 2))
        unseen_items = list(range(1, n, 2))
        errors, errors2, cand, failed = [], [], [], {}
        for D in DS:
            try:                                                   # ADAPTATION 14
                A, Bp, _ = _train_and_load(responses_train[train_ind], workdir / f'irt_D{D}_val', D, n)
            except Exception as e:
                failed[D] = f'{type(e).__name__}: {e}'[:200]
                continue
            cand.append(D)
            thetas = []
            for j in range(len(val_ind)):                          # ADAPTATION 4
                rj = responses_train[val_ind][j]
                sj = [s for s in seen_items if not np.isnan(rj[s])]   # ADAPTATION 3
                thetas.append(disco_irt.estimate_ability_parameters(rj, sj, A, Bp))
            errors2.append([])
            for scenario in chosen_scenarios:
                iu = [u for u in unseen_items if u in scenarios_position[scenario]]
                errors2[-1].append(np.mean([
                    abs((balance_weights * disco_experiments.item_curve(thetas[j], A, Bp))[0, iu].mean()
                        - np.nanmean(scores_train[val_ind][j, iu]))
                    for j in range(len(val_ind))]))
            errors.append(np.mean(errors2[-1]))
        if not cand:
            raise RuntimeError(f'disco: py-irt diverged for every D in {DS}: {failed}')
        ind_D = int(np.argmax(np.array(errors) - np.min(errors) < 0.0025))   # experiments.py:300
        D = cand[ind_D]

        # --- experiments.py:313-316, the two ingredients of the G-PIRT lambda ---
        vs, bs = {}, {}
        for i, scenario in enumerate(chosen_scenarios):
            vs[scenario] = float(np.nanvar(scores_train[:, scenarios_position[scenario]], axis=1).mean())
            bs[scenario] = float(np.mean(errors2[ind_D][i]))

        # --- experiments.py:340-348, final IRT model ------------------------
        order = [cand[ind_D]] + [d for d in cand if d != cand[ind_D]]        # ADAPTATION 14
        final_failed = {}
        for Dq in order:
            try:
                A, Bp, Theta = _train_and_load(responses_train, workdir / f'irt_D{Dq}_final', Dq, n)
                D = Dq
                break
            except Exception as e:
                final_failed[Dq] = f'{type(e).__name__}: {e}'[:200]
        else:
            raise RuntimeError(f'disco: the final py-irt fit diverged for every D: {final_failed}')

        # --- acc.py:compute_true_acc on the source models (ADAPTATION 3) ----
        train_model_true_accs = {j: {SCENARIO: float(np.nanmean((balance_weights[None, :] * scores_train)[j, ind]))}
                                 for j in range(K)}

        info = {'K_cal': K, 'n_bank': int(n), 'nan_cells': int((~recorded).sum()),
                'threshold_c': float(c), 'responses_equal_scores': same,
                'Ds': DS, 'D_candidates': cand, 'D_errors': [float(e) for e in errors],
                'D_diverged': failed, 'D_final_diverged': final_failed, 'D': int(D), 'ind_D': ind_D,
                'pyirt_seed': PYIRT_SEED, 'epochs': EPOCHS, 'lr': LR, 'device': DEVICE,
                'priors': 'hierarchical', 'model_type': 'multidim_2pl',
                'sampling_name': SAMPLING_NAME, 'high_first': HIGH_FIRST,
                'n_guiding_models': None, 'stratified': False,
                'headline_estimator': HEADLINE_EST,
                'pds_n_disagreeing': int((dis['pds'] > 1.5).sum()),
                'pds_distinct_values': int(np.unique(np.round(dis['pds'], 9)).size),
                'jsd_max': float(np.nanmax(dis['jsd'])),
                'gpirt_b': float(bs[SCENARIO]), 'gpirt_v': float(vs[SCENARIO]),
                'gpirt_lambda_B30': float(get_lambda(bs[SCENARIO], vs[SCENARIO] / (4 * 30))),
                'a_mean': float(A.mean()), 'a_sd': float(A.std()),
                'b_mean': float(Bp.mean()), 'b_sd': float(Bp.std()),
                'cal_abilities_mean': float(np.asarray(Theta).mean()),
                'train_true_acc_mean': float(np.mean([v[SCENARIO] for v in train_model_true_accs.values()]))}
        return {'A': A, 'B': Bp, 'Theta': Theta, 'dis': dis, 'gpirt_b': bs, 'gpirt_v': vs,
                'predictions_train': predictions_train, 'train_true_accs': train_model_true_accs,
                'n': int(n), 'K': int(K), 'info': info}

    # ----------------------------------------------------------- estimate ---
    def estimate(self, model, y, budgets, seed):
        apply_random_seed(seed)                                    # ADAPTATION 11
        A, Bp = np.asarray(model['A']), np.asarray(model['B'])
        n, K = model['n'], model['K']
        budgets = [int(b) for b in budgets]
        chosen_scenarios, scenarios, scenarios_position, subscenarios_position, balance_weights = _positions(n)
        predictions_train = np.asarray(model['predictions_train'])
        train_true = {int(k): v for k, v in model['train_true_accs'].items()}
        train_true_list = [train_true[i] for i in range(K)]

        # --- experiments.py:317-336, the G-PIRT lambda grid ------------------
        opt_lambds = {k: {SCENARIO: {}} for k in ('random_gpirt', 'anchor_gpirt',
                                                  'anchor-irt_gpirt', 'adaptive_gpirt')}
        b_, v_ = model['gpirt_b'][SCENARIO], model['gpirt_v'][SCENARIO]
        for key in opt_lambds:
            for B in budgets:
                opt_lambds[key][SCENARIO][B] = (get_lambda(b_, v_ / B) if key == 'random_gpirt'
                                                else get_lambda(b_, v_ / (4 * B)))

        # --- selection.py:sample_by_disagreement, per selection and budget ---
        seen_items_dic, unseen_items_dic, item_weights_dic = {}, {}, {}
        for sel in SELECTIONS:
            seen_items_dic[sel], unseen_items_dic[sel], item_weights_dic[sel] = {}, {}, {}
            for B in budgets:
                iw, seen, unseen = disco_selection.sample_by_disagreement(
                    SAMPLING_NAME, chosen_scenarios, scenarios, B, subscenarios_position,
                    num_samples_in_test=n, predictions_train=predictions_train,
                    balance_weights=balance_weights,
                    disagreement_scores_dict={DISAGREEMENT_KEY: model['dis'][sel]},
                    random_seed=0, high_first=HIGH_FIRST)
                seen_items_dic[sel][B] = {0: [int(s) for s in seen]}
                unseen_items_dic[sel][B] = {0: [int(u) for u in unseen]}
                item_weights_dic[sel][B] = {0: iw}

        # --- the only reads of the evaluation planner's outcomes ------------
        guard = LeakGuard(y)
        read = sorted({i for sel in SELECTIONS for B in budgets for i in seen_items_dic[sel][B][0]})
        yv = np.full(n, np.nan)
        yv[read] = guard.values_at(read)
        predictions_test = np.stack([1.0 - yv, yv], axis=-1)[None, :, :]

        # --- experiments.py:make_train_test_model_embeddings (ADAPTATION 7/8) ---
        pca = int(min(PCA_DIM, K, 2 * min(budgets)))
        emb_tr, emb_te = {}, {}
        for sel in SELECTIONS:
            with _quiet():                                          # ADAPTATION 12 (tqdm bars)
                tr, te = disco_experiments.make_train_test_model_embeddings({
                    'sampling_names': [SAMPLING_NAME], 'number_items': budgets, 'iterations': 1,
                    'predictions_train': predictions_train, 'predictions_test': predictions_test,
                    'seen_items_dic': {SAMPLING_NAME: seen_items_dic[sel]}, 'pca': pca,
                    'apply_softmax': True})
            emb_tr[sel], emb_te[sel] = tr[SAMPLING_NAME], te[SAMPLING_NAME]

        # --- experiments.py:make_fitted_model per selection/budget/method ----
        fitted, rmse = {}, {}
        for sel in SELECTIONS:
            fitted[sel], rmse[sel] = {}, {}
            for B in budgets:
                X = emb_tr[sel][B][0].numpy()
                yt = np.array([train_true_list[i][SCENARIO] for i in range(K)])
                fitted[sel][B], rmse[sel][B] = {}, {}
                for mname, builder in FITTING_METHODS:
                    kw = dict(builder[1])
                    if builder[0] is disco_acc.MLPRegressor:
                        kw['device'] = 'cpu'                        # ADAPTATION 10
                    with _quiet():
                        fm = disco_experiments.make_fitted_model({
                            'builder': (builder[0], kw), 'sampling_name': SAMPLING_NAME,
                            'number_item': B, 'it': 0, 'cur_train_models_embeddings_np': X,
                            'train_model_true_accs_np': yt, 'fitted_weights': {}, 'model_name': mname})
                    fitted[sel][B][mname] = fm
                    rmse[sel][B][mname] = float(np.sqrt(np.mean((fm.predict(X) - yt) ** 2)))

        mean_train_score = float(np.nanmean(np.asarray([train_true_list[i][SCENARIO] for i in range(K)])))
        out = {}
        for B in budgets:
            variants, diag = {}, {}
            for sel in SELECTIONS:
                seen = seen_items_dic[sel][B][0]
                unseen = unseen_items_dic[sel][B][0]
                iw = item_weights_dic[sel][B][0]
                sub = [s for s in seen if s in scenarios_position[SCENARIO]]
                # acc.py:494-497
                naive = float((iw[SCENARIO] * yv[sub]).sum())
                # acc.py:481-483, 566-598
                theta = disco_irt.estimate_ability_parameters(yv, seen, A, Bp)
                data_part = float((balance_weights * yv)[sub].mean())
                pirt = float(disco_acc.compute_acc_pirt(data_part, SCENARIO, scenarios_position,
                                                        seen, unseen, A, Bp, theta, balance_weights, thresh=None))
                cirt = float(disco_acc.compute_acc_pirt(data_part, SCENARIO, scenarios_position,
                                                        seen, unseen, A, Bp, theta, balance_weights, thresh=0.5))
                lambd = opt_lambds['anchor-irt_gpirt'][SCENARIO][B]              # acc.py:600-607
                gpirt = float(lambd * naive + (1 - lambd) * pirt)
                # acc.py:522-529
                emb_j = emb_te[sel][B][0][0]                                      # acc.py:517-519 (j = 0)
                knn = float(disco_acc.compute_acc_knn(emb_j, emb_tr[sel][B][0],
                                                      SCENARIO, train_true_list))
                variants[f'{sel}_naive'] = naive
                variants[f'{sel}_pirt'] = pirt
                variants[f'{sel}_cirt'] = cirt
                variants[f'{sel}_gpirt'] = gpirt
                variants[f'{sel}_KNN'] = knn
                for mname, _ in FITTING_METHODS:                                  # acc.py:555-563
                    variants[f'{sel}_{mname}'] = float(
                        fitted[sel][B][mname].predict(emb_j.numpy().reshape(1, -1))[0])
                diag[f'{sel}_theta'] = [float(t) for t in np.asarray(theta).reshape(-1)]
            variants['mean_train_score'] = mean_train_score
            est = variants.pop(f'pds_{HEADLINE_EST}')
            items = seen_items_dic['pds'][B][0]
            diag['jsd_items'] = seen_items_dic['jsd'][B][0]
            diag['n_read_union'] = len(set(items) | set(diag['jsd_items']))
            diag['gpirt_lambda'] = float(opt_lambds['anchor-irt_gpirt'][SCENARIO][B])
            diag['fit_train_rmse'] = {f'{s}_{m}': rmse[s][B][m] for s in SELECTIONS for m, _ in FITTING_METHODS}
            note = (f'est = pds selection + {HEADLINE_EST} (the shipped Table 1 "fit" column), '
                    f'a signature metamodel fitted on {K} source planners; jsd_* variants read '
                    f'their own top-{B} routes')
            out[B] = {'est': est, 'items': items, 'variants': variants, 'note': note, 'diag': diag}
        return out

    def stop(self, model, y, seed):
        return {}                        # DISCO publishes no stopping rule


METHOD = Disco()
