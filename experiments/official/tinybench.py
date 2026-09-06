"""tinyBenchmarks (Maia Polo et al., 2024) run through the OFFICIAL tutorial
pipeline, /data2/jeongtae/official_baselines/tinyBenchmarks (felipemaiapolo/
tinyBenchmarks, commit e9a8b10), tutorials/{training_irt, anchor_points,
estimating_performance}.ipynb with tutorials/irt.py + tutorials/utils.py.

The three notebooks are executed in order on every protocol cell, with the
whole bank playing the role of ONE scenario (so balance_weights == 1, the
notebooks' MMLU sub-scenario weighting is vacuous), the K_cal calibration
planners playing Y_train (the models the IRT model and the anchors are learnt
from) and the evaluation planner playing one row of Y_test.

What is called (file:function)
  fit    (a) tutorials/irt.py:create_irt_dataset  -- replicated with the NaN
             skip of ADAPTATION 1 (lines 9-33 otherwise verbatim)
         (a) tutorials/irt.py:train_irt_model     -- called verbatim (its own
             os.system("py-irt train 'multidim_2pl' ... --dims D --lr .1
             --epochs 2000 --priors 'hierarchical' --seed 42 --deterministic
             --log-every 200"), irt.py:50-51), once per D in Ds = [5, 10] on
             the validation-training subjects and once more on all calibration
             subjects with the chosen D -- training_irt.ipynb cells 24-29
         (a) tutorials/irt.py:load_irt_parameters (l.53-70)
         (a) tutorials/irt.py:estimate_ability_parameters (l.74-110, BFGS MLE)
             and tutorials/utils.py:item_curve (l.18-32) for the D-validation
             error of training_irt.ipynb cell 24
  select (b) sklearn.cluster.KMeans(n_clusters=B, n_init="auto",
             random_state=42).fit(X, sample_weight=norm_balance_weights) and
             sklearn.metrics.pairwise.pairwise_distances(centers, X,
             metric='euclidean').argmin(axis=1) on
             X = np.vstack((A.squeeze(), B.squeeze().reshape((1,-1)))).T
             -- anchor_points.ipynb cell 27, clustering = 'irt', verbatim
  est    (c) anchor  = (Y_anchor*anchor_weights).sum()  -- anchor_points.ipynb
             cell 34 / estimating_performance.ipynb cell 26 (IRT-free)
         (c) p-IRT   = lambd*data_part + (1-lambd)*irt_part with
             lambd = Y_anchor.shape[1]/n_bank, data_part = the anchors' mean
             outcome and irt_part = item_curve(theta) averaged over the unseen
             items, theta = estimate_ability_parameters(y[anchors], A[.,.,
             anchors], B[.,.,anchors]) -- estimating_performance.ipynb cell 33
         (c) gp-IRT  = lambds*anchor + (1-lambds)*pirt with
             lambds = get_lambda(b, v/(4*number_item)) = b^2/(v/(4B) + b^2),
             b = the chosen D's validation error, v = the calibration
             planners' mean within-planner variance over the bank
             -- training_irt.ipynb cells 31-34 (which produce data/lambds.pickle)
             and estimating_performance.ipynb cell 38

Readouts
  est       gp-IRT (the paper's headline estimator).
  variants  'anchor' (anchor-weighted, IRT-free), 'pirt', 'gpirt' (= est).
  stop()    {} -- tinyBenchmarks is fixed-budget and publishes no stopping rule.

Not adaptive: each budget is its own KMeans, so the B = 30 anchors are not a
prefix of the B = 165 anchors and the budgets are not nested; the overlap is
reported (not asserted) by selftest_tinybench.py.

One property of the official readout is kept as published rather than
"fixed": utils.py:31 computes the item curve as
np.clip(a*theta - b, -30, 30).sum(axis=1) with a of shape (1, D, n) and b of
shape (1, 1, n), so the difficulty is subtracted once PER DIMENSION
(sum_d a_d theta_d - D*b), whereas the fitted multidim_2pl model's logit is
disc*ability - diff (one b). Every tinyBenchmarks readout that uses item_curve
-- the D-validation error and the p-IRT / gp-IRT estimates -- inherits this, so
the wrapper inherits it too.

Seeds: every random element of the official pipeline carries the official
code's own fixed seed -- py-irt --seed 42 --deterministic (irt.py:50) and
KMeans(random_state=42) (anchor_points.ipynb cell 3, `random_state = 42`).
The harness's cell `seed` is therefore unused; a cell is reproducible from its
data alone (verified by the determinism test of the self-test).

ADAPTATIONS
1. tutorials/irt.py:9-33 (create_irt_dataset) writes `int(responses[i, j])`
   for EVERY (subject, item) cell. A NaN of R (no rollout record; up to 38 of
   ~215*12 cells here) cannot be written as an integer, and py-irt's jsonlines
   format has no missing-value code, so the wrapper's copy of the function
   omits those keys from the subject's `responses` dict (py-irt accepts a
   partial response set: the observation list is built per (subject, item)
   pair present). Everything else -- the dict layout, `subject_id` as str(i),
   'q<j>' item names, the jsonlines writer -- is line-for-line the official
   function. Unavoidable: the official data (leaderboard correctness) is a
   complete matrix, ours is not.
2. tutorials/irt.py:50-51 shells out to whatever `py-irt` is first on PATH,
   and tutorials/README.md says it must be Polo's MODIFIED py-irt
   (github.com/felipemaiapolo/py-irt). That fork is not one of the six cloned
   baselines, so it was cloned to OFFICIAL_ROOT/py-irt-polo (commit dc3d2a0)
   and installed --no-deps into a --system-site-packages venv,
   /data2/jeongtae/envs/tinybench_pyirt, whose `py-irt` binary is put first on
   PATH for the duration of the (verbatim) train_irt_model call. This is not
   cosmetic: upstream py-irt 0.6.5 (the one in the atdrive_official bin dir)
   has a DIFFERENT multidim_2pl -- per-dimension difficulties
   (loc_diff shape (n_items, dims) instead of (n_items, 1)), logits
   disc*(ability - diff) instead of disc*ability - diff, and N(0, 1e6) priors
   instead of N(0, 1e1) -- so its output cannot be read by
   load_irt_parameters (which needs diff.T to be a (1, n_items) row), and on
   these cells it diverges to NaN and crashes at ~epoch 300. Polo's fork
   reproduces the shapes and key set of the repo's own
   tutorials/data/irt_model/best_parameters.json exactly. Polo's five commits
   on top of py-irt 0.4.10 touch only py_irt/models/multidim_2pl.py.
3. tutorials/irt.py:53-70 (load_irt_parameters) assumes row i of
   best_parameters.json is item i. py-irt numbers items by first appearance in
   the jsonlines file, which with ADAPTATION 1's omissions need not be bank
   order, so the wrapper reorders the disc/diff rows through
   best_parameters['item_ids'] before calling the official reshaping
   (l.67-69). With complete data this is the identity.
4. training_irt.ipynb cell 22 (binarization) is skipped: it picks the
   threshold c in linspace(.01, .99, 100) minimizing |(Y>c).mean - Y.mean|,
   and for a 0/1 matrix (Y > c) == Y for every c in that range, so the step is
   the identity on our data -- and running it as written would additionally
   turn the NaNs of ADAPTATION 1 into zeros.
5. training_irt.ipynb cell 24: the validation subjects are val_ind =
   range(0, K_cal, 5) and the IRT model is fitted on the remaining subjects,
   exactly as written, but K_cal is 4/8/12 rather than 295, so the validation
   set holds 1/2/3 planners and the fit 3/6/9. Nothing is changed; the numbers
   are just small, and 'val_ind'/'train_ind' are reported in model['info'].
6. training_irt.ipynb cell 24: `estimate_ability_parameters(Y_bin_train[
   val_ind][j][seen_items], ...)` is called with seen_items = the even bank
   indices RESTRICTED to the items that validation planner j actually has a
   record for (a NaN response would make the official neg_log_like NaN and
   BFGS return the initial theta); the unseen-item error term likewise uses
   np.nanmean over the odd indices instead of np.mean. Same consequence of the
   incomplete matrix as ADAPTATION 1.
7. training_irt.ipynb cell 34: `v = np.var(Y_train[:, pos], axis=1).mean()` is
   computed as np.nanvar(R, axis=1).mean() for the same reason; `number_item`
   is the budget B (in the notebook it is the number of anchor points, 100,
   which is exactly the role B plays here), so lambds is recomputed per
   budget. `b` is errors2[ind_D][0], the chosen D's validation error of the
   single scenario, as in the notebook.
8. anchor_points.ipynb cell 27 / estimating_performance.ipynb cells 26-38 loop
   over six scenarios; here the bank is one scenario, so balance_weights is
   np.ones(n_bank) and norm_balance_weights = 1/n_bank per item -- the
   notebook's own code for a single-subscenario scenario.
9. Duplicate anchors: `pairwise_distances(centers, X).argmin(1)` can return
   the same bank index for two clusters when the (a, b) embedding has
   duplicate rows (routes with identical calibration responses). The official
   code keeps the duplicates and the estimators use them with multiplicity
   (Y_anchor.shape[1] = number_item either way), so all three readouts are
   computed exactly as published; only the reported 'items' are deduplicated
   (the harness wants bank indices), which makes n_items < B in those cells --
   flagged in 'note' with the duplicate count. No item outside the anchor set
   is ever read.
10. The official modules are loaded from tutorials/ by absolute path and then
   removed from sys.modules, because 'irt' and 'utils' are also the module
   names of another cloned baseline (disco-public/{irt,utils}.py) that the
   harness imports into the same process. Plumbing only.
11. OMP_NUM_THREADS = MKL_NUM_THREADS = 1 for the py-irt subprocess only.
   irt.py:50 passes --deterministic, but torch's CPU reductions are not
   bit-reproducible across thread counts, so on a shared machine the same cell
   could otherwise return slightly different item parameters depending on the
   load. Pinning the subprocess to one thread makes a cell reproducible from
   its data alone (the determinism test of the self-test re-runs the three
   py-irt fits) and is also faster for a model this small. Nothing else about
   the command or the algorithm changes.
12. Console output of the py-irt CLI (irt.py:51 uses os.system, whose output
   cannot be captured by the caller) is redirected at the file-descriptor
   level into workdir/py_irt_<tag>.log, so that 192 cells x 3 fits do not
   flood the harness log. The command string is untouched.
"""
import contextlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

from official.base import LeakGuard, OFFICIAL_ROOT, OfficialMethod

TB_ROOT = OFFICIAL_ROOT / 'tinyBenchmarks'
TUTORIALS = TB_ROOT / 'tutorials'
PYIRT_FORK = OFFICIAL_ROOT / 'py-irt-polo'                 # ADAPTATION 2
PYIRT_BIN = Path('/data2/jeongtae/envs/tinybench_pyirt/bin')

_FORK_COMMIT = None


def pyirt_fork_commit():
    """Commit of Polo's py-irt (ADAPTATION 2); base.provenance() only covers
    the six cloned baselines, so the fork's id travels in model['info']."""
    global _FORK_COMMIT
    if _FORK_COMMIT is None:
        try:
            _FORK_COMMIT = subprocess.run(['git', '-C', str(PYIRT_FORK), 'rev-parse', 'HEAD'],
                                          capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            _FORK_COMMIT = 'unknown'
    return _FORK_COMMIT


DS = [5, 10]              # training_irt.ipynb cell 24
LR = .1                   # training_irt.ipynb cell 24 (py-irt default)
EPOCHS = 2000             # training_irt.ipynb cell 24 (py-irt default)
DEVICE = 'cpu'
RANDOM_STATE = 42         # anchor_points.ipynb cell 3


def _official_modules():
    """tutorials/irt.py + tutorials/utils.py, loaded without leaving 'irt' /
    'utils' in sys.modules (ADAPTATION 10)."""
    tut = str(TUTORIALS)
    saved = {k: sys.modules.pop(k, None) for k in ('irt', 'utils')}
    sys.path.insert(0, tut)
    try:
        utils = importlib.import_module('utils')
        irt = importlib.import_module('irt')
    finally:
        sys.path.remove(tut)
        sys.modules.pop('irt', None)
        sys.modules.pop('utils', None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
    return irt, utils


_IRT, _UTILS = _official_modules()


def get_lambda(b, v):
    """training_irt.ipynb cell 31, verbatim."""
    return (b ** 2) / (v + (b ** 2))


def create_irt_dataset(responses, dataset_name):
    """tutorials/irt.py:9-33 with the NaN skip of ADAPTATION 1."""
    import jsonlines
    dataset = []
    for i in range(responses.shape[0]):
        aux = {}
        aux_q = {}
        for j in range(responses.shape[1]):
            if np.isnan(responses[i, j]):                       # ADAPTATION 1
                continue
            aux_q['q' + str(j)] = int(responses[i, j])
        aux['subject_id'] = str(i)
        aux['responses'] = aux_q
        dataset.append(aux)
    with jsonlines.open(dataset_name, mode='w') as writer:
        writer.write_all([dataset[i] for i in range(len(dataset))])


@contextlib.contextmanager
def _official_cli(logpath):
    """Polo's py-irt first on PATH (ADAPTATION 2), single-threaded
    (ADAPTATION 11), os.system output to a log file (ADAPTATION 12)."""
    saved = {k: os.environ.get(k) for k in ('PATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
    os.environ['PATH'] = str(PYIRT_BIN) + os.pathsep + (saved['PATH'] or '')
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'   # ADAPTATION 11
    sys.stdout.flush()
    sys.stderr.flush()
    lf = open(logpath, 'ab')
    so, se = os.dup(1), os.dup(2)
    try:
        os.dup2(lf.fileno(), 1)
        os.dup2(lf.fileno(), 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(so, 1)
        os.dup2(se, 2)
        os.close(so)
        os.close(se)
        lf.close()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _train(dataset_name, model_dir, D, workdir, tag):
    """irt.py:35-51 called verbatim, then the parameters loaded in bank order."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log = Path(workdir) / f'py_irt_{tag}.log'
    t0 = time.time()
    with _official_cli(log):
        _IRT.train_irt_model(str(dataset_name), str(model_dir) + '/', D, LR, EPOCHS, DEVICE)
    out = model_dir / 'best_parameters.json'
    if not out.exists():
        raise RuntimeError(f'py-irt produced no best_parameters.json for D={D}; see {log}')
    return float(time.time() - t0)


def _load(model_dir, n):
    """irt.py:53-70 (load_irt_parameters) with the bank-order reindexing of
    ADAPTATION 3."""
    path = Path(model_dir) / 'best_parameters.json'
    params = json.load(open(path))
    ix_to_item = params['item_ids']                          # {'0': 'q7', ...}
    pos = {v: int(k) for k, v in ix_to_item.items()}
    if len(pos) != n or any(f'q{j}' not in pos for j in range(n)):
        raise RuntimeError('tinybench: py-irt item set does not cover the bank '
                           f'({len(pos)} items for a bank of {n})')
    order = [pos[f'q{j}'] for j in range(n)]
    disc = np.array(params['disc'])[order]
    diff = np.array(params['diff'])[order]
    A = disc.T[None, :, :]                                   # irt.py:67
    Bm = diff.T[None, :, :]                                  # irt.py:68
    Theta = np.array(params['ability'])[:, :, None]          # irt.py:69
    return A, Bm, Theta


class TinyBenchmarks(OfficialMethod):
    name = 'tinybench'
    adaptive = False

    # ---------------------------------------------------------------- fit (a)
    def fit(self, R, seed, workdir):
        R = np.asarray(R, float)
        K, n = R.shape
        if np.isnan(R).all(0).any():
            raise RuntimeError('tinybench: bank item(s) without any calibration record')
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        # training_irt.ipynb cell 24: validation split over SUBJECTS
        val_ind = list(range(0, K, 5))
        train_ind = [i for i in range(K) if i not in val_ind]
        if not train_ind:
            raise RuntimeError(f'tinybench: no training subjects left for K_cal={K}')
        ds_val = workdir / 'irt_val_dataset.jsonlines'
        create_irt_dataset(R[train_ind], str(ds_val))

        seen_items = list(range(0, n, 2))                    # cell 24
        unseen_items = list(range(1, n, 2))
        errors, errors2, fit_s = [], [], {}
        for D in DS:
            mdir = workdir / f'irt_val_model_D{D}'
            fit_s[f'val_D{D}'] = _train(ds_val, mdir, D, workdir, f'val_D{D}')
            A, Bm, _ = _load(mdir, n)
            thetas = []
            for j in range(len(val_ind)):
                row = R[val_ind[j]]
                si = [i for i in seen_items if not np.isnan(row[i])]     # ADAPTATION 6
                thetas.append(_IRT.estimate_ability_parameters(row[si], A[:, :, si], Bm[:, :, si]))
            with np.errstate(invalid='ignore'):
                e = float(np.mean([abs(_UTILS.item_curve(thetas[j], A, Bm)[0, unseen_items].mean()
                                       - np.nanmean(R[val_ind[j], unseen_items]))
                                   for j in range(len(val_ind))]))
            errors2.append([e])                              # one scenario (ADAPTATION 8)
            errors.append(e)
        ind_D = int(np.argmin(np.array(errors)))             # cell 25
        D = DS[ind_D]

        ds_all = workdir / 'irt_dataset.jsonlines'           # cell 27
        create_irt_dataset(R, str(ds_all))
        mdir = workdir / 'irt_model'
        fit_s['final'] = _train(ds_all, mdir, D, workdir, f'final_D{D}')   # cell 29
        A, Bm, Theta = _load(mdir, n)

        v = float(np.nanvar(R, axis=1).mean())               # cell 34 (ADAPTATION 7)
        b_err = float(np.mean(errors2[ind_D][0]))            # cell 34

        info = {'D': D, 'Ds': list(DS), 'val_errors': [float(x) for x in errors],
                'b_val_error': b_err, 'v_within_planner_var': v,
                'K_cal': int(K), 'n_bank': int(n), 'val_ind': val_ind, 'train_ind': train_ind,
                'n_nan_cells': int(np.isnan(R).sum()),
                'epochs': EPOCHS, 'lr': LR, 'py_irt_seed': 42, 'kmeans_random_state': RANDOM_STATE,
                'fit_seconds': fit_s,
                'a_mean': float(A.mean()), 'a_sd': float(A.std()),
                'b_mean': float(Bm.mean()), 'b_sd': float(Bm.std()),
                'theta_mean': float(Theta.mean()),
                'pyirt_fork': str(PYIRT_FORK), 'pyirt_fork_commit': pyirt_fork_commit(),
                'official_dir': str(TUTORIALS)}
        return {'A': A, 'B': Bm, 'D': D, 'b': b_err, 'v': v, 'n': int(n), 'info': info}

    # -------------------------------------------------- anchors (b) + est (c)
    def estimate(self, model, y, budgets, seed):
        A = np.asarray(model['A'], float)
        Bm = np.asarray(model['B'], float)
        n = int(model['n'])
        b_err, v = float(model['b']), float(model['v'])
        yg = LeakGuard(y)

        # anchor_points.ipynb cell 27, clustering = 'irt'
        X = np.vstack((A.squeeze(), Bm.squeeze().reshape((1, -1)))).T
        balance_weights = np.ones(n)                                   # ADAPTATION 8
        out = {}
        for Bud in budgets:
            number_item = int(Bud)
            norm_balance_weights = balance_weights.copy()
            norm_balance_weights /= norm_balance_weights.sum()
            kmeans = KMeans(n_clusters=number_item, n_init="auto", random_state=RANDOM_STATE)
            kmeans.fit(X, sample_weight=norm_balance_weights)
            anchor_points = pairwise_distances(kmeans.cluster_centers_, X,
                                               metric='euclidean').argmin(axis=1)
            anchor_weights = np.array([np.sum(norm_balance_weights[kmeans.labels_ == c])
                                       for c in range(number_item)])

            seen = [int(i) for i in anchor_points]                     # duplicates kept (ADAPTATION 9)
            unseen = [i for i in range(n) if i not in seen]
            Y_anchor = yg.values_at(seen)                              # the ONLY read of y

            # estimating_performance.ipynb cell 26 (anchor / IRT-free)
            anchor_est = float((Y_anchor * anchor_weights).sum())
            # estimating_performance.ipynb cells 31-33 (p-IRT)
            theta = _IRT.estimate_ability_parameters(Y_anchor, A[:, :, seen], Bm[:, :, seen])
            pirt_lambd = len(seen) / n
            data_part = float(Y_anchor.mean())
            irt_part = float(_UTILS.item_curve(theta, A, Bm)[0, unseen].mean())
            pirt_est = float(pirt_lambd * data_part + (1 - pirt_lambd) * irt_part)
            # training_irt.ipynb cell 34 + estimating_performance.ipynb cell 38 (gp-IRT)
            lambds = float(get_lambda(b_err, v / (4 * number_item)))
            gpirt_est = float(lambds * anchor_est + (1 - lambds) * pirt_est)

            items, seenset = [], set()
            for i in seen:
                if i not in seenset:
                    seenset.add(i)
                    items.append(i)
            note = '' if len(items) == number_item else (
                f'{number_item - len(items)} of {number_item} anchors are duplicate bank '
                'indices (identical (a, b) embedding rows); the official estimators use '
                'them with multiplicity, the reported items are deduplicated')
            out[int(Bud)] = {'est': gpirt_est, 'items': items,
                             'variants': {'anchor': anchor_est, 'pirt': pirt_est, 'gpirt': gpirt_est},
                             'note': note,
                             'diag': {'lambda': lambds, 'pirt_lambd': pirt_lambd,
                                      'n_unique_anchors': len(items),
                                      'kmeans_inertia': float(kmeans.inertia_),
                                      'theta_norm': float(np.linalg.norm(theta))}}
        return out

    def stop(self, model, y, seed):
        return {}


METHOD = TinyBenchmarks()
