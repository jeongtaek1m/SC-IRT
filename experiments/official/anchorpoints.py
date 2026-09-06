"""AnchorPoints (Vivek et al., EACL 2024) — Anchor Points Weighted (APW)
through the official implementation's own algorithmic lines.

Official code: OFFICIAL_ROOT/AnchorPoints/optimal_valset_validation.py,
function anchor_points_weighted (lines 548-633, commit 64b6087). Its core, in
order, is

    corrs = np.corrcoef(val_slice1, rowvar=False)                       # l.595
    selected_idxs = kmedoids.fasterpam(1 - corrs, num_medoids,
                                       init="random").medoids           # l.597-599
    cluster_members = np.argmax(corrs[selected_idxs, :], axis=0)        # l.602
    unique, cluster_sizes = np.unique(cluster_members, return_counts=True)
    for i in range(num_medoids): if i not in unique: cluster_sizes.insert(i, 0)
    weights = cluster_sizes / np.sum(cluster_sizes)                     # l.603-611
    votes = val_slice2[model_idx, selected_idxs]                         # l.615
    est_score = np.sum(weights * votes)                                  # l.616

val_slice1 = the SOURCE models' per-example values (rows = models, columns =
examples), val_slice2 = the TARGET models' values. Here the source models are
the K_cal calibration planners (R), the single target model is the evaluation
planner (y), num_medoids = B, 'items' = the medoids and 'est' = the weighted
vote. No IRT, no stopping rule, not adaptive (each budget is its own PAM run,
so the selections of two budgets are not nested). The official function cannot
be called as-is: it is the whole experiment loop (num_runs random source /
target model draws, returns a rank correlation against known true scores, and
its module imports `datasets`, `anchor_point_predictor`, matplotlib, ...). The
statements above are therefore executed here verbatim, in the same order,
including the official indentation of `weights` inside the empty-cluster loop
(l.611), with the same library calls; `_check_official_source` asserts at fit
time that every one of them is still present in the official file, so a drift
of the official code is caught rather than silently diverged from.

VARIANTS
`anchors_unweighted` is the official *naive* estimator formula (l.234,
metric = accuracy_score by default, l.909: the plain accuracy over the
selected points) read on the same anchors — i.e. APW without the cluster-size
weights. It reads no item outside 'items'. The official experiment never pairs
the anchor selection with the naive estimator (the techniques dict, l.895-904,
offers Anchor_Points_Weighted only), so this readout is ours and is reported
as a variant, never as 'est'. Anchor_Points_Predictor (l.903) is not offered:
it regresses per-example class-probability vectors (AnchorPointPredictor,
use_logit=True), which do not exist for a binary driving rollout.

ADAPTATIONS
1. optimal_valset_validation.py:563-577, 583-584 — the num_runs loop with
   random source/target model draws is replaced by the protocol cell: source
   models = the K_cal calibration planners, target model = the evaluation
   planner (one model, so no correlation-with-true-scores readout; the
   weighted vote est_score of line 616 is returned instead of the Kendall tau
   of line 619). Unavoidable: the protocol fixes who is source and who is
   target.
2. optimal_valset_validation.py:597-599 — kmedoids.fasterpam is called with
   random_state=seed (the official call is unseeded and the repository never
   seeds numpy either: `grep -r 'random.seed|random_state' AnchorPoints/*.py`
   is empty); needed so that a cell is reproducible. Same init='random', same
   default max_iter=100, same default n_cpu.
3. optimal_valset_validation.py:974-1047 (gold_data = all_data[:, arange, gt],
   the probability of the correct class) — a driving rollout has no class
   probabilities; the per-example value of a source model is its binary
   outcome (1 pass, 0 fail), and the target model's 'votes' (l.615) are its
   binary outcomes. The estimate is then the APW estimate of the success rate.
4. Missing records (NaN cells of R; the official data is complete and has no
   counterpart) are filled, before line 595, with the route's mean over the
   calibration planners that do have a record (if a bank route were unrecorded
   by every calibration planner, the bank-wide mean of R is used; this does
   not occur in the 192 protocol cells — 'allnan_cols' in 'info' reports it).
5. optimal_valset_validation.py:595 — np.corrcoef returns NaN for a column
   that is constant across the source models (all pass / all fail: 10 of 219
   routes at K_cal = 12, 62 of 219 at K_cal = 4 in cell (0, *, 0)).
   kmedoids.fasterpam does not raise on a NaN dissimilarity but its result is
   meaningless (loss = NaN, and np.argmax at l.602 then assigns every route to
   one cluster), so a NaN correlation is replaced by 0 (distance 1 to every
   other route) and the diagonal by 1 (distance 0 to itself, the value
   np.corrcoef gives for every non-constant column). The number of NaN
   correlation entries is reported in model['info'].
6. The budget is capped at the bank size (fasterpam cannot draw more medoids
   than points); never binding here (B <= 165 < 210 <= n_bank), and reported
   in 'note' if it ever were.
"""
import importlib.metadata
from pathlib import Path

import kmedoids
import numpy as np

from official.base import LeakGuard, OFFICIAL_ROOT, OfficialMethod

OFFICIAL_FILE = OFFICIAL_ROOT / 'AnchorPoints' / 'optimal_valset_validation.py'
# The core statements of anchor_points_weighted, as they appear in the official file.
_CORE_LINES = [
    'corrs = np.corrcoef(val_slice1, rowvar=False)',
    'selected_idxs = kmedoids.fasterpam(',
    '1 - corrs, num_medoids, init="random"',
    ').medoids',
    'cluster_members = np.argmax(corrs[selected_idxs, :], axis=0)',
    'unique, cluster_sizes = np.unique(cluster_members, return_counts=True)',
    'cluster_sizes = list(cluster_sizes)',
    'cluster_sizes.insert(i, 0)',
    'weights = cluster_sizes / np.sum(cluster_sizes)',
    'votes = val_slice2[model_idx, selected_idxs]',
    'est_score = np.sum(weights * votes)',
]


def _check_official_source():
    src = Path(OFFICIAL_FILE).read_text()
    missing = [l for l in _CORE_LINES if l not in src]
    if missing:
        raise RuntimeError(f'official AnchorPoints source drifted; core lines not found: {missing}')


class AnchorPointsOfficial(OfficialMethod):
    name = 'anchorpoints'
    adaptive = False

    def fit(self, R, seed, workdir):
        _check_official_source()
        R = np.asarray(R, float)
        K, n = R.shape
        # ADAPTATION 4: fill missing records with the route mean over recorded planners.
        with np.errstate(all='ignore'):
            col_mean = np.nanmean(R, axis=0)
        allnan = np.isnan(col_mean)
        col_mean[allnan] = np.nanmean(R)
        val_slice1 = np.where(np.isnan(R), col_mean[None, :], R)       # ADAPTATION 1/3: source models = calibration planners
        # official line 595
        with np.errstate(invalid='ignore', divide='ignore'):
            corrs = np.corrcoef(val_slice1, rowvar=False)
        nan_mask = np.isnan(corrs)
        n_nan = int(nan_mask.sum())
        const_cols = int((val_slice1.std(axis=0) == 0).sum())
        # ADAPTATION 5: NaN correlation (constant column) -> 0, self-correlation -> 1.
        corrs = np.where(nan_mask, 0.0, corrs)
        np.fill_diagonal(corrs, 1.0)
        info = {'n_source': int(K), 'n_bank': int(n), 'nan_cells_filled': int(np.isnan(R).sum()),
                'allnan_cols': int(allnan.sum()), 'const_cols': const_cols,
                'corrcoef_nan_entries': n_nan, 'corrcoef_nan_frac': n_nan / float(n * n),
                'kmedoids_version': importlib.metadata.version('kmedoids'),
                'official_file': str(OFFICIAL_FILE), 'official_function': 'anchor_points_weighted',
                'empty_clusters': {}}
        return {'corrs': corrs, 'n': int(n), 'info': info}

    def estimate(self, model, y, budgets, seed):
        corrs = model['corrs']
        n = model['n']
        yg = LeakGuard(y)                                              # target model (ADAPTATION 1)
        out = {}
        for B in budgets:
            num_medoids = min(int(B), n)                               # ADAPTATION 6
            # official lines 597-599 (ADAPTATION 2: random_state=seed)
            selected_idxs = kmedoids.fasterpam(1 - corrs, num_medoids, init="random", random_state=seed).medoids
            # official lines 602-611, verbatim (weights is assigned inside the loop there)
            cluster_members = np.argmax(corrs[selected_idxs, :], axis=0)
            unique, cluster_sizes = np.unique(cluster_members, return_counts=True)
            cluster_sizes = list(cluster_sizes)
            for i in range(num_medoids):
                if i not in unique:
                    cluster_sizes.insert(i, 0)
                weights = cluster_sizes / np.sum(cluster_sizes)
            # official lines 615-616: the target model's values at the medoids only
            votes = yg.values_at(selected_idxs)
            est_score = np.sum(weights * votes)
            model['info']['empty_clusters'][int(B)] = int(np.sum(np.asarray(weights) == 0))
            out[int(B)] = {'est': float(est_score), 'items': [int(i) for i in selected_idxs],
                           'variants': {'anchors_unweighted': float(np.mean(votes))},   # official l.234 formula
                           'note': '' if num_medoids == B else f'budget {B} capped at bank size {n}'}
        return out


METHOD = AnchorPointsOfficial()
