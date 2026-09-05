"""Pooled metrics and the resampling conventions of the paper.

Bootstrap unit: (planner) the unique planner ids are resampled with
replacement for planner-side paired deltas — the same planners recur across
the 16 draws, so draws are not independent clusters; every evaluation of a
resampled planner enters with equal weight. Differences smaller than
about .005 SR-MAE are inside the paired 95% intervals at n = 64 evaluations
per cell (16 draws x 4 evaluation planners).
"""
import numpy as np


def mean_se(v):
    v = np.array(v, float)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def paired_cluster_boot(a, b, clusters, B=4000, seed=0):
    """Paired delta a-b with a cluster bootstrap over planners: the unique
    ids in `clusters` (parallel to a, b) are resampled with replacement and
    the delta is averaged over every evaluation of the resampled planners
    with equal weight. Returns (mean, lo95, hi95)."""
    d = np.array(a, float) - np.array(b, float)
    clusters = np.asarray(clusters)
    ids = np.unique(clusters)
    sums = np.array([d[clusters == c].sum() for c in ids])
    cnts = np.array([(clusters == c).sum() for c in ids])
    rng = np.random.RandomState(seed)
    idx = rng.randint(len(ids), size=(B, len(ids)))
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d.mean(), lo, hi


def ies(mae, rollouts, mae_ref, rollouts_ref=55):
    """Information Efficiency Score (ATLAS-style) with a declared reference:
    (MAE / MAE_ref) x (rollouts / rollouts_ref). The paper's reference is the
    random order at a fixed budget of 55 under the common readout."""
    return mae / mae_ref * rollouts / rollouts_ref
