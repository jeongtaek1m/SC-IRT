"""Pooled metrics and the resampling conventions of the paper.

Bootstrap units: (seed) 16 clusters of 6 for planner-side paired deltas.
Differences smaller than
about .005 SR-MAE are inside the paired 95% intervals at n = 96.
"""
import numpy as np


def mean_se(v):
    v = np.array(v, float)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def paired_seed_boot(a, b, n_seeds=16, per_seed=3, B=4000, seed=0):
    """Paired delta a-b with a cluster bootstrap over evaluation seeds.
    Returns (mean, lo95, hi95)."""
    dd = (np.array(a) - np.array(b)).reshape(n_seeds, per_seed)
    rng = np.random.RandomState(seed)
    bs = np.array([dd[rng.randint(n_seeds, size=n_seeds)].mean() for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dd.mean(), lo, hi


def ies(mae, rollouts, mae_ref, rollouts_ref=55):
    """Information Efficiency Score (ATLAS-style) with a declared reference:
    (MAE / MAE_ref) x (rollouts / rollouts_ref). The paper's reference is the
    random order at a fixed budget of 55 under the common readout."""
    return mae / mae_ref * rollouts / rollouts_ref
