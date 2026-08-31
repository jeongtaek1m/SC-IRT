"""Pooled metrics and the resampling conventions of the paper.

Bootstrap units: (seed) 16 clusters of 6 for planner-side paired deltas,
(draw, type) 128 clusters for the US pooled deltas. Differences smaller than
about .005 SR-MAE are inside the paired 95% intervals at n = 96.
"""
import numpy as np
from scipy.stats import spearmanr


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


def cluster_boot_rho_delta(bt_a, bt_b, fail, clusters, B=10000, seed=0):
    """Delta Spearman rho (a - b) against observed fail rates, cluster
    bootstrap over the given (draw, type) labels. Returns
    (delta, lo95, hi95, P(delta > 0))."""
    bt_a, bt_b, fail = map(np.asarray, (bt_a, bt_b, fail))
    cl = np.asarray(clusters)
    cls = sorted(set(cl))
    byc = {c: np.where(cl == c)[0] for c in cls}
    rng = np.random.RandomState(seed)
    ds = []
    for _ in range(B):
        pick = [cls[i] for i in rng.randint(0, len(cls), len(cls))]
        ii = np.concatenate([byc[c] for c in pick])
        ds.append(spearmanr(bt_a[ii], fail[ii]).correlation
                  - spearmanr(bt_b[ii], fail[ii]).correlation)
    ds = np.array(ds)
    delta = spearmanr(bt_a, fail).correlation - spearmanr(bt_b, fail).correlation
    return delta, np.percentile(ds, 2.5), np.percentile(ds, 97.5), float((ds > 0).mean())


def ies(mae, rollouts, mae_ref, rollouts_ref=55):
    """Information Efficiency Score (ATLAS-style) with a declared reference:
    (MAE / MAE_ref) x (rollouts / rollouts_ref). The paper's reference is the
    random order at a fixed budget of 55 under the common readout."""
    return mae / mae_ref * rollouts / rollouts_ref
