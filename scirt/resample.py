"""Cluster bootstrap over scenario types.

Routes of the same scenario type are not independent observations: they share a
generator, a map region and a failure mode. Resampling individual routes would
therefore understate the uncertainty. Every interval in this package instead
resamples whole scenario types with replacement, which is also why the effective
sample size for a confidence interval is 44, not 219.

Comparisons between two predictors are **paired**: both are scored on the same
resampled index set within a replicate, so the interval is on the difference and
not on two independently noisy estimates.
"""

import numpy as np
from scipy.stats import spearmanr


def cluster_bootstrap_idx(clusters, B=10000, seed=0):
    """Yield B resampled index arrays, drawing whole clusters with replacement.

    The generator is constructed inside the function on purpose. Every contrast
    in an experiment therefore replays the identical stream of replicates, which
    is what keeps separate comparisons mutually paired rather than merely paired
    within one call.

    Each replicate draws its cluster picks with a single vectorised `randint`.
    Replacing that with scalar draws or with `rng.choice` consumes the Mersenne
    Twister stream differently and changes every published interval.
    """
    rng = np.random.RandomState(seed)
    k = len(clusters)
    for _ in range(B):
        pick = rng.randint(k, size=k)
        yield np.concatenate([clusters[i] for i in pick])


def bootstrap_rho(gold, pred, clusters, B=10000, seed=0):
    """Percentile CI of Spearman rho against gold."""
    vals = [
        spearmanr(gold[idx], pred[idx]).correlation
        for idx in cluster_bootstrap_idx(clusters, B, seed)
    ]
    return np.percentile(vals, [2.5, 97.5])


def paired_delta_rho(gold, a, b, clusters, B=10000, seed=0):
    """Paired CI and exceedance probability for rho(a) - rho(b).

    Returns:
        (ci, p) where ci is the percentile interval of the difference and p is
        the fraction of replicates with a positive difference. A comparison is
        reported as significant only when the interval excludes zero.
    """
    deltas = np.array(
        [
            spearmanr(gold[idx], a[idx]).correlation
            - spearmanr(gold[idx], b[idx]).correlation
            for idx in cluster_bootstrap_idx(clusters, B, seed)
        ]
    )
    return np.percentile(deltas, [2.5, 97.5]), float((deltas > 0).mean())
