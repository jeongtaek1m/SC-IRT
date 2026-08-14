"""Summary statistics shared across experiments."""

import numpy as np
from scipy.stats import spearmanr


def spearman(x, y):
    """Spearman rank correlation as a plain float."""
    return spearmanr(x, y).correlation


def pct_ci(v, lo=2.5, hi=97.5):
    """Percentile confidence interval over bootstrap replicates."""
    return np.percentile(v, [lo, hi])


def mean_se(v):
    """Mean and standard error over folds.

    The n == 1 guard matters only for degenerate configurations; at the live fold
    counts (16 and 48) it is identical to the unguarded form.
    """
    v = np.array(v, dtype=float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


def between_within(gold, pred, cluster_labels):
    """Decompose rank agreement into between-cluster and within-cluster parts.

    A predictor can score well by ranking scenario *types* correctly while being
    blind to variation between routes of the same type. Reporting the two parts
    separately makes that failure visible.

    Returns:
        (between, within) Spearman correlations. `within` is the mean over
        clusters of at least four members; smaller clusters give an undefined or
        hopelessly noisy correlation.
    """
    gold = np.asarray(gold, dtype=float)
    pred = np.asarray(pred, dtype=float)
    labels = sorted(set(cluster_labels))
    idx = {t: [i for i, c in enumerate(cluster_labels) if c == t] for t in labels}

    between = spearmanr(
        [gold[idx[t]].mean() for t in labels],
        [pred[idx[t]].mean() for t in labels],
    ).correlation

    per_cluster = []
    for t in labels:
        members = idx[t]
        if len(members) >= 4:
            per_cluster.append(
                spearmanr(list(gold[members]), list(pred[members])).correlation
            )
    return between, float(np.nanmean(per_cluster))
