"""Ability estimation for adaptive testing.

During a CAT run the item parameters are already fixed, so ability is estimated
by Newton MAP on a handful of responses rather than by the full Adam fit in
`scirt.irt`. The prior is standard normal, giving the -t and -1.0 terms in the
gradient and Hessian.
"""

import numpy as np


def sig(x):
    """Logistic function, clipped to keep exp() in range for extreme abilities."""
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def map_theta(bs, ys, aa, it):
    """Newton MAP estimate of a single rater's ability.

    Args:
        bs: difficulties of the administered items.
        ys: observed binary responses to those items.
        aa: discriminations of those items. Pass ones for a 1PL bank; the
            multiplication by 1.0 is exact under IEEE 754, so the 1PL and 2PL
            code paths are bit-identical rather than merely close.
        it: Newton steps. Required — the CAT experiments use 50 and the
            descriptor table uses 40, and a shared default would silently move
            the reported UP macro-AUROC and success-rate MAE.

    Returns:
        Ability estimate, clipped to [-6, 6].
    """
    t = 0.0
    for _ in range(it):
        p = sig(aa * (t - bs))
        g = (aa * (ys - p)).sum() - t
        h = -(aa * aa * p * (1 - p)).sum() - 1.0
        t -= g / h
    return float(np.clip(t, -6, 6))
