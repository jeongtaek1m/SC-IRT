"""Item response curves on the theta grid, with difficulty uncertainty
marginalised — the inference engine shared by every regime.

THETA_GRID / PRIOR / the 21-node Gauss-Hermite rule are protocol constants.
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

GX, GW = hermegauss(21)
GW = GW / GW.sum()
THG = np.linspace(-6, 6, 241)
PRIOR = np.exp(-0.5 * THG ** 2)
PRIOR = PRIOR / PRIOR.sum()


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def h_(p):
    """Binary entropy (nats), clipped."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def marginal_curves(mu, sd, a=None):
    """m_i(theta) = E_{b_i ~ N(mu_i, sd_i^2)}[ sigmoid(a_i (theta - b_i)) ]
    on the grid, via Gauss-Hermite. a=None -> Rasch (the paper's main model).
    Returns (grid, n_items)."""
    n = len(mu)
    aa = np.ones(n) if a is None else a
    return np.stack([(sig(aa[i] * (THG[:, None] - (mu[i] + sd[i] * GX)[None, :]))
                      * GW[None, :]).sum(1) for i in range(n)], 1)


def point_curves_2pl(a, b):
    """sigmoid(a (theta - b)) point curves (Fluid-style native scoring)."""
    return sig(a[None, :] * (THG[:, None] - b[None, :]))
