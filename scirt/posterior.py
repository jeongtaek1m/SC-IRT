"""Bayesian ability posterior for the calibrated-bank regime (UP).

    b_i | x_i ~ N(b_hat_i, s_i^2)          difficulty is estimated, not known
    theta_j   ~ N(0, 1)
    Y_ij      ~ Bernoulli( sigmoid( a_i (theta_j - b_i) ) )

Difficulty is marginalised out *before* the likelihood is formed, so an item
contributes its marginal curve

    m_i(theta) = E_{b_i}[ sigmoid( a_i (theta - b_i) ) ]

rather than a point-estimate curve. This is what buys calibrated intervals:
marginalising costs rollouts (+2.2 at +-10%, +6.2 at +-5%) and does not move
the point estimate, but raises coverage from 0.88 to 0.94 and 0.81 to 0.88.
See PROTOCOL.md section 2.1b.

The success-rate interval is a posterior-predictive quantile, not a normal
approximation: the two disagree on the same model (0.88 vs 0.94 coverage), and
the quantile form is the one the protocol fixes as canonical.
"""

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from .theta import sig

# Gauss-Hermite nodes for the difficulty integral, and the ability grid the
# posterior lives on. Both are module constants so every call site shares them.
_GH_X, _GH_W = hermegauss(21)
_GH_W = _GH_W / _GH_W.sum()
THETA_GRID = np.linspace(-6, 6, 241)
_PRIOR = np.exp(-0.5 * THETA_GRID**2)
THETA_PRIOR = _PRIOR / _PRIOR.sum()


def _h(p):
    """Binary entropy, in nats."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def item_curves(b, a, s):
    """Marginal response curves m_i(theta) on THETA_GRID.

    Args:
        b, a, s: per-item difficulty, discrimination, and difficulty SD. Pass
            s = zeros for the plug-in (unmarginalised) variant.

    Returns:
        (len(THETA_GRID), n_items) array.
    """
    b, a, s = np.asarray(b, float), np.asarray(a, float), np.asarray(s, float)
    out = np.empty((len(THETA_GRID), len(b)))
    for i in range(len(b)):
        if s[i] <= 0:
            out[:, i] = sig(a[i] * (THETA_GRID - b[i]))
        else:
            draws = b[i] + s[i] * _GH_X
            out[:, i] = (sig(a[i] * (THETA_GRID[:, None] - draws[None, :])) * _GH_W[None, :]).sum(1)
    return out


def theta_posterior(curves, y, administered):
    """Posterior over the ability grid given the administered responses.

    Args:
        curves: output of `item_curves`.
        y: binary outcomes aligned with the item axis of `curves`.
        administered: indices of the items actually run.

    Returns:
        Probability vector over THETA_GRID.
    """
    if len(administered) == 0:
        return THETA_PRIOR.copy()
    idx = np.asarray(administered, int)
    yy = np.asarray(y, float)[idx]
    m = curves[:, idx]
    ll = (yy[None, :] * np.log(m + 1e-12) + (1 - yy[None, :]) * np.log(1 - m + 1e-12)).sum(1)
    q = np.exp(ll - ll.max()) * THETA_PRIOR
    return q / q.sum()


def success_rate(post, curves, y, administered, n_bank, n_draws=4000, seed=0):
    """Posterior-predictive distribution of the *realised* success rate.

    Observed outcomes are kept as-is; unadministered items are sampled from
    their marginal curves. The spread therefore carries both ability
    uncertainty and unobserved-response predictive uncertainty -- the second is
    not simulator noise, it is not knowing outcomes that were never run.

    Returns:
        {'mean', 'lo', 'hi', 'se'} with a 95% quantile interval.
    """
    admin = set(int(i) for i in administered)
    rest = np.array([i for i in range(curves.shape[1]) if i not in admin], int)
    obs = float(np.asarray(y, float)[np.asarray(sorted(admin), int)].sum()) if admin else 0.0
    rng = np.random.RandomState(seed)
    draw_theta = rng.choice(len(THETA_GRID), size=n_draws, p=post)
    if len(rest):
        m = curves[draw_theta][:, rest]
        sr = (obs + (rng.random(m.shape) < m).sum(1)) / n_bank
    else:
        sr = np.full(n_draws, obs / n_bank)
    return {"mean": float(sr.mean()), "lo": float(np.percentile(sr, 2.5)),
            "hi": float(np.percentile(sr, 97.5)), "se": float(sr.std(ddof=1))}


def expected_information_gain(post, curves, candidates):
    """Target-EIG for each candidate item: I(theta; Y_i), with b marginalised.

        A_i = h( E_theta[m_i] ) - E_theta[ h(m_i) ]

    Difficulty is integrated out *inside* m_i, so this is the information the
    response carries about ability alone. Scoring the joint form
    E_{theta,b}[H(Y|theta,b)] instead computes I((theta,b); Y) and answers a
    different question -- it rewards items that teach us about b.
    """
    idx = np.asarray(candidates, int)
    m = curves[:, idx]
    m_bar = (post[:, None] * m).sum(0)
    return _h(m_bar) - (post[:, None] * _h(m)).sum(0)
