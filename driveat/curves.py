"""Item response curves on the ability axis with the difficulty posterior
marginalised exactly — the inference engine shared by every regime.

Grids (protocol constants):
  XG      extended ability axis [-9, 9], step .05 (361 points): theta + u
  THG     ability grid [-6, 6], 241 points = XG[I0 : I0 + 241]
  PRIOR   N(0, 1) on THG, normalised
  BG      difficulty grid [-10, 10], 801 points (exact item posterior)
  UG      testlet-effect grid [-3, 3], step .1 (61 points); USHIFT[j] is the
          index offset on XG with XG[I0 + i + USHIFT[j]] = THG[i] + UG[j]
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import logsumexp

GX, GW = hermegauss(21)
GW = GW / GW.sum()
XG = np.linspace(-9, 9, 361)
I0 = 60
THG = XG[I0:I0 + 241]
PRIOR = np.exp(-0.5 * THG ** 2)
PRIOR = PRIOR / PRIOR.sum()
BG = np.linspace(-10, 10, 801)
UG = np.linspace(-3, 3, 61)
USHIFT = np.rint(UG / (XG[1] - XG[0])).astype(int)


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def h_(p):
    """Binary entropy (nats), clipped."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def item_loglik(R, th, a=None):
    """log p(y_s | theta_cal, b) on BG for every item: (n, 801).
    R: (n, K) responses with nan for unobserved cells; th: (K,) calibration
    abilities; a: optional (n,) discriminations (Rasch when None)."""
    mk = ~np.isnan(R)
    Y = np.nan_to_num(R)
    aa = np.ones(R.shape[0]) if a is None else np.asarray(a)
    Z = aa[:, None, None] * (th[None, :, None] - BG[None, None, :])
    P = sig(Z)
    return (mk[:, :, None] * (Y[:, :, None] * np.log(P + 1e-12)
                              + (1 - Y[:, :, None]) * np.log(1 - P + 1e-12))).sum(1)


def item_posteriors(ll, sigma_b):
    """Exact conditional posterior of every difficulty given the calibration
    abilities, prior b ~ N(0, sigma_b^2): W (n, 801) on BG. `ll` is built from
    the centred theta_hat, so this prior is anchored at zero in the centred
    frame — the frame the evaluation posterior also uses (PROTOCOL section 3)."""
    lw = ll - 0.5 * (BG[None, :] / sigma_b) ** 2
    W = np.exp(lw - lw.max(1, keepdims=True))
    return W / W.sum(1, keepdims=True)


def item_marginal_loglik(ll, sigma_b):
    """sum_s log int p(y_s | theta, b) N(b; 0, sigma_b^2) db on BG — the
    empirical-Bayes objective for sigma_b."""
    db = BG[1] - BG[0]
    lprior = -0.5 * (BG / sigma_b) ** 2 - np.log(sigma_b * np.sqrt(2 * np.pi)) + np.log(db)
    return float(logsumexp(ll + lprior[None, :], axis=1).sum())


def posterior_sd(W):
    """Posterior SD of each difficulty from its grid posterior."""
    m = W @ BG
    return np.sqrt(np.maximum(W @ BG ** 2 - m ** 2, 0.0))


def curves_from_posterior(W):
    """m_s(x) = E_{b_s ~ W_s}[sigmoid(x - b_s)] on XG: (361, n)."""
    return sig(XG[:, None] - BG[None, :]) @ W.T


def marginal_curves(mu, sd):
    """Gaussian-prior curves on XG via Gauss-Hermite: m_s(x) =
    E_{b ~ N(mu_s, sd_s^2)}[sigmoid(x - b)]. Used for encoder priors and,
    with sd -> 0, for point curves. Returns (361, n)."""
    n = len(mu)
    return np.stack([(sig(XG[:, None] - (mu[i] + sd[i] * GX)[None, :]) * GW[None, :]).sum(1)
                     for i in range(n)], 1)

