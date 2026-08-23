"""Grid posterior, posterior-predictive success rate, and the certified-SR
credible interval — the pieces every CAT loop composes.

The experiment scripts keep their loops inline on purpose: the order in which
these primitives consume random numbers is part of the reproduction contract,
and hiding it inside a generic runner is how silent drift happens.
"""
import numpy as np

from .curves import THG, PRIOR


def post_from(M, y, S, prior=PRIOR):
    """Posterior over the theta grid given administered items S.
    M: (grid, n_items) response curves; y: observed 0/1 responses (full row,
    only S entries are read)."""
    if not len(S):
        return prior.copy()
    ll = (y[S][None, :] * np.log(M[:, S] + 1e-12)
          + (1 - y[S][None, :]) * np.log(1 - M[:, S] + 1e-12)).sum(1)
    q = np.exp(ll - ll.max()) * prior
    return q / q.sum()


def sr_ci(M, y, S, q, rng, n_draws=4000):
    """Posterior-predictive distribution of the realised full-bank success
    rate: draw theta from q, fill unobserved responses as Bernoulli(m_i),
    add the observed successes. Returns (lo95, hi95, mean).

    This consumes exactly len==2 draws from `rng` (choice + random) — the
    draw order is part of the protocol."""
    n = M.shape[1]
    un = [i for i in range(n) if i not in S]
    ti = rng.choice(len(THG), size=n_draws, p=q)
    mm = M[ti][:, un] if un else np.zeros((n_draws, 0))
    sr = (y[S].sum() + (rng.random(mm.shape) < mm).sum(1)) / n
    return np.percentile(sr, 2.5), np.percentile(sr, 97.5), sr.mean()


def theta_sd(q):
    """Posterior SD of theta on the grid (ATLAS-style SE(theta) stopping)."""
    m = (q * THG).sum()
    return float(np.sqrt((q * THG ** 2).sum() - m ** 2))


def posterior_mean_sr(M, y, S, q):
    """Point estimate of the full-bank SR: observed successes + posterior-mean
    fill of the unobserved items."""
    n = M.shape[1]
    un = [i for i in range(n) if i not in S]
    if not un:
        return float(y[S].sum() / n)
    return float((y[S].sum() + (M * q[:, None]).sum(0)[un].sum()) / n)
