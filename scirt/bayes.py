"""Evaluation-side inference: grid posterior over the new planner's ability,
the MAP-fill success-rate readout, and the posterior L1 risk that drives
adaptive stopping (PROTOCOL sections 2 and 5).

All of it runs on the difficulty-marginalised Rasch curves
m_i(theta) = E_{b_i | A}[sigmoid(theta - b_i)] (`scirt.curves`). The
acquisition model never enters here.
"""
import numpy as np
from scipy.stats import norm

from .curves import PRIOR


def post_from(M, y, S, prior=PRIOR):
    """Posterior over the theta grid given administered items S.
    M: (grid, n_items) curves; y: 0/1 responses (full row, only S read)."""
    if not len(S):
        return prior.copy()
    ll = (y[S][None, :] * np.log(M[:, S] + 1e-12)
          + (1 - y[S][None, :]) * np.log(1 - M[:, S] + 1e-12)).sum(1)
    q = np.exp(ll - ll.max()) * prior
    return q / q.sum()


def map_fill(M, y, S, q=None):
    """Success-rate readout: observed successes + curve fill of the
    unobserved items at the MAP ability."""
    n = M.shape[1]
    S = list(S)
    if q is None:
        q = post_from(M, y, S)
    un = [i for i in range(n) if i not in set(S)]
    ih = int(np.argmax(q))
    return float((y[S].sum() + M[ih, un].sum()) / n)


def r1_risk(M, y, S, q=None, s_hat=None):
    """Posterior expected absolute error of the readout, E|S - S_hat | D|,
    in closed form: unobserved responses given theta are approximated by a
    normal with mean sum m_i and variance sum m_i(1 - m_i); the mixture over
    the theta grid then gives E|X - c| = sigma [2 phi(z) + z (2 Phi(z) - 1)]."""
    n = M.shape[1]
    S = list(S)
    if q is None:
        q = post_from(M, y, S)
    if s_hat is None:
        s_hat = map_fill(M, y, S, q)
    un = [i for i in range(n) if i not in set(S)]
    if not un:
        return 0.0
    Mu = M[:, un]
    mu = Mu.sum(1)
    sd = np.sqrt((Mu * (1 - Mu)).sum(1) + 1e-9)
    z = (s_hat * n - y[S].sum() - mu) / sd
    return float((q * sd * (2 * norm.pdf(z) + z * (2 * norm.cdf(z) - 1))).sum() / n)


def track(M, y, order):
    """Readout and risk along a bank order: (S_hat[t], R1[t]) for
    t = 1..len(order). Fixed budgets read S_hat[B-1]; the stopping rule
    reads the first t with R1[t] <= tau."""
    Sh, R1 = [], []
    for t in range(1, len(order) + 1):
        S = order[:t]
        q = post_from(M, y, S)
        sh = map_fill(M, y, S, q)
        Sh.append(sh)
        R1.append(r1_risk(M, y, S, q, sh))
    return Sh, R1


def stop_at(R1, tau):
    """First index (1-based rollout count) with R1 <= tau; the full length
    if never reached."""
    hit = np.where(np.asarray(R1) <= tau)[0]
    return int(hit[0]) + 1 if len(hit) else len(R1)
