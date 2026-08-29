"""Target-aligned acquisition (PROTOCOL section 4).

The acquisition model is a joint 2PL fit of the calibration panel used
*only* to choose the next scene; every reported number is read out under the
Rasch evaluation model (`scirt.bayes`). Two regimes, two targets:

  UP   the quantity is the full-bank success rate S -> localize the new
       planner for K rollouts (2PL Fisher at its Newton-MAP ability), then
       cover the bank by population-Fisher order (`localize_cover`).
  UPS  the quantity is the evaluation-scale ability to be transported ->
       theta-EIG under the evaluation model itself (`eig_pick`).

K is a bank constant estimated at calibration time by leave-one-planner-out
simulation (`experiments/run_k_calibration.py`); on Bench2Drive the
simulated loss is flat for K in [15, 30] and K = 20 is used.
"""
import numpy as np

from .curves import THG, h_, sig

K_LOCALIZE = 20


def theta_newton(b, y, a, it=50):
    """Newton MAP of the acquisition-model ability from the administered
    items (N(0,1) prior, clipped to the theta grid)."""
    t = 0.0
    for _ in range(it):
        p = sig(a * (t - b))
        g = (a * (y - p)).sum() - t
        h = -((a ** 2) * p * (1 - p)).sum() - 1.0
        t -= g / h
    return float(np.clip(t, -6, 6))


def fisher_pick(theta_hat, a, b, rem):
    """Localize: argmax 2PL Fisher information at theta_hat."""
    p = sig(a[rem] * (theta_hat - b[rem]))
    return rem[int(np.argmax((a[rem] ** 2) * p * (1 - p)))]


def population_fisher(a, b, th_cal):
    """Cover: mean 2PL Fisher information over the calibration planners'
    (acquisition-model) abilities — a planner-independent bank order."""
    return np.array([np.mean([(a[i] ** 2) * sig(a[i] * (t - b[i])) * (1 - sig(a[i] * (t - b[i])))
                              for t in th_cal]) for i in range(len(a))])


def localize_cover(a, b, th_cal, y, K=K_LOCALIZE, T=120):
    """The UP acquisition: K localize picks (adaptive, reads y only for the
    administered items) followed by the population-Fisher static order.
    Returns the first T items."""
    n = len(a)
    T = min(T, n)
    S, t0 = [], 0.0
    for _ in range(min(K, T)):
        rem = [i for i in range(n) if i not in S]
        S.append(fisher_pick(t0, a, b, rem))
        idx = np.array(S)
        t0 = theta_newton(b[idx], y[idx], a[idx])
    cover = [int(i) for i in np.argsort(-population_fisher(a, b, th_cal)) if i not in set(S)]
    return S + cover[:T - len(S)]


def eig_pick(q, M, rem):
    """UPS probe rule: expected information gain about the evaluation-model
    ability, h(E[m]) - E[h(m)] on the theta grid."""
    mi = M[:, rem]
    mbar = (q[:, None] * mi).sum(0)
    return rem[int(np.argmax(h_(mbar) - (q[:, None] * h_(mi)).sum(0)))]
