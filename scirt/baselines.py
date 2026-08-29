"""Published-baseline selectors and their native readouts (Table 1 rows).

"-style / -lite / -adapted" marks re-implementations from the method
descriptions on this bank; every baseline is calibrated from the same
planner panel as SC-IRT. The static orders have the prefix property
(order[:B] is the budget-B subset), which is what lets one bank order serve
both the fixed-budget table and the common stopping machine.
"""
import warnings

import numpy as np

from .acquisition import theta_newton, population_fisher
from .curves import sig, THG


def fluid_order(a, b, y, T):
    """Fluid-style: 2PL Fisher argmax at the Newton-MAP ability, adaptively."""
    n = len(a)
    S, t0 = [], 0.0
    for _ in range(min(T, n)):
        rem = [i for i in range(n) if i not in S]
        p = sig(a[rem] * (t0 - b[rem]))
        S.append(rem[int(np.argmax((a[rem] ** 2) * p * (1 - p)))])
        idx = np.array(S)
        t0 = theta_newton(b[idx], y[idx], a[idx])
    return S


def total_fisher_order(a, b, th_cal):
    """Total-Fisher static: sum of 2PL information over the calibration
    planners (the same order as SC-IRT's cover phase)."""
    return [int(i) for i in np.argsort(-population_fisher(a, b, th_cal))]


def marginal_fisher_order(a, b):
    """Marginal-Fisher static: E_{theta ~ N(0,1)} 2PL information."""
    GXg = np.linspace(-3, 3, 61)
    w = np.exp(-0.5 * GXg ** 2)
    w /= w.sum()
    info = np.array([((a[i] ** 2) * sig(a[i] * (GXg - b[i])) * (1 - sig(a[i] * (GXg - b[i]))) * w).sum()
                     for i in range(len(a))])
    return [int(i) for i in np.argsort(-info)]


def disco_order(pbar):
    """DISCO-adapted: inter-planner disagreement p(1-p), descending."""
    return [int(i) for i in np.argsort(-(pbar * (1 - pbar)))]


def kmeans_anchors(a2, b2, budget, n_items):
    """tinyBenchmarks-style anchors: K-means on (a-hat, b-hat), K = budget,
    medoid per cluster (budget-specific, not a prefix order)."""
    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    E = np.stack([a2, b2], 1)
    warnings.simplefilter("ignore", ConvergenceWarning)
    km = KMeans(n_clusters=min(budget, n_items), n_init=4, random_state=0).fit(E)
    anchors = []
    for cl in range(km.n_clusters):
        mem = np.where(km.labels_ == cl)[0]
        if len(mem):
            anchors.append(int(mem[np.argmin(((E[mem] - km.cluster_centers_[cl]) ** 2).sum(1))]))
    return anchors[:budget]


def metabench_order(a2, b2, budget, n_items):
    """metabench-lite: greedy max 2PL information over a quantile grid of
    ability points (insertion order = prefix order)."""
    grid = np.quantile(b2, np.linspace(0.02, 0.98, 25))
    order, pool = [], list(range(n_items))
    while len(order) < budget and pool:
        for g in grid:
            if len(order) >= budget or not pool:
                break
            info = [a2[i] ** 2 * sig(a2[i] * (g - b2[i])) * (1 - sig(a2[i] * (g - b2[i])))
                    for i in pool]
            pk = pool[int(np.argmax(info))]
            order.append(pk)
            pool.remove(pk)
    return order


def anchorpoints_estimate(Rb, yy, budget):
    """AnchorPoints-adapted: K-means on calibration response vectors,
    cluster-weighted medoid mean (its own estimator, no IRT)."""
    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    warnings.simplefilter("ignore", ConvergenceWarning)
    km = KMeans(n_clusters=min(budget, len(Rb)), n_init=4, random_state=0).fit(Rb)
    est, tot = 0.0, 0
    for cl in range(km.n_clusters):
        mem = np.where(km.labels_ == cl)[0]
        if not len(mem):
            continue
        med = mem[np.argmin(((Rb[mem] - km.cluster_centers_[cl]) ** 2).sum(1))]
        est += len(mem) * yy[med]
        tot += len(mem)
    return est / tot


def stratified_order(types, rng):
    """Type-stratified random order (round-robin across scenario types)."""
    byt = {}
    for i in range(len(types)):
        byt.setdefault(types[i], []).append(i)
    for t in byt:
        rng.shuffle(byt[t])
    order, k = [], 0
    while len(order) < len(types):
        for t in sorted(byt):
            if k < len(byt[t]):
                order.append(byt[t][k])
        k += 1
    return order


def pirt(bs, aa, yy, S):
    """Plug-in IRT readout used natively by the IRT baselines: Newton-MAP
    ability on the administered items, point-curve fill of the rest."""
    n = len(bs)
    S = np.array(S)
    t = theta_newton(bs[S], yy[S], aa[S])
    un = [i for i in range(n) if i not in set(S.tolist())]
    return (yy[S].sum() + sig(aa[un] * (t - bs[un])).sum()) / n
