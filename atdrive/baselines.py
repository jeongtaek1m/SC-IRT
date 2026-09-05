"""Published-baseline selectors and their native readouts (Table 1 rows).

"-style / -lite / -adapted" marks re-implementations from the method
descriptions on this bank; every baseline is calibrated from the same
planner panel as ATDrive. The static orders have the prefix property
(order[:B] is the budget-B subset), which is what lets one bank order serve
both the fixed-budget table and the common stopping machine.
"""
import warnings

import numpy as np

from .curves import sig, THG


def theta_newton(b, y, a, it=50):
    """Newton MAP ability under a 2PL curve set (N(0,1) prior, clipped)."""
    t = 0.0
    for _ in range(it):
        p = sig(a * (t - b))
        g = (a * (y - p)).sum() - t
        h = -((a ** 2) * p * (1 - p)).sum() - 1.0
        t -= g / h
    return float(np.clip(t, -6, 6))


def population_fisher(a, b, th_cal):
    """Mean 2PL Fisher information over the calibration planners' abilities."""
    return np.array([np.mean([(a[i] ** 2) * sig(a[i] * (t - b[i])) * (1 - sig(a[i] * (t - b[i]))) for t in th_cal])
                     for i in range(len(a))])


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
    planners, applied as a static order."""
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
    one medoid per non-empty cluster (budget-specific, not a prefix order).
    Duplicate (a, b) points (routes with identical calibration responses)
    leave clusters empty; the budget is then filled with the remaining
    routes closest to their centroid, so exactly min(budget, n) routes are
    rolled out."""
    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    E = np.stack([a2, b2], 1)
    warnings.simplefilter("ignore", ConvergenceWarning)
    k = min(budget, n_items)
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(E)
    d = ((E - km.cluster_centers_[km.labels_]) ** 2).sum(1)
    anchors = []
    for cl in range(km.n_clusters):
        mem = np.where(km.labels_ == cl)[0]
        if len(mem):
            anchors.append(int(mem[np.argmin(d[mem])]))
    rest = [int(i) for i in np.argsort(d, kind='stable') if int(i) not in set(anchors)]
    return (anchors + rest)[:k]


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


def phi_distance(Rb):
    """1 - phi (Pearson on binary rows) between calibration response vectors;
    identical rows are at distance 0, a constant row is at distance 1 from
    every non-identical row (its correlation is undefined)."""
    R = np.asarray(Rb, float)
    n = len(R)
    Rc = R - R.mean(1, keepdims=True)
    nrm = np.sqrt((Rc ** 2).sum(1))
    D = np.ones((n, n))
    ok = nrm > 1e-12
    if ok.any():
        C = (Rc[ok] @ Rc[ok].T) / np.outer(nrm[ok], nrm[ok])
        D[np.ix_(ok, ok)] = 1 - np.clip(C, -1, 1)
    same = (np.abs(R[:, None, :] - R[None, :, :]).sum(2) == 0)
    D[same] = 0.0
    np.fill_diagonal(D, 0.0)
    return D


def pam_medoids(D, k, max_iter=50):
    """Partitioning Around Medoids (BUILD + SWAP) on a distance matrix;
    returns (medoid indices, cluster label of every point)."""
    n = len(D)
    k = min(k, n)
    med = [int(np.argmin(D.sum(1)))]
    while len(med) < k:                                     # BUILD
        cur = D[:, med].min(1)
        gain = np.maximum(cur[:, None] - D, 0).sum(0)
        gain[med] = -1
        med.append(int(np.argmax(gain)))
    med = list(med)
    for _ in range(max_iter):                               # SWAP
        Dm = D[:, med]
        order = np.argsort(Dm, 1)
        near = np.array(med)[order[:, 0]]
        d1 = Dm[np.arange(n), order[:, 0]]
        d2 = Dm[np.arange(n), order[:, 1]] if k > 1 else np.full(n, np.inf)
        best, best_delta = None, -1e-12
        for j, m in enumerate(med):
            newd = np.where((near == m)[:, None], np.minimum(d2[:, None], D), np.minimum(d1[:, None], D))
            delta = newd.sum(0) - d1.sum()
            delta[med] = np.inf
            h = int(np.argmin(delta))
            if delta[h] < best_delta:
                best, best_delta = (j, h), delta[h]
        if best is None:
            break
        med[best[0]] = best[1]
    med = sorted(int(m) for m in med)
    labels = np.argmin(D[:, med], 1)
    return med, labels


def anchorpoints_select(Rb, budget):
    """AnchorPoints (Vivek et al.): K-medoids on 1 - correlation of the
    calibration response vectors, K = budget; returns (anchors, weights)
    with weights = cluster sizes (exactly min(budget, n) anchors)."""
    med, labels = pam_medoids(phi_distance(Rb), budget)
    w = np.array([(labels == c).sum() for c in range(len(med))], float)
    return med, w


def anchorpoints_estimate(Rb, yy, budget):
    """AnchorPoints readout: cluster-size-weighted mean of the anchors'
    outcomes (its own estimator, no IRT)."""
    med, w = anchorpoints_select(Rb, budget)
    return float((w * yy[med]).sum() / w.sum())


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
