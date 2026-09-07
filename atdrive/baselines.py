"""Published-baseline selectors and their native readouts (Table 1 rows).

"-style / -lite / -adapted" marks re-implementations from the method
descriptions on this bank; every baseline is calibrated from the same
planner panel as ATDrive. The static orders have the prefix property
(order[:B] is the budget-B subset), which is what lets one bank order serve
both the fixed-budget table and the common stopping machine; the two
budget-sized selectors (`kmeans_anchors`, `metabench_order`) return one subset
per requested size, as their papers do.
"""
import warnings

import numpy as np

from .acquisition import TIE_DECIMALS
from .curves import sig, THG


def theta_newton(b, y, a, it=100, theta0=0.0, rng=(-4.0, 4.0), tol=1e-6):
    """MAP ability under a 2PL curve set with a N(0, 1) prior.

    Damped Newton on the score, with range projection, backtracking and a
    bisection fallback — the same safeguards as
    fluid_benchmarking.estimators.ability_estimate (allenai/fluid-benchmarking),
    which is also ported verbatim as run_system_comparison.fluid_map. The
    undamped version this replaces walked into the saturated region and cycled:
    on the recorded trajectory of seed 9, K_cal 4, evaluation planner 11 it
    returned +5.94 where the MAP is -2.98, a plug-in SR error of .73 that by
    itself set the Fluid K4 B30 cell. `rng` is the official (-4, 4); the MAP on
    this bank never exceeds 2.6 in magnitude, so the bound never binds."""
    lo, hi = rng

    def score(t):
        return (0.0 - t) + float((a * (y - sig(a * (t - b)))).sum())

    def score_prime(t):
        p = sig(a * (t - b))
        return -1.0 - float(((a ** 2) * p * (1 - p)).sum())

    t = float(np.clip(theta0, lo, hi))
    for _ in range(it):
        T = score(t)
        if abs(T) < tol:
            return t
        Tp = score_prime(t)
        if not np.isfinite(Tp) or Tp == 0.0:
            break
        nt = float(np.clip(t - T / Tp, lo, hi))
        for _bt in range(15):                       # backtrack until |score| decreases
            if abs(score(nt)) < abs(T) or not np.isfinite(score(nt)):
                break
            nt = 0.5 * (nt + t)
        t = nt
    sL, sH = score(lo), score(hi)                   # bisection fallback: the score is decreasing
    if sL * sH <= 0:
        a_, b_ = lo, hi
        for _ in range(80):
            m = 0.5 * (a_ + b_)
            sM = score(m)
            if abs(sM) < tol:
                return m
            if sL * sM > 0:
                a_, sL = m, sM
            else:
                b_ = m
        return 0.5 * (a_ + b_)
    return hi if (sL > 0 and sH > 0) else lo


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
    """Total-Fisher static: total 2PL information over the calibration
    planners, applied as a static order (the code averages rather than sums —
    the same ranking, since the planner count is common to every route).
    Ties are broken by the project convention (round to TIE_DECIMALS, then
    lowest bank index), not by the sort's internals."""
    info = np.round(population_fisher(a, b, th_cal), TIE_DECIMALS)
    return [int(i) for i in np.argsort(-info, kind='stable')]


def marginal_fisher_order(a, b):
    """Marginal-Fisher static: E_{theta ~ N(0,1)} 2PL information, ties broken
    by the project convention (round to TIE_DECIMALS, then lowest bank index)."""
    GXg = np.linspace(-3, 3, 61)
    w = np.exp(-0.5 * GXg ** 2)
    w /= w.sum()
    info = np.array([((a[i] ** 2) * sig(a[i] * (GXg - b[i])) * (1 - sig(a[i] * (GXg - b[i]))) * w).sum()
                     for i in range(len(a))])
    return [int(i) for i in np.argsort(-np.round(info, TIE_DECIMALS), kind='stable')]


def disco_order(pbar):
    """DISCO-adapted: inter-planner disagreement p(1-p), descending; ties
    (p(1-p) takes 3-4 distinct values at K_cal = 4) are broken by the project
    convention (round to TIE_DECIMALS, then lowest bank index)."""
    return [int(i) for i in np.argsort(-np.round(pbar * (1 - pbar), TIE_DECIMALS), kind='stable')]


def kmeans_anchors(a2, b2, budget, n_items, seed=0):
    """tinyBenchmarks-style anchors: K-means on the paper's (discrimination,
    difficulty) embedding, K = budget, one medoid per non-empty cluster
    (budget-specific, not a prefix order). The source logit is
    sig(disc * theta - diff) (Polo's py-irt fork; experiments/official/
    tinybench.py:57-59, :89-90) and it clusters X = (disc, diff)
    (tinybench.py:379-381); ours is sig(a (theta - b)) = a theta - a b, so
    disc = a, diff = a b and the embedding is (a, a b) here — clustering (a, b)
    would be a different geometry. (Reflecting the second coordinate is a
    KMeans isometry, so the sign convention does not change the partition.)
    Duplicate embedding points (routes with identical calibration responses)
    leave clusters empty; the budget is then filled with the remaining routes
    closest to their centroid, ties among equidistant routes drawn at random
    under `seed` rather than taken in bank order, so exactly min(budget, n)
    routes are rolled out. `seed` sets both the K-means restart and that fill;
    every table uses the single default seed, as the official code uses a
    single fixed random_state (tinybench.py:25-27) — averaging over seeds would
    read the union of the selections and break the budget the row is priced
    at."""
    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    E = np.stack([a2, a2 * b2], 1)
    k = min(budget, n_items)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(E)
    d = ((E - km.cluster_centers_[km.labels_]) ** 2).sum(1)
    anchors = []
    for cl in range(km.n_clusters):
        mem = np.where(km.labels_ == cl)[0]
        if len(mem):
            anchors.append(int(mem[np.argmin(d[mem])]))
    rem = np.array([i for i in range(n_items) if i not in set(anchors)], int)
    rng = np.random.RandomState(seed)
    rest = rem[np.lexsort((rng.permutation(len(rem)), np.round(d[rem], TIE_DECIMALS)))]
    return (anchors + [int(i) for i in rest])[:k]


def _spread_visits(k):
    """Bit-reversal permutation of 0..k-1: the visit order for the ability
    grid. The grid itself is ascending, so visiting it in index order would
    make every prefix the lowest-ability points only; bit reversal keeps each
    prefix spread over the ability GRID (rank correlation between position and
    grid index .03 at k = 165, .17 at k = 30), which is what a length-t prefix
    has to be to read as a length-t administration. Spread over the grid is
    not range-completeness in b_hat: the routes the first 30 visits pick still
    stop short of the hardest end of the bank, which longer lengths reach."""
    nb = max(1, int(np.ceil(np.log2(k))))
    rev = np.array([int(format(i, '0%db' % nb)[::-1], 2) for i in range(k)])
    return [int(i) for i in np.argsort(rev, kind='stable')]


def metabench_order(a2, b2, th_cal, budget, n_items, prefix=False):
    """metabench-lite: greedy max 2PL information over a grid of ABILITY
    points, one route per grid point in a single pass, as select.items does
    (metabench's reduce.R grids over the fitted respondent abilities, not over
    the item difficulties). The grid is `budget` evenly spaced ability points
    over the range of the calibration planners' abilities `th_cal`, one route
    each -- the lite stand-in for reduce.R's B quantile bins of a 500-point
    uniform grid (get.info.quantiles at :132-137, the within-bin max at :166),
    whose maximum information over the many grid points inside a bin is
    reduced here to the information at a single representative theta. Uniform
    over the calibration range is reduce.R:205-210 grid.type = 2, the same
    grid the official wrapper uses and for the same reason
    (experiments/official/metabench.py:84-91: grid.type = 1, the K_cal fitted
    abilities themselves, gives at most K_cal non-empty bins and cannot reach
    the budget).

    Like metabench, the grid is sized to the request, so the subsets of two
    budgets are not nested.

    `prefix` is the one place this can depart from select.items, and it is off
    by default. reduce.R:154-177 visits the bins ASCENDING and deletes each
    pick from the pool, so the default here visits ascending too and the
    returned SET is the faithful lite subset (its order carries no meaning --
    metabench returns a subset, not an order). With prefix=True the grid
    points are visited in bit-reversed order (`_spread_visits`) so that
    order[:t] reads as a length-t administration spread over the ability range
    instead of its easy end; the adaptive scripts, which walk one metabench
    order over growing t, need that and pass it. Because each visit removes
    its pick from the pool, the visit order changes WHICH routes are selected
    and not only their order: against the ascending construction the
    prefix=True set differs by up to 3 of 30, 9 of 55, 20 of 110 and 30 of 165
    routes (measured over seeds 0 / 3 / 7 at K_cal = 12). Fixed-budget rows
    therefore take the default.
    """
    k = min(budget, n_items)
    grid = np.linspace(float(np.min(th_cal)), float(np.max(th_cal)), k)
    order, pool = [], list(range(n_items))
    for gi in (_spread_visits(k) if prefix else range(k)):
        g = grid[gi]
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
