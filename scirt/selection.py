"""Item-selection strategies for adaptive testing.

A CAT run administers routes one at a time until the ability estimate is precise
enough. What separates the strategies is how the next route is chosen:

* **non-adaptive** orderings fix the full sequence up front (random, difficulty
  spread, anchor clustering) and simply reveal it one item at a time;
* **adaptive** rules re-rank the remaining pool after every response, using the
  current ability estimate.

The two `order_*` random strategies deliberately stay separate functions even
though both build a type -> routes dict and draw from `RandomState(0)`. They
consume different numbers of draws in different shapes, so merging them would
change which routes are administered.
"""

import numpy as np

from .theta import sig


def order_strat_random(routes, route_types, seed=0):
    """Type-stratified random order: shuffle within each type, then round-robin.

    Dict insertion order is load-bearing. `setdefault` yields first-appearance
    order in `routes` (that is, response-matrix column order), and the generator
    is consumed per type in exactly that order. Sorting the keys, or hoisting the
    generator out of the function, changes the resulting sequence.
    """
    rng = np.random.RandomState(seed)
    by_type = {}
    for r in routes:
        by_type.setdefault(route_types[r], []).append(r)
    for t in by_type:
        rng.shuffle(by_type[t])
    keys = list(by_type)
    rng.shuffle(keys)

    out, i = [], 0
    while len(out) < len(routes):
        for t in keys:
            if i < len(by_type[t]):
                out.append(by_type[t][i])
        i += 1
    return out


def order_spread(pool, difficulty):
    """Cover the difficulty spectrum by recursive bisection.

    Emits the median of the difficulty-sorted pool, then the medians of the two
    halves, and so on, so any prefix of the sequence is spread across the whole
    range rather than clustered at one end. This is the strongest non-adaptive
    baseline: it needs no ability estimate yet still avoids wasting early items
    on routes far from the planner's level.
    """
    srt = sorted(pool, key=lambda r: difficulty[r])
    out, segs = [], [(0, len(srt) - 1)]
    while segs and len(out) < len(srt):
        new = []
        for a, b in segs:
            m = (a + b) // 2
            if srt[m] not in out:
                out.append(srt[m])
            if a < m:
                new.append((a, m - 1))
            if m < b:
                new.append((m + 1, b))
        segs = new
    for r in srt:
        if r not in out:
            out.append(r)
    return out


def order_anchor(routes, difficulty, K=40, seed=0):
    """tinyBenchmarks-style anchor set: K difficulty clusters, one exemplar each.

    Note the degeneracy this inherits: on a 1PL bank the difficulties take only
    about thirty distinct values across 219 routes, so the k-means objective is
    near-flat and cluster assignment is decided at float noise. See
    REPRODUCIBILITY.md — this is one of the outputs that does not transfer
    across environments.
    """
    from sklearn.cluster import KMeans

    K = min(K, len(routes))
    X = np.array([[difficulty[r]] for r in routes])
    km = KMeans(n_clusters=K, n_init=4, random_state=seed).fit(X)

    out = []
    for cid in range(K):
        members = [i for i in range(len(routes)) if km.labels_[i] == cid]
        if not members:
            continue
        out.append(
            routes[min(members, key=lambda i: abs(X[i, 0] - km.cluster_centers_[cid, 0]))]
        )
    for r in routes:
        if r not in out:
            out.append(r)
    return out


def probe_strat(routes, route_types, n=20, seed=0):
    """Fixed type-stratified probe set of n routes.

    Distinct from `order_strat_random`: this draws with `rng.choice(rs, k,
    replace=False)` per type — which internally consumes a permutation sized by
    that type — and truncates to n, rather than shuffling and round-robining
    into a full ordering.
    """
    rng = np.random.RandomState(seed)
    by_type = {}
    for r in routes:
        by_type.setdefault(route_types[r], []).append(r)
    picked = []
    for t, rs in by_type.items():
        k = max(1, round(n * len(rs) / len(routes)))
        picked += list(rng.choice(rs, min(k, len(rs)), replace=False))
    return picked[:n]


def fisher_information(theta_hat, b, a):
    """2PL Fisher information at the current ability estimate.

    The classical adaptive rule: administer the item that discriminates most
    sharply at the estimate. Requires calibrated (b, a), so it is unavailable in
    the unseen-scenario regime.
    """
    p = sig(a * (theta_hat - b))
    return a * a * p * (1 - p)


#: Denominator of the probit-logit matching constant in `shrunk_information`.
#: PROTOCOL section 4.4 writes this as 1.7^2 = 2.89; the code that produced the
#: published numbers uses the rounded 2.9. The value is kept as-run, and the
#: discrepancy is recorded rather than silently resolved — it shifts the reported
#: item counts in the third significant figure.
SHRINK_SCALE_SQ = 2.9


def shrunk_information(theta_hat, b_tilde, resid_var, scale_sq=SHRINK_SCALE_SQ):
    """Information under predicted difficulty, discounted by its own uncertainty.

    Greedy Fisher selection on *predicted* difficulty fails: the items that look
    most informative are disproportionately those whose difficulty was
    over-predicted, so selection systematically buys prediction error rather than
    information — a winner's curse.

    Marginalising a logistic over a Gaussian difficulty posterior has the
    standard closed-form approximation

        p ~= sigmoid( (theta - b_tilde) / sqrt(1 + s^2 / scale_sq) )

    with `s^2` the Stage-2 residual variance. Flattening the probability toward
    0.5 in proportion to prediction uncertainty is what removes the bias.
    """
    p = sig((theta_hat - np.asarray(b_tilde)) / np.sqrt(1 + resid_var / scale_sq))
    return p * (1 - p)
