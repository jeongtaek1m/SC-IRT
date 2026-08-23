"""Next-item selection rules.

`srvar_pick` is the paper's acquisition (PROTOCOL section 4.2 (0)): pick the
item whose observation most reduces the posterior variance of the realised
full-bank success rate S — the same quantity the stopping rule certifies.
Closed form on the theta grid, no Monte Carlo:

    Var(N S | D) = E_q[ sum_U m_j (1 - m_j) ]  +  Var_q( sum_U m_j )

The theta-EIG rule is retained for the UPS-extend bank probes (there the
probe's purpose is theta transport, not bank SR) and as the acquisition
ablation. Fisher/ATLAS/static rules exist to run the published baselines.
"""
import numpy as np

from .curves import THG, h_, sig


def srvar_pick(M, q, rem):
    """argmax expected reduction of Var(S | D) (= argmin expected posterior
    variance) over candidate items `rem`."""
    Mi = M[:, rem]
    T1 = (Mi * (1 - Mi)).sum(1)
    Ms = Mi.sum(1)
    p1 = q @ Mi
    q1 = (q[:, None] * Mi) / np.maximum(p1[None, :], 1e-12)
    q0 = (q[:, None] * (1 - Mi)) / np.maximum(1 - p1[None, :], 1e-12)
    T1p = T1[:, None] - Mi * (1 - Mi)
    Mp = Ms[:, None] - Mi

    def var_of(qq):
        return (qq * T1p).sum(0) + (qq * Mp ** 2).sum(0) - ((qq * Mp).sum(0)) ** 2

    ev = p1 * var_of(q1) + (1 - p1) * var_of(q0)
    return rem[int(np.argmin(ev))]


def eig_pick(q, M, rem):
    """Theta-information EIG: h(E[m]) - E[h(m)]. The pre-revision main
    acquisition; still canonical for UPS-extend bank probes."""
    mi = M[:, rem]
    mbar = (q[:, None] * mi).sum(0)
    return rem[int(np.argmax(h_(mbar) - (q[:, None] * h_(mi)).sum(0)))]


def fisher_2pl_pick(q, a, b, rem):
    """Fluid-style: argmax a^2 p (1-p) at the posterior-mean theta."""
    tb = (THG * q).sum()
    p = sig(a[rem] * (tb - b[rem]))
    return rem[int(np.argmax((a[rem] ** 2) * p * (1 - p)))]


def atlas_3pl_pick(q, a, b, c, rem, rng):
    """ATLAS-style: 3PL Fisher information, top-5 randomesque sampling."""
    tb = (THG * q).sum()
    base = sig(a[rem] * (tb - b[rem]))
    p = c[rem] + (1 - c[rem]) * base
    info = (a[rem] ** 2) * ((1 - c[rem]) ** 2) * (base * (1 - base)) ** 2 \
        / np.maximum(p * (1 - p), 1e-6)
    top = np.argsort(-info)[:5]
    return rem[int(top[rng.randint(len(top))])]


def kmeans_anchors(a2, b2, budget, n_items):
    """tinyBenchmarks-style static anchors: K-means on (a-hat, b-hat) with
    K = budget, medoid per cluster."""
    from sklearn.cluster import KMeans
    E = np.stack([a2, b2], 1)
    km = KMeans(n_clusters=min(budget, n_items), n_init=4, random_state=0).fit(E)
    anchors = []
    for cl in range(km.n_clusters):
        mem = np.where(km.labels_ == cl)[0]
        if len(mem):
            anchors.append(int(mem[np.argmin(((E[mem] - km.cluster_centers_[cl]) ** 2).sum(1))]))
    return anchors[:budget]


def metabench_order(a2, b2, budget, n_items):
    """metabench-lite static order: greedy max 2PL information over a
    quantile grid of ability points."""
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
