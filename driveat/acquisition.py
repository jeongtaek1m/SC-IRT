"""Target-aligned acquisition (PROTOCOL section 4): one posterior, one target.

The same posterior that produces the readout and the stopping risk also
chooses the next scene. No auxiliary model, no phase switch.

  UP   the reported quantity is the full-bank success rate SR, so the next
       scene is the one whose outcome most reduces the posterior L1 risk of
       the reported estimate (`r1_pick`):

           Delta R1_s = R1(D_t) - E_{Y_s | D_t}[ R1(D_t + (s, Y_s)) ]

       Each branch is scored at its own posterior median — the same L1
       Bayes action the readout reports — in the closed form of
       `driveat.bayes.mix_l1`. Candidates of one scenario type share the
       type's (theta x u) table, so the branch posteriors are vectorised
       per type.

  UPS  the quantity that must generalise is the block-D success rate, so the
       probe rule is the same Delta-R1 with the risk evaluated on the D bank
       (`r1_pick_transfer`); theta-EIG (`eig_pick`) is kept as an ablation.
"""
import numpy as np

from .curves import h_, THG
from .bayes import mix_median, mix_l1, State, EPS


TIE_DECIMALS = 10


def argmin_stable(ev, rem):
    """Lowest-index candidate among the minimisers of `ev` rounded to
    TIE_DECIMALS (all-pass / all-fail routes of one type share a posterior,
    so exact ties are common; rounding keeps the choice invariant to
    floating-point rescaling)."""
    evr = np.round(np.asarray(ev, float), TIE_DECIMALS)
    return int(rem[int(np.argmin(evr))])


def r1_pick(state, rem):
    """argmin over candidate scenes `rem` of the expected branch risk."""
    return argmin_stable(r1_scores(state, rem), rem)


def r1_scores(state, rem):
    """Expected branch risk E_{Y_s}[R1(D + (s, Y_s))] for every candidate in
    `rem` (same order), on the per-route scale of `State.readout`;
    Delta R1_s = R1(D) - this."""
    b = state.bank
    q = state.q
    mu_all, sd_all = state.stats()
    var_all = sd_all ** 2
    n_left = b.n - len(state.S) - 1
    by_type = {}
    for s in rem:
        by_type.setdefault(b.types[s], []).append(int(s))
    pos = {int(s): k for k, s in enumerate(rem)}
    out = np.zeros(len(rem))
    for t, cands in by_type.items():
        cands = np.array(cands)
        S1, S2, mean_t, var_t = state.cache[t]
        A = state.A.get(t)
        logl_t = state.logl.get(t, 0.0)
        w = state.w(t)
        M, V = b.M3[:, :, cands], b.V3[:, :, cands]                        # (241, J, C)
        p1 = np.einsum('i,ij,ijc->c', q, w, M)
        S1c, S2c = S1[:, :, None] - M, S2[:, :, None] - V
        mu_o = (mu_all - mean_t)[:, None]
        var_o = (var_all - var_t)[:, None]
        ev = np.zeros(len(cands))
        for py, L in ((p1, b.lM[:, :, cands]), (1 - p1, b.l1M[:, :, cands])):
            A1 = (A[:, :, None] if A is not None else 0.0) + L                # (241, J, C)
            m = A1.max(1, keepdims=True)
            E = np.exp(A1 - m) * b.pu[None, :, None]
            l1 = E.sum(1)                                                    # (241, C)
            lq = state.logq[:, None] + m[:, 0, :] + np.log(l1 + EPS) - (logl_t[:, None] if A is not None else 0.0)
            q1 = np.exp(lq - lq.max(0, keepdims=True))
            q1 = q1 / q1.sum(0, keepdims=True)
            w1 = E / (l1[:, None, :] + EPS)
            mean1 = (w1 * S1c).sum(1)
            var1 = (w1 * S2c).sum(1) + (w1 * S1c ** 2).sum(1) - mean1 ** 2
            mu1 = mu_o + mean1
            sd1 = np.sqrt(np.maximum(var_o + var1, 0.0) + 1e-9)
            c = mix_median(q1, mu1, sd1, float(max(n_left, 0)))
            ev += py * mix_l1(q1, mu1, sd1, c)
        for k, c_ in enumerate(cands):
            out[pos[int(c_)]] = ev[k]
    return out / b.n


def r1_pick_transfer(state, bank_d, rem):
    """UPS probe rule (`Delta-R1 on D`): the probe-bank candidate whose
    outcome most reduces the posterior L1 risk of the transported block-D
    success rate. The candidate's branch posteriors come from the probe
    bank; the risk is evaluated on the D bank, whose types are unobserved
    (prior u), so its per-theta statistics are fixed."""
    b = state.bank
    q = state.q
    st_d = State(bank_d, np.zeros(bank_d.n))
    mu_d, sd_d = st_d.stats()
    mu_d, sd_d = mu_d[:, None], sd_d[:, None]

    def risk(qq):
        c = mix_median(qq, mu_d, sd_d, float(bank_d.n))
        return mix_l1(qq, mu_d, sd_d, c) / bank_d.n

    by_type = {}
    for s in rem:
        by_type.setdefault(b.types[s], []).append(int(s))
    pos = {int(s): k for k, s in enumerate(rem)}
    out = np.zeros(len(rem))
    for t, cands in by_type.items():
        cands = np.array(cands)
        A = state.A.get(t)
        logl_t = state.logl.get(t, 0.0)
        w = state.w(t)
        p1 = np.einsum('i,ij,ijc->c', q, w, b.M3[:, :, cands])
        ev = np.zeros(len(cands))
        for py, L in ((p1, b.lM[:, :, cands]), (1 - p1, b.l1M[:, :, cands])):
            A1 = (A[:, :, None] if A is not None else 0.0) + L
            m = A1.max(1, keepdims=True)
            l1 = (np.exp(A1 - m) * b.pu[None, :, None]).sum(1)
            lq = state.logq[:, None] + m[:, 0, :] + np.log(l1 + EPS) - (logl_t[:, None] if A is not None else 0.0)
            q1 = np.exp(lq - lq.max(0, keepdims=True))
            q1 = q1 / q1.sum(0, keepdims=True)
            ev += py * risk(q1)
        for k, c_ in enumerate(cands):
            out[pos[int(c_)]] = ev[k]
    return argmin_stable(out, rem)


def eig_pick(state, rem):
    """theta-EIG probe rule (UPS ablation): expected information gain about
    the evaluation-model ability, h(E[p]) - E[h(p)] on the theta grid, with
    the testlet effect of each candidate's type integrated out."""
    q = state.q
    b = state.bank
    ws = {}
    for s in rem:
        if b.types[s] not in ws:
            ws[b.types[s]] = state.w(b.types[s])
    ps = np.stack([(ws[b.types[s]] * b.M3[:, :, s]).sum(1) for s in rem], 1)
    mbar = (q[:, None] * ps).sum(0)
    return argmin_stable(-(h_(mbar) - (q[:, None] * h_(ps)).sum(0)), rem)


def r1_traj(bank, y, T):
    """Greedy Delta-R1 rollout order of length T on a bank (the DriveAT order)."""
    st, S = State(bank, y), []
    for _ in range(min(T, bank.n)):
        s = r1_pick(st, [i for i in range(bank.n) if i not in S])
        S.append(s)
        st.add(s)
    return S


def fisher_pick(state, rem):
    """1PL maximum-information rule: the item whose marginal success
    probability at the posterior-mean ability is closest to 1/2. This is the
    classic CAT selection rule, written against the same posterior DriveAT
    uses, so only the objective differs."""
    b = state.bank
    q = state.q
    i = int(np.argmin(np.abs(THG - float(q @ THG))))
    ws = {}
    for s in rem:
        t = b.types[s]
        if t not in ws:
            ws[t] = state.w(t)[i]
    p = np.array([float(ws[b.types[s]] @ b.M3[i, :, s]) for s in rem])
    return argmin_stable(-(p * (1 - p)), rem)


def traj(bank, y, T, pick):
    """Greedy rollout order of length T under any of the pick rules above."""
    st, S = State(bank, y), []
    for _ in range(min(T, bank.n)):
        s = pick(st, [i for i in range(bank.n) if i not in S])
        S.append(s)
        st.add(s)
    return S
