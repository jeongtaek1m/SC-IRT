"""Evaluation-side inference (PROTOCOL sections 3-4): one posterior over the
new planner's ability theta and its planner x scenario-type testlet effects
u_g, the posterior-median success-rate readout (the L1 Bayes action), and
the posterior L1 risk R1 that drives both acquisition and stopping.

    logit P(y_s = 1) = theta - b_s + u_{g(s)},   u_g ~ N(0, sigma_g^2)

Difficulties enter through their exact conditional posteriors: the bank
curves m_s(x) = E_{b_s | A}[sigmoid(x - b_s)] are tabulated on the extended
axis XG so that m_s(theta + u) is an index lookup (`Bank.M3`). Given theta,
types are independent and each u_g integrates on the grid UG, so the
posterior factorises as q(theta) prod_g l_g(theta) with per-type
(theta x u) log-likelihood tables (`State.A`).
"""
import numpy as np
from scipy.stats import norm

from .curves import PRIOR, THG, UG, USHIFT, I0, curves_from_posterior

EPS = 1e-12


class Bank:
    """Curves of one bank on the (theta, u) grid.
    C: (361, n) curves on XG; types: (n,) scenario-type ids; sigma_g: testlet SD
    (0 -> independent items, the single-grid special case)."""

    def __init__(self, C, types, sigma_g=0.0):
        self.n = C.shape[1]
        self.types = np.asarray(types)
        self.tids = {t: [int(i) for i in np.where(self.types == t)[0]] for t in np.unique(self.types)}
        if sigma_g < 1e-9:
            self.pu, shifts = np.ones(1), np.zeros(1, int)
        else:
            lp = -0.5 * (UG / sigma_g) ** 2
            self.pu = np.exp(lp - lp.max())
            self.pu = self.pu / self.pu.sum()
            shifts = USHIFT
        self.lpu = np.log(self.pu + EPS)
        idx = I0 + np.arange(len(THG))[:, None] + shifts[None, :]            # (241, J)
        self.M3 = C[idx]                                                      # (241, J, n)
        self.V3 = self.M3 * (1 - self.M3)
        self.lM = np.log(self.M3 + EPS)
        self.l1M = np.log(1 - self.M3 + EPS)


def bank_from_fit(f, bi, types, sigma_g=None):
    """Bank of one planner's observed bank rows `bi` from a Rasch calibration
    `f` (needs f['W'], f['sigma_g']); `types` indexes the full calibration
    route list. sigma_g overrides the fitted testlet SD."""
    return Bank(curves_from_posterior(f['W'][bi]), np.asarray(types)[bi],
                f['sigma_g'] if sigma_g is None else sigma_g)


def _logmix(A, lpu):
    """log sum_j pu_j exp(A_ij) row-wise."""
    m = A.max(1, keepdims=True)
    return m[:, 0] + np.log((np.exp(A - m) * np.exp(lpu)[None, :]).sum(1) + EPS)


def mix_median(q, mu, sd, hi, iters=40):
    """Median of the normal mixture sum_i q_i N(mu_i, sd_i^2) on [0, hi];
    columns are independent problems. q, mu, sd: (241, C)."""
    lo = np.zeros(q.shape[1])
    hi = np.full(q.shape[1], float(hi))
    for _ in range(iters):
        c = (lo + hi) / 2
        F = (q * norm.cdf((c[None, :] - mu) / sd)).sum(0)
        m_ = F < 0.5
        lo = np.where(m_, c, lo)
        hi = np.where(m_, hi, c)
    return (lo + hi) / 2


def mix_l1(q, mu, sd, c):
    """E|X - c| under the normal mixture, closed form per component."""
    z = (c[None, :] - mu) / sd
    return (q * sd * (2 * norm.pdf(z) + z * (2 * norm.cdf(z) - 1))).sum(0)


class State:
    """Posterior state of one evaluation planner on a bank after observing
    the items in `S`. `add` is incremental; `readout` returns the
    posterior-median success rate and its posterior expected absolute
    error; `predictive_all` gives P(Y_s = 1 | D) for every item."""

    def __init__(self, bank, y):
        self.bank, self.y = bank, np.asarray(y, float)
        self.S, self.yo = [], 0.0
        self.logq = np.log(PRIOR)
        self.A, self.logl = {}, {}
        self.cache = {}
        for t, ix in bank.tids.items():
            self._set(t, bank.M3[:, :, ix].sum(2), bank.V3[:, :, ix].sum(2))

    def w(self, t):
        """Posterior over u_t on the theta grid: (241, J)."""
        if t in self.A:
            W = self.bank.pu[None, :] * np.exp(self.A[t] - self.A[t].max(1, keepdims=True))
            return W / W.sum(1, keepdims=True)
        return np.broadcast_to(self.bank.pu[None, :], (len(THG), len(self.bank.pu)))

    def _set(self, t, S1, S2):
        w = self.w(t)
        mean = (w * S1).sum(1)
        var = (w * S2).sum(1) + (w * S1 ** 2).sum(1) - mean ** 2
        self.cache[t] = (S1, S2, mean, np.maximum(var, 0.0))

    def add(self, s, ys=None):
        s = int(s)
        ys = self.y[s] if ys is None else float(ys)
        t = self.bank.types[s]
        b = self.bank
        ll = b.lM[:, :, s] if ys > 0.5 else b.l1M[:, :, s]
        A = self.A.get(t, 0.0) + ll
        new = _logmix(A, b.lpu)
        self.logq = self.logq + new - self.logl.get(t, 0.0)
        self.A[t], self.logl[t] = A, new
        self.S.append(s)
        self.yo += ys
        S1, S2, _, _ = self.cache[t]
        self._set(t, S1 - b.M3[:, :, s], S2 - b.V3[:, :, s])
        return self

    @property
    def q(self):
        q = np.exp(self.logq - self.logq.max())
        return q / q.sum()

    def stats(self):
        """Mean and SD of the unobserved-success total given theta: (241,), (241,)."""
        mu = sum(c[2] for c in self.cache.values())
        var = sum(c[3] for c in self.cache.values())
        return mu, np.sqrt(var + 1e-9)

    def readout(self):
        """(S_hat, R1): posterior-median success rate and E|SR - S_hat | D|."""
        n = self.bank.n
        if len(self.S) == n:
            return float(self.yo / n), 0.0
        q = self.q[:, None]
        mu, sd = self.stats()
        c = mix_median(q, mu[:, None], sd[:, None], float(n - len(self.S)))
        r = mix_l1(q, mu[:, None], sd[:, None], c)
        return float((self.yo + c[0]) / n), float(r[0] / n)

    def predictive_all(self):
        """P(Y_s = 1 | D) for every item (observed ones return their curve fill)."""
        q = self.q
        out = np.zeros(self.bank.n)
        for t, ix in self.bank.tids.items():
            wt = self.w(t)
            out[ix] = np.einsum('i,ij,ijs->s', q, wt, self.bank.M3[:, :, ix])
        return out


def state_from(bank, y, S):
    st = State(bank, y)
    for s in S:
        st.add(s)
    return st


def readout(bank, y, S):
    """Posterior-median success rate after observing items S."""
    return state_from(bank, y, S).readout()[0]


def track(bank, y, order):
    """Readout and risk along a bank order: (S_hat[t], R1[t]) for
    t = 1..len(order). Fixed budgets read S_hat[B-1]; the stopping rule
    reads the first t with the (calibrated) risk below its target."""
    return track3(bank, y, order)[:2]


def track3(bank, y, order):
    """As `track`, plus the ability posterior SD: (S_hat[t], R1[t], SD(theta)[t]).
    SD(theta) is what an ability-precision CAT rule stops on; R1 is the
    posterior L1 risk of the reported SR, which is what ATDrive stops on."""
    st = State(bank, y)
    Sh, R1, SE = [], [], []
    for s in order:
        sh, r = st.add(s).readout()
        q = st.q
        m = float(q @ THG)
        Sh.append(sh)
        R1.append(r)
        SE.append(float(np.sqrt(max(q @ (THG ** 2) - m * m, 0.0))))
    return Sh, R1, SE


def transfer(q, bank_d):
    """Transport a theta posterior to a block with no observations (UPS):
    returns (posterior-median success rate on the block, per-item P(Y=1))."""
    st = State(bank_d, np.zeros(bank_d.n))
    st.logq = np.log(np.asarray(q) + EPS)
    return st.readout()[0], st.predictive_all()


def stop_at(R1, tau, tmin=1):
    """First index (1-based rollout count) t >= tmin with R1[t] <= tau; the full
    length if never reached. tmin matches the window on which the risk scale c is
    fitted (t >= 10, PROTOCOL section 4); it never binds on this panel (earliest
    stop 29)."""
    v = np.asarray(R1)
    hit = np.where(v[tmin - 1:] <= tau)[0]
    return int(hit[0]) + tmin if len(hit) else len(v)
