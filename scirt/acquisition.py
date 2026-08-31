"""Target-aligned acquisition (PROTOCOL section 4): one posterior, one target.

The same uncertainty-aware Rasch posterior that produces the readout and the
stopping risk also chooses the next scene. No auxiliary model, no phase
switch, no localisation budget.

  UP   the reported quantity is the full-bank success rate SR, so the next
       scene is the one whose outcome most reduces the posterior L1 risk of
       the reported estimate (`r1_pick`):

           Delta R1_s = R1(D_t) - E_{Y_s | D_t}[ R1(D_t + (s, Y_s)) ]

       R1 inside the acquisition is evaluated at the branch posterior median
       (the L1-optimal point) in the same closed form as `scirt.bayes.r1_risk`;
       the reported point estimate is the MAP fill, which differs from the
       median by < .0005 SR on this panel.

  UPS  the quantity that must generalise is the evaluation-scale ability, so
       the probe rule maximises expected information about theta under the
       same curves (`eig_pick`).

"""
import numpy as np
from scipy.stats import norm

from .curves import h_


def r1_pick(M, y, S, q, rem):
    """argmax Delta R1 over candidate scenes `rem`, vectorised over the
    candidate set (closed form; a bisection over the mixture CDF finds each
    branch's posterior median)."""
    n = M.shape[1]
    un = [i for i in range(n) if i not in S]
    yo = y[S].sum() if len(S) else 0.0
    Mu = M[:, un]
    idx = {j: k for k, j in enumerate(un)}
    mu_all = Mu.sum(1)
    var_all = (Mu * (1 - Mu)).sum(1)
    cols = np.array([idx[i] for i in rem])
    mi = Mu[:, cols]                                        # (grid, K)
    p1 = q @ mi
    q1 = (q[:, None] * mi) / np.maximum(p1[None, :], 1e-12)
    q0 = (q[:, None] * (1 - mi)) / np.maximum(1 - p1[None, :], 1e-12)
    mup = mu_all[:, None] - mi
    sdp = np.sqrt(np.maximum(var_all[:, None] - mi * (1 - mi), 0) + 1e-9)

    def branch(qb, yb):
        lo, hi = np.zeros(len(rem)), np.full(len(rem), float(n))
        for _ in range(30):
            c = (lo + hi) / 2
            F = (qb * norm.cdf((c[None, :] - yb - mup) / sdp)).sum(0)
            m_ = F < 0.5
            lo = np.where(m_, c, lo)
            hi = np.where(m_, hi, c)
        c = (lo + hi) / 2
        z = (c[None, :] - yb - mup) / sdp
        return (qb * sdp * (2 * norm.pdf(z) + z * (2 * norm.cdf(z) - 1))).sum(0)

    ev = p1 * branch(q1, yo + 1.0) + (1 - p1) * branch(q0, yo)
    return rem[int(np.argmin(ev))]


def eig_pick(q, M, rem):
    """UPS probe rule: expected information gain about the evaluation-model
    ability, h(E[m]) - E[h(m)] on the theta grid."""
    mi = M[:, rem]
    mbar = (q[:, None] * mi).sum(0)
    return rem[int(np.argmax(h_(mbar) - (q[:, None] * h_(mi)).sum(0)))]
