#!/usr/bin/env python3
"""Table 5 — complete-system comparison: each published method run as its OWN system.

Tables 1 and 2 each hold something fixed. Table 1 gives every method the same
budget B and compares SR-MAE (it isolates selection); Table 2 gives every bank
order the same readout and stopping rule and compares where each stops. Neither
answers the question a reader of the ATLAS or Fluid paper actually has: what do
you get if you run *their whole pipeline* — their IRT parameterisation, their
ability estimator, their selection, their stopping rule, their score
reconstruction — on this benchmark, against ours run as ours?

That is this table. Nothing is shared between the rows except the response
matrix, the protocol split (`splits.up_split`, 12 calibration : 4 evaluation
planners, bank = all 220 routes), the K_cal subsample RNG, and the metric and
its bootstrap (`metrics.paired_cluster_boot` over planner ids).

    ATLAS     (arXiv:2511.04689)  MMLE/EM 3PL with a bank-wide guessing constant,
                                  EAP ability on a 33-point trapezoid grid,
                                  top-5 randomesque Fisher selection,
                                  SE(theta) <= tau after a 30-item minimum,
                                  p-IRT accuracy reconstruction.
    Fluid     (arXiv:2509.11106)  2PL point (a, b), MAP ability by Newton with
                                  Fluid's own safeguards, greedy Fisher argmax,
                                  fixed length (its published design) and an
                                  ability-precision stop built from its own MAP
                                  machinery, p-IRT readout supplied by us.
    DriveAT  (this repo)         1PL + testlet + exact difficulty posterior,
                                  Delta-R1 acquisition, posterior-median readout,
                                  stop at c * R1 <= eps with c fixed by
                                  leave-one-planner-out on the calibration
                                  planners (mirrors run_tau_calibration.py).
    Random    (reference)         uniform order at a fixed budget, read with each
                                  system's own readout — the IES denominator.

FAIRNESS RULES (each honoured below, and each violation would be printed loudly)

 1. Every system sees exactly the same calibration data — the K_cal calibration
    planners x the 220 routes of the draw — and the same 4 evaluation planners.
    No system looks at an evaluation planner outside its own adaptive run: the
    only evaluation-planner information any system ever touches is the outcome
    of a route it has itself administered.
 2. Every stopping threshold is fixed a priori or calibrated on the calibration
    planners ONLY. ATLAS's tau in {.1, .2, .3} and its 30-item minimum are its
    published constants; ATLAS's guessing constant c and difficulty prior SD are
    profiled on the calibration block; Fluid's delta is the mean rank-adjacent
    gap of the *calibration* planners' abilities; Fluid's fixed lengths are its
    published default (n_max = 100) and a cost match to DriveAT's
    leave-one-out mean stop on the *calibration* planners; DriveAT's c is the
    leave-one-planner-out 90th percentile on the calibration planners and its
    eps in {.05, .03} is fixed a priori. No threshold is tuned on an evaluation
    planner. The report prints the provenance of every threshold.
 3. Every system is reported at its own native operating points. ATLAS runs at
    all three published tau; Fluid runs at its published fixed-length design and
    at the precision stop its own machinery supports; DriveAT runs at its two
    published eps. The cost-matched Fluid row is labelled as a cost match and is
    the only row whose length is set by another system's behaviour.

DECLARED DEVIATIONS (all forced; see the per-system notes in the code)

  ATLAS  - No partition calibration and no common-person mean-sigma linking:
           inapplicable by scale (220 items, 4-12 persons), so one calibration
           and no linking.
         - Per-item guessing c_i is not estimable from 4-12 Bernoulli draws per
           route, so c is a single bank-wide constant profiled on the
           calibration block over {0, .05, .10, .15} (c = 0 collapses to the
           2PL variant ATLAS itself reports in G.3). Reported per cell.
         - No item filtering and no model screening: all 220 routes stay in the
           bank and in the p-IRT denominator, so the estimand is unmoved and
           every system sees the same bank. The protocol's 12:4 planner split
           replaces ATLAS's ability-percentile screening.
         - No ability-space MAE and therefore no WLE anywhere: our estimand is a
           rate, and a whole-bank theta would be a re-encoding of it.
         - SE uses the released code's Bayes-modal form 1/sqrt(sum I + 1), not
           Algorithm 1 line 13's 1/sqrt(sum I); the code produced the published
           numbers and is the coherent partner of the EAP point estimate.
         - max_items = the planner's bank (<= 220), not 500.
         - Ties in the top-5 randomesque set are broken deterministically
           (information rounded to 1e-10, lowest bank index) before the uniform
           draw, because all-pass / all-fail routes share exact parameters.
  Fluid  - Fluid's own fitting code (py-irt SVI, hierarchical Normal-Gamma
           priors) does not identify on 4-12 subjects, so the 2PL item
           parameters come from a MAP fit with explicit priors
           (log a ~ N(0, .5^2), b ~ N(0, sigma_b^2), sigma_b by empirical Bayes
           on this row's own calibration block). The model form is Fluid's; the
           fitting estimator is ours. Said plainly, not hidden.
         - Fluid has no stopping rule and no score reconstruction. The
           precision stop is OUR construction from Fluid's own MAP posterior SE
           and its own leaderboard-gap threshold definition, and the p-IRT
           readout is OURS. Both are labelled as ours everywhere they appear.
  Both   - Each evaluation planner's bank is its recorded routes (210-220 of
           220); no system may select a route with no recorded outcome.

IES.  ATLAS's definition, IES = (MAE_method / MAE_ref) x (Items_method /
Items_ref), with the reference restated for a 220-route bank: uniform Random at
110 routes (half the bank) is the DECLARED reference, in place of ATLAS's
Random_100 on a 1000+ item bank. IES against Random at 55 routes is printed
alongside so the number can be compared with `driveat.metrics.ies`, whose
reference is the random order at B = 55. Each system's reference is read with
that system's own readout, so the IES of a row is that system's own efficiency
score; the reference MAEs are printed so the denominators are visible.

    python experiments/run_system_comparison.py --seeds 0 2    # shard
    python experiments/run_system_comparison.py --merge        # Table 5 + anchors
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveat.b2d import Panel
from driveat.splits import up_split, R_DRAWS
from driveat.calibration import calibrate
from driveat.bayes import bank_from_fit, track, state_from, stop_at
from driveat.acquisition import r1_traj
from driveat.metrics import paired_cluster_boot, ies

OUT = Path(os.environ.get('DRIVEAT_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
KCALS = tuple(int(x) for x in os.environ.get('DRIVEAT_KCALS', '4,8,12').split(','))
DEVICE = os.environ.get('DRIVEAT_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

NROUTES = 220               # the benchmark; per-planner banks are its recorded subset (210-220)
IES_REF = 110               # declared IES reference budget: uniform Random at half the bank
IES_REF2 = 55               # second reference, comparable with driveat.metrics.ies
ETOL = 0.05                 # error tolerance kept in the json only, not printed
ATLAS_TAUS = (0.1, 0.2, 0.3)
ATLAS_MIN = 30              # ATLAS's published minimum item count
FLUID_NMAX = 100            # Fluid's published default n_max (README)
EPS = (0.05, 0.03)          # DriveAT's published risk targets
RISK_T0 = 10                # first step of the LOO tracks used for the risk scale (run_tau_calibration)
MATCH_EPS = 0.05            # the DriveAT operating point the cost-matched Fluid row matches

# Pinned on the 16-draw run of record (results/syscmp.json, 192 planner
# evaluations). Entries are (K_cal, row label, field, value, tolerance);
# fields: 'rollouts', 'mae', 'coverage'. The DriveAT rows are also the
# cross-check against RESULTS.md Table 2 (83.5 / .0272, 79.0 / .0281,
# 70.3 / .0207 at eps = .05), which this script reproduces from its own
# leave-one-planner-out risk scale (c = 1.97 / 2.12 / 1.95).
ANCHORS = ((4, 'DriveAT eps=0.05', 'rollouts', 83.5, 1.0),
           (4, 'DriveAT eps=0.05', 'mae', .0272, .002),
           (8, 'DriveAT eps=0.05', 'rollouts', 79.0, 1.0),
           (12, 'DriveAT eps=0.05', 'rollouts', 70.3, 1.0),
           (12, 'DriveAT eps=0.05', 'mae', .0207, .002),
           (12, 'DriveAT eps=0.05', 'rollouts', 70.3, .5),
           (12, 'DriveAT eps=0.03', 'mae', .0138, .002),
           (4, 'ATLAS  tau=0.1', 'rollouts', 217.4, 1.0),
           (4, 'ATLAS  tau=0.1', 'mae', .0000, .001),
           (8, 'ATLAS  tau=0.2', 'rollouts', 66.4, 1.5),
           (8, 'ATLAS  tau=0.2', 'mae', .0311, .002),
           (4, 'ATLAS  tau=0.3', 'mae', .0514, .002),
           (12, 'ATLAS  tau=0.3', 'rollouts', 30.0, 0.5),
           (8, 'Fluid  fixed B=100', 'mae', .0230, .002),
           (12, 'Fluid  fixed B=match', 'rollouts', 69.7, 1.0),
           (12, 'Fluid  fixed B=match', 'mae', .0264, .002),
           (4, 'Fluid  SE<=delta*', 'rollouts', 3.2, 0.5),
           (4, 'Fluid  SE<=delta*', 'mae', .1406, .005),
           (12, 'Fluid  SE<=delta*', 'rollouts', 19.0, 0.5))


def subsample(cols, seed, Kc):
    """The protocol's K_cal subsample of the calibration planners."""
    if Kc >= len(cols):
        return list(cols)
    rs = np.random.RandomState(9000 + seed * 100 + Kc * 10 + 0)
    return sorted(np.array(cols)[rs.choice(len(cols), Kc, replace=False)].tolist())


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def response_matrix(Y, routes, cols):
    """(len(routes) x len(cols)) responses with nan for unobserved cells."""
    M = np.full((len(routes), len(cols)), np.nan)
    for i, r in enumerate(routes):
        for j, pi in enumerate(cols):
            if (r, pi) in Y:
                M[i, j] = Y[(r, pi)]
    return M


def first_le(v, thr, tmin=1):
    """First 1-based index t >= tmin with v[t-1] <= thr; len(v) if never."""
    v = np.asarray(v, float)
    tmin = min(max(tmin, 1), len(v))
    hit = np.where(v[tmin - 1:] <= thr)[0]
    return int(hit[0]) + tmin if len(hit) else len(v)


# ===========================================================================
# ATLAS — its own calibration, ability estimator, selection, stop and readout
# ===========================================================================
A_QX = np.linspace(-6.0, 6.0, 61)                 # MMLE/EM quadrature (mirt quadpts = 61)
A_QW = np.exp(-0.5 * A_QX ** 2)
A_QW = A_QW / A_QW.sum()
A_BG = np.linspace(-10.0, 10.0, 801)              # difficulty grid for the empirical-Bayes criterion
A_EAP = np.linspace(-4.0, 4.0, 33)                # catR eapEst default: 33 nodes on [-4, 4]
A_TRAP = np.full(33, 1.0)
A_TRAP[0] = A_TRAP[-1] = 0.5                      # trapezoidal rule (catR integrate.catR)
A_EAPW = A_TRAP * np.exp(-0.5 * A_EAP ** 2)       # x prior N(0, 1)
A_SIG_LOGA = 0.5
A_C_GRID = (0.0, 0.05, 0.10, 0.15)                # bank-wide guessing constant, profiled on calibration
A_SB_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
A_RANDOMESQUE = 5


def atlas_icc(a, b, c, th):
    """3PL with D = 1: c + (1 - c) / (1 + exp(-a (theta - b)))."""
    return c + (1 - c) * sig(np.asarray(a) * (th - np.asarray(b)))


def atlas_info(a, b, c, th):
    """Fisher information of the 3PL at theta (ATLAS Eq. 2, catR::Ii)."""
    P = np.clip(atlas_icc(a, b, c, th), 1e-9, 1 - 1e-9)
    return (np.asarray(a) ** 2) * (P - c) ** 2 * (1 - P) / ((1 - c) ** 2 * P)


def atlas_em(M, c, sigma_b, n_em=200, tol=1e-5):
    """Marginal maximum likelihood by EM, unidimensional, theta ~ N(0, 1), c fixed
    bank-wide; the M step is Fisher scoring on (log a, b) with the regularising
    priors log a ~ N(0, .5^2), b ~ N(0, sigma_b^2) that ATLAS's mirt call leaves
    implicit and that 4-12 persons per item make mandatory."""
    mk = ~np.isnan(M)
    Y1 = (np.nan_to_num(M) * mk).astype(float)
    Y0 = (mk & (np.nan_to_num(M) < 0.5)).astype(float)
    mkf = mk.astype(float)
    al, b = np.zeros(M.shape[0]), np.zeros(M.shape[0])
    for it in range(n_em):
        a = np.exp(al)
        P = np.clip(atlas_icc(a[:, None], b[:, None], c, A_QX[None, :]), 1e-9, 1 - 1e-9)
        lg = (Y1.T @ np.log(P) + Y0.T @ np.log(1 - P)) + np.log(A_QW)[None, :]        # (K, Q)
        G = np.exp(lg - logsumexp(lg, 1, keepdims=True))
        Nq, Rq = mkf @ G, Y1 @ G                                                       # (n, Q)
        al0, b0 = al.copy(), b.copy()
        for _ in range(8):
            a = np.exp(al)
            s = sig(a[:, None] * (A_QX[None, :] - b[:, None]))
            P = np.clip(c + (1 - c) * s, 1e-9, 1 - 1e-9)
            sp = (1 - c) * s * (1 - s)
            dPa = sp * a[:, None] * (A_QX[None, :] - b[:, None])                       # dP/dlog a
            dPb = -sp * a[:, None]                                                     # dP/db
            g = Rq / P - (Nq - Rq) / (1 - P)
            ga = (g * dPa).sum(1) - al / A_SIG_LOGA ** 2
            gb = (g * dPb).sum(1) - b / sigma_b ** 2
            w = Nq / (P * (1 - P))                                                     # expected information
            Iaa = (w * dPa * dPa).sum(1) + 1 / A_SIG_LOGA ** 2
            Ibb = (w * dPb * dPb).sum(1) + 1 / sigma_b ** 2
            Iab = (w * dPa * dPb).sum(1)
            det = Iaa * Ibb - Iab ** 2
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            al = np.clip(al + np.clip((Ibb * ga - Iab * gb) / det, -1, 1), -3, 3)
            b = np.clip(b + np.clip((Iaa * gb - Iab * ga) / det, -2, 2), -10, 10)
        if max(np.abs(al - al0).max(), np.abs(b - b0).max()) < tol:
            break
    return dict(a=np.exp(al), b=b, c=float(c), sigma_b=float(sigma_b), em_iters=it + 1)


def atlas_eap(a, b, c, y, adm):
    """EAP ability (ATLAS Eq. 3) on the 33-point [-4, 4] trapezoid grid, N(0, 1)
    prior; `adm` are the administered bank indices and `y` their outcomes."""
    if len(adm) == 0:
        return 0.0
    P = np.clip(atlas_icc(a[adm][:, None], b[adm][:, None], c, A_EAP[None, :]), 1e-9, 1 - 1e-9)
    ll = (y[:, None] * np.log(P) + (1 - y[:, None]) * np.log(1 - P)).sum(0)
    w = A_EAPW * np.exp(ll - ll.max())
    return float((w @ A_EAP) / w.sum())


def atlas_pirt(a, b, c, yy, adm, th):
    """p-IRT accuracy reconstruction (App. F): observed sample sum on the
    administered routes + the calibrated model's predicted probability on the
    rest, over the planner's full recorded bank."""
    n = len(a)
    m = np.ones(n, bool)
    m[list(adm)] = False
    return float((yy[list(adm)].sum() + atlas_icc(a[m], b[m], c, th).sum()) / n)


def atlas_calibrate(M):
    """ATLAS's own bank calibration: profile the bank-wide guessing constant and
    the difficulty prior SD on the CALIBRATION BLOCK ONLY, by the marginal
    likelihood with the difficulty integrated on A_BG (the analogue of the
    protocol's empirical-Bayes criterion, so the profile is not self-fulfilling)."""
    mk = ~np.isnan(M)
    Y = np.nan_to_num(M)
    db = A_BG[1] - A_BG[0]
    best = None
    for c in A_C_GRID:
        for sb in A_SB_GRID:
            f = atlas_em(M, c, sb)
            # EAP calibration abilities on the EM quadrature grid
            P = np.clip(atlas_icc(f['a'][:, None], f['b'][:, None], c, A_QX[None, :]), 1e-9, 1 - 1e-9)
            lg = ((Y * mk).T @ np.log(P) + (mk & (Y < 0.5)).T @ np.log(1 - P)) + np.log(A_QW)[None, :]
            f['th'] = np.exp(lg - logsumexp(lg, 1, keepdims=True)) @ A_QX
            Pb = np.clip(atlas_icc(f['a'][:, None, None], A_BG[None, None, :], c, f['th'][None, :, None]),
                         1e-9, 1 - 1e-9)
            ll = (mk[:, :, None] * (Y[:, :, None] * np.log(Pb)
                                    + (1 - Y[:, :, None]) * np.log(1 - Pb))).sum(1)
            lpri = -0.5 * (A_BG / sb) ** 2 - np.log(sb * np.sqrt(2 * np.pi)) + np.log(db)
            f['eb'] = float(logsumexp(ll + lpri[None, :], axis=1).sum())
            if best is None or f['eb'] > best['eb']:
                best = f
    return best


def atlas_run(f, yy, rng):
    """One ATLAS adaptive run through the whole bank. Returns per-step
    (SR_hat by p-IRT, SE(theta) by the released code's Bayes-modal form).
    Selection: item 1 uniform on {|b_i - 0| < .5} (03_atlas_cat.r; the paper's
    Alg. 1 line 5 says argmin |b_i|, the code produced the published numbers),
    then a uniform draw among the top-5 unadministered items by Fisher
    information at the running EAP, ties broken deterministically."""
    a, b, c = f['a'], f['b'], f['c']
    n = len(a)
    rem = np.ones(n, bool)
    adm, ys, Sh, SE = [], [], [], []
    th = 0.0
    win = np.where(np.abs(b - th) < 0.5)[0]
    nxt = int(rng.choice(win)) if len(win) else int(np.argmin(np.abs(b - th)))
    for _ in range(n):
        adm.append(nxt)
        ys.append(float(yy[nxt]))
        rem[nxt] = False
        th = atlas_eap(a, b, c, np.array(ys), adm)
        Sh.append(atlas_pirt(a, b, c, yy, adm, th))
        SE.append(float(1.0 / np.sqrt(atlas_info(a[adm], b[adm], c, th).sum() + 1.0)))
        if not rem.any():
            break
        cand = np.where(rem)[0]
        info = np.round(atlas_info(a[cand], b[cand], c, th), 10)
        top = cand[np.lexsort((cand, -info))[:A_RANDOMESQUE]]
        nxt = int(rng.choice(top))
    return Sh, SE


def atlas_static_readout(f, yy, order, B):
    """ATLAS's own readout on a fixed prefix of an order (its Random_B reference)."""
    adm = [int(i) for i in order[:B]]
    th = atlas_eap(f['a'], f['b'], f['c'], yy[adm], adm)
    return atlas_pirt(f['a'], f['b'], f['c'], yy, adm, th)


# ===========================================================================
# Fluid — its own ability estimator, selection and (constructed) stop
# ===========================================================================
F_MU0, F_SIGMA0, F_D = 0.0, 1.0, 1.0
F_RANGE = (-4.0, 4.0)                             # Fluid's own theta_range, not ours (-6, 6)


def fluid_map(a, b, y, theta0=None, tol=1e-6, max_iter=100):
    """Verbatim port of fluid_benchmarking.estimators.ability_estimate(method='map'):
    Newton on the MAP score with range projection, 15-step backtracking and an
    80-step bisection fallback. N(0, 1) prior, item parameters fixed."""
    inv_s2 = 1.0 / (F_SIGMA0 * F_SIGMA0)
    lo, hi = F_RANGE

    def score(t):
        return (F_MU0 - t) * inv_s2 + F_D * float((a * (y - sig(F_D * a * (t - b)))).sum())

    def score_prime(t):
        P = sig(F_D * a * (t - b))
        return -inv_s2 - F_D ** 2 * float((a * a * P * (1 - P)).sum())

    t = float(np.clip(F_MU0 if theta0 is None else theta0, lo, hi))
    for _ in range(max_iter):
        T = score(t)
        if abs(T) < tol:
            return t
        Tp = score_prime(t)
        if not np.isfinite(Tp) or Tp == 0.0:
            break
        nt = t - T / Tp
        if nt < lo or nt > hi or not np.isfinite(nt):
            nt = float(np.clip(nt, lo, hi))
        for _bt in range(15):
            Tn = score(nt)
            if abs(Tn) < abs(T) or not np.isfinite(Tn):
                break
            nt = 0.5 * (nt + t)
        t = nt
    sL, sH = score(lo), score(hi)
    if sL * sH <= 0:
        l, h = lo, hi
        for _ in range(80):
            mid = 0.5 * (l + h)
            sM = score(mid)
            if abs(sM) < tol:
                return mid
            if sL * sM > 0:
                l, sL = mid, sM
            else:
                h = mid
        return 0.5 * (l + h)
    return hi if (sL > 0 and sH > 0) else lo


def fluid_se(a, b, th):
    """Posterior-mode SE of Fluid's own MAP estimate: [1/sigma0^2 + D^2 sum a^2 P(1-P)]^-1/2.
    This is -score_prime at the estimate, i.e. machinery Fluid already computes;
    the SE and the stop built on it are OURS, not Fluid's."""
    P = sig(F_D * a * (th - b))
    return float(1.0 / np.sqrt(1.0 / F_SIGMA0 ** 2 + F_D ** 2 * float((a * a * P * (1 - P)).sum())))


def fluid_pirt(a, b, yy, adm, th):
    """p-IRT plug-in on Fluid's own 2PL parameters and MAP theta. OURS, not Fluid's:
    Fluid reports the ability itself and reconstructs no benchmark score."""
    n = len(a)
    m = np.ones(n, bool)
    m[list(adm)] = False
    return float((yy[list(adm)].sum() + sig(F_D * a[m] * (th - b[m])).sum()) / n)


def fluid_run(a, b, yy):
    """One Fluid run through the whole bank: greedy 2PL Fisher argmax at the
    running MAP ability (Eq. 5), first item at start_ability = 0, ties to the
    lowest bank index. Returns per-step (SR_hat by our p-IRT, SE of the MAP)."""
    n = len(a)
    rem = np.ones(n, bool)
    adm, Sh, SE = [], [], []
    th = 0.0
    for _ in range(n):
        cand = np.where(rem)[0]
        P = sig(F_D * a[cand] * (th - b[cand]))
        nxt = int(cand[int(np.argmax(F_D ** 2 * a[cand] ** 2 * P * (1 - P)))])
        adm.append(nxt)
        rem[nxt] = False
        ix = np.array(adm)
        th = fluid_map(a[ix], b[ix], yy[ix], theta0=th)
        Sh.append(fluid_pirt(a, b, yy, adm, th))
        SE.append(fluid_se(a[ix], b[ix], th))
    return Sh, SE


def fluid_static_readout(a, b, yy, order, B):
    """Fluid's ability on a fixed prefix, read with our p-IRT (its Random_B reference)."""
    adm = [int(i) for i in order[:B]]
    ix = np.array(adm)
    return fluid_pirt(a, b, yy, adm, fluid_map(a[ix], b[ix], yy[ix]))


# ===========================================================================
# DriveAT — the repo method, with its risk scale fixed on the calibration panel
# ===========================================================================
def drivecat_loo(panel, calR, typ, cs, f0):
    """Leave-one-planner-out on the CALIBRATION planners (mirrors
    run_tau_calibration.py): the bank is re-calibrated from the other K_cal - 1
    planners, the Delta-R1 order is run through the whole bank on the left-out
    planner, and

        c     = 90th percentile of |SR_hat_t - SR| / R1_t over the left-out
                planners and t in [RISK_T0, bank size]   (the risk scale)
        Bmatch= mean stop of c * R1_t <= MATCH_EPS over the left-out planners
                (the cost the Fluid fixed-length row is matched to)

    Both are calibration-side quantities: no evaluation planner is touched."""
    raw, act, stops = [], [], []
    tracks = []
    for j in cs:
        csl = [c for c in cs if c != j]
        f1 = calibrate(panel.Y, calR, csl, mode='1pl', device=DEVICE, sigma_b=f0['sigma_b'])
        bi, yy = panel.bank_rows(calR, j)
        bank = bank_from_fit(f1, bi, typ, sigma_g=f0['sigma_g'])
        Sh, R1 = track(bank, yy, r1_traj(bank, yy, len(bi)))
        R1 = np.array(R1)
        er = np.abs(np.array(Sh) - yy.mean())
        raw.append(R1[RISK_T0 - 1:])
        act.append(er[RISK_T0 - 1:])
        tracks.append(R1)
    raw, act = np.concatenate(raw), np.concatenate(act)
    ok = raw > 1e-6
    c = float(np.percentile(act[ok] / raw[ok], 90))
    for R1 in tracks:
        stops.append(stop_at(c * R1, MATCH_EPS))
    return c, int(round(float(np.mean(stops))))


# ===========================================================================
def run(seeds):
    panel = Panel()
    recs = []
    for seed in seeds:
        hp, ht = up_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        typ = np.array([panel.sn[r] for r in calR])
        for Kc in KCALS:
            cs = subsample(cols, seed, Kc)
            M = response_matrix(panel.Y, calR, cs)
            fa = atlas_calibrate(M)                                          # ATLAS's own bank
            ff = calibrate(panel.Y, calR, cs, mode='2pl', device=DEVICE)     # Fluid's model form, our MAP fit
            f0 = calibrate(panel.Y, calR, cs, mode='1pl', device=DEVICE, types=typ)   # DriveAT's bank
            delta = float(np.mean(np.diff(np.sort(ff['th']))))               # Fluid's calibration-side threshold
            c_risk, Bmatch = drivecat_loo(panel, calR, typ, cs, f0)
            for js in hp:
                bi, yy = panel.bank_rows(calR, js)
                n = len(bi)
                rng = np.random.RandomState(5000 + seed * 1000 + Kc * 100 + js)
                ao = np.random.RandomState(100 + seed * panel.J + js).permutation(n)   # the Random reference order
                fa_bi = {'a': fa['a'][bi], 'b': fa['b'][bi], 'c': fa['c']}
                aSh, aSE = atlas_run(fa_bi, yy, rng)
                fSh, fSE = fluid_run(ff['a'][bi], ff['b'][bi], yy)
                bank = bank_from_fit(f0, bi, typ)
                dSh, dR1 = track(bank, yy, r1_traj(bank, yy, n))
                rec = {'seed': seed, 'K': Kc, 'js': int(js), 'SR': float(yy.mean()), 'n': n,
                       'atlas_c': fa['c'], 'atlas_sigma_b': fa['sigma_b'], 'fluid_sigma_b': ff['sigma_b'],
                       'delta': delta, 'c_risk': c_risk, 'Bmatch': Bmatch, 'sigma_g': f0['sigma_g'],
                       'ATLAS': {'Shat': [float(x) for x in aSh], 'SE': [float(x) for x in aSE]},
                       'Fluid': {'Shat': [float(x) for x in fSh], 'SE': [float(x) for x in fSE]},
                       'DriveAT': {'Shat': [float(x) for x in dSh], 'R1': [float(x) for x in dR1]},
                       'ref': {'ATLAS': [atlas_static_readout(fa_bi, yy, ao, min(B, n)) for B in (IES_REF2, IES_REF)],
                               'Fluid': [fluid_static_readout(ff['a'][bi], ff['b'][bi], yy, ao, min(B, n))
                                         for B in (IES_REF2, IES_REF)],
                               'DriveAT': [float(state_from(bank, yy, ao[:min(B, n)]).readout()[0])
                                            for B in (IES_REF2, IES_REF)]}}
                recs.append(rec)
            print(f'seed {seed} K{Kc} done  (ATLAS c={fa["c"]:.2f} sigma_b={fa["sigma_b"]:.2f}; '
                  f'Fluid sigma_b={ff["sigma_b"]:.2f} delta={delta:.3f}; DriveAT c={c_risk:.2f} Bmatch={Bmatch})',
                  flush=True)
    return recs


# ===========================================================================
def _cell(rs, sysname, stops):
    """(rollouts, |err| at the stop, hit-the-bank flag) for one operating point."""
    t = np.array(stops)
    return (t, np.array([abs(r[sysname]['Shat'][ti - 1] - r['SR']) for r, ti in zip(rs, t)]),
            np.array([float(ti == r['n']) for r, ti in zip(rs, t)]))


def rows_for(rs):
    """Per-system operating points of one K_cal cell.
    Returns {label: (rollouts, |err|, hit-the-bank flag) arrays over evaluations}."""
    out = {}
    for tau in ATLAS_TAUS:
        out[f'ATLAS  tau={tau:.1f}'] = _cell(rs, 'ATLAS', [first_le(r['ATLAS']['SE'], tau, ATLAS_MIN) for r in rs])
    out['Fluid  fixed B=100'] = _cell(rs, 'Fluid', [min(FLUID_NMAX, r['n']) for r in rs])
    out['Fluid  fixed B=match'] = _cell(rs, 'Fluid', [min(r['Bmatch'], r['n']) for r in rs])
    out['Fluid  SE<=delta*'] = _cell(rs, 'Fluid', [first_le(r['Fluid']['SE'], r['delta']) for r in rs])
    for eps in EPS:
        out[f'DriveAT eps={eps:.2f}'] = _cell(
            rs, 'DriveAT', [first_le(np.array(r['DriveAT']['R1']) * r['c_risk'], eps) for r in rs])
    return out


def report(recs):
    recs = sorted(recs, key=lambda r: (r['seed'], r['K'], r['js']))
    ndraw = len(set(r['seed'] for r in recs))
    print(f'\n{len(recs)} planner evaluations over {ndraw} draws '
          f'({len(KCALS)} K_cal x {ndraw} draws x 4 evaluation planners)')
    print('\n===== threshold provenance (fairness rule 2) =====')
    print(f'  ATLAS  tau in {ATLAS_TAUS}, min {ATLAS_MIN} items, max = bank : published constants, fixed a priori')
    print('  ATLAS  guessing c, difficulty prior SD               : profiled on the calibration block only')
    print('  Fluid  fixed B=100                                   : its published default n_max')
    print(f'  Fluid  fixed B=match                                 : DriveAT\'s LOO mean stop at eps={MATCH_EPS}'
          ' on the CALIBRATION planners (a cost match, not an accuracy tuning)')
    print('  Fluid  delta (SE stop)                               : mean rank-adjacent gap of the CALIBRATION '
          'planners\' abilities   [* = our construction: Fluid states no stop]')
    print(f'  DriveAT c                                           : LOO 90th pct |err|/R1 on the CALIBRATION planners')
    print(f'  DriveAT eps in {EPS}                          : published, fixed a priori')
    print('  NO threshold on this table was tuned on an evaluation planner.')
    print(f'\n===== IES references: uniform Random at {IES_REF} routes (declared, half the bank) and at '
          f'{IES_REF2} routes (comparable with driveat.metrics.ies); each read with the row system\'s own readout =====')
    print('      * = an operating point we constructed, not the method\'s own; readout for both Fluid rows is ours')
    res = {}
    for K in KCALS:
        rs = [r for r in recs if r['K'] == K]
        js = [r['js'] for r in rs]
        ref = {s: [float(np.mean([abs(r['ref'][s][i] - r['SR']) for r in rs])) for i in (0, 1)]
               for s in ('ATLAS', 'Fluid', 'DriveAT')}
        R = rows_for(rs)
        base = R[f'DriveAT eps={EPS[0]:.2f}'][1]
        print(f'\n-- K_cal = {K} --  ATLAS c={sorted(set(r["atlas_c"] for r in rs))} '
              f'sigma_b={sorted(set(r["atlas_sigma_b"] for r in rs))}; Fluid delta '
              f'{np.mean([r["delta"] for r in rs]):.3f}; DriveAT c {np.median([r["c_risk"] for r in rs]):.2f}, '
              f'B_match {sorted(set(r["Bmatch"] for r in rs))}')
        print(f'   Random reference SR-MAE  ' + '  '.join(
            f'{s}: {ref[s][1]:.4f} @{IES_REF} / {ref[s][0]:.4f} @{IES_REF2}' for s in ('ATLAS', 'Fluid', 'DriveAT')))
        print(f'   {"row":20s} {"routes":>7s} {"of220":>6s} {"cap":>5s} {"SR-MAE":>8s} '
              f'{"IES@" + str(IES_REF):>8s} {"IES@" + str(IES_REF2):>8s}   delta vs DriveAT eps={EPS[0]:.2f}')
        for lab, (t, e, cap) in R.items():
            s = lab.split()[0]
            isbase = lab == f'DriveAT eps={EPS[0]:.2f}'
            d, lo, hi = (0.0, 0.0, 0.0) if isbase else paired_cluster_boot(e, base, js)
            i1 = ies(e.mean(), t.mean(), ref[s][1], IES_REF)
            i2 = ies(e.mean(), t.mean(), ref[s][0], IES_REF2)
            print(f'   {lab:20s} {t.mean():7.1f} {t.mean() / NROUTES:5.0%} {cap.mean():5.0%} {e.mean():8.4f} '
                  f'{i1:8.2f} {i2:8.2f}   '
                  + ('   —' if isbase else f'{d:+.4f} [{lo:+.4f},{hi:+.4f}]'))
            res[f'K{K}|{lab.strip()}'] = {'rollouts': float(t.mean()), 'frac': float(t.mean() / NROUTES),
                                          'cap': float(cap.mean()), 'mae': float(e.mean()),
                                          'coverage': float(np.mean(e <= ETOL)), 'ies110': float(i1),
                                          'ies55': float(i2), 'delta_vs_drivecat': [float(d), float(lo), float(hi)],
                                          'ref_mae': ref[s]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs=2, type=int, default=None)
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.merge:
        recs = sum([json.load(open(f)) for f in sorted(glob.glob(str(OUT / 'syscmp_*_*.json')))], [])
        json.dump(recs, open(OUT / 'syscmp.json', 'w'))
    elif args.seeds:
        lo, hi = args.seeds
        recs = run(range(lo, hi))
        json.dump(recs, open(OUT / f'syscmp_{lo}_{hi}.json', 'w'))
        print('shard saved; run with --merge after all shards')
        return
    else:
        recs = run(range(R_DRAWS))
        json.dump(recs, open(OUT / 'syscmp.json', 'w'))
    res = report(recs)
    json.dump(res, open(OUT / 'syscmp_table.json', 'w'), indent=1)
    print(f'\nwritten: {OUT / "syscmp_table.json"}')
    if not ANCHORS:
        print('anchors: TODO — pin after the 16-draw run of record')
        return
    for K, lab, field, v, tol in ANCHORS:
        got = res[f'K{K}|{lab}'][field]
        assert abs(got - v) < tol, (K, lab, field, got)
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
