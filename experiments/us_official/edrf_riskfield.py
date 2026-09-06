#!/usr/bin/env python3
"""EDRF risk-field descriptor for the ATDrive US table ("Risk field" row), rebuilt
faithfully from the published definition.

Source of the method
--------------------
J. Jiang, Z. Han, Y. Wang, M. Cai, Q. Meng, Q. Xu, J. Wang,
"EDRF: Enhanced Driving Risk Field Based on Multimodal Trajectory Prediction and
Its Applications", arXiv:2410.14996v1 (19 Oct 2024); an IEEE conference version
exists (IEEE Xplore 10919970) whose venue was NOT verified here, so cite the
arXiv version unless the venue is checked -- the shipped row's citation to
Westhofen et al. 2022 (a survey that defines no field) is wrong either way.
Implemented here: Sec. 3.2 (DRP along a predicted trajectory), Sec. 3.3 (complete
EDRF model, virtual mass), Sec. 4.1 (traffic risk monitoring: IR and F), Sec. 4.2
(ego-vehicle risk analysis: kinematic look-ahead + Laplace cross-section).
Constants: Table II (q, b, k, c, alpha, beta, gamma) and Table III (q_ego, b_ego,
k_ego, c_ego).  Look-ahead t_la = 6 s (Sec. 4.2).

Formulas as printed in the paper
--------------------------------
  agent (Gaussian cross-section, Sec. 3.2)
      DRP(s, d)   = a(s) * exp(-d^2 / (2 sigma(s)^2))
      a(s)        = q * (s - s_pt)^2
      sigma(s)    = (b + k * kappa_bar_pt) * s + c
  multimodal combination (Sec. 3.3)
      DRP^C(x,y)  = sum_i p_i * a(s_i) * exp(-d_i^2 / (2 sigma(s_i)^2))
  virtual mass and field (Sec. 3.3)
      M           = m * T * (alpha * v^beta + gamma)
      EDRF(x,y)   = DRP^C(x,y) * M
  ego (Laplace cross-section, Sec. 4.2)
      DRP_ego     = a_ego(s) * exp(-|d| / lambda(s))
      a_ego(s)    = q_ego * |s - v * t_la|
      lambda(s)   = (b_ego + k_ego * |delta|) * s + c_ego
      R           = L / tan(delta)                (simplified bicycle model)
  interaction risk (Sec. 4.1)
      IR_ij(x,y)  = EDRF_i(x,y) * EDRF_j(x,y)
      F_ij        = max_{x,y} IR_ij(x,y)

Input
-----
The PDM-Lite reference rollout of each of the 220 Bench2Drive routes,
/data1/jeongtae/b2d_eval_sensors/route_<id>/anno/*.json.gz (10 Hz; ego x, y,
theta, speed, steer, angular_velocity + per-agent bounding_boxes with location,
speed, base_type, extent).  PDM-Lite is EXCLUDED_PLANNER in the ATDrive panel and
the RelGraph encoder reads the same rollout, so this is the protocol's route
representation, not planner leakage.  No response matrix, planner outcome or
scenario-type label is read anywhere in this file.

Documented deviations from the paper are listed in DEVIATIONS below and copied
into the provenance JSON.
"""
import argparse
import glob
import gzip
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

# ----------------------------------------------------------------- constants --
DT = 0.1                     # anno logging period (10 Hz)
T_LA = 6.0                   # look-ahead time, Sec. 4.2 ("6 seconds in this paper")
H = int(round(T_LA / DT))    # 60 frames of future used for an agent trajectory

# Table II — EDRF model parameters (agents)
Q, B, K, C = 1e-4, 0.04, 1.0, 0.5
ALPHA, BETA, GAMMA = 1.566e-14, 6.687, 0.3345
# Table III — EDRF model parameters of the ego vehicle
Q_EGO, B_EGO, K_EGO, C_EGO = 0.004, 0.05, 1.0, 0.5

WHEELBASE = 2.85             # m, CARLA vehicle.lincoln.mkz_2020 (the B2D ego)
DELTA_MAX = math.radians(70.0)   # CARLA front-wheel steering limit

# actual mass m [kg] by Bench2Drive base_type / class.  The type coefficient T of
# M = m * T * (alpha v^beta + gamma) is not tabulated in EDRF nor in the cited
# Wang et al. (2015) virtual-mass paper, so T = 1 for every entity (see
# DEVIATIONS); type enters only through m.
MASS_KG = {'car': 1500.0, 'van': 2500.0, 'truck': 8000.0, 'bus': 12000.0,
           'bicycle': 90.0, 'motorcycle': 250.0, 'walker': 70.0, 'pedestrian': 70.0}
TYPE_COEF = 1.0
DEFAULT_MASS = 1500.0

# numerical settings of the maximiser (not part of the model)
RIDGE_STEP = 0.25            # m, seed spacing along each trajectory
MAX_RIDGE_PTS = 768          # cap on ridge seeds per trajectory
GRID_N = 64                  # area-seed grid: cells per axis over the padded bbox
GRID_MIN_STEP = 1.0          # m, coarsest area-seed spacing
N_REFINE = 4                 # seeds carried into the local refinement
REFINE_H0 = 0.5              # m, minimum half width of the first refinement window
REFINE_N = 9                 # points per axis in a refinement window
REFINE_ROUNDS = 4            # -> final step ~ 2*0.5/(3^3 * 8) ~ 0.005 m
SKIP_BOUND = 1e-300          # rigorous upper bound below which a pair is F = 0
REL_TOL = 1e-9               # relative tolerance of the bound-ordered early exit
LOG_FLOOR = 1e-12            # floor inside log10(. + floor) for the aggregates

DEVIATIONS = [
    "Multimodal prediction (Sec. 3.1, QCNet on Argoverse 2) is replaced by the "
    "single realised future of each agent taken from the same PDM-Lite rollout, "
    "i.e. n = 1 and p_1 = 1 in DRP^C. No trajectory predictor is available in this "
    "setting; the deterministic substitution removes the multimodal spread that "
    "the paper's Gaussian widths are meant to carry.",
    "An agent's realised future is truncated when its track ends (log end or the "
    "agent leaves the annotation set); the remaining part of the 6 s horizon is "
    "filled by a constant-velocity straight extension from the last observed "
    "step, so that s_pt stays ~ v * t_la and the aggregate is not biased by route "
    "length. The realised fraction of the horizon is reported per route.",
    "The type coefficient T in M = m * T * (alpha v^beta + gamma) is not given in "
    "EDRF (Table II lists only q, b, k, c, alpha, beta, gamma) and no table for it "
    "exists in the cited Wang et al. (2015) virtual-mass paper, so T = 1 for every "
    "entity and the entity type enters only through the actual mass m [kg].",
    "The ego steering angle delta of the simplified bicycle model is not logged by "
    "Bench2Drive (only the normalised CARLA steer command), so delta = "
    "atan(L * omega_z / v) from the logged yaw rate, L = 2.85 m (Lincoln MKZ 2020), "
    "clipped to |delta| <= 70 deg; the look-ahead arc uses the same curvature.",
    "kappa_bar_pt ('average curvature of points along the trajectory') is taken as "
    "the arclength-weighted mean |curvature|, i.e. total absolute heading change "
    "divided by trajectory length.",
    "The paper gives no footprint / extent term: entities are points on their "
    "centre trajectory and their size enters only through the initial cross-section "
    "width c (c_ego). Bench2Drive bounding-box extents are therefore used only to "
    "identify entity type, never to inflate the field.",
    "F_ij = max_{x,y} IR_ij is maximised by exact evaluation on ridge seeds (every "
    "0.25 m along both trajectories) plus a padded coarse area grid (<= 1 m), "
    "followed by 4 rounds of 9x9 local refinement around the 4 best seeds (final "
    "step ~0.005 m), rather than by one uniform grid: resolving sigma(0) = c = 0.5 m "
    "over the whole region a 6 s trajectory pair spans is unaffordable at 63k "
    "frames. Agreement with a brute-force 0.05 m grid is reported by --validate. "
    "The paper does not specify a discretisation.",
    "Route-level aggregation of the per-frame F values into 6 descriptor columns "
    "is ours; the paper defines F per entity pair and per instant only, and its "
    "F_thld is left unset (no threshold-based column is produced).",
]

COLS = ['edrf_logFego_sum_mean', 'edrf_logFego_sum_max', 'edrf_logFego_pair_max',
        'edrf_Fego_tint', 'edrf_logFtraffic_mean', 'edrf_logFtraffic_max']


# ------------------------------------------------------------------ geometry --
def polyline_arclength(P):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1) if len(P) > 1 else np.zeros(0)
    return np.concatenate([[0.0], np.cumsum(seg)]), seg


def mean_abs_curvature(P, S):
    """Arclength-weighted mean |curvature| of a polyline (kappa_bar_pt)."""
    if len(P) < 3 or S[-1] <= 1e-9:
        return 0.0
    hd = np.arctan2(np.diff(P[:, 1]), np.diff(P[:, 0]))
    dh = np.abs((np.diff(hd) + np.pi) % (2 * np.pi) - np.pi)
    return float(dh.sum() / S[-1])


def project(X, P, S, seg):
    """Nearest point on a polyline. X (m,2) -> (d (m,), s (m,), proj (m,2))."""
    if len(P) == 1:
        v = X - P[0]
        d = np.linalg.norm(v, axis=1)
        return d, np.zeros(len(X)), np.repeat(P, len(X), axis=0)
    A = P[:-1]
    E = np.diff(P, axis=0)
    L2 = np.maximum((E ** 2).sum(1), 1e-12)
    W = X[:, None, :] - A[None, :, :]
    t = np.clip((W * E[None, :, :]).sum(2) / L2[None, :], 0.0, 1.0)
    PR = A[None, :, :] + t[:, :, None] * E[None, :, :]
    D = np.linalg.norm(X[:, None, :] - PR, axis=2)
    k = np.argmin(D, axis=1)
    r = np.arange(len(X))
    return D[r, k], S[k] + t[r, k] * seg[k], PR[r, k]


def resample(P, S, step, max_pts=None):
    """Points along a polyline every `step` metres -> (points, their arclengths)."""
    if S[-1] <= step:
        return P, S
    if max_pts is not None:
        step = max(step, S[-1] / (max_pts - 1))
    u = np.arange(0.0, S[-1] + 1e-9, step)
    return np.stack([np.interp(u, S, P[:, 0]), np.interp(u, S, P[:, 1])], 1), u


# -------------------------------------------------------------------- fields --
class Field:
    """One entity's EDRF: trajectory + cross-section model + virtual mass."""

    def __init__(self, P, mass, speed, kind, delta=0.0):
        self.P = np.asarray(P, float)
        self.S, self.seg = polyline_arclength(self.P)
        self.spt = float(self.S[-1])
        self.segmax = float(self.seg.max()) if len(self.seg) else 0.0
        self.kind = kind
        self.M = mass * TYPE_COEF * (ALPHA * abs(speed) ** BETA + GAMMA)
        if kind == 'gauss':
            self.kbar = mean_abs_curvature(self.P, self.S)
            self.wslope = B + K * self.kbar
            self.w0 = C
            self.amax = Q * self.spt ** 2
        else:                                   # ego, Laplace cross-section
            self.delta = float(delta)
            self.wslope = B_EGO + K_EGO * abs(self.delta)
            self.w0 = C_EGO
            self.amax = Q_EGO * self.spt
        self.wmax = self.wslope * self.spt + self.w0

    def profile(self, s):
        """(height a(s), width sigma(s) or lambda(s)) along the trajectory."""
        if self.kind == 'gauss':
            return Q * (s - self.spt) ** 2, self.wslope * s + C
        return Q_EGO * np.abs(s - self.spt), self.wslope * s + C_EGO

    def kernel(self, d, w):
        """Cross-section shape: Gaussian for agents, Laplace for the ego."""
        if self.kind == 'gauss':
            return np.exp(-(d ** 2) / (2.0 * w ** 2))
        return np.exp(-np.abs(d) / w)

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if self.spt <= 1e-9:
            return np.zeros(len(X))
        d, s, _ = project(X, self.P, self.S, self.seg)
        if self.kind == 'gauss':
            a = Q * (s - self.spt) ** 2
            sig = self.wslope * s + C
            return self.M * a * np.exp(-(d ** 2) / (2.0 * sig ** 2))
        a = Q_EGO * np.abs(s - self.spt)
        lam = self.wslope * s + C_EGO
        return self.M * a * np.exp(-np.abs(d) / lam)


def pair_upper_bound(f1, f2, dmin):
    """Rigorous upper bound on max_x EDRF_1(x) EDRF_2(x).

    Every height is <= a_max (a(s) peaks at s = 0), every width is <= w_max, and
    for any point x the two cross-section distances satisfy d_1 + d_2 >= dmin, the
    distance between the two trajectories. Minimising the exponent over that split
    is closed form in each of the three kernel combinations:

      Gaussian x Gaussian   dmin^2 / (2 (s1^2 + s2^2))
      Laplace  x Laplace    dmin / max(l1, l2)
      Gaussian x Laplace    dmin^2 / (2 s^2)                  if s^2 / l >= dmin
                            dmin / l - s^2 / (2 l^2)          otherwise
    """
    peak = f1.M * f1.amax * f2.M * f2.amax
    if dmin <= 0 or peak <= 0.0:
        return peak
    k1, k2 = f1.kind, f2.kind
    if k1 == 'gauss' and k2 == 'gauss':
        e = dmin ** 2 / (2.0 * (f1.wmax ** 2 + f2.wmax ** 2))
    elif k1 != 'gauss' and k2 != 'gauss':
        e = dmin / max(f1.wmax, f2.wmax)
    else:
        sg = f1.wmax if k1 == 'gauss' else f2.wmax          # Gaussian width
        lp = f2.wmax if k1 == 'gauss' else f1.wmax          # Laplace width
        e = (dmin ** 2 / (2.0 * sg ** 2) if sg ** 2 / lp >= dmin
             else dmin / lp - sg ** 2 / (2.0 * lp ** 2))
    return peak * math.exp(-e)


def pair_max(f1, f2):
    """max_{x,y} EDRF_1(x,y) * EDRF_2(x,y).

    A single uniform grid cannot do this: it must resolve a cross-section of
    sigma(0) = c = 0.5 m (the aliasing the shipped 2.0 m grid suffered from) over a
    region that two 6 s trajectories can span by 150 m. The maximum is therefore
    searched with an exact evaluation (nearest-point projection, no approximation
    of the field) on three candidate sets and then refined:

      ridge seeds   points every 0.25 m along both trajectories -- catches the
                    narrow peaks that sit on a trajectory, which a coarse grid
                    aliases away;
      area seeds    a coarse grid (<= 1 m, at most 64 cells per axis) over the
                    bounding box of both trajectories padded by 3 cross-section
                    widths -- catches the wide far-field maxima that lie between
                    the trajectories, where both cross-sections are broad;
      refinement    4 rounds of 9x9 local search around each of the best 4 seeds,
                    window shrinking by 3 each round, final step ~0.006 m.

    Validated against a brute-force 0.05 m grid over the same padded region
    (--validate).
    """
    if f1.spt <= 1e-9 or f2.spt <= 1e-9:
        return 0.0
    R1, _ = resample(f1.P, f1.S, RIDGE_STEP, MAX_RIDGE_PTS)
    R2, _ = resample(f2.P, f2.S, RIDGE_STEP, MAX_RIDGE_PTS)
    P = np.vstack([f1.P, f2.P])
    pad = 3.0 * max(f1.wmax, f2.wmax)
    lo, hi = P.min(0) - pad, P.max(0) + pad
    g = max(GRID_MIN_STEP, float((hi - lo).max()) / GRID_N)
    ax = np.arange(lo[0], hi[0] + 1e-9, g)
    ay = np.arange(lo[1], hi[1] + 1e-9, g)
    gx, gy = np.meshgrid(ax, ay)
    X = np.vstack([R1, R2, np.stack([gx.ravel(), gy.ravel()], 1)])
    v = f1(X) * f2(X)
    keep = X[np.argsort(v)[-N_REFINE:]]
    best, bx = float(v.max()), X[int(np.argmax(v))]
    off = np.linspace(-1.0, 1.0, REFINE_N)
    gg, hh = np.meshgrid(off, off)
    G = np.stack([gg.ravel(), hh.ravel()], 1)
    for x0 in keep:
        cx, h = x0.copy(), max(g / 2.0, REFINE_H0)
        for _ in range(REFINE_ROUNDS):
            Xr = cx[None, :] + h * G
            vr = f1(Xr) * f2(Xr)
            j = int(np.argmax(vr))
            if vr[j] > best:
                best, bx = float(vr[j]), Xr[j].copy()
            cx = Xr[j]
            h /= 3.0
    return best


def pair_bound(f1, f2):
    """Rigorous upper bound on F_12 from the distance between the two polylines.

    The vertex-to-vertex minimum can overshoot the true polyline distance by at
    most half of the two longest segments, which is subtracted before bounding.
    """
    d = float(np.linalg.norm(f1.P[:, None, :] - f2.P[None, :, :], axis=2).min())
    d = max(0.0, d - 0.5 * (f1.segmax + f2.segmax))
    return pair_upper_bound(f1, f2, d)


def pair_F(f1, f2):
    if pair_bound(f1, f2) < SKIP_BOUND:
        return 0.0
    return pair_max(f1, f2)


def ego_pair_values(ego, flds):
    """F(ego, a) for every agent a, evaluating in decreasing upper-bound order and
    stopping once the remaining bounds cannot change the sum by more than
    REL_TOL: every skipped pair is <= the bound at which we stopped."""
    bnd = [pair_bound(ego, f) for f in flds]
    vals, tot, n_eval = np.zeros(len(flds)), 0.0, 0
    for i in sorted(range(len(flds)), key=lambda i: -bnd[i]):
        b = bnd[i]
        if b < SKIP_BOUND or (tot > 0.0 and b <= REL_TOL * tot):
            break
        vals[i] = pair_max(ego, flds[i])
        tot += vals[i]
        n_eval += 1
    return vals, n_eval


def traffic_max(flds):
    """max over agent-agent pairs (Sec. 4.1), bound-ordered with early exit."""
    pairs = [(i, j) for i in range(len(flds)) for j in range(i + 1, len(flds))]
    if not pairs:
        return 0.0, 0
    bnd = [pair_bound(flds[i], flds[j]) for i, j in pairs]
    best, n_eval = 0.0, 0
    for k in np.argsort(bnd)[::-1]:
        if bnd[k] < SKIP_BOUND or bnd[k] <= best:
            break
        i, j = pairs[k]
        best = max(best, pair_max(flds[i], flds[j]))
        n_eval += 1
    return best, n_eval


# ---------------------------------------------------------------- rollout IO --
def load_route(route_dir):
    files = sorted(glob.glob(os.path.join(route_dir, 'anno', '*.json.gz')))
    return [json.load(gzip.open(f)) for f in files]


def mass_of(bb):
    if bb.get('class') == 'walker':
        return MASS_KG['walker']
    bt = (bb.get('base_type') or '').lower()
    for key, m in MASS_KG.items():
        if key and key in bt:
            return m
    tid = (bb.get('type_id') or '').lower()
    for key, m in MASS_KG.items():
        if key and key in tid:
            return m
    return DEFAULT_MASS


def build_tracks(frames):
    """{agent_id: {frame: (x, y)}} plus {agent_id: {frame: speed}} and mass."""
    pos, spd, mass = {}, {}, {}
    for t, fr in enumerate(frames):
        for bb in fr.get('bounding_boxes', []):
            if bb.get('class') not in ('vehicle', 'walker'):
                continue
            loc = bb.get('location')
            aid = bb.get('id')
            if not loc or aid is None:
                continue
            pos.setdefault(aid, {})[t] = (float(loc[0]), float(loc[1]))
            spd.setdefault(aid, {})[t] = float(bb.get('speed') or 0.0)
            mass[aid] = mass_of(bb)
    return pos, spd, mass


def agent_path(track, t0, n_frames):
    """Realised future over [t0, t0 + H], constant-velocity extension to H steps.

    Returns (P, realised_steps).  Gaps in the track end the realised part.
    """
    pts = [track[t0]]
    t = t0
    while t + 1 <= t0 + H and (t + 1) in track and (t + 1) < n_frames:
        t += 1
        pts.append(track[t])
    real = len(pts) - 1
    if real < H:
        P = np.array(pts, float)
        if len(P) >= 2:
            v = (P[-1] - P[-2]) / DT
        else:
            v = np.zeros(2)
        for _ in range(H - real):
            pts.append(tuple(np.array(pts[-1]) + v * DT))
    return np.array(pts, float), real


def ego_arc(fr):
    """Sec. 4.2 kinematic look-ahead: constant speed, constant steering angle.

    Returns (P, delta).  Heading in the anno (x, y) plane is theta - pi/2
    (verified against finite differences of the logged positions).
    """
    v = float(fr['speed'])
    psi = float(fr['theta']) - math.pi / 2.0
    om = float(fr['angular_velocity'][2])
    if v < 1e-3:
        delta = 0.0
    else:
        delta = math.atan(WHEELBASE * om / v)
        delta = max(-DELTA_MAX, min(DELTA_MAX, delta))
    kap = math.tan(delta) / WHEELBASE
    x0, y0 = float(fr['x']), float(fr['y'])
    tt = np.arange(0.0, T_LA + 1e-9, DT)
    s = v * tt
    if abs(kap) < 1e-6:
        X = x0 + s * math.cos(psi)
        Y = y0 + s * math.sin(psi)
    else:
        ph = psi + kap * s
        X = x0 + (np.sin(ph) - math.sin(psi)) / kap
        Y = y0 - (np.cos(ph) - math.cos(psi)) / kap
    return np.stack([X, Y], 1), delta


# --------------------------------------------------------------- per-route ---
def route_features(route_dir, want_traffic=True, verbose=False):
    frames = load_route(route_dir)
    n = len(frames)
    rid = str(json.load(open(os.path.join(route_dir, 'meta.json')))['route_id'])
    pos, spd, mass = build_tracks(frames)
    Fsum, Fpair, Ftr = [], [], []
    real_frac, n_pair, n_skip = [], 0, 0
    n_tpair, n_teval = 0, 0
    for t in range(n):
        eP, delta = ego_arc(frames[t])
        ego = Field(eP, MASS_KG['car'], frames[t]['speed'], 'laplace', delta)
        flds = []
        for aid, tr in pos.items():
            if t not in tr:
                continue
            P, real = agent_path(tr, t, n)
            real_frac.append(real / H)
            flds.append(Field(P, mass[aid], spd[aid][t], 'gauss'))
        vals, ne = ego_pair_values(ego, flds)
        n_pair += len(flds)
        n_skip += len(flds) - ne
        Fsum.append(float(vals.sum()) if len(vals) else 0.0)
        Fpair.append(float(vals.max()) if len(vals) else 0.0)
        if want_traffic:
            g, net = traffic_max(flds)
            Ftr.append(g)
            n_tpair += len(flds) * (len(flds) - 1) // 2
            n_teval += net
        else:
            Ftr.append(0.0)
    Fsum = np.array(Fsum)
    Fpair = np.array(Fpair)
    Ftr = np.array(Ftr)
    lsum = np.log10(Fsum + LOG_FLOOR)
    stats = np.array([lsum.mean(), lsum.max(),
                      np.log10(Fpair + LOG_FLOOR).max(),
                      float(Fsum.sum() * DT),
                      np.log10(Ftr + LOG_FLOOR).mean(),
                      np.log10(Ftr + LOG_FLOOR).max()], float)
    diag = {'route_id': rid, 'n_frames': n, 'n_agents_tracked': len(pos),
            'n_ego_agent_pairs': n_pair,
            'frac_ego_pairs_bound_pruned': n_skip / max(n_pair, 1),
            'n_agent_agent_pairs': n_tpair,
            'frac_traffic_pairs_bound_pruned': 1 - n_teval / max(n_tpair, 1),
            'frac_frames_Fsum_zero': float((Fsum <= 0).mean()),
            'mean_realised_horizon_frac': float(np.mean(real_frac)) if real_frac else 1.0,
            'Fsum_median': float(np.median(Fsum)), 'Fsum_max': float(Fsum.max()),
            'Ftraffic_max': float(Ftr.max())}
    if verbose:
        print(rid, diag, flush=True)
    return rid, stats, diag


def _work(args):
    rd, want_traffic = args
    t0 = time.time()
    try:
        rid, stats, diag = route_features(rd, want_traffic)
        diag['seconds'] = round(time.time() - t0, 1)
        print(f'ok {rid} {diag["n_frames"]}f {diag["seconds"]}s', flush=True)
        return rid, stats, diag
    except Exception as e:                       # noqa: BLE001 - report and skip
        print(f'ERR {rd}: {e}', flush=True)
        raise


# ------------------------------------------------------------- verification --
def brute_grid_max(f1, f2, step):
    P = np.vstack([f1.P, f2.P])
    pad = 3.0 * max(f1.wmax, f2.wmax)
    xs = np.arange(P[:, 0].min() - pad, P[:, 0].max() + pad + 1e-9, step)
    ys = np.arange(P[:, 1].min() - pad, P[:, 1].max() + pad + 1e-9, step)
    best = 0.0
    for y in ys:                                  # chunked to bound memory
        X = np.stack([xs, np.full_like(xs, y)], 1)
        best = max(best, float((f1(X) * f2(X)).max()))
    return best


def validate(route_dir, n_pairs=40, step=0.05, seed=0):
    """Compare the seeded maximiser with a brute-force fine grid."""
    rng = np.random.default_rng(seed)
    frames = load_route(route_dir)
    pos, spd, mass = build_tracks(frames)
    n = len(frames)
    rel = []
    tried = 0
    while len(rel) < n_pairs and tried < 20 * n_pairs:
        tried += 1
        t = int(rng.integers(0, n))
        cand = [a for a in pos if t in pos[a]]
        if not cand:
            continue
        aid = cand[int(rng.integers(0, len(cand)))]
        eP, delta = ego_arc(frames[t])
        ego = Field(eP, MASS_KG['car'], frames[t]['speed'], 'laplace', delta)
        P, _ = agent_path(pos[aid], t, n)
        f = Field(P, mass[aid], spd[aid][t], 'gauss')
        if ego.spt <= 1e-9 or f.spt <= 1e-9:
            continue
        fast = pair_max(ego, f)
        slow = brute_grid_max(ego, f, step)
        if max(fast, slow) < 1e-12:
            continue
        rel.append((fast - slow) / max(slow, 1e-30))
        print(f't={t} agent={aid} seeded={fast:.6e} grid{step}={slow:.6e} '
              f'rel={rel[-1]:+.3e}', flush=True)
    rel = np.array(rel)
    print(f'\n{len(rel)} pairs: seeded/grid-{step}m relative difference '
          f'min {rel.min():+.2e} max {rel.max():+.2e} mean {rel.mean():+.2e} '
          f'(negative = seeded below grid)')
    return {'n_pairs': int(len(rel)), 'grid_step_m': step,
            'rel_min': float(rel.min()), 'rel_max': float(rel.max()),
            'rel_mean': float(rel.mean())}


# -------------------------------------------------------------------- main ---
def check_heading(rollouts, out_json=''):
    """Sanity check of the two rollout conventions this file relies on:

    (a) the heading of the ego in the anno (x, y) plane is theta - pi/2, and
    (b) angular_velocity[2] is the yaw rate in rad/s with the same sign as theta.
    """
    err, corr = [], []
    for rd in sorted(glob.glob(os.path.join(rollouts, 'route_*'))):
        fs = sorted(glob.glob(os.path.join(rd, 'anno', '*.json.gz')))[:400]
        F = [json.load(gzip.open(f)) for f in fs[::4]]
        if len(F) < 5:
            continue
        x = np.array([f['x'] for f in F])
        y = np.array([f['y'] for f in F])
        th = np.array([f['theta'] for f in F]) - math.pi / 2
        sp = np.array([f['speed'] for f in F])
        av = np.array([f['angular_velocity'][2] for f in F])
        hd = np.arctan2(np.diff(y), np.diff(x))
        m = sp[:-1] > 2.0
        if m.sum() > 3:
            e = np.abs((th[:-1][m] - hd[m] + np.pi) % (2 * np.pi) - np.pi)
            err.append(float(np.median(e)))
        dth = ((np.diff(th) + np.pi) % (2 * np.pi) - np.pi) / (4 * DT)
        if np.std(dth) > 1e-6 and np.std(av[:-1]) > 1e-6:
            corr.append(float(np.corrcoef(dth, av[:-1])[0, 1]))
    res = {'n_routes': len(err),
           'median_heading_error_rad': float(np.median(err)),
           'p95_heading_error_rad': float(np.percentile(err, 95)),
           'median_corr_dtheta_vs_angular_velocity_z': float(np.median(corr)),
           'min_corr': float(np.min(corr)),
           'note': 'ego heading in the (x, y) plane = theta - pi/2; '
                   'angular_velocity[2] is the yaw rate in rad/s (positive corr).'}
    print(json.dumps(res, indent=1))
    if out_json:
        json.dump(res, open(out_json, 'w'), indent=1)
    return res


def check_ego_arc(rollouts, out_json='', n_routes=40, seed=0):
    """Does the Sec. 4.2 kinematic look-ahead (constant speed, constant delta from
    the logged yaw rate) actually track the ego? Median displacement error of the
    arc against the realised ego position at +1 s and +2 s, next to the same
    prediction with delta forced to 0 (straight line). The arc must be the better
    of the two, which also pins the sign of the curvature."""
    rng = np.random.default_rng(seed)
    rds = sorted(glob.glob(os.path.join(rollouts, 'route_*')))
    rds = [rds[i] for i in rng.choice(len(rds), min(n_routes, len(rds)), replace=False)]
    e_arc, e_str = {10: [], 20: []}, {10: [], 20: []}
    turn = []
    for rd in rds:
        F = load_route(rd)
        for t in range(0, len(F) - 21, 7):
            if F[t]['speed'] < 2.0:
                continue
            P, delta = ego_arc(F[t])
            fr0 = dict(F[t])
            fr0['angular_velocity'] = [0.0, 0.0, 0.0]
            Q, _ = ego_arc(fr0)
            for k in (10, 20):
                g = np.array([F[t + k]['x'], F[t + k]['y']])
                e_arc[k].append(float(np.linalg.norm(P[k] - g)))
                e_str[k].append(float(np.linalg.norm(Q[k] - g)))
            turn.append(abs(F[t]['angular_velocity'][2]) > 0.1)
    turn = np.array(turn)
    res = {'n_routes': len(rds), 'n_samples': len(e_arc[10]),
           'median_err_m_arc_1s': float(np.median(e_arc[10])),
           'median_err_m_straight_1s': float(np.median(e_str[10])),
           'median_err_m_arc_2s': float(np.median(e_arc[20])),
           'median_err_m_straight_2s': float(np.median(e_str[20])),
           'n_turning_samples': int(turn.sum()),
           'turning_median_err_m_arc_1s': float(np.median(np.array(e_arc[10])[turn])),
           'turning_median_err_m_straight_1s': float(np.median(np.array(e_str[10])[turn])),
           'turning_median_err_m_arc_2s': float(np.median(np.array(e_arc[20])[turn])),
           'turning_median_err_m_straight_2s': float(np.median(np.array(e_str[20])[turn])),
           'note': 'constant-speed constant-steering arc of Sec. 4.2 vs the realised '
                   'ego position; the straight-line column is the same prediction with '
                   'delta = 0 and must be worse if the yaw-rate sign is right.'}
    print(json.dumps(res, indent=1))
    if out_json:
        json.dump(res, open(out_json, 'w'), indent=1)
    return res


def attach_check(prov_json, check_json, key):
    p = json.load(open(prov_json))
    p.setdefault('checks', {})[key] = json.load(open(check_json))
    json.dump(p, open(prov_json, 'w'), indent=1)
    print(f'attached {key} to {prov_json}')


V1_COLS = ['v1_tot_mean', 'v1_tot_max', 'v1_grad_max', 'v1_front_max', 'v1_side_max', 'v1_tint']
SSM_COLS = ['ssm_min_ttc', 'ssm_ttc_lt3_frac', 'ssm_max_drac', 'haz_min_dist',
            'haz_egospeed_at_min', 'haz_max_closing', 'haz_close_frac']


def emit_combined(edrf_npz, csv_path, out):
    """EDRF + the shipped in-house v1 potential field + the shipped SSM/hazard block.

    Reporting variant only (the recommended row is the EDRF-only file): it lets
    Table 3A be read both ways, since the shipped 'Risk field' row was those 17
    columns. The v1 and SSM/hazard columns are copied verbatim from the shipped
    CSV; nothing about them is re-derived here.
    """
    import csv as _csv
    d = np.load(edrf_npz, allow_pickle=True)
    rows = list(_csv.reader(open(csv_path)))
    hdr = rows[0]
    ci = {c: i for i, c in enumerate(hdr)}
    take = V1_COLS + SSM_COLS
    old = {r[0]: np.array([float(r[ci[c]]) for c in take]) for r in rows[1:]}
    names = [str(x) for x in d['names']]
    S = np.hstack([d['stats'], np.vstack([old[n.replace('route_', '')] for n in names])])
    np.savez(out, stats=S.astype(np.float64), names=np.array(names),
             columns=np.array(list(d['columns']) + take))
    prov = {'feature_file': out,
            'family': 'Risk field (EDRF + in-house v1 potential field + SSM/hazard)',
            'purpose': 'reporting variant of the Table 3A row: the shipped row was '
                       'these 17 in-house columns; this file replaces its 4 EDRF-labelled '
                       'columns by the 6 faithful EDRF columns and keeps the other 13 '
                       'verbatim, so the row can be read both ways.',
            'produced_by': {'script': os.path.abspath(__file__),
                            'entrypoint': 'edrf_riskfield.py:emit_combined',
                            'command': ' '.join(sys.argv)},
            'components': {
                'EDRF (rebuilt)': {'columns': list(map(str, d['columns'])),
                                   'provenance': edrf_npz.replace('.npz', '_provenance.json')},
                'in-house v1 potential field (NOT from any paper)': {
                    'columns': V1_COLS, 'copied_from': csv_path,
                    'produced_by': '/home/jeongtae/SCIRT/b2d_irt/extract_b2d_traffic_features_220.py'
                                   ' (v1_features): F(x) = sum_a m_a (|v_rel|^2 + V0^2) / (d + D0)^2 '
                                   '(1 + ALP * closing), D0 = 5.0, ALP = 0.5, V0 = 2.0 -- an '
                                   'uncalibrated in-house prototype with no published source.'},
                'SSM / hazard block': {
                    'columns': SSM_COLS, 'copied_from': csv_path,
                    'note': 'surrogate safety measures (TTC, DRAC, min distance); they belong '
                            'to the Min-TTC family, kept here only to reproduce the shipped '
                            '17-column row.'}},
            'columns': list(map(str, d['columns'])) + take}
    json.dump(prov, open(out.replace('.npz', '_provenance.json'), 'w'), indent=1)
    print(f'wrote {out} {S.shape} and its provenance')


def write_provenance(out, names, stats, diags, rollouts):
    prov = {
        'feature_file': out,
        'family': 'Risk field (EDRF)',
        'produced_by': {'script': os.path.abspath(__file__),
                        'entrypoint': 'edrf_riskfield.py:main -> route_features',
                        'python': sys.version.split()[0],
                        'numpy': np.__version__,
                        'command': ' '.join(sys.argv),
                        'regenerate': 'python edrf_riskfield.py --workers 16 --out '
                                      '<out.npz>  (220 routes, ~37 min on 16 cores), then '
                                      '--provenance-only to refresh this record and '
                                      '--emit-combined <plus_inhouse.npz> for the '
                                      'EDRF + v1 + SSM reporting variant'},
        'official_code': None,
        'official_code_note':
            'The EDRF authors release no code (arXiv:2410.14996 lists no repository; '
            'no GitHub project under the authors for EDRF). The model is therefore '
            'implemented from the published equations; the multimodal predictor it '
            'builds on (QCNet, Zhou et al. 2023) is open source but is not usable '
            'here (Argoverse-2 model, no Bench2Drive/CARLA agent-history input '
            'pipeline in this repo).',
        'method_source': {
            'paper': 'J. Jiang, Z. Han, Y. Wang, M. Cai, Q. Meng, Q. Xu, J. Wang, '
                     '"EDRF: Enhanced Driving Risk Field Based on Multimodal Trajectory '
                     'Prediction and Its Applications", arXiv:2410.14996v1, 19 Oct 2024',
            'venue_caveat': 'An IEEE conference version exists (IEEE Xplore document '
                            '10919970); its venue was not verified in this run, so the '
                            'arXiv record is the citation of record here. The shipped '
                            'row cited Westhofen et al. 2022 (arXiv:2108.02403), a '
                            'survey of criticality metrics that defines no risk field '
                            'and none of these constants.',
            'sections_implemented': {
                'Sec. 3.2': 'DRP(s,d)=a(s)exp(-d^2/2sigma(s)^2), a(s)=q(s-s_pt)^2, '
                            'sigma(s)=(b+k*kappa_bar_pt)s+c',
                'Sec. 3.3': 'DRP^C = sum_i p_i(...); M = m*T*(alpha v^beta + gamma); '
                            'EDRF = DRP^C * M',
                'Sec. 4.1': 'IR_ij = EDRF_i*EDRF_j; F_ij = max_{x,y} IR_ij',
                'Sec. 4.2': 'ego kinematic look-ahead t_la = 6 s, bicycle radius '
                            'R = L/tan(delta), Laplace cross-section '
                            'DRP_ego = a_ego(s)exp(-|d|/lambda(s)), '
                            'a_ego = q_ego|s - v t_la|, '
                            'lambda = (b_ego + k_ego|delta|)s + c_ego'},
            'tables': {'Table II (agents)': {'q': Q, 'b': B, 'k': K, 'c': C,
                                             'alpha': ALPHA, 'beta': BETA, 'gamma': GAMMA},
                       'Table III (ego)': {'q_ego': Q_EGO, 'b_ego': B_EGO,
                                           'k_ego': K_EGO, 'c_ego': C_EGO}}},
        'parameters': {
            't_la_s': T_LA, 'dt_s': DT, 'horizon_frames': H,
            'wheelbase_m': WHEELBASE, 'delta_max_rad': DELTA_MAX,
            'type_coefficient_T': TYPE_COEF, 'mass_kg': MASS_KG,
            'default_mass_kg': DEFAULT_MASS,
            'entity_classes_used': ['vehicle', 'walker'],
            'speed_units': 'm/s (v in alpha v^beta + gamma, SI throughout)',
            'ridge_seed_step_m': RIDGE_STEP, 'max_ridge_points': MAX_RIDGE_PTS,
            'area_grid_cells_per_axis': GRID_N, 'area_grid_min_step_m': GRID_MIN_STEP,
            'n_seeds_refined': N_REFINE, 'refine_h0_m': REFINE_H0,
            'refine_points_per_axis': REFINE_N, 'refine_rounds': REFINE_ROUNDS,
            'effective_resolution_m': 2 * REFINE_H0 / (3.0 ** (REFINE_ROUNDS - 1)) / (REFINE_N - 1),
            'skip_upper_bound': SKIP_BOUND, 'early_exit_rel_tol': REL_TOL,
            'log_floor': LOG_FLOOR,
            'columns': COLS},
        'input_data': {
            'rollouts': rollouts,
            'source': 'PDM-Lite reference rollout, anno/*.json.gz at 10 Hz',
            'fields_read': ['x', 'y', 'theta', 'speed', 'angular_velocity',
                            'bounding_boxes[location, speed, base_type, type_id, class, id]'],
            'leakage_note': 'PDM-Lite is EXCLUDED_PLANNER in the ATDrive panel and the '
                            'RelGraph encoder reads the same rollout; no response '
                            'matrix, planner outcome or scenario type is read.',
            'n_routes': int(len(names))},
        'deviations_from_source': DEVIATIONS,
        'per_route_diagnostics': diags,
        'sanity': {
            'columns': COLS,
            'mean': stats.mean(0).tolist(), 'std': stats.std(0).tolist(),
            'min': stats.min(0).tolist(), 'max': stats.max(0).tolist(),
            'n_nonfinite': int((~np.isfinite(stats)).sum()),
            'mean_realised_horizon_frac': float(np.mean([d['mean_realised_horizon_frac'] for d in diags])),
            'mean_frac_frames_Fsum_zero': float(np.mean([d['frac_frames_Fsum_zero'] for d in diags]))},
    }
    prov['checks'] = {}
    for key, fn in (('maximiser_vs_brute_grid', 'edrf_grid_validation.json'),
                    ('ego_kinematic_arc', 'edrf_ego_arc_check.json'),
                    ('rollout_frame_conventions', 'edrf_heading_convention_check.json')):
        f = os.path.join(os.path.dirname(out), fn)
        if os.path.exists(f):
            prov['checks'][key] = json.load(open(f))
    pj = out.replace('.npz', '_provenance.json')
    json.dump(prov, open(pj, 'w'), indent=1)
    print('wrote', pj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rollouts', default='/data1/jeongtae/b2d_eval_sensors')
    ap.add_argument('--out', default='/data2/jeongtae/official_baselines/us_features/edrf_riskfield.npz')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no-traffic', action='store_true',
                    help='skip the Sec. 4.1 agent-agent monitoring columns')
    ap.add_argument('--validate', default='', metavar='ROUTE_DIR')
    ap.add_argument('--validate-json', default='')
    ap.add_argument('--validate-pairs', type=int, default=12)
    ap.add_argument('--validate-step', type=float, default=0.05)
    ap.add_argument('--emit-combined', default='', metavar='OUT_NPZ',
                    help='write the EDRF + v1 + SSM reporting variant from --out and --csv')
    ap.add_argument('--csv', default='/home/jeongtae/SC-IRT/data/b2d/traffic_features_220.csv')
    ap.add_argument('--provenance-only', action='store_true',
                    help='rewrite <out>_provenance.json from the existing npz')
    ap.add_argument('--check-heading', action='store_true')
    ap.add_argument('--check-ego-arc', action='store_true')
    ap.add_argument('--check-json', default='')
    ap.add_argument('--attach', nargs=3, default=None,
                    metavar=('PROVENANCE_JSON', 'CHECK_JSON', 'KEY'))
    args = ap.parse_args()

    if args.check_heading:
        check_heading(args.rollouts, args.check_json)
        return
    if args.check_ego_arc:
        check_ego_arc(args.rollouts, args.check_json)
        return
    if args.attach:
        attach_check(*args.attach)
        return
    if args.emit_combined:
        emit_combined(args.out, args.csv, args.emit_combined)
        return
    if args.provenance_only:                       # rewrite the record from the npz
        d = np.load(args.out, allow_pickle=True)
        write_provenance(args.out, d['names'], d['stats'],
                         json.loads(str(d['diag'])), args.rollouts)
        return

    if args.validate:
        res = validate(args.validate, args.validate_pairs, args.validate_step)
        if args.validate_json:
            json.dump(res, open(args.validate_json, 'w'), indent=1)
        return

    routes = sorted(glob.glob(os.path.join(args.rollouts, 'route_*')))
    if args.limit:
        routes = routes[:args.limit]
    t0 = time.time()
    jobs = [(r, not args.no_traffic) for r in routes]
    with Pool(args.workers) as p:
        out = p.map(_work, jobs, chunksize=1)
    names = np.array([f'route_{r}' for r, _, _ in out])
    stats = np.vstack([s for _, s, _ in out]).astype(np.float64)
    diags = [d for _, _, d in out]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, stats=stats, names=names, columns=np.array(COLS),
             diag=np.array(json.dumps(diags)))
    print(f'wrote {args.out}  {stats.shape}  in {time.time() - t0:.0f}s')

    write_provenance(args.out, names, stats, diags, args.rollouts)


if __name__ == '__main__':
    main()
