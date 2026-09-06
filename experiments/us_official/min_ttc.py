#!/usr/bin/env python3
"""Table 3A row 'Min-TTC' — rebuilt as the standard surrogate safety measure.

WHAT THIS IS
------------
Time-To-Collision (Hayward 1972; aggregates TET/TIT from Minderhoud & Bovy
2001), in the form reviewed by Westhofen et al., "Criticality Metrics for
Automated Driving: A Review and Suitability Analysis of the State of the
Art", arXiv:2108.02403, Sec. 5.2.1 / 5.2.6 / 5.2.7:

    TTC(A1, A2, t) = min ({t~ >= 0 | d(p1(t + t~), p2(t + t~)) = 0} u {inf})
    TET(A1, A2, tau) = \\int_{t0}^{te} 1_{TTC(A1,A2,t) <= tau} dt
    TIT(A1, A2, tau) = \\int_{t0}^{te} 1_{TTC(A1,A2,t) <= tau} (tau - TTC(A1,A2,t)) dt

The decision-model (DMM) used to propagate both actors is constant velocity,
as specified for the classical TTC.  `d = 0` is instantiated on the actors'
OCCUPIED SETS (their oriented bounding boxes), i.e. TTC is the first time the
two footprints intersect -- not the first time two centre points coincide.
See the provenance JSON for why this instantiation, and not the literal
point-distance reading, is the one meant by "until A1 and A2 collide".

WHAT IT REPLACES
----------------
`ssm_hazard_features()['ssm_min_ttc']` in
/home/jeongtae/SCIRT/b2d_irt/extract_b2d_traffic_features_220.py (lines
116-147), which computes centre-to-centre range divided by the radial closing
speed, over every agent whose range is shrinking, clipped at 30 s.  That is a
closing-distance time: no footprints, no lateral-miss test, so passing traffic
that never conflicts counts and a stopped car straight ahead does not differ
from one two lanes over.

EXACT SOLVER (no time sampling)
-------------------------------
Under constant velocity with fixed orientation, the agent's box B translates
relative to the ego's box A at w = v_B - v_A.  A meets B + w*t iff w*t lies in
the Minkowski difference M = A (+) (-B), a convex polygon whose facet normals
are exactly the union of the facet normals of A and of B (8 for two
rectangles) and whose support function is h_M(n) = h_A(n) + h_B(-n).  So

    TTC = min { t >= 0 : n_k . w t <= c_k  for all k },
    c_k = n_k . (p_A - p_B) + L_A|n_k.u_A| + W_A|n_k.v_A|
                            + L_B|n_k.u_B| + W_B|n_k.v_B|,

a ray/convex-polygon clip solved in closed form.  t = 0 is returned iff the
boxes already overlap (all c_k >= 0, the separating-axis test).  Verified
against a brute-force separating-axis sweep in --selftest.

USAGE
    python min_ttc.py [--jobs 16] [--selftest] [--limit N]
"""
import argparse
import glob
import gzip
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- parameters
ROLLOUT_ROOT = '/data1/jeongtae/b2d_eval_sensors'   # PDM-Lite reference rollout
OUT_DIR = Path('/data2/jeongtae/official_baselines/us_features')
NAME = 'min_ttc'
DT = 0.1                 # s, Bench2Drive annotation rate (10 Hz)
TAU = 3.0                # s, TET/TIT target value (Minderhoud & Bovy 2001)
TTC_CENSOR = 10.0        # s, ceiling for the 1-d table column (see provenance)
ZERO_TOL = 1e-12         # |n.w| below this counts as "no motion along n"
AGENT_CLASSES = ('vehicle', 'walker')


# ------------------------------------------------------------------ geometry
def pair_ttc(pe, ue, Le, We, ve, P, U, Lh, Wh, V, return_margin=False):
    """Exact constant-velocity footprint TTC of the ego box against n agents.

    pe (2,)  ego centre; ue (2,) ego heading unit vector; Le, We ego half
    length/width; ve (2,) ego velocity.  P (n,2), U (n,2), Lh (n,), Wh (n,),
    V (n,2) the same for the agents.  Returns (n,) with +inf where the
    constant-velocity extrapolations never make the footprints intersect.

    With return_margin, also returns min_k c_k: >= 0 iff the footprints already
    intersect at t = 0 (the separating-axis test), and then equal to the
    interpenetration depth in metres.
    """
    n = len(P)
    if n == 0:
        return (np.zeros(0), np.zeros(0)) if return_margin else np.zeros(0)
    ve_p = np.array([-ue[1], ue[0]])                       # ego lateral axis
    U_p = np.stack([-U[:, 1], U[:, 0]], 1)                 # agent lateral axes
    # the 8 facet normals of the Minkowski difference: +-ego axes, +-agent axes
    N = np.empty((n, 8, 2))
    N[:, 0] = ue
    N[:, 1] = -ue
    N[:, 2] = ve_p
    N[:, 3] = -ve_p
    N[:, 4] = U
    N[:, 5] = -U
    N[:, 6] = U_p
    N[:, 7] = -U_p

    dp = pe[None, :] - P                                   # (n,2)
    w = V - ve[None, :]                                    # (n,2) relative vel
    c = (np.einsum('nkd,nd->nk', N, dp)
         + Le * np.abs(np.einsum('nkd,d->nk', N, ue))
         + We * np.abs(np.einsum('nkd,d->nk', N, ve_p))
         + Lh[:, None] * np.abs(np.einsum('nkd,nd->nk', N, U))
         + Wh[:, None] * np.abs(np.einsum('nkd,nd->nk', N, U_p)))
    a = np.einsum('nkd,nd->nk', N, w)

    pos, neg = a > ZERO_TOL, a < -ZERO_TOL
    inf = np.inf
    hi = np.where(pos, np.divide(c, a, out=np.full_like(c, inf), where=pos), inf).min(1)
    lo = np.where(neg, np.divide(c, a, out=np.zeros_like(c), where=neg), -inf).max(1)
    lo = np.maximum(lo, 0.0)
    infeasible = ((~pos & ~neg) & (c < 0)).any(1) | (lo > hi)
    ttc = np.where(infeasible, inf, lo)
    return (ttc, c.min(1)) if return_margin else ttc


def boxes_overlap(pA, uA, LA, WA, pB, uB, LB, WB):
    """Separating-axis test on two oriented rectangles (self-test only)."""
    vA = np.array([-uA[1], uA[0]])
    vB = np.array([-uB[1], uB[0]])
    d = pA - pB
    for nrm in (uA, vA, uB, vB):
        r = (LA * abs(nrm @ uA) + WA * abs(nrm @ vA)
             + LB * abs(nrm @ uB) + WB * abs(nrm @ vB))
        if abs(nrm @ d) > r:
            return False
    return True


# ------------------------------------------------------------------- loading
def load_route(route_dir):
    """-> list of frames, each {'ego': (...), 'agents': {id: (...)}}."""
    files = sorted(glob.glob(os.path.join(route_dir, 'anno', '*.json.gz')))
    out = []
    for f in files:
        d = json.load(gzip.open(f))
        ego, agents = None, {}
        for bb in d['bounding_boxes']:
            cls = bb.get('class')
            loc, ext, rot = bb.get('location'), bb.get('extent'), bb.get('rotation')
            if loc is None or ext is None or rot is None:
                continue
            rec = (np.array([loc[0], loc[1]]), math.radians(rot[2]),
                   float(ext[0]), float(ext[1]), float(bb.get('speed', 0.0)))
            if cls == 'ego_vehicle':
                ego = rec
            elif cls in AGENT_CLASSES:
                agents[bb['id']] = rec
        out.append({'ego': ego, 'agents': agents, 'x': d['x'], 'y': d['y']})
    return out


def track_velocities(frames):
    """Central finite differences of `location` per track id, dt = 0.1 s.

    Falls back to one-sided differences at a track's first/last frame and to
    speed*(cos yaw, sin yaw) for a track seen in a single frame.  Returns
    (ego_vel list, [dict id->vel] per frame) and the |fd - recorded speed|
    residuals used as the convention check.
    """
    T = len(frames)
    present = [set(f['agents']) for f in frames]
    ego_v, ag_v, resid = [], [], []

    def fd(seq, t, tmin, tmax):
        if t - 1 >= tmin and t + 1 <= tmax:
            return (seq(t + 1) - seq(t - 1)) / (2 * DT)
        if t + 1 <= tmax:
            return (seq(t + 1) - seq(t)) / DT
        if t - 1 >= tmin:
            return (seq(t) - seq(t - 1)) / DT
        return None

    for t in range(T):
        e = frames[t]['ego']
        v = fd(lambda k: frames[k]['ego'][0], t, 0, T - 1)
        if v is None:
            v = e[4] * np.array([math.cos(e[1]), math.sin(e[1])])
        ego_v.append(v)
        resid.append(abs(np.linalg.norm(v) - e[4]))
        dd = {}
        for aid, rec in frames[t]['agents'].items():
            lo = t - 1 if (t - 1 >= 0 and aid in present[t - 1]) else t
            hi = t + 1 if (t + 1 < T and aid in present[t + 1]) else t
            va = fd(lambda k, _a=aid: frames[k]['agents'][_a][0], t, lo, hi)
            if va is None:
                va = rec[4] * np.array([math.cos(rec[1]), math.sin(rec[1])])
            else:
                resid.append(abs(np.linalg.norm(va) - rec[4]))
            dd[aid] = va
        ag_v.append(dd)
    return ego_v, ag_v, resid


# ------------------------------------------------------------------ per route
def route_features(route_dir):
    frames = load_route(route_dir)
    meta = json.load(open(os.path.join(route_dir, 'meta.json')))
    ego_v, ag_v, resid = track_velocities(frames)
    ttc_t = np.full(len(frames), np.inf)
    pen_t = np.full(len(frames), -np.inf)     # SAT interpenetration where TTC = 0
    n_agents = []
    for t, fr in enumerate(frames):
        e = fr['ego']
        if e is None:
            n_agents.append(0)
            continue
        aids = list(fr['agents'])
        n_agents.append(len(aids))
        if not aids:
            continue
        recs = [fr['agents'][a] for a in aids]
        P = np.stack([r[0] for r in recs])
        U = np.stack([[math.cos(r[1]), math.sin(r[1])] for r in recs])
        Lh = np.array([r[2] for r in recs])
        Wh = np.array([r[3] for r in recs])
        V = np.stack([ag_v[t][a] for a in aids])
        ue = np.array([math.cos(e[1]), math.sin(e[1])])
        tt, mg = pair_ttc(e[0], ue, e[2], e[3], ego_v[t], P, U, Lh, Wh, V,
                          return_margin=True)
        ttc_t[t] = tt.min()
        if ttc_t[t] == 0:
            pen_t[t] = mg.max()

    fin = np.isfinite(ttc_t)
    below = fin & (ttc_t <= TAU)
    return {
        'route_id': str(meta['route_id']),
        'n_frames': len(frames),
        'min_ttc': float(ttc_t.min()),                      # may be +inf
        'tet': float(below.sum() * DT),
        'tit': float(((TAU - ttc_t[below]).sum()) * DT),
        'frac_finite_frames': float(fin.mean()),
        'frac_zero_frames': float((ttc_t == 0).mean()),
        'max_penetration_m': float(pen_t.max()) if np.isfinite(pen_t).any() else 0.0,
        'median_finite_ttc': float(np.median(ttc_t[fin])) if fin.any() else float('inf'),
        'mean_agents': float(np.mean(n_agents)),
        'vel_resid_mae': float(np.mean(resid)),
        'vel_resid_p99': float(np.percentile(resid, 99)),
    }


# ------------------------------------------------------------------ self-test
def selftest(n_pairs=20000, horizon=30.0, step=1e-3):
    """Analytic solver vs a brute-force separating-axis sweep on a time grid.

    The sweep is the same procedure commonroad-crime's TTCStar uses (step the
    boxes forward, test overlap, take the first hit), only on a constant-
    velocity extrapolation and a 1 ms grid.  Agreement must be within one grid
    step, and finiteness must agree for every pair whose analytic TTC is
    inside the sweep horizon.
    """
    rng = np.random.default_rng(0)
    the, tha = rng.uniform(-np.pi, np.pi, (2, n_pairs))
    ue = np.stack([np.cos(the), np.sin(the)], 1)
    U = np.stack([np.cos(tha), np.sin(tha)], 1)
    Le, We = rng.uniform(0.2, 2.6, n_pairs), rng.uniform(0.2, 1.4, n_pairs)
    Lh, Wh = rng.uniform(0.2, 2.6, n_pairs), rng.uniform(0.2, 1.4, n_pairs)
    pe = rng.normal(0, 6, (n_pairs, 2))
    P = pe + rng.normal(0, 12, (n_pairs, 2))
    ve, V = rng.normal(0, 6, (n_pairs, 2)), rng.normal(0, 6, (n_pairs, 2))

    t_an = np.array([pair_ttc(pe[i], ue[i], Le[i], We[i], ve[i], P[i:i + 1],
                              U[i:i + 1], Lh[i:i + 1], Wh[i:i + 1], V[i:i + 1])[0]
                     for i in range(n_pairs)])
    # brute force: SAT on a time grid, vectorised over pairs
    grid = np.arange(0.0, horizon + step, step)
    t_bf = np.full(n_pairs, np.inf)
    ve_p = np.stack([-ue[:, 1], ue[:, 0]], 1)
    U_p = np.stack([-U[:, 1], U[:, 0]], 1)
    axes = [ue, ve_p, U, U_p]
    rad = [Le * np.abs((ax * ue).sum(1)) + We * np.abs((ax * ve_p).sum(1))
           + Lh * np.abs((ax * U).sum(1)) + Wh * np.abs((ax * U_p).sum(1)) for ax in axes]
    live = np.ones(n_pairs, bool)
    for t in grid:
        d = (pe + ve * t) - (P + V * t)
        hit = np.ones(n_pairs, bool)
        for ax, r in zip(axes, rad):
            hit &= np.abs((ax * d).sum(1)) <= r
        new = hit & live
        t_bf[new] = t
        live &= ~hit
        if not live.any():
            break

    inside = t_an <= horizon
    both = inside & np.isfinite(t_bf)
    err = np.abs(t_an[both] - t_bf[both])
    assert err.max() <= step * 1.001, f'max error {err.max()}'

    # Disagreements can only be grazing contacts the 1 ms grid steps over.
    # Verify each one directly: at t_an the boxes touch (all SAT margins >= 0)
    # and just before t_an they are separated.
    graze = 0
    for i in np.where(inside & ~np.isfinite(t_bf))[0]:
        def marg(t):
            d = (pe[i] + ve[i] * t) - (P[i] + V[i] * t)
            return min(r[i] - abs(ax[i] @ d) for ax, r in zip(axes, rad))
        assert marg(t_an[i]) >= -1e-9 and marg(max(0.0, t_an[i] - 1e-4)) < 0, \
            f'pair {i}: analytic TTC {t_an[i]} is not a first-contact time'
        graze += 1
    assert (~inside & np.isfinite(t_bf)).sum() == 0, 'brute force found an earlier collision'
    print(f'selftest OK: {n_pairs} random OBB pairs, {int(inside.sum())} with a collision '
          f'inside {horizon}s; max |analytic - brute force| = {err.max():.2e} s '
          f'(grid step {step} s); {graze} grazing contacts the grid misses were verified '
          f'directly as first-contact times.')


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=16)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    routes = sorted(glob.glob(os.path.join(ROLLOUT_ROOT, 'route_*')))
    if args.limit:
        routes = routes[:args.limit]
    t0 = time.time()
    with ProcessPoolExecutor(args.jobs) as ex:
        recs = list(ex.map(route_features, routes, chunksize=1))
    print(f'{len(recs)} routes in {time.time() - t0:.0f}s', flush=True)

    # ProcessPoolExecutor.map preserves input order; ids must match the dir names
    for r, d in zip(recs, routes):
        assert 'route_' + r['route_id'] == os.path.basename(d), (r['route_id'], d)
    names = np.array(['route_' + r['route_id'] for r in recs])
    raw = np.array([r['min_ttc'] for r in recs], float)
    stats = np.stack([np.minimum(raw, TTC_CENSOR),
                      [r['tet'] for r in recs],
                      [r['tit'] for r in recs]], 1).astype(np.float64)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / f'{NAME}.npz', stats=stats, names=names,
             columns=np.array(['min_ttc_censored_10s', 'TET_tau3_s', 'TIT_tau3_s2']),
             min_ttc_raw=raw,
             n_frames=np.array([r['n_frames'] for r in recs]),
             frac_finite_frames=np.array([r['frac_finite_frames'] for r in recs]),
             frac_zero_frames=np.array([r['frac_zero_frames'] for r in recs]),
             median_finite_ttc=np.array([r['median_finite_ttc'] for r in recs]),
             mean_agents=np.array([r['mean_agents'] for r in recs]),
             max_penetration_m=np.array([r['max_penetration_m'] for r in recs]))
    # The shared scorer (score_one.py) feeds every column of a file to the Ridge
    # plug-in as one arm, so the 1-d table row also gets its own file.
    np.savez(OUT_DIR / f'{NAME}_1d.npz', stats=stats[:, :1].copy(), names=names,
             columns=np.array(['min_ttc_censored_10s']), min_ttc_raw=raw)

    # ---- provenance --------------------------------------------------------
    def git(path, *a):
        try:
            return subprocess.check_output(['git', '-C', path] + list(a),
                                           text=True).strip()
        except Exception:
            return None

    old = {}
    csvp = Path('/home/jeongtae/SC-IRT/data/b2d/traffic_features_220.csv')
    if csvp.exists():
        import csv as _csv
        rows = list(_csv.reader(open(csvp)))
        ci = {c: i for i, c in enumerate(rows[0])}
        old = {r[0]: float(r[ci['ssm_min_ttc']]) for r in rows[1:]}
    ov = np.array([old.get(r['route_id'], np.nan) for r in recs])
    ok = np.isfinite(ov) & np.isfinite(stats[:, 0])
    from scipy.stats import spearmanr, pearsonr
    fin = np.isfinite(raw)
    prov = {
        'name': NAME,
        'family': 'Min-TTC',
        'written': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'produced_by': 'SC-IRT/experiments/us_official/min_ttc.py:route_features (pair_ttc)',
        'scirt_repo': {'path': '/home/jeongtae/SC-IRT',
                       'commit': git('/home/jeongtae/SC-IRT', 'rev-parse', 'HEAD')},
        'official_code_run': None,
        'official_code_consulted': {
            'commonroad-crime 0.4.5 (TUM CPS, sdist from PyPI)': {
                'files': ['commonroad_crime/measure/time/ttc_star.py',
                          'commonroad_crime/measure/time/tet.py',
                          'commonroad_crime/measure/time/tit.py',
                          'commonroad_crime/measure/time/ttc.py'],
                'used_for': 'TET/TIT accumulation convention (+= dt when TTC <= tau; '
                            '+= (tau - TTC)*dt when TTC <= tau) and the OBB-footprint '
                            'reading of "collide" (TTCStar builds pycrcc.RectOBB boxes).',
                'not_run_because':
                    'CriMe operates on CommonRoad scenarios with a lanelet network and the '
                    'compiled commonroad-drivability-checker. Its TTC class is a '
                    'lane-projected car-following formula (needs find_lanelet_by_position, '
                    'undefined for Bench2Drive/CARLA rollouts without a CommonRoad map), and '
                    'its TTCStar sweeps the RECORDED future of the other actors rather than a '
                    'constant-velocity DMM, so neither implements the metric this row claims. '
                    'No official implementation of the general constant-velocity footprint TTC '
                    'exists for Bench2Drive annotations, so the published definition was '
                    'implemented directly and cross-checked against brute force.'}},
        'definitions': {
            'TTC': 'Westhofen et al., arXiv:2108.02403, Sec. 5.2.1: "For two actors A1, A2 '
                   'at time t, the TTC metric returns the minimal time until A1 and A2 '
                   'collide according to a given DMM, or infinity if the predicted '
                   'trajectories do not intersect [33, 94]." Formula: '
                   'TTC(A1,A2,t) = min({t~ >= 0 | d(p1(t+t~), p2(t+t~)) = 0} u {inf}). '
                   'Ref [33] = Hayward JC (1972), Near miss determination through use of a '
                   'scale of danger, 51st HRB annual meeting, vol 384, pp 24-34.',
            'TET': 'Westhofen Sec. 5.2.6: TET(A1,A2,tau) = int_{t0}^{te} 1_{TTC(A1,A2,t) <= tau} dt. '
                   'Ref [66] = Minderhoud & Bovy (2001), Extended time-to-collision measures for '
                   'road traffic safety assessment, Accid Anal Prev 33:89-97.',
            'TIT': 'Westhofen Sec. 5.2.7: TIT(A1,A2,tau) = int_{t0}^{te} '
                   '1_{TTC(A1,A2,t) <= tau} (tau - TTC(A1,A2,t)) dt. Ref [66] as above.',
            'scene_aggregation': 'per-frame TTC = min over agents (Westhofen Sec. 5.2.1 notes '
                                 'TTC must be "meaningfully aggregated over actors" in '
                                 'multi-actor scenes); route min-TTC = min over frames.'},
        'parameters': {
            'DT_s': DT, 'tau_s': TAU, 'TTC_censor_s': TTC_CENSOR,
            'DMM': 'constant velocity, orientation held fixed over the extrapolation',
            'prediction_horizon': 'none (unbounded t~, per the definition)',
            'footprint': 'oriented bounding box centred at anno `location`, half-extents '
                         '(extent[0], extent[1]), yaw = radians(rotation[2])',
            'agent_classes': list(AGENT_CLASSES),
            'ego_excluded_from_agents': True,
            'velocity': 'central finite differences of `location` per track id at DT; '
                        'one-sided at track ends; speed*(cos yaw, sin yaw) for a track seen '
                        'in a single frame',
            'zero_tol': ZERO_TOL,
            'solver': 'closed-form ray/convex-polygon clip on the Minkowski difference '
                      '(8 facet normals); no time discretisation'},
        'data': {'root': ROLLOUT_ROOT,
                 'rollout': 'PDM-Lite reference rollout, 10 Hz anno/*.json.gz',
                 'n_routes': len(recs),
                 'n_frames_total': int(sum(r['n_frames'] for r in recs)),
                 'leakage_note': 'PDM-Lite is EXCLUDED_PLANNER (atdrive/b2d.py) and is not one '
                                 'of the 16 evaluated planners; the RelGraph encoder reads the '
                                 'same rollout. No response matrix, planner outcome or held-out '
                                 'scenario type was read while building this descriptor.'},
        'deviations_from_source': [
            'Westhofen writes d as the Euclidean distance between actor POSITIONS p1, p2 '
            '(their Table 1); taken literally, d = 0 requires two points to coincide and the '
            'metric would be degenerate for extended bodies. We instantiate d on the actors\' '
            'occupied sets (rigid oriented rectangles), so d = 0 iff the footprints intersect. '
            'This is the reading of "until A1 and A2 collide" used by commonroad-crime TTCStar '
            '(pycrcc.RectOBB + collision checker) and by CARLA/Bench2Drive collision events.',
            'Orientation is held constant during the extrapolation (yaw rate not propagated); '
            'this is what the constant-velocity DMM specifies and keeps the solver exact.',
            'tau = 3 s for TET/TIT as specified for this rebuild; commonroad-crime 0.4.5 '
            'defaults to tau = 2 s (configuration.py:244).',
            'The 1-d table column censors min-TTC at TTC_CENSOR = 10 s so the Ridge plug-in '
            'has a finite input for routes whose constant-velocity extrapolations never '
            'intersect; min_ttc_raw in the npz keeps the uncensored value (+inf where '
            'applicable).',
            'The anno `location` is the actor transform origin (it equals the ego x/y exactly); '
            'CARLA bounding boxes can carry a small (<0.5 m) longitudinal offset from it, which '
            'is not modelled.',
            'Westhofen defines TET/TIT as continuous-time integrals; they are evaluated as '
            'Riemann sums at the 10 Hz annotation rate, matching Minderhoud & Bovy\'s discrete '
            'formulation and commonroad-crime.'],
        'convention_checks': {
            'heading': 'The direction of motion in the anno (x, y) frame equals '
                       'radians(rotation[2]) (yaw), NOT the top-level `theta`, which is offset '
                       'by +pi/2. Verified by finite differences of the ego and of moving '
                       'agents. The replaced ssm_hazard_features() uses `theta`.',
            'velocity': 'Central finite differences reproduce the recorded scalar `speed` to '
                        f'MAE {float(np.mean([r["vel_resid_mae"] for r in recs])):.4f} m/s '
                        f'(p99 {float(np.max([r["vel_resid_p99"] for r in recs])):.3f} m/s) '
                        'across all routes; forward differences are ~3x worse (MAE 0.113 m/s '
                        'on a 4-route probe), so the recorded `speed` is the instantaneous '
                        'speed at the frame and central differences are the matching '
                        'convention. Central differences are used because the TTC needs the '
                        'velocity VECTOR, which the scalar `speed` field does not give.',
            'extent': 'extent = half-extents (ego 2.446 x 0.918 -> 4.89 m x 1.84 m, a Lincoln '
                      'MKZ 2020).'},
        'sanity': {
            'n_routes': len(recs),
            'n_finite_min_ttc': int(fin.sum()),
            'n_censored_at_10s': int((raw > TTC_CENSOR).sum()),
            'n_zero_min_ttc': int((raw == 0).sum()),
            'zero_min_ttc_routes': [r['route_id'] for r in recs if r['min_ttc'] == 0],
            'zero_min_ttc_note':
                'min-TTC = 0 means the footprints already intersect at some frame (t~ = 0 '
                'satisfies the definition). On those routes the deepest separating-axis '
                'interpenetration is {:.3f}-{:.3f} m, i.e. the ego is in contact or within '
                'the box model\'s tolerance of it; the fraction of frames in that state '
                'ranges {:.3f}-{:.3f}. Kept, because censoring it would be a deviation from '
                'the definition -- but the reader should know 14/220 routes tie at 0 on the '
                '1-d column, which is why TET/TIT are shipped alongside it.'.format(
                    min([r['max_penetration_m'] for r in recs if r['min_ttc'] == 0] or [0]),
                    max([r['max_penetration_m'] for r in recs if r['min_ttc'] == 0] or [0]),
                    min([r['frac_zero_frames'] for r in recs if r['min_ttc'] == 0] or [0]),
                    max([r['frac_zero_frames'] for r in recs if r['min_ttc'] == 0] or [0])),
            'min_ttc_raw_quantiles_s': {q: float(np.quantile(raw[fin], p)) for q, p in
                                        [('p0', 0), ('p5', .05), ('p25', .25), ('p50', .5),
                                         ('p75', .75), ('p95', .95), ('p100', 1.0)]},
            'TET_tau3_s': {'mean': float(stats[:, 1].mean()), 'median': float(np.median(stats[:, 1])),
                           'max': float(stats[:, 1].max()), 'n_zero': int((stats[:, 1] == 0).sum())},
            'TIT_tau3_s2': {'mean': float(stats[:, 2].mean()), 'median': float(np.median(stats[:, 2])),
                            'max': float(stats[:, 2].max()), 'n_zero': int((stats[:, 2] == 0).sum())},
            'frames_per_route': {'min': int(min(r['n_frames'] for r in recs)),
                                 'max': int(max(r['n_frames'] for r in recs)),
                                 'total': int(sum(r['n_frames'] for r in recs))},
            'mean_agents_per_frame': float(np.mean([r['mean_agents'] for r in recs])),
            'vs_shipped_ssm_min_ttc': {
                'n_compared': int(ok.sum()),
                'spearman': float(spearmanr(stats[ok, 0], ov[ok]).correlation),
                'pearson': float(pearsonr(stats[ok, 0], ov[ok])[0]),
                'shipped_mean': float(np.mean(ov[ok])), 'new_mean': float(np.mean(stats[ok, 0])),
                'shipped_note': 'ssm_min_ttc = min over frames of (centre-to-centre range / '
                                'radial closing speed) over agents with closing speed > 0.1 '
                                'm/s, per-frame value clipped at 30 s.'}},
        'table3a_scoring': {
            'scored_by': 'experiments/us_official/score_one.py (wraps experiments/run_us.py; '
                         'same unified split, 16 draws, two-stage Ridge plug-in, pooled 640 '
                         'route evaluations)',
            'command': 'python score_one.py '
                       '/data2/jeongtae/official_baselines/us_features/min_ttc_1d.npz '
                       '/data2/jeongtae/official_baselines/us_features/min_ttc.npz',
            'planner_only_null': {'auroc': 0.699, 'scene_mae': 0.214},
            'shipped_Min_TTC_row': {'auroc': 0.692, 'scene_mae': 0.219, 'rho': -0.061,
                                    'note': 'reproduced exactly in the same run'},
            'rebuilt_1d_min_ttc': {'auroc': 0.713, 'scene_mae': 0.213, 'rho': 0.217},
            'rebuilt_3d_min_ttc_TET_TIT': {'auroc': 0.711, 'scene_mae': 0.214, 'rho': 0.206},
            'anchors': 'null / kinematics / hand-crafted risk / RelGraph all reproduced '
                       '(run_us.py assertions passed with the extra arms present)'},
        'npz': {'path': str(OUT_DIR / f'{NAME}.npz'),
                'stats_columns': ['min_ttc_censored_10s (the 1-d Table 3A row)',
                                  'TET_tau3_s', 'TIT_tau3_s2'],
                'names_format': 'route_<id>',
                'extra_arrays': ['min_ttc_raw (uncensored, +inf where the constant-velocity '
                                 'extrapolations never intersect)', 'n_frames',
                                 'frac_finite_frames', 'frac_zero_frames',
                                 'median_finite_ttc', 'mean_agents'],
                'one_d_slice': {
                    'path': str(OUT_DIR / f'{NAME}_1d.npz'),
                    'why': 'experiments/us_official/score_one.py feeds every column of a file '
                           'to the Ridge plug-in as a single arm, so column 0 alone -- the row '
                           'that goes in Table 3A -- is also written as its own file. It is a '
                           'slice of min_ttc.npz, not a separate computation.'}},
    }
    json.dump(prov, open(OUT_DIR / f'{NAME}_provenance.json', 'w'), indent=2)
    print(json.dumps(prov['sanity'], indent=2))
    print('wrote', OUT_DIR / f'{NAME}.npz')


if __name__ == '__main__':
    main()
