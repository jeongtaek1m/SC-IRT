#!/usr/bin/env python3
"""H-rel — ego plus HAND-CRAFTED relational summaries (PROTOCOL_R.md section 1).

This is the reviewer's arm: "do you need a graph encoder, or would counting a few
route-related agents do?".  It must be a GOOD-FAITH strong baseline, so it gets the
same splits, the same gold, the same recipe and the same 3 seeds as R0, and it is
allowed to read every channel of the frozen graph -- it just reads them through 20
scalars per window instead of a learned encoder.

Two readouts on the SAME features, both reported (PROTOCOL_R.md section 1):
  ridge  closed-form linear map onto the train-split Rasch b, alpha chosen by inner
         group-CV inside the training rows only.
  mlp    R0's SeqNet with a second, identical branch over the per-window relational
         vectors; d = 64, same depth, same optimizer / epochs / batch / weight decay.

Item granularity follows R0 exactly:
  B2D     item = route.  Ego is R0's UNIQUE 2 Hz sequence over the route (windows
          overlap by 8 of 12 steps).  The relational scalars live on the WINDOW axis
          (one graph snapshot per window, anchored at its step 3) and are pooled over
          windows, so every window carries equal weight.  Pooling the two blocks on
          their own axes is the only reason this file is not a one-line edit of R0.
  NavSim  item = window = scored scene, so the window axis has length 1.

route_rel_valid == False (NavSim 2.857%, B2D 0.000%): the 16 route-derived scalars are
UNDEFINED there, not zero -- reading a zeroed route_rel as "no route-related agent"
is exactly the false-negative supervision KEYS.md forbids.  They are imputed with the
mean of the VALID TRAINING rows (per split, so no leakage), which lands on ~0 after
standardization, i.e. the "no evidence" point of a centred linear model.  The validity
flag itself is NEVER an input feature: NavSim has 2.86% missingness and B2D has 0%, so
a model given the flag could use it as a domain identifier.

Usage: hrel.py --domain b2d|navsim --gpu G --seed S
       hrel.py --domain b2d|navsim --features        (descriptive stats, no training)
"""
import argparse, os, re, sys

import numpy as np

RG = '/data2/jeongtae/relgraph'
sys.path.insert(0, RG)
sys.path.insert(0, '/home/jeongtae/SC-IRT')
sys.path.insert(0, '/home/jeongtae/SCIRT/b2d_irt')

import b2d_earlystop as es                        # the ONE B2D checkpoint-selection rule

# ego side is literally R0's, imported rather than re-typed so it cannot drift
from r0_ego import (wrap, step_feats, DT, STRIDE, NIN, B2D_NPZ, NAVSIM_NPZ,
                    B2D_MAT, B2D_TYPES, NAVSIM_MAT, NAVSIM_LOGS,
                    NUPLAN_NPZ, NUPLAN_EGO, NUPLAN_OOF_DIR,
                    build_nuplan, nuplan_scn_readout,
                    EGO_DX, EGO_DY, EGO_COS, EGO_SIN, EGO_SPEED, EGO_ISFUT)

# ── symbolic channel constants (KEYS.md: never numeric indices) ───────────────
LANE_X, LANE_Y, LANE_DXSEG, LANE_DYSEG = 0, 1, 2, 3
LF_JUNCTION, LF_ARCLEN = 0, 1
L2L_SUCC, L2L_LEFT, L2L_RIGHT, L2L_OPPOSITE = 0, 1, 2, 3
A2L_DLAT, A2L_DLON, A2L_COS, A2L_SIN = 0, 1, 2, 3
R_ON_ROUTE, R_REACH, R_SHARES, R_XSECT, R_ORDER = 0, 1, 2, 3, 4
AG_DX, AG_DY, AG_COS, AG_SIN, AG_SPEED, AG_HALFLEN, AG_HALFWID, AG_ISVEH = range(8)
ANCHOR_T = 3                   # the frozen graph is a single-frame snapshot at t = 3

MOVE_MS = 0.5                  # "moving": KEYS.md uses 0.5 m/s for a driving vehicle
R_QUERY = 60.0                 # lane query radius = the crop, so it is the distance sentinel
TTC_CAP = 20.0                 # seconds; also the "nobody is closing" sentinel
CLOS_EPS = 0.1                 # m/s below which range rate is not a closing rate

# ── the 20 scalars ───────────────────────────────────────────────────────────
# name, route_derived?, why it is here
FEATS = [
    ('n_moving',          False, 'moving agents in the window; denominator for the fractions'),
    ('n_static',          False, 'present but stopped agents; separates "busy" from "blocked"'),
    ('n_mov_corridor',    True,  'PROTOCOL: moving agents assigned to a route corridor lane'),
    ('n_mov_reach3hop',   True,  'PROTOCOL: moving agents on a lane reaching the route in 3 hops'),
    ('n_mov_shares_down', True,  'PROTOCOL: moving agents on a lane sharing downstream with route'),
    ('n_mov_xsect',       True,  'PROTOCOL: moving agents on a lane geometrically crossing route'),
    ('f_mov_corridor',    True,  'PROTOCOL: fraction of moving agents carrying each relation'),
    ('f_mov_reach3hop',   True,  'PROTOCOL: ditto'),
    ('f_mov_shares_down', True,  'PROTOCOL: ditto'),
    ('f_mov_xsect',       True,  'PROTOCOL: ditto'),
    ('d_min_route_agent', True,  'PROTOCOL: min ego distance to a route-related moving agent'),
    ('d_mean_route_agent',True,  'PROTOCOL: mean ego distance to route-related moving agents'),
    ('d_lead_corridor',   True,  'nearest AHEAD moving agent on a corridor lane (lead vehicle)'),
    ('ttc_min_route',     True,  'min range/closing-rate over route-related moving agents'),
    ('closing_max_route', True,  'max closing rate over route-related moving agents'),
    ('offcentre_max_route',True, 'largest lane-centre offset among route-related moving agents'),
    ('n_lanes',           False, 'lanes in the window; map clutter and denominator for f_junction'),
    ('n_corridor_lanes',  True,  'corridor width/length: how much of the map is the ego route'),
    ('f_junction_lanes',  False, 'junction vs open road, the strongest map-only difficulty cue'),
    ('route_order_span',  True,  'route_order_norm range over corridor lanes = route extent'),
]
RELF = [f[0] for f in FEATS]
ROUTE_DERIVED = np.array([f[1] for f in FEATS], bool)
D_REL = len(FEATS)

# ── section 19.1 (Z2/Z3): the ALIGNED input family — the channels whose B2D/NavSim gap is
# within +-0.42 sd (PROTOCOL_R 18.5.4 / 19): interaction geometry + ego kinematics only.
# NO counts (n_moving, n_lanes, n_corridor_lanes, ...), NO map granularity (offcentre,
# route_order_span, f_junction), NO command one-hots.  --feat-subset aligned selects it.
EGO_NAMES = ['speed', 'acc', 'yawrate', 'abs_acc', 'abs_yawrate', 'cmd_0', 'cmd_1', 'cmd_2', 'cmd_3']
assert len(EGO_NAMES) == NIN
ALIGNED_REL = ['n_mov_corridor', 'f_mov_reach3hop', 'f_mov_xsect', 'd_min_route_agent',
               'd_mean_route_agent', 'd_lead_corridor', 'ttc_min_route', 'closing_max_route']
assert all(n in RELF for n in ALIGNED_REL)
FEAT_SUBSETS = {'all': (EGO_NAMES, RELF), 'aligned': (EGO_NAMES[:5], ALIGNED_REL)}
# section 19.1 (Z1/Z3): B2D collision-only target (transfer/zs/build_zs_targets.py)
ZS_COLLISION_MAT = f'{RG}/transfer/zs/b2d_e2e16_collision_matrix.csv'
B2D_TARGET_MATS = {'full': B2D_MAT, 'collision': ZS_COLLISION_MAT}


def window_features(npz, chunk=2048):
    """frozen graph  ->  (N, D_REL) float32 scalars, (N,) route_rel_valid."""
    lane_mask = npz['lane_mask']
    lane_feat = npz['lane_feat']
    route_rel = npz['route_rel']
    rvalid = npz['route_rel_valid'].astype(bool)
    a2l_idx = npz['a2l_idx']
    a2l_mask = npz['a2l_mask']
    a2l_rel = npz['a2l_rel']
    agents = npz['agents']
    agent_mask = npz['agent_mask']
    ego = npz['ego']
    N, M = lane_mask.shape
    A, K = a2l_idx.shape[1], a2l_idx.shape[2]
    H = np.zeros((N, D_REL), np.float32)
    col = {n: i for i, n in enumerate(RELF)}

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        n = e - s
        lm = lane_mask[s:e]
        rr = route_rel[s:e].astype(np.float32)
        lf = lane_feat[s:e].astype(np.float32)
        idx = a2l_idx[s:e].astype(np.int64)
        cm = a2l_mask[s:e] & (a2l_idx[s:e] >= 0)                       # (n,A,K)
        arel = a2l_rel[s:e].astype(np.float32)
        ag = agents[s:e, :, ANCHOR_T].astype(np.float32)               # (n,A,8) anchor snapshot
        pres = agent_mask[s:e, :, ANCHOR_T]
        egsp = ego[s:e, ANCHOR_T, EGO_SPEED].astype(np.float32)        # anchor: heading is +x

        # per-agent relation flags: does ANY surviving lane candidate carry the relation
        gid = np.clip(idx, 0, M - 1)
        relk = rr[np.arange(n)[:, None, None], gid, :]                 # (n,A,K,5)
        relk = relk * cm[..., None]
        has = relk[..., :4].max(2) > 0.5                               # (n,A,4)
        rel_any = has.any(-1)

        spd = ag[..., AG_SPEED]
        mov = pres & (spd > MOVE_MS)
        sta = pres & ~mov
        px, py = ag[..., AG_DX], ag[..., AG_DY]
        dist = np.hypot(px, py)
        rmov = mov & rel_any                                           # route-related moving

        n_mov = mov.sum(1).astype(np.float32)
        H[s:e, col['n_moving']] = n_mov
        H[s:e, col['n_static']] = sta.sum(1)
        den = np.maximum(n_mov, 1.0)
        for r, nm, fm in ((R_ON_ROUTE, 'n_mov_corridor', 'f_mov_corridor'),
                          (R_REACH, 'n_mov_reach3hop', 'f_mov_reach3hop'),
                          (R_SHARES, 'n_mov_shares_down', 'f_mov_shares_down'),
                          (R_XSECT, 'n_mov_xsect', 'f_mov_xsect')):
            c = (mov & has[..., r]).sum(1).astype(np.float32)
            H[s:e, col[nm]] = c
            H[s:e, col[fm]] = c / den

        any_r = rmov.any(1)
        H[s:e, col['d_min_route_agent']] = np.where(
            any_r, np.where(rmov, dist, np.inf).min(1), R_QUERY)
        H[s:e, col['d_mean_route_agent']] = np.where(
            any_r, (np.where(rmov, dist, 0.0).sum(1) / np.maximum(rmov.sum(1), 1)), R_QUERY)

        lead = mov & has[..., R_ON_ROUTE] & (px > 0.0)
        H[s:e, col['d_lead_corridor']] = np.where(
            lead.any(1), np.where(lead, px, np.inf).min(1), R_QUERY)

        # closing geometry at the anchor: ego velocity is (speed, 0) in the anchor frame
        rvx = spd * ag[..., AG_COS] - egsp[:, None]
        rvy = spd * ag[..., AG_SIN]
        rng = np.maximum(dist, 1e-3)
        clos = -(px * rvx + py * rvy) / rng                            # + = approaching
        ttc = np.where(clos > CLOS_EPS, rng / np.maximum(clos, CLOS_EPS), TTC_CAP)
        ttc = np.clip(ttc, 0.0, TTC_CAP)
        H[s:e, col['ttc_min_route']] = np.where(
            any_r, np.where(rmov, ttc, TTC_CAP).min(1), TTC_CAP)
        H[s:e, col['closing_max_route']] = np.where(
            any_r, np.where(rmov, clos, 0.0).max(1), 0.0).clip(0.0)

        off = np.where(cm, np.abs(arel[..., A2L_DLAT]), np.inf).min(2)  # (n,A) nearest centreline
        off = np.where(np.isfinite(off), off, 0.0)
        H[s:e, col['offcentre_max_route']] = np.where(
            any_r, np.where(rmov, off, 0.0).max(1), 0.0)

        n_lane = lm.sum(1).astype(np.float32)
        n_corr = (rr[..., R_ON_ROUTE] * lm).sum(1)
        H[s:e, col['n_lanes']] = n_lane
        H[s:e, col['n_corridor_lanes']] = n_corr
        H[s:e, col['f_junction_lanes']] = (lf[..., LF_JUNCTION] * lm).sum(1) / np.maximum(n_lane, 1)
        cor = (rr[..., R_ON_ROUTE] > 0.5) & lm
        oo = rr[..., R_ORDER]
        H[s:e, col['route_order_span']] = np.where(
            cor.any(1), np.where(cor, oo, -np.inf).max(1) - np.where(cor, oo, np.inf).min(1), 0.0)

    assert np.isfinite(H).all(), 'non-finite relational feature'
    return H, rvalid


def impute_route(H, valid, fit_rows, rd=ROUTE_DERIVED):
    """Route-derived columns of invalid rows <- mean of the VALID rows in fit_rows.

    fit_rows is the training mask, so the imputed value never sees the evaluation rows.
    No validity flag is added: it would be a domain identifier (KEYS.md)."""
    H = H.copy()
    bad = ~valid
    if not bad.any():
        return H
    src = fit_rows & valid
    fill = H[src][:, rd].mean(0) if src.any() else H[:, rd].mean(0)
    H[np.ix_(bad, rd)] = fill
    return H


# ── pooling / ridge ──────────────────────────────────────────────────────────
def pool4(X, mask):
    """(n, L, C) + (n, L) -> (n, 4C) = [mean, max, min, std] over the masked axis."""
    m = mask[..., None].astype(np.float32)
    k = np.maximum(m.sum(1), 1.0)
    mean = (X * m).sum(1) / k
    mx = np.where(mask[..., None], X, -np.inf).max(1)
    mn = np.where(mask[..., None], X, np.inf).min(1)
    var = (((X - mean[:, None]) ** 2) * m).sum(1) / k
    return np.concatenate([mean, mx, mn, np.sqrt(var)], -1).astype(np.float32)


def _ridge_solve(Xtr, ytr, Xte, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    keep = sd > 1e-8
    A = (Xtr[:, keep] - mu[keep]) / sd[keep]
    B = (Xte[:, keep] - mu[keep]) / sd[keep]
    ym = ytr.mean()
    G = A.T @ A + alpha * np.eye(A.shape[1])
    w = np.linalg.solve(G, A.T @ (ytr - ym))
    return B @ w + ym


def group_folds(groups, k, rng):
    g = np.array(sorted(set(groups)))
    chunks = np.array_split(rng.permutation(len(g)), k)
    return [np.isin(groups, g[c]) for c in chunks]


ALPHAS = np.logspace(-2, 6, 17)          # wide enough that the CV optimum is interior


def ridge_predict(Xtr, ytr, Xte, groups_tr, rng, k=5):
    """alpha by inner group-CV inside the training rows only, then refit on all of them."""
    folds = [f for f in group_folds(groups_tr, k, rng) if f.any() and (~f).any()]
    err = np.zeros(len(ALPHAS))
    for f in folds:
        for i, al in enumerate(ALPHAS):
            p = _ridge_solve(Xtr[~f], ytr[~f], Xtr[f], al)
            err[i] += float(((p - ytr[f]) ** 2).mean())
    al = float(ALPHAS[int(np.argmin(err))])
    return _ridge_solve(Xtr, ytr, Xte, al), al


# ── model: R0's SeqNet with a second, identical branch on the window axis ─────
def build(torch, nn, d=64, tau=0.5, nin=NIN, drel=D_REL):
    class HRelNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(nin, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU())
            self.psi = nn.Sequential(nn.Linear(drel, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU())
            self.head = nn.Sequential(nn.LayerNorm(d * 8), nn.Linear(d * 8, d), nn.SiLU(),
                                      nn.Linear(d, 1))
            self.tau = tau

        def pool(self, z, mask):
            m = mask[..., None].float()
            n = m.sum(1).clamp(min=1)
            mean = (z * m).sum(1) / n
            mx = z.masked_fill(~mask[..., None], -1e9).max(1).values
            neg = torch.where(mask[..., None], -z / self.tau, torch.full_like(z, -float('inf')))
            smin = -self.tau * torch.logsumexp(neg, 1)
            var = ((z - mean[:, None]) ** 2 * m).sum(1) / n
            # a single-window item (all of NavSim, 8 B2D routes) has var identically 0,
            # and d var/d z is 0 there too, so raw sqrt backprops inf * 0 = NaN.
            return torch.cat([mean, mx, smin, (var + 1e-12).sqrt()], -1)

        def forward(self, x, mask, h, hmask):
            return self.head(torch.cat([self.pool(self.phi(x), mask),
                                        self.pool(self.psi(h), hmask)], -1)).squeeze(-1)
    return HRelNet()


# ── B2D ──────────────────────────────────────────────────────────────────────
def fixed_k_select(sp, W, K):
    """section 19.1 Z2: the K window indices (0..W-1) nearest to normalised arclength
    0, 1/(K-1), ..., 1.  Arclength of window w = distance travelled by the route's UNIQUE 2 Hz
    sequence up to the window's anchor step (STRIDE*w + ANCHOR_T), i.e. cumsum(|speed|*DT);
    normalised over the anchors' span.  A route whose anchors span no distance (static) falls
    back to index spacing w/(W-1).  W < K (or coincident anchors) -> the nearest window is
    REPEATED, so every route contributes exactly K windows.  -> (sel (K,), used_arclength)."""
    anchors = STRIDE * np.arange(W) + ANCHOR_T
    cum = np.concatenate([[0.0], np.cumsum(np.abs(sp, dtype=np.float64) * DT)])
    s = cum[anchors]
    span = s[-1] - s[0]
    if span > 1e-6:
        u = (s - s[0]) / span
    else:
        u = np.arange(W) / max(W - 1, 1)
    sel = np.array([int(np.argmin(np.abs(u - t))) for t in np.linspace(0.0, 1.0, K)])
    return sel, bool(span > 1e-6)


def build_b2d(mat=B2D_MAT, fixed_k=None):
    """R0's build_b2d, plus the per-window relational block on its own axis.
    mat      : response matrix csv (default the e2e16 full-fail panel; section 19.1 Z1 passes
               the collision-only panel).
    fixed_k  : section 19.1 Z2 — replace the stitched route sequence and the full window axis
               by exactly K windows per route (fixed_k_select): ego = the K windows' 12-step
               WINDOW-LOCAL features (tc.window_ego_feats, as NavSim items are built) laid
               end to end (K*12 steps, all valid), rel = the K windows' scalars.  Window count
               and route length are then invisible to the model."""
    import csv
    d = np.load(B2D_NPZ, allow_pickle=True)
    ids = [str(x) for x in d['item_id']]
    mm = [re.match(r'route_(\d+)_(\d+)$', s) for s in ids]
    assert all(mm), 'unexpected B2D item_id format'
    rid = np.array([m.group(1) for m in mm])
    widx = np.array([int(m.group(2)) for m in mm])
    ego = d['ego'].astype(np.float32)
    cmd = d['command'].astype(np.float32)
    sp_w = ego[:, :, EGO_SPEED]
    psi_w = np.arctan2(ego[:, :, EGO_SIN], ego[:, :, EGO_COS])
    Hw, rvalid = window_features(d)
    print(f'[b2d] relational scalars {Hw.shape}, route_rel_valid=False windows '
          f'{int((~rvalid).sum())}/{len(rvalid)}', flush=True)

    routes = sorted(set(rid))
    ov_sp, ov_psi, npair = 0.0, 0.0, 0
    for r in routes:
        idx = np.where(rid == r)[0][np.argsort(widx[rid == r])]
        for a, b in zip(idx[:-1], idx[1:]):
            ov_sp = max(ov_sp, np.abs(sp_w[a, STRIDE:] - sp_w[b, :12 - STRIDE]).max())
            da = psi_w[a, STRIDE:] - psi_w[a, STRIDE]
            db = psi_w[b, :12 - STRIDE] - psi_w[b, 0]
            ov_psi = max(ov_psi, np.abs(wrap(da - db)).max())
            npair += 1
    print(f'[b2d] overlap check on {npair} consecutive window pairs (stride {STRIDE}): '
          f'max |dspeed| {ov_sp:.4f} m/s, max |dpsi| {ov_psi:.5f} rad', flush=True)
    assert ov_sp < 0.05 and ov_psi < 0.05, 'window stride is not 4 — do not build the sequence'

    seqs, lens, static, hs, valid_r, sps = [], [], [], [], [], []
    for r in routes:
        sel = np.where(rid == r)[0]
        idx = sel[np.argsort(widx[sel])]
        W = len(idx)
        L = STRIDE * (W - 1) + 12
        sp = np.zeros(L, np.float32); psi = np.zeros(L, np.float32)
        cm = np.zeros((L, 4), np.float32); got = np.zeros(L, bool)
        for w, row in enumerate(idx):
            g0 = STRIDE * w
            off = 0.0 if w == 0 else float(psi[g0 + 3])
            for t in range(12):
                g = g0 + t
                if got[g]:
                    continue
                sp[g] = sp_w[row, t]; psi[g] = off + psi_w[row, t]
                cm[g] = cmd[row]; got[g] = True
        assert got.all()
        seqs.append(step_feats(sp, psi, cm)); lens.append(L)
        static.append(bool((sp_w[idx].max(1) < 0.5).any()))
        hs.append(Hw[idx]); valid_r.append(rvalid[idx]); sps.append(sp)
    T_MAX = max(lens)
    W_MAX = max(len(h) for h in hs)
    X = np.zeros((len(routes), T_MAX, NIN), np.float32)
    MK = np.zeros((len(routes), T_MAX), bool)
    HR = np.zeros((len(routes), W_MAX, D_REL), np.float32)
    HM = np.zeros((len(routes), W_MAX), bool)
    HV = np.zeros((len(routes), W_MAX), bool)
    for i, (s, h, v) in enumerate(zip(seqs, hs, valid_r)):
        X[i, :len(s)] = s; MK[i, :len(s)] = True
        HR[i, :len(h)] = h; HM[i, :len(h)] = True; HV[i, :len(h)] = v
    lens = np.array(lens)
    print(f'[b2d] {len(routes)} routes, unique 2 Hz steps p50 {int(np.median(lens))} '
          f'p90 {int(np.percentile(lens,90))} max {lens.max()} | windows per route p50 '
          f'{int(np.median([len(h) for h in hs]))} max {W_MAX}', flush=True)

    if fixed_k:
        import transfer_common as tc
        K = int(fixed_k)
        Xw = tc.window_ego_feats(ego, cmd)                    # (N, 12, NIN) window-local, as NavSim
        X = np.zeros((len(routes), K * 12, NIN), np.float32)
        MK = np.ones((len(routes), K * 12), bool)
        HR = np.zeros((len(routes), K, D_REL), np.float32)
        HM = np.ones((len(routes), K), bool)
        HV = np.zeros((len(routes), K), bool)
        n_short = n_rep = n_idxfb = 0; n_distinct = []
        for i, r in enumerate(routes):
            sel_r = np.where(rid == r)[0]
            idx = sel_r[np.argsort(widx[sel_r])]
            k_sel, used_arc = fixed_k_select(sps[i], len(idx), K)
            rows_k = idx[k_sel]
            X[i] = Xw[rows_k].reshape(K * 12, NIN)
            HR[i] = Hw[rows_k]; HV[i] = rvalid[rows_k]
            n_short += len(idx) < K; n_rep += len(set(k_sel.tolist())) < K
            n_idxfb += not used_arc; n_distinct.append(len(set(k_sel.tolist())))
        print(f'[b2d fixed-K] K={K} windows/route at normalised arclength '
              f'{", ".join(f"{t:.3f}" for t in np.linspace(0, 1, K))} (nearest window; '
              f'repeat-nearest when W<K or anchors coincide): routes with W<K {n_short}/{len(routes)}, '
              f'routes with a repeated window {n_rep}, distinct windows/route p50 '
              f'{int(np.median(n_distinct))} min {min(n_distinct)}, zero-arclength routes on index '
              f'spacing {n_idxfb} | ego = {K}x12 window-local steps (mask all True), rel = {K} '
              f'windows -> X {X.shape}, HR {HR.shape}', flush=True)

    rows = list(csv.reader(open(mat)))
    rids_m = rows[0][1:]
    body = [r for r in rows[1:] if r[0] != 'PDM-Lite']
    Yf = np.full((len(body), len(rids_m)), np.nan)
    for pi, row in enumerate(body):
        for j in range(len(rids_m)):
            if row[1 + j] != '':
                Yf[pi, j] = 1.0 - float(row[1 + j])
    col = {r: j for j, r in enumerate(rids_m)}
    tmap = dict(csv.reader(open(B2D_TYPES)))
    assert all(r in col and r in tmap for r in routes)
    Y = Yf[:, [col[r] for r in routes]]
    types = np.array([tmap[r] for r in routes])
    print(f'[b2d] response panel {Y.shape[0]} planners x {Y.shape[1]} routes, '
          f'missing cells {int(np.isnan(Y).sum())}, {len(set(types))} scenario types, '
          f'static routes {int(np.sum(static))}'
          + (f' | target matrix {mat}' if mat != B2D_MAT else ''), flush=True)
    return X, MK, HR, HM, HV, np.array(routes), types, Y, np.array(static)


def run_b2d(a):
    import torch, torch.nn as nn
    from numpy.polynomial.hermite_e import hermegauss
    from scipy.stats import spearmanr
    from scirt.encoder import rasch
    from b2d_splits import unified_split, R_DRAWS

    dev = 'cuda'
    # section 19.2 same-domain sanity: the run_full levers, all default-inert
    b2d_mat = B2D_TARGET_MATS[getattr(a, 'target', 'full')]
    fixed_k = getattr(a, 'fixed_k_windows', None)
    ego_names, rel_names = FEAT_SUBSETS[getattr(a, 'feat_subset', 'all')]
    xsel = [EGO_NAMES.index(n) for n in ego_names]; hsel = [RELF.index(n) for n in rel_names]
    nin, drel, rd = len(xsel), len(hsel), ROUTE_DERIVED[hsel]
    subset = getattr(a, 'feat_subset', 'all') != 'all'
    zs_active = subset or bool(fixed_k) or b2d_mat != B2D_MAT
    X, MK, HR, HM, HV, routes, types, Y, static = build_b2d(mat=b2d_mat, fixed_k=fixed_k)
    if subset:
        X, HR = np.ascontiguousarray(X[..., xsel]), np.ascontiguousarray(HR[..., hsel])
    if zs_active:
        print(f'[zs] OOF INPUT COLUMNS ({getattr(a, "feat_subset", "all")}): ego({nin}) '
              f'{ego_names} | rel({drel}) {rel_names} | X {X.shape} HR {HR.shape} | b2d target '
              f'{getattr(a, "target", "full")} | fixed-K windows {fixed_k}', flush=True)
    R, J = len(routes), Y.shape[0]
    fail = np.nanmean(Y, 0)
    _, b_ref = rasch(Y, it=800)
    print(f'[b2d] rho(observed failure rate, full-panel Rasch b) '
          f'{spearmanr(fail, b_ref).correlation:+.4f}', flush=True)

    gxn, gwn = hermegauss(15); gwn = gwn / gwn.sum()
    gx = torch.tensor(gxn, dtype=torch.float32, device=dev)
    lgw = torch.log(torch.tensor(gwn, dtype=torch.float32, device=dev))
    utypes = sorted(set(types))
    pred = {'ridge': np.full((R_DRAWS, R), np.nan), 'mlp': np.full((R_DRAWS, R), np.nan)}
    per_draw = {'ridge': [], 'mlp': []}
    alphas = []
    led = es.Ledger(R_DRAWS, a.epochs, a.early_stop)           # sigma / e* / curve / leak counts

    for draw in range(min(a.draws, R_DRAWS)):
        hp, ht = unified_split(draw, utypes, J)
        keepJ = np.array([j for j in range(J) if j not in hp])
        te = np.isin(types, list(ht)); tr = ~te
        # *** WHAT THE TWO-STAGE RULE MEANS FOR THE RIDGE: STAGE 2 IS ALL OF IT. ***
        # The ridge is closed-form with alpha chosen by inner group-CV; it has no epochs, so
        # there is no checkpoint for stage 1 to select and stage 1 contributes NOTHING to it.
        # Stage 2 -- "refit on the full outer-training block under theta_outer" -- is exactly
        # what the frozen ridge already does, so the ridge in es/ is expected to reproduce the
        # frozen ridge to floating-point noise, not to differ from it.  That is a check, not a
        # disappointment: if the es/ ridge moved, the two-stage plumbing would be wrong.
        # Concretely, the ridge is fitted ONCE per draw, on the FINAL stage's rows and b_tr.
        plan = es.TwoStage(draw, types, tr, a.epochs, a.early_stop)
        guard = es.HeldOutGuard(np.where(te)[0], None, hp, keepJ)
        guard.selftest()
        # ONE guard for BOTH stages: opened here, closed only after stage 2's last epoch.
        for stg in plan.stages():
            trn = stg.train                     # stage 1: A_train.  stage 2: the FULL A block.
            torch.manual_seed(a.seed); np.random.seed(a.seed)  # same seed -> same fresh init
            th_f, b_tr = rasch(Y[keepJ][:, trn])  # stage 1 theta_inner / stage 2 theta_outer
            plan.note_theta(stg, th_f)

            # route-derived scalars of invalid windows <- mean of the valid TRAINING windows
            HRi = HR.reshape(-1, drel)
            fitrow = np.repeat(trn, HR.shape[1]) & HM.reshape(-1)
            HRi = impute_route(HRi, HV.reshape(-1) | ~HM.reshape(-1), fitrow, rd).reshape(HR.shape)
            HRi = HRi * HM[..., None]

            mu = X[trn][MK[trn]].mean(0); sd = X[trn][MK[trn]].std(0) + 1e-6
            Xn = ((X - mu) / sd) * MK[..., None]
            hmu = HRi[trn][HM[trn]].mean(0); hsd = HRi[trn][HM[trn]].std(0) + 1e-6
            Hn = ((HRi - hmu) / hsd) * HM[..., None]

            # ── ridge features: [4 stats of ego over steps] + [4 stats of rel over windows] ──
            # The fit/predict itself is below the stage loop, after guard.end_training(): the
            # ridge is closed-form, so its training is already over by then, and this way
            # NOTHING reads a held-out row while the guard is open.  It uses its own
            # default_rng, so the move is numerically inert.
            F = np.concatenate([pool4(Xn, MK), pool4(Hn, HM)], 1)

            # ── mlp: R0's recipe, unchanged ──
            m = build(torch, nn, a.d, nin=nin, drel=drel).to(dev)
            ls = torch.tensor(-0.5, device=dev, requires_grad=True)
            opt = torch.optim.AdamW(list(m.parameters()) + [ls], lr=1e-3, weight_decay=0.1)
            THE = torch.tensor(th_f, dtype=torch.float32, device=dev)
            Yk = es.erase_heldout(Y[keepJ], te) if a.early_stop else Y[keepJ]
            Yd = torch.tensor(np.nan_to_num(Yk), dtype=torch.float32, device=dev)
            Md = torch.tensor((~np.isnan(Yk)).astype(np.float32), device=dev)
            Xd = torch.tensor(Xn, device=dev); Md_ = torch.tensor(MK, device=dev)
            Hd = torch.tensor(Hn, device=dev); Hm_ = torch.tensor(HM, device=dev)
            idx = guard.routes(np.where(trn)[0], f'{stg.name} train columns')
            iv_c = guard.routes(np.where(stg.iv)[0], f'{stg.name} inner-val columns')
            if a.early_stop:
                es.assert_masked(torch, Yd, Md, te, dev)
            print(plan.head(stg), flush=True)

            def fwd(sel):
                s = torch.tensor(sel, device=dev)
                return m(Xd[s], Md_[s], Hd[s], Hm_[s])

            for ep in range(stg.epochs):
                m.train(); np.random.shuffle(idx); tl = nb = 0
                for i0 in range(0, len(idx), a.bs):
                    sel = guard.routes(idx[i0:i0 + a.bs], f'{stg.name} train batch')
                    bt = fwd(sel); sg = torch.exp(ls)
                    z = (bt[None, :, None] + sg * gx[None, None, :]) - THE[:, None, None]
                    p = torch.sigmoid(z)
                    s = torch.tensor(sel, device=dev)
                    yy = Yd[:, s]; mm = Md[:, s]
                    llc = (yy[:, :, None] * torch.log(p + 1e-7)
                           + (1 - yy[:, :, None]) * torch.log(1 - p + 1e-7)) * mm[:, :, None]
                    loss = -torch.logsumexp(llc.sum(0) + lgw[None, :], 1).sum() / mm.sum() \
                        + 0.05 * ls.pow(2)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                    tl += float(loss); nb += 1
                if stg.select:
                    m.eval(); sg = torch.exp(ls).detach()
                    with torch.no_grad():
                        bv = fwd(guard.routes(iv_c, 'inner-val forward'))
                    nll = es.cell_nll(torch, bv, guard.routes(iv_c, 'inner-val likelihood'),
                                      Yd, Md, THE, gx, lgw, sg)
                    plan.update(stg, ep, tl / max(nb, 1), nll, float(sg))
                    print(f'  [b2d draw {draw}] s{stg.no} ep {ep:2d} loss {tl/max(nb,1):.4f} '
                          f'innerval nll {nll:.4f} sigma {float(sg):.3f}', flush=True)
                elif ep in (0, 4, stg.epochs - 1):
                    print(f'  [b2d draw {draw}] s{stg.no} ep {ep} loss {tl/max(nb,1):.4f} '
                          f'sigma {float(torch.exp(ls)):.3f}', flush=True)
            plan.end_stage(stg, float(torch.exp(ls)))
        if a.early_stop:
            print(plan.line(int(te.sum())), flush=True)
        guard.end_training()                     # held-out readable HERE, after STAGE 2 only
        m.eval()
        with torch.no_grad():
            ii = guard.heldout(np.where(te)[0])  # the ONE held-out pass of the whole draw
            pm = fwd(ii).cpu().numpy()
        pred['mlp'][draw, ii] = pm
        per_draw['mlp'].append(spearmanr(pm, fail[te]).correlation)
        # ── ridge, closed form: alpha by inner group-CV inside the FINAL stage's rows.
        # `trn`, `F` and `b_tr` are the last stage's, i.e. the full outer-training block under
        # theta_outer whenever the two-stage rule is on, and the frozen block when it is off.
        pr, al = ridge_predict(F[trn], b_tr, F[te], types[trn],
                               np.random.default_rng(100 * a.seed + draw))
        alphas.append(al)
        pred['ridge'][draw, ii] = pr
        per_draw['ridge'].append(spearmanr(pr, fail[te]).correlation)
        led.record(draw, plan, guard)
        print(guard.leak_line(draw), flush=True)
        print(f'  [b2d draw {draw}] held-out {te.sum()} routes  ridge (alpha {al:g}) '
              f'{per_draw["ridge"][-1]:+.4f}  mlp {per_draw["mlp"][-1]:+.4f}', flush=True)

    if getattr(a, 'zs_suffix', ''):        # section 19.2: lever runs never touch es/ or the root
        os.makedirs(f'{RG}/transfer/zs', exist_ok=True)
        out = f'{RG}/transfer/zs/hrel_b2d_s{a.seed}{a.zs_suffix}.npz'
    else:
        out = es.out_path(RG, f'hrel_b2d_s{a.seed}.npz', a.early_stop)
    np.savez(out, pred_ridge=pred['ridge'], pred_mlp=pred['mlp'], routes=routes, types=types,
             Y=Y, fail=fail, b_ref=b_ref, static=static, alphas=np.array(alphas),
             feat_names=np.array(rel_names),
             **(dict(ego_names=np.array(ego_names), feat_subset=getattr(a, 'feat_subset', 'all'),
                     fixed_k_windows=int(fixed_k or 0), b2d_target=getattr(a, 'target', 'full'),
                     b2d_target_mat=b2d_mat) if zs_active else {}),
             **led.fields())
    for key in ('ridge', 'mlp'):
        P, F, B, S = [], [], [], []
        for dd in range(R_DRAWS):
            k = np.isfinite(pred[key][dd])
            P.append(pred[key][dd][k]); F.append(fail[k]); B.append(b_ref[k]); S.append(static[k])
        P, F, B, S = map(np.concatenate, (P, F, B, S))
        pdv = per_draw[key]
        print(f'HREL_B2D seed={a.seed} model={key}  '
              f'per-draw rho {np.mean(pdv):+.4f} +/- {np.std(pdv, ddof=1):.4f}  '
              f'pooled rho {spearmanr(P, F).correlation:+.4f}  '
              f'rho_ref {spearmanr(P, B).correlation:+.4f}  '
              f'static {spearmanr(P[S], F[S]).correlation:+.4f}  '
              f'non-static {spearmanr(P[~S], F[~S]).correlation:+.4f}  '
              f'(pooled {len(P)} held-out cells)', flush=True)
    print(f'WROTE {out}', flush=True)


# ── NavSim ───────────────────────────────────────────────────────────────────
def build_navsim():
    import pandas as pd
    d = np.load(NAVSIM_NPZ, allow_pickle=True)
    toks = np.array([str(x) for x in d['item_id']])
    assert len(set(toks)) == len(toks)
    ego = d['ego'].astype(np.float32)
    cmd = d['command'].astype(np.float32)
    sp = ego[:, :, EGO_SPEED]
    psi = np.arctan2(ego[:, :, EGO_SIN], ego[:, :, EGO_COS])
    acc = np.gradient(sp, DT, axis=1)
    yr = np.concatenate([np.zeros((len(sp), 1), np.float32),
                         wrap(np.diff(psi, axis=1)) / DT], 1)
    X = np.concatenate([sp[..., None], acc[..., None], yr[..., None],
                        np.abs(acc)[..., None], np.abs(yr)[..., None],
                        np.repeat(cmd[:, None, :], 12, 1)], -1).astype(np.float32)
    MK = np.ones(X.shape[:2], bool)
    Hw, rvalid = window_features(d)
    HR = Hw[:, None, :]                                     # window axis has length 1
    HM = np.ones(HR.shape[:2], bool)
    HV = rvalid[:, None]
    print(f'[navsim] relational scalars {Hw.shape}, route_rel_valid=False windows '
          f'{int((~rvalid).sum())}/{len(rvalid)} ({100*(~rvalid).mean():.3f}%) -> route-derived '
          f'columns imputed with the valid-train mean, no validity flag as input', flush=True)

    t = np.load(NAVSIM_LOGS, allow_pickle=True)
    log_by_tok = {str(n): str(l) for n, l in zip(t['names'], t['log'])}
    assert set(toks) <= set(log_by_tok)
    logs = np.array([log_by_tok[x] for x in toks])

    M = pd.read_csv(NAVSIM_MAT, index_col=0)
    tok2col = {str(c): i for i, c in enumerate(M.columns)}
    assert set(toks) <= set(tok2col)
    V = M.values.astype(float)[:, [tok2col[x] for x in toks]]
    Y = (V < 0.5).astype(np.float32)
    Y[np.isnan(V)] = np.nan
    print(f'[navsim] {len(toks)} scenes, {len(set(logs))} logs, panel {Y.shape[0]} planners, '
          f'missing cells {int(np.isnan(Y).sum())}, overall failure rate {np.nanmean(Y):.4f}',
          flush=True)
    return X, MK, HR, HM, HV, toks, logs, Y


def run_navsim(a):
    import math
    import torch, torch.nn as nn
    from scipy.stats import spearmanr
    from scirt.encoder import rasch

    dev = 'cuda'
    X, MK, HR, HM, HV, toks, logs, Y = build_navsim()
    N = len(toks)
    fail = np.nanmean(Y, 0)
    _, b_ref = rasch(Y, it=800)
    print(f'[navsim] rho(observed failure rate, full-panel Rasch b) '
          f'{spearmanr(fail, b_ref).correlation:+.4f}', flush=True)

    ulogs = np.array(sorted(set(logs)))
    chunks = np.array_split(np.random.default_rng(0).permutation(len(ulogs)), a.nfolds)
    pred = {'ridge': np.full(N, np.nan), 'mlp': np.full(N, np.nan)}
    per_fold = {'ridge': [], 'mlp': []}
    alphas = []
    Yd = torch.tensor(np.nan_to_num(Y), device=dev)
    Md = torch.tensor((~np.isnan(Y)).astype(np.float32), device=dev)

    for fold in range(a.nfolds):
        te_logs = set(ulogs[chunks[fold]])
        rest = [l for l in ulogs if l not in te_logs]
        iv_logs = set(np.random.default_rng(100 + fold).choice(rest, 15, replace=False))
        te = np.array([l in te_logs for l in logs])
        iv = np.array([l in iv_logs for l in logs])
        tr = ~te & ~iv
        torch.manual_seed(a.seed + 1000); np.random.seed(a.seed)
        th_tr, b_tr = rasch(Y[:, tr])
        THE = torch.tensor(th_tr, dtype=torch.float32, device=dev)

        HRi = impute_route(HR[:, 0], HV[:, 0], tr)[:, None, :]
        mu = X[tr][MK[tr]].mean(0); sd = X[tr][MK[tr]].std(0) + 1e-6
        Xn = ((X - mu) / sd) * MK[..., None]
        hmu = HRi[tr][HM[tr]].mean(0); hsd = HRi[tr][HM[tr]].std(0) + 1e-6
        Hn = ((HRi - hmu) / hsd) * HM[..., None]

        F = np.concatenate([pool4(Xn, MK), pool4(Hn, HM)], 1)
        pr, al = ridge_predict(F[tr], b_tr, F[te], logs[tr],
                               np.random.default_rng(100 * a.seed + fold))
        alphas.append(al)
        pred['ridge'][te] = pr
        per_fold['ridge'].append(spearmanr(pr, fail[te]).correlation)

        Xd = torch.tensor(Xn, device=dev); Mk = torch.tensor(MK, device=dev)
        Hd = torch.tensor(Hn, device=dev); Hm_ = torch.tensor(HM, device=dev)
        m = build(torch, nn, a.d).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
        idx_tr = np.where(tr)[0]
        steps = a.epochs * max(len(idx_tr) // a.bs, 1)
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(s / 200 + 1e-2, 0.5 * (1 + math.cos(math.pi * s / max(steps, 1)))))
        shuf = np.random.default_rng(a.seed)

        def fwd(sel):
            s = torch.tensor(sel, device=dev)
            return m(Xd[s], Mk[s], Hd[s], Hm_[s])

        def pred_on(mask):
            ii = np.where(mask)[0]
            with torch.no_grad():
                return np.concatenate([fwd(ii[i:i + 2048]).cpu().numpy()
                                       for i in range(0, len(ii), 2048)])

        def nll_of(bv, mask):
            ii = torch.tensor(np.where(mask)[0], device=dev)
            p = torch.sigmoid(torch.tensor(bv, dtype=torch.float32, device=dev)[None, :]
                              - THE[:, None])
            yy = Yd[:, ii]; mm = Md[:, ii]
            return float((-(yy * torch.log(p + 1e-7)
                            + (1 - yy) * torch.log(1 - p + 1e-7)) * mm).sum() / mm.sum())

        best = (9e9, None, -1)
        for ep in range(a.epochs):
            m.train(); shuf.shuffle(idx_tr); tl = nb = 0
            for i0 in range(0, len(idx_tr) - a.bs + 1, a.bs):
                sel = idx_tr[i0:i0 + a.bs]
                s = torch.tensor(sel, device=dev)
                p = torch.sigmoid(fwd(sel)[None, :] - THE[:, None])
                yy = Yd[:, s]; mm = Md[:, s]
                loss = (-(yy * torch.log(p + 1e-7)
                          + (1 - yy) * torch.log(1 - p + 1e-7)) * mm).sum() / mm.sum()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); sch.step(); tl += float(loss); nb += 1
            m.eval()
            nll = nll_of(pred_on(iv), iv)
            if nll < best[0]:
                best = (nll, pred_on(te), ep)
            if ep in (0, 4, a.epochs - 1):
                print(f'  [navsim fold {fold}] ep {ep} train nll {tl/max(nb,1):.4f} '
                      f'innerval nll {nll:.4f}', flush=True)
        pred['mlp'][te] = best[1]
        per_fold['mlp'].append(spearmanr(best[1], fail[te]).correlation)
        print(f'  [navsim fold {fold}] train {tr.sum()} innerval {iv.sum()} test {te.sum()} | '
              f'best ep {best[2]} nll {best[0]:.4f} | ridge (alpha {al:g}) '
              f'{per_fold["ridge"][-1]:+.4f}  mlp {per_fold["mlp"][-1]:+.4f}', flush=True)

    out = f'{RG}/hrel_navsim_s{a.seed}.npz'
    np.savez(out, pred_ridge=pred['ridge'], pred_mlp=pred['mlp'], tokens=toks, logs=logs,
             fail=fail, b_ref=b_ref, Y=Y, alphas=np.array(alphas), feat_names=np.array(RELF))
    for key in ('ridge', 'mlp'):
        p = pred[key]; k = np.isfinite(p)
        pf = per_fold[key]
        print(f'HREL_NAVSIM seed={a.seed} model={key}  '
              f'per-fold rho {np.mean(pf):+.4f} +/- {np.std(pf, ddof=1):.4f}  '
              f'pooled rho {spearmanr(p[k], fail[k]).correlation:+.4f}  '
              f'rho_ref {spearmanr(p[k], b_ref[k]).correlation:+.4f}  '
              f'(out-of-fold {k.sum()}/{N})', flush=True)
    print(f'WROTE {out}', flush=True)


# ── nuPlan val14 same-domain OOF (section 20.2 S3) ───────────────────────────
def build_nuplan_hrel(which='logged'):
    """r0_ego.build_nuplan (item = window, logged ego default) plus the per-window relational
    scalars read off the frozen S1 graph tensor (row order asserted equal)."""
    X, MK, ids, toks, widx, logs, types, Y = build_nuplan(which)
    d = np.load(NUPLAN_NPZ, allow_pickle=True)
    gids = np.array([str(x) for x in d['item_id']])
    assert np.array_equal(gids, ids), 'graph rows must be the ego-tensor rows'
    Hw, rvalid = window_features(d)
    HR = Hw[:, None, :]                                     # window axis has length 1
    HM = np.ones(HR.shape[:2], bool)
    HV = rvalid[:, None]
    print(f'[nuplan] relational scalars {Hw.shape}, route_rel_valid=False windows '
          f'{int((~rvalid).sum())}/{len(rvalid)} ({100*(~rvalid).mean():.3f}%) -> route-derived '
          f'columns imputed with the valid-train mean, no validity flag as input', flush=True)
    return X, MK, HR, HM, HV, ids, toks, widx, logs, types, Y


def run_nuplan(a):
    """S3 H-rel: hrel.run_navsim verbatim on nuPlan val14 windows, folds log-disjoint over the
    218 logs.  Writes ONLY to transfer/nuplan/."""
    import math
    import torch, torch.nn as nn
    from scipy.stats import spearmanr
    from scirt.encoder import rasch

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, MK, HR, HM, HV, ids, toks, widx, logs, types, Y = build_nuplan_hrel(a.nuplan_ego)
    N = len(ids)
    fail = np.nanmean(Y, 0)
    _, b_ref = rasch(Y, it=800)
    print(f'[nuplan] rho(observed failure rate, full-panel Rasch b) '
          f'{spearmanr(fail, b_ref).correlation:+.4f}', flush=True)

    ulogs = np.array(sorted(set(logs)))
    chunks = np.array_split(np.random.default_rng(0).permutation(len(ulogs)), a.nfolds)
    pred = {'ridge': np.full(N, np.nan), 'mlp': np.full(N, np.nan)}
    fold_of = np.full(N, -1, np.int64)
    per_fold = {'ridge': [], 'mlp': []}
    alphas = []
    Yd = torch.tensor(np.nan_to_num(Y), device=dev)
    Md = torch.tensor((~np.isnan(Y)).astype(np.float32), device=dev)
    b_by_tok = {t: b for t, b in zip(toks, b_ref)}

    def scn_rho(pw, mask):
        scn, ro = nuplan_scn_readout(pw, toks, widx, mask)
        return float(spearmanr(ro[0], np.array([b_by_tok[t] for t in scn])).correlation)

    for fold in range(a.nfolds):
        te_logs = set(ulogs[chunks[fold]])
        rest = [l for l in ulogs if l not in te_logs]
        iv_logs = set(np.random.default_rng(100 + fold).choice(rest, 15, replace=False))
        te = np.array([l in te_logs for l in logs])
        iv = np.array([l in iv_logs for l in logs])
        tr = ~te & ~iv
        torch.manual_seed(a.seed + 1000); np.random.seed(a.seed)
        th_tr, b_tr = rasch(Y[:, tr])
        THE = torch.tensor(th_tr, dtype=torch.float32, device=dev)

        HRi = impute_route(HR[:, 0], HV[:, 0], tr)[:, None, :]
        mu = X[tr][MK[tr]].mean(0); sd = X[tr][MK[tr]].std(0) + 1e-6
        Xn = ((X - mu) / sd) * MK[..., None]
        hmu = HRi[tr][HM[tr]].mean(0); hsd = HRi[tr][HM[tr]].std(0) + 1e-6
        Hn = ((HRi - hmu) / hsd) * HM[..., None]

        F = np.concatenate([pool4(Xn, MK), pool4(Hn, HM)], 1)
        pr, al = ridge_predict(F[tr], b_tr, F[te], logs[tr],
                               np.random.default_rng(100 * a.seed + fold))
        alphas.append(al)
        pred['ridge'][te] = pr

        Xd = torch.tensor(Xn, device=dev); Mk = torch.tensor(MK, device=dev)
        Hd = torch.tensor(Hn, device=dev); Hm_ = torch.tensor(HM, device=dev)
        m = build(torch, nn, a.d).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
        idx_tr = np.where(tr)[0]
        steps = a.epochs * max(len(idx_tr) // a.bs, 1)
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(s / 200 + 1e-2, 0.5 * (1 + math.cos(math.pi * s / max(steps, 1)))))
        shuf = np.random.default_rng(a.seed)

        def fwd(sel):
            s = torch.tensor(sel, device=dev)
            return m(Xd[s], Mk[s], Hd[s], Hm_[s])

        def pred_on(mask):
            ii = np.where(mask)[0]
            with torch.no_grad():
                return np.concatenate([fwd(ii[i:i + 2048]).cpu().numpy()
                                       for i in range(0, len(ii), 2048)])

        def nll_of(bv, mask):
            ii = torch.tensor(np.where(mask)[0], device=dev)
            p = torch.sigmoid(torch.tensor(bv, dtype=torch.float32, device=dev)[None, :]
                              - THE[:, None])
            yy = Yd[:, ii]; mm = Md[:, ii]
            return float((-(yy * torch.log(p + 1e-7)
                            + (1 - yy) * torch.log(1 - p + 1e-7)) * mm).sum() / mm.sum())

        best = (9e9, None, -1)
        for ep in range(a.epochs):
            m.train(); shuf.shuffle(idx_tr); tl = nb = 0
            for i0 in range(0, len(idx_tr) - a.bs + 1, a.bs):
                sel = idx_tr[i0:i0 + a.bs]
                s = torch.tensor(sel, device=dev)
                p = torch.sigmoid(fwd(sel)[None, :] - THE[:, None])
                yy = Yd[:, s]; mm = Md[:, s]
                loss = (-(yy * torch.log(p + 1e-7)
                          + (1 - yy) * torch.log(1 - p + 1e-7)) * mm).sum() / mm.sum()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); sch.step(); tl += float(loss); nb += 1
            m.eval()
            nll = nll_of(pred_on(iv), iv)
            if nll < best[0]:
                best = (nll, pred_on(te), ep)
            if ep in (0, 4, a.epochs - 1):
                print(f'  [nuplan fold {fold}] ep {ep} train nll {tl/max(nb,1):.4f} '
                      f'innerval nll {nll:.4f}', flush=True)
        pred['mlp'][te] = best[1]
        fold_of[te] = fold
        per_fold['ridge'].append(scn_rho(pred['ridge'], te))
        per_fold['mlp'].append(scn_rho(pred['mlp'], te))
        print(f'  [nuplan fold {fold}] train {tr.sum()} innerval {iv.sum()} test {te.sum()} | '
              f'best ep {best[2]} nll {best[0]:.4f} | rho_scene(w0) ridge (alpha {al:g}) '
              f'{per_fold["ridge"][-1]:+.4f}  mlp {per_fold["mlp"][-1]:+.4f}', flush=True)

    os.makedirs(NUPLAN_OOF_DIR, exist_ok=True)
    out = f'{NUPLAN_OOF_DIR}/hrel_nuplan_oof_s{a.seed}.npz'
    np.savez(out, pred_ridge=pred['ridge'], pred_mlp=pred['mlp'], item_id=ids, tokens=toks,
             widx=widx, logs=logs, types=types, fold=fold_of, fail=fail, b_ref=b_ref, Y=Y,
             alphas=np.array(alphas), feat_names=np.array(RELF), nuplan_ego=a.nuplan_ego,
             per_fold_ridge=np.array(per_fold['ridge']), per_fold_mlp=np.array(per_fold['mlp']))
    for key in ('ridge', 'mlp'):
        scn, ro = nuplan_scn_readout(pred[key], toks, widx)
        bt = np.array([b_by_tok[t] for t in scn])
        pf = per_fold[key]
        print(f'HREL_NUPLAN seed={a.seed} ego={a.nuplan_ego} model={key}  pooled rho_scene w0 '
              f'{spearmanr(ro[0], bt).correlation:+.4f}  mean {spearmanr(ro[2], bt).correlation:+.4f}  '
              f'w1 {spearmanr(ro[1], bt).correlation:+.4f}  per-fold(w0) {np.mean(pf):+.4f} '
              f'+/- {np.std(pf, ddof=1):.4f}', flush=True)
    print(f'WROTE {out}', flush=True)


# ── feature-only diagnostic ──────────────────────────────────────────────────
def show_features(domain):
    from scipy.stats import spearmanr
    d = np.load(B2D_NPZ if domain == 'b2d' else NAVSIM_NPZ, allow_pickle=True)
    H, v = window_features(d)
    print(f'{domain}: N={len(H)}  route_rel_valid=False {int((~v).sum())} '
          f'({100*(~v).mean():.3f}%)')
    print(f'{"feature":22s} {"route?":6s} {"mean":>9s} {"sd":>9s} {"p50":>9s} '
          f'{"p95":>9s} {"min":>8s} {"max":>8s}  zero-frac')
    for i, (nm, rd, _) in enumerate(FEATS):
        c = H[v, i]
        print(f'{nm:22s} {"R" if rd else "-":6s} {c.mean():9.3f} {c.std():9.3f} '
              f'{np.median(c):9.3f} {np.percentile(c,95):9.3f} {c.min():8.2f} {c.max():8.2f}'
              f'   {(c==0).mean():.3f}')
    C = np.corrcoef(H[v].T)
    np.fill_diagonal(C, 0)
    print('\nmost collinear pairs:')
    for _ in range(6):
        i, j = np.unravel_index(np.argmax(np.abs(C)), C.shape)
        print(f'  {RELF[i]:22s} {RELF[j]:22s} r={C[i,j]:+.3f}')
        C[i, j] = C[j, i] = 0


# ── section 18: ONE model on ALL source items, forwarded on the other domain ──────────────
def run_full(a):
    """--full-train --transfer-to DOM [--subsample-logs N].  Additive: run_b2d / run_navsim are
    untouched.  Both readouts: the MLP under the nested rule (schedule / e* / recipe from
    transfer_common, es.TwoStage underneath) and the ridge fitted closed-form on ALL source
    items -> source Rasch b (alpha by inner group-CV in the source), then applied to the target
    under the SOURCE standardisation.  The target is loaded AFTER training; its responses are
    never read or saved."""
    import torch, torch.nn as nn
    from scipy.stats import spearmanr
    from scirt.encoder import rasch
    import transfer_common as tc

    dev, devname = tc.device(torch)
    src, tgt = a.domain, a.transfer_to
    tag = f'hrel {tc.src_tag(a)}->{tgt} s{a.seed}'
    sub_logs = np.array([])
    # section 19.1 flags (all default-inert): B2D target matrix, fixed-K window pooling, column subset
    b2d_mat = B2D_TARGET_MATS[getattr(a, 'target', 'full')]
    fixed_k = getattr(a, 'fixed_k_windows', None)
    ego_names, rel_names = FEAT_SUBSETS[getattr(a, 'feat_subset', 'all')]
    xsel = [EGO_NAMES.index(n) for n in ego_names]; hsel = [RELF.index(n) for n in rel_names]
    nin, drel, rd = len(xsel), len(hsel), ROUTE_DERIVED[hsel]
    subset = getattr(a, 'feat_subset', 'all') != 'all'
    zs_active = subset or bool(fixed_k) or b2d_mat != B2D_MAT
    if src == 'b2d':
        X, MK, HR, HM, HV, items, grp, Y, _ = build_b2d(mat=b2d_mat, fixed_k=fixed_k)
    else:
        X, MK, HR, HM, HV, items, grp, Y = build_navsim()
        if a.subsample_logs:
            keep, sub_logs = tc.subsample_logs(grp, a.subsample_logs, a.subsample_seed)
            X, MK, HR, HM, HV, items, grp = (X[keep], MK[keep], HR[keep], HM[keep], HV[keep],
                                             items[keep], grp[keep])
            Y = Y[:, keep]
    if subset:
        X, HR = np.ascontiguousarray(X[..., xsel]), np.ascontiguousarray(HR[..., hsel])
    print(f'[zs] INPUT COLUMNS ({getattr(a, "feat_subset", "all")}): ego({nin}) {ego_names} | '
          f'rel({drel}) {rel_names} | X {X.shape} HR {HR.shape} | b2d target '
          f'{getattr(a, "target", "full")} | fixed-K windows {fixed_k}', flush=True)
    N = len(items)
    fail = np.nanmean(Y, 0)
    _, b_src = rasch(Y, it=800)                   # SOURCE full-panel reference, sanity rho only
    itr, iv = tc.inner_split(src, grp)
    plan = tc.FullTrainPlan(grp, itr, iv, a.epochs)
    Yd = torch.tensor(np.nan_to_num(Y), dtype=torch.float32, device=dev)
    Md = torch.tensor((~np.isnan(Y)).astype(np.float32), device=dev)
    Mk = torch.tensor(MK, device=dev)

    for stg in plan.stages():
        trn = stg.train                           # stage 1: inner-train.  stage 2: ALL items.
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        th_f, b_tr = rasch(Y[:, trn])             # theta / b on this stage's SOURCE columns
        torch.manual_seed(a.seed)                 # proper-init (section 12): re-seed AFTER rasch
        plan.note_theta(stg, th_f)
        # route-derived scalars of invalid windows <- mean of the valid TRAINING windows (as run_b2d)
        HR2 = HR.reshape(-1, drel)
        fitrow = np.repeat(trn, HR.shape[1]) & HM.reshape(-1)
        valid2 = HV.reshape(-1) | ~HM.reshape(-1)
        HRi = impute_route(HR2, valid2, fitrow, rd).reshape(HR.shape) * HM[..., None]
        h_fill = HR2[fitrow & valid2][:, rd].mean(0)   # the same fill, for the target
        mu = X[trn][MK[trn]].mean(0); sd = X[trn][MK[trn]].std(0) + 1e-6
        Xn = ((X - mu) / sd) * MK[..., None]
        hmu = HRi[trn][HM[trn]].mean(0); hsd = HRi[trn][HM[trn]].std(0) + 1e-6
        Hn = ((HRi - hmu) / hsd) * HM[..., None]
        F = np.concatenate([pool4(Xn, MK), pool4(Hn, HM)], 1)
        m = build(torch, nn, a.d, nin=nin, drel=drel).to(dev)
        print(f'  [init] s{stg.no} seed {a.seed} proper_init=True weight-hash {es.init_hash(m)}',
              flush=True)
        THE = torch.tensor(th_f, dtype=torch.float32, device=dev)
        Xd = torch.tensor(Xn, device=dev); Hd = torch.tensor(Hn, device=dev)
        Hm_ = torch.tensor(HM, device=dev)
        print(plan.head(stg), flush=True)

        def fwd(sel):
            s = torch.tensor(sel, device=dev)
            return m(Xd[s], Mk[s], Hd[s], Hm_[s])
        tc.fit_stage(src, torch, nn, m, fwd, stg, plan, Yd, Md, THE, a.bs, a.seed, dev, tag,
                     warmup=a.warmup_steps)
    print(plan.line(0), flush=True)
    # ridge, closed form, on the FINAL stage's rows (= all source items) and b_tr, as the loop
    pr_src, al = ridge_predict(F, b_tr, F, grp, np.random.default_rng(100 * a.seed))
    pm_src = tc.predict(torch, m, fwd, N)
    rho_m = float(spearmanr(pm_src, b_src).correlation)
    rho_r = float(spearmanr(pr_src, b_src).correlation)
    print(f'  [{tag}] e* {plan.best_ep}, stage 2 {plan.stage2_epochs} epochs on {N} items | '
          f'source in-sample rho_ref mlp {rho_m:+.4f} ridge (alpha {al:g}) {rho_r:+.4f} | '
          f'rho_fail mlp {spearmanr(pm_src, fail).correlation:+.4f} ridge '
          f'{spearmanr(pr_src, fail).correlation:+.4f}', flush=True)

    # ── target: loaded only now; imputation fill, mu/sd, hmu/hsd and the ridge are all SOURCE ──
    out = dict(arm='hrel', src=tc.src_tag(a), tgt=tgt, seed=a.seed, device=devname, bs=a.bs,
               epochs=a.epochs, pred_src_mlp=pm_src, pred_src_ridge=pr_src, src_item_id=items,
               src_groups=grp, src_b_ref=b_src, rho_src_insample_mlp=rho_m,
               rho_src_insample_ridge=rho_r, ridge_alpha=al, x_mu=mu, x_sd=sd, h_mu=hmu,
               h_sd=hsd, h_fill=h_fill, feat_names=np.array(rel_names), sub_logs=sub_logs,
               subsample_seed=a.subsample_seed, warmup_steps=a.warmup_steps,
               **tc.plan_fields(plan))
    if zs_active:                       # section 19.1 bookkeeping; absent under the defaults
        out.update(ego_names=np.array(ego_names), feat_subset=getattr(a, 'feat_subset', 'all'),
                   fixed_k_windows=int(fixed_k or 0), b2d_target=getattr(a, 'target', 'full'),
                   b2d_target_mat=b2d_mat)

    def feats_t(Xt, MKt, HRt, HMt, HVt):
        if subset:
            Xt, HRt = np.ascontiguousarray(Xt[..., xsel]), np.ascontiguousarray(HRt[..., hsel])
        H2 = HRt.reshape(-1, drel).copy()
        bad = ~(HVt.reshape(-1) | ~HMt.reshape(-1))
        H2[np.ix_(bad, rd)] = h_fill
        HRit = H2.reshape(HRt.shape) * HMt[..., None]
        Xnt = ((Xt - mu) / sd) * MKt[..., None]
        Hnt = ((HRit - hmu) / hsd) * HMt[..., None]
        return Xnt, Hnt, np.concatenate([pool4(Xnt, MKt), pool4(Hnt, HMt)], 1)

    def mlp_t(Xnt, MKt, Hnt, HMt):
        Xd_t, Mk_t = torch.tensor(Xnt, device=dev), torch.tensor(MKt, device=dev)
        Hd_t, Hm_t = torch.tensor(Hnt, device=dev), torch.tensor(HMt, device=dev)
        return tc.predict(torch, m, lambda sel: m(*(z[torch.tensor(sel, device=dev)]
                                                    for z in (Xd_t, Mk_t, Hd_t, Hm_t))), len(Xnt))

    ridge_t = lambda Ft: _ridge_solve(F, b_tr, Ft, al)
    if tgt == 'navsim':
        Xt, MKt, HRt, HMt, HVt, toks, logs, _ = build_navsim()   # its Y is discarded, never saved
        Xnt, Hnt, Ft = feats_t(Xt, MKt, HRt, HMt, HVt)
        out.update(pred_mlp=mlp_t(Xnt, MKt, Hnt, HMt), pred_ridge=ridge_t(Ft), item_id=toks,
                   tgt_groups=logs)
    elif tgt == 'nuplan':
        # section 20.2 S2: window-level forward on val14 under the SOURCE statistics; rel scalars
        # from the frozen S1 graph tensor, ego from BOTH tensors (routed = shipped, synthetic
        # future; logged = rebuild).  No nuPlan response is read here.
        dn = np.load(NUPLAN_NPZ, allow_pickle=True)
        wids = np.array([str(x) for x in dn['item_id']])
        Hw_n, rv_n = window_features(dn)
        HMw = np.ones((len(Hw_n), 1), bool)
        wtok = np.array([s.rsplit('_', 1)[0] for s in wids])
        wdx = np.array([int(s.rsplit('_', 1)[1]) for s in wids])
        out.update(item_id=wids, tgt_groups=wtok, widx=wdx)
        for which in ('routed', 'logged'):
            e = np.load(NUPLAN_EGO[which], allow_pickle=True)
            gid = np.array([f'{str(t)}_{int(w)}' for t, w in zip(e['token'], e['widx'])])
            assert np.array_equal(wids, gid), 'ego-tensor rows must be the graph rows'
            Xw_n = tc.window_ego_feats(e['ego'].astype(np.float32), e['cmd'].astype(np.float32))
            MKw = np.ones(Xw_n.shape[:2], bool)
            Xnw, Hnw, Fw = feats_t(Xw_n, MKw, Hw_n[:, None], HMw, rv_n[:, None])
            out[f'pred_mlp_{which}'] = mlp_t(Xnw, MKw, Hnw, HMw)
            out[f'pred_ridge_{which}'] = ridge_t(Fw)
    else:
        # route level = the loop's OWN B2D item: stitched ego sequence + the route's window axis
        Xr, MKr, HRr, HMr, HVr, routes, types, _, _ = build_b2d()
        Xnr, Hnr, Fr = feats_t(Xr, MKr, HRr, HMr, HVr)
        # window level = one window per item, as NavSim rows are
        d = np.load(B2D_NPZ, allow_pickle=True)
        wids = np.array([str(x) for x in d['item_id']])
        Xw = tc.window_ego_feats(d['ego'].astype(np.float32), d['command'].astype(np.float32))
        Hw, rv = window_features(d)
        MKw = np.ones(Xw.shape[:2], bool); HMw = np.ones((len(Hw), 1), bool)
        Xnw, Hnw, Fw = feats_t(Xw, MKw, Hw[:, None], HMw, rv[:, None])
        W = tc.b2d_windows_by_route(wids, routes)
        pm_w, pr_w = mlp_t(Xnw, MKw, Hnw, HMw), ridge_t(Fw)
        out.update(pred_mlp=pm_w, pred_ridge=pr_w, item_id=wids,
                   pred_route_mlp=mlp_t(Xnr, MKr, Hnr, HMr), pred_route_ridge=ridge_t(Fr),
                   pred_route_winmean_mlp=tc.route_mean(pm_w, W, len(routes)),
                   pred_route_winmean_ridge=tc.route_mean(pr_w, W, len(routes)),
                   routes=routes, tgt_groups=types)
    npred = len(out['pred_mlp_routed' if tgt == 'nuplan' else 'pred_mlp'])
    print(f'  [{tag}] target {tgt}: {npred} window preds (mlp + ridge'
          + (', routed + logged ego)' if tgt == 'nuplan' else ')')
          + (f', {len(out["pred_route_mlp"])} route preds' if 'pred_route_mlp' in out else ''),
          flush=True)
    tc.save(tc.out_path('hrel', a), **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, choices=['b2d', 'navsim', 'nuplan'])
    ap.add_argument('--gpu', default='2')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--nuplan-ego', default='logged', choices=['routed', 'logged'],
                    help='section 20.2 S3, --domain nuplan (same-domain OOF) only: which val14 '
                         'ego tensor feeds the ego branch. Default logged (the shipped/routed '
                         'ego future is synthetic, S0 caveat).')
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--bs', type=int, default=None)
    ap.add_argument('--nfolds', type=int, default=5)
    ap.add_argument('--draws', type=int, default=99)
    ap.add_argument('--features', action='store_true')
    ap.add_argument('--early-stop', action='store_true',
                    help='B2D only: two-stage nested CHECKPOINT SELECTION BY INNER-VALIDATION '
                         'NLL. Stage 1 carves an inner-val split of whole scenario types out of '
                         'the training block, fits theta on the A_train responses, runs all 30 '
                         'epochs and takes e* = argmin inner-val NLL. Stage 2 refits theta on '
                         'the FULL outer-training responses, re-initialises the model with the '
                         'same seed and trains on the FULL outer-training block for exactly e*+1 '
                         'epochs. The held-out block is forwarded once, after stage 2. So the '
                         'only difference from the frozen arm is 30 -> e*+1 epochs. Nothing is '
                         'stopped early; all 30 epochs are always run. Writes to RG/es/, never '
                         'over frozen30. NAVSIM already selects on inner-val and is not run.')
    ap.add_argument('--full-train', action='store_true',
                    help='PROTOCOL_R section 18: ONE model per seed on ALL source items. Stage 1 '
                         'selects e* by inner-val NLL on a grouped inner split (B2D: 8 of 44 '
                         'types; NavSim: 1/5 of the logs), stage 2 refits on every source item '
                         'for e*+1 epochs with theta refitted on every source column (the '
                         'section-12 nested rule). Re-seeds after rasch (proper-init) always. '
                         'Writes only under RG/transfer/. Requires --transfer-to.')
    ap.add_argument('--transfer-to', default=None, choices=['b2d', 'navsim', 'nuplan'],
                    help='after --full-train, forward the OTHER domain tensor under the SOURCE '
                         'normalisation statistics and save window-level (and, for a B2D '
                         'target, route-level) predictions plus the source in-sample prediction. '
                         'nuplan (section 20.2 S2): both ego tensors are forwarded, keys '
                         'pred_{mlp,ridge}_{routed,logged}; no nuPlan response is read.')
    ap.add_argument('--subsample-logs', type=int, default=None,
                    help='NavSim source only: keep whole logs (fixed seed) until ~N windows '
                         '(section 18.2 T3, N=2656). The kept logs are recorded in the output.')
    ap.add_argument('--subsample-seed', type=int, default=None,
                    help='section 18.6 (a): RNG seed of the --subsample-logs log draw. Default '
                         '3000 (= section 18 T3, file name unchanged); any other value appends '
                         '_sub<seed> to the output name.')
    ap.add_argument('--feat-subset', default='all', choices=sorted(FEAT_SUBSETS),
                    help='PROTOCOL_R section 19.1 Z2/Z3: "aligned" restricts the inputs to the '
                         'cross-domain-aligned family (8 H-rel interaction scalars + 5 ego '
                         'kinematics, no command one-hots, no counts, no map granularity). '
                         'Default "all" = every column, byte-identical to section 18. '
                         '--full-train only; output name gets _fal and goes to transfer/zs/.')
    ap.add_argument('--fixed-k-windows', type=int, default=None,
                    help='section 19.1 Z2/Z3, B2D source only: pool each route over exactly K '
                         'windows sampled at normalised arclength 0..1 (nearest window, repeated '
                         'when the route has fewer than K), ego = those windows\' window-local '
                         '12-step features; removes the window-count / route-length cue. '
                         'Default None = stitched sequence + all windows (section 18). '
                         '--full-train only; output name gets _fk<K> and goes to transfer/zs/.')
    ap.add_argument('--target', default='full', choices=sorted(B2D_TARGET_MATS),
                    help='section 19.1 Z1/Z3, B2D source only: "collision" trains on the '
                         'collision-only panel transfer/zs/b2d_e2e16_collision_matrix.csv '
                         '(Rasch theta refitted on it). Default "full" = e2e16 full-fail panel. '
                         '--full-train only; output name gets _tcol and goes to transfer/zs/.')
    ap.add_argument('--warmup-steps', type=int, default=None,
                    help='section 18.6 (b)/(c): NavSim-recipe LR warmup length in optimizer '
                         'steps. Default 200 (= the frozen recipe, file name unchanged); any '
                         'other value appends _wu<n> to the output name.')
    a = ap.parse_args()
    if a.features:
        show_features(a.domain)
        return
    # section 19.1: additive flags, --full-train only, B2D-source only for the two B2D levers
    a.zs_suffix = ((f'_tcol' if a.target == 'collision' else '')
                   + (f'_fal' if a.feat_subset == 'aligned' else '')
                   + (f'_fk{a.fixed_k_windows}' if a.fixed_k_windows else ''))
    if a.zs_suffix:
        assert (a.full_train and a.transfer_to) or a.domain == 'b2d', \
            'section-19.1 flags: --full-train/--transfer-to, or the B2D 16-draw OOF sanity (19.2)'
        if a.target != 'full' or a.fixed_k_windows:
            assert a.domain == 'b2d', '--target / --fixed-k-windows are B2D-source levers'
        assert not a.fixed_k_windows or a.fixed_k_windows >= 1
        eg, rl = FEAT_SUBSETS[a.feat_subset]
        print(f'[zs] section 19.1 arm flags: target={a.target} feat_subset={a.feat_subset} '
              f'fixed_k_windows={a.fixed_k_windows} suffix={a.zs_suffix}\n'
              f'[zs] INPUT COLUMNS ego({len(eg)}): {eg}\n[zs] INPUT COLUMNS rel({len(rl)}): {rl}',
              flush=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    if a.domain == 'nuplan':                       # section 20.2 S3: same-domain OOF only
        assert not (a.full_train or a.transfer_to or a.subsample_logs or a.early_stop
                    or a.zs_suffix), '--domain nuplan is the S3 same-domain OOF only'
        a.epochs = a.epochs or 40; a.bs = a.bs or 256   # frozen NavSim-style recipe
        run_nuplan(a)
        return
    if a.transfer_to == 'nuplan':
        a.nuplan_ego = 'routed'   # both ego variants land in ONE output file; no _logged suffix
    if a.full_train or a.transfer_to or a.subsample_logs \
            or a.subsample_seed is not None or a.warmup_steps is not None:
        import transfer_common as tc
        a.subsample_seed = tc.SUBSAMPLE_SEED if a.subsample_seed is None else a.subsample_seed
        a.warmup_steps = tc.WARMUP_STEPS if a.warmup_steps is None else a.warmup_steps
        tc.check_args(a)
        assert not a.early_stop, '--full-train carries its own nested selection; drop --early-stop'
        a.epochs = a.epochs or (30 if a.domain == 'b2d' else 40)
        a.bs = a.bs or (64 if a.domain == 'b2d' else 256)
        run_full(a)
        return
    assert not (a.early_stop and a.domain == 'navsim'), \
        'NAVSIM already selects the best-inner-val checkpoint; --early-stop is B2D only'
    if a.domain == 'b2d':
        a.epochs = a.epochs or 30; a.bs = a.bs or 64
        run_b2d(a)
    else:
        a.epochs = a.epochs or 40; a.bs = a.bs or 256
        run_navsim(a)


if __name__ == '__main__':
    main()
