#!/usr/bin/env python3
"""R0 — the ego-only rung of the relational ablation ladder (R0 -> R1 -> R2).

Definition (fixed, = E0-g with LSE removed, the +0.4677 arm of the earlier study):
  the scene's UNIQUE 2 Hz ego sequence -> per-timestep MLP -> distribution pool
  [mean, max, softmin, std] -> MLP -> scalar b_tilde.
  NO logsumexp branch: LSE over T near-equal values is ~ log T + value, so it smuggles
  the route length into the readout and confounds with route duration.
  command IS an input.  cmdkin25 is NOT.

Rows are taken from the FROZEN relgraph tensors so R0 reads exactly the rows R1/R2 will
read; only ego / command / item_id are touched, nothing is written back.

B2D windows OVERLAP: stride is 4 of 12 steps (verified here, not assumed), so consecutive
windows share 8 steps. Flattening the window axis would count middle timesteps three times
and the per-window (x, y, psi) are anchor-relative anyway. We rebuild one unique 2 Hz
sequence per route: window 0 contributes all 12 steps, every later window only its last 4,
and headings are chained through the overlap so psi is continuous along the route.
NavSim rows are one window per scored scene, so their 12 steps are already unique.

Usage: r0_ego.py --domain b2d|navsim --gpu G --seed S
"""
import argparse, os, re, sys

import numpy as np

sys.path.insert(0, '/home/jeongtae/SC-IRT')
sys.path.insert(0, '/home/jeongtae/SCIRT/b2d_irt')
sys.path.insert(0, '/data2/jeongtae/relgraph')

import b2d_earlystop as es               # noqa: E402  the ONE B2D checkpoint-selection rule

RG = '/data2/jeongtae/relgraph'
B2D_NPZ = f'{RG}/b2d_relgraph_v2.npz'
NAVSIM_NPZ = f'{RG}/navsim_relgraph_v2.npz'
B2D_MAT = '/home/jeongtae/SC-IRT/data/matrices/b2d_e2e16_response_matrix.csv'
B2D_TYPES = '/home/jeongtae/SC-IRT/data/matrices/b2d_route_types.csv'
NAVSIM_MAT = '/data2/jeongtae/navsim_response_matric/navsim_navtest_pdms_v1_matrix.csv'
NAVSIM_LOGS = '/data2/jeongtae/navsim_interact/navtest_tensors_v2.npz'
# section 20.2 S2: nuPlan val14 ego/cmd windows (2 per scenario, widx 0/1; KEYS.md ego layout).
# 'routed' = the upstream tensor (ego future = synthetic centerline ego, S0 caveat);
# 'logged' = same windows rebuilt from the logged ego (transfer/nuplan/build_val14_ego_logged.py).
NUPLAN_EGO = {'routed': '/data2/jeongtae/navsim_interact/val14_tensors.npz',
              'logged': f'{RG}/transfer/nuplan/val14_ego_logged.npz'}
# section 20.2 S3 (same-domain OOF): the validated S1 graph tensor, the k11 response matrix
# (read ONLY by the same-domain OOF loops — the transfer arms never touch it), and the S0
# scenario -> (type, log) map for the log-disjoint folds and cluster bootstraps.
NUPLAN_NPZ = f'{RG}/nuplan_val14_relgraph_v2.npz'
NUPLAN_K11 = '/home/jeongtae/SCIRT/SC-IRT/result/nuplan_val14_k11_response_matrix.csv'
NUPLAN_META = f'{RG}/transfer/nuplan/val14_584_scenario_log_map.csv'
NUPLAN_OOF_DIR = f'{RG}/transfer/nuplan'

# ego channel symbols (KEYS.md: ego (N,T,6) [dx, dy, cos dpsi, sin dpsi, speed, is_future])
EGO_DX, EGO_DY, EGO_COS, EGO_SIN, EGO_SPEED, EGO_ISFUT = 0, 1, 2, 3, 4, 5
DT = 0.5                       # 2 Hz
STRIDE = 4                     # B2D window stride in steps; asserted below
NIN = 9                        # [speed, acc, yawrate, |acc|, |yawrate|, cmd(4)]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def step_feats(sp, psi, cmd):
    """(L,) speed, (L,) heading, (L,4) command  ->  (L, NIN)."""
    n = len(sp)
    acc = np.gradient(sp, DT) if n > 1 else np.zeros(n, np.float32)
    yr = np.r_[0.0, wrap(np.diff(psi)) / DT].astype(np.float32) if n > 1 else np.zeros(n, np.float32)
    return np.c_[sp, acc, yr, np.abs(acc), np.abs(yr), cmd].astype(np.float32)


# ── model ────────────────────────────────────────────────────────────────────
def build(torch, nn, d=64, tau=0.5):
    class SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(NIN, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU())
            self.head = nn.Sequential(nn.LayerNorm(d * 4), nn.Linear(d * 4, d), nn.SiLU(),
                                      nn.Linear(d, 1))
            self.tau = tau

        def forward(self, x, mask):
            z = self.phi(x)
            m = mask[..., None].float()
            n = m.sum(1).clamp(min=1)
            mean = (z * m).sum(1) / n
            mx = z.masked_fill(~mask[..., None], -1e9).max(1).values
            neg = torch.where(mask[..., None], -z / self.tau, torch.full_like(z, -float('inf')))
            smin = -self.tau * torch.logsumexp(neg, 1)
            var = ((z - mean[:, None]) ** 2 * m).sum(1) / n
            return self.head(torch.cat([mean, mx, smin, var.sqrt()], -1)).squeeze(-1)
    return SeqNet()


# ── B2D ──────────────────────────────────────────────────────────────────────
def build_b2d():
    """-> X (R,T_MAX,NIN), MK (R,T_MAX), routes, types, Y (J,R) fail=1/NaN, static (R,)"""
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

    routes = sorted(set(rid))
    # stride is a fact about the tensor, not a belief: verify on every consecutive pair
    ov_sp, ov_psi, npair = 0.0, 0.0, 0
    for r in routes:
        idx = np.where(rid == r)[0][np.argsort(widx[rid == r])]
        for a, b in zip(idx[:-1], idx[1:]):
            ov_sp = max(ov_sp, np.abs(sp_w[a, STRIDE:] - sp_w[b, :12 - STRIDE]).max())
            # heading agrees only up to each window's own anchor offset
            da = psi_w[a, STRIDE:] - psi_w[a, STRIDE]
            db = psi_w[b, :12 - STRIDE] - psi_w[b, 0]
            ov_psi = max(ov_psi, np.abs(wrap(da - db)).max())
            npair += 1
    print(f'[b2d] overlap check on {npair} consecutive window pairs (stride {STRIDE}): '
          f'max |dspeed| {ov_sp:.4f} m/s, max |dpsi| {ov_psi:.5f} rad', flush=True)
    assert ov_sp < 0.05 and ov_psi < 0.05, 'window stride is not 4 — do not build the sequence'

    seqs, lens, static = [], [], []
    for r in routes:
        sel = np.where(rid == r)[0]
        idx = sel[np.argsort(widx[sel])]
        W = len(idx)
        L = STRIDE * (W - 1) + 12
        sp = np.zeros(L, np.float32); psi = np.zeros(L, np.float32)
        cm = np.zeros((L, 4), np.float32); got = np.zeros(L, bool)
        for w, row in enumerate(idx):
            g0 = STRIDE * w
            off = 0.0 if w == 0 else float(psi[g0 + 3])      # anchor of window w is step g0+3
            for t in range(12):
                g = g0 + t
                if got[g]:
                    continue
                sp[g] = sp_w[row, t]; psi[g] = off + psi_w[row, t]
                cm[g] = cmd[row]; got[g] = True
        assert got.all()
        seqs.append(step_feats(sp, psi, cm)); lens.append(L)
        static.append(bool((sp_w[idx].max(1) < 0.5).any()))
    T_MAX = max(lens)
    X = np.zeros((len(routes), T_MAX, NIN), np.float32)
    MK = np.zeros((len(routes), T_MAX), bool)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s; MK[i, :len(s)] = True
    lens = np.array(lens)
    print(f'[b2d] {len(routes)} routes, unique 2 Hz steps p50 {int(np.median(lens))} '
          f'p90 {int(np.percentile(lens,90))} max {lens.max()} '
          f'(sum {lens.sum()} vs {12*len(rid)} if windows were flattened)', flush=True)

    rows = list(csv.reader(open(B2D_MAT)))
    rids_m = rows[0][1:]
    body = [r for r in rows[1:] if r[0] != 'PDM-Lite']
    Yf = np.full((len(body), len(rids_m)), np.nan)
    for pi, row in enumerate(body):
        for j in range(len(rids_m)):
            if row[1 + j] != '':
                Yf[pi, j] = 1.0 - float(row[1 + j])           # matrix holds success -> fail=1
    col = {r: j for j, r in enumerate(rids_m)}
    tmap = dict(csv.reader(open(B2D_TYPES)))
    assert all(r in col and r in tmap for r in routes)
    Y = Yf[:, [col[r] for r in routes]]
    types = np.array([tmap[r] for r in routes])
    print(f'[b2d] response panel {Y.shape[0]} planners x {Y.shape[1]} routes, '
          f'missing cells {int(np.isnan(Y).sum())}, {len(set(types))} scenario types, '
          f'static routes {int(np.sum(static))}', flush=True)
    return X, MK, np.array(routes), types, Y, np.array(static)


def run_b2d(a):
    import torch, torch.nn as nn
    from numpy.polynomial.hermite_e import hermegauss
    from scipy.stats import spearmanr
    from scirt.encoder import rasch
    from b2d_splits import unified_split, R_DRAWS

    dev = 'cuda'
    X, MK, routes, types, Y, static = build_b2d()
    R, J = len(routes), Y.shape[0]
    fail = np.nanmean(Y, 0)
    _, b_ref = rasch(Y, it=800)                                # full-panel reference, eval only
    print(f'[b2d] rho(observed failure rate, full-panel Rasch b) '
          f'{spearmanr(fail, b_ref).correlation:+.4f}', flush=True)

    gxn, gwn = hermegauss(15); gwn = gwn / gwn.sum()
    gx = torch.tensor(gxn, dtype=torch.float32, device=dev)
    lgw = torch.log(torch.tensor(gwn, dtype=torch.float32, device=dev))
    utypes = sorted(set(types))
    pred = np.full((R_DRAWS, R), np.nan)
    per_draw = []
    led = es.Ledger(R_DRAWS, a.epochs, a.early_stop)           # sigma / e* / curve / leak counts

    for draw in range(min(a.draws, R_DRAWS)):
        hp, ht = unified_split(draw, utypes, J)
        keepJ = np.array([j for j in range(J) if j not in hp])
        te = np.isin(types, list(ht)); tr = ~te
        plan = es.TwoStage(draw, types, tr, a.epochs, a.early_stop)
        guard = es.HeldOutGuard(np.where(te)[0], None, hp, keepJ)
        guard.selftest()
        # ONE guard for BOTH stages: opened here, closed only after stage 2's last epoch.
        for stg in plan.stages():
            trn = stg.train                     # stage 1: A_train.  stage 2: the FULL A block.
            torch.manual_seed(a.seed); np.random.seed(a.seed)  # same seed -> same fresh init
            th_f, _ = rasch(Y[keepJ][:, trn])   # stage 1 theta_inner / stage 2 theta_outer
            if a.proper_init:
                torch.manual_seed(a.seed)       # re-seed AFTER rasch: rasch reseeds the torch RNG
                                                # internally, so without this every seed builds the
                                                # SAME initial weights.  Stage 1 and stage 2 still
                                                # start from identical weights within a run.
            plan.note_theta(stg, th_f)
            mu = X[trn][MK[trn]].mean(0); sd = X[trn][MK[trn]].std(0) + 1e-6
            Xn = ((X - mu) / sd) * MK[..., None]
            m = build(torch, nn, a.d).to(dev)
            if draw == 0:
                print(f'  [init] draw 0 s{stg.no} seed {a.seed} '
                      f'proper_init={a.proper_init} weight-hash {es.init_hash(m)}',
                      flush=True)
            ls = torch.tensor(-0.5, device=dev, requires_grad=True)
            opt = torch.optim.AdamW(list(m.parameters()) + [ls], lr=1e-3, weight_decay=0.1)
            THE = torch.tensor(th_f, dtype=torch.float32, device=dev)
            Yk = es.erase_heldout(Y[keepJ], te) if a.early_stop else Y[keepJ]
            Yd = torch.tensor(np.nan_to_num(Yk), dtype=torch.float32, device=dev)
            Md = torch.tensor((~np.isnan(Yk)).astype(np.float32), device=dev)
            Xd = torch.tensor(Xn, device=dev); Md_ = torch.tensor(MK, device=dev)
            idx = guard.routes(np.where(trn)[0], f'{stg.name} train columns')
            iv_c = guard.routes(np.where(stg.iv)[0], f'{stg.name} inner-val columns')
            if a.early_stop:
                es.assert_masked(torch, Yd, Md, te, dev)
            print(plan.head(stg), flush=True)

            def fwd(sel):
                s = torch.tensor(sel, device=dev)
                return m(Xd[s], Md_[s])

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
            pred[draw, ii] = fwd(ii).cpu().numpy()
        led.record(draw, plan, guard)
        print(guard.leak_line(draw), flush=True)
        P = pred[draw][te]
        per_draw.append(spearmanr(P, fail[te]).correlation)
        print(f'  [b2d draw {draw}] held-out {te.sum()} routes  '
              f'rho_scene {per_draw[-1]:+.4f}', flush=True)

    out = es.out_path(RG, f'r0_b2d_s{a.seed}.npz', a.early_stop, a.proper_init)
    np.savez(out, pred=pred, routes=routes, types=types, Y=Y, fail=fail,
             b_ref=b_ref, static=static, **led.fields())
    P, F, B, S = [], [], [], []
    for dd in range(R_DRAWS):
        k = np.isfinite(pred[dd])
        P.append(pred[dd][k]); F.append(fail[k]); B.append(b_ref[k]); S.append(static[k])
    P, F, B, S = map(np.concatenate, (P, F, B, S))
    print(f'R0_B2D seed={a.seed}  rho_scene {spearmanr(P, F).correlation:+.4f}  '
          f'rho_ref {spearmanr(P, B).correlation:+.4f}  '
          f'static {spearmanr(P[S], F[S]).correlation:+.4f}  '
          f'non-static {spearmanr(P[~S], F[~S]).correlation:+.4f}  '
          f'(pooled {len(P)} held-out cells)  '
          f'per-draw mean {np.mean(per_draw):+.4f} +/- {np.std(per_draw, ddof=1):.4f}', flush=True)
    print(f'WROTE {out}', flush=True)


# ── NavSim ───────────────────────────────────────────────────────────────────
def build_navsim(target='pdms'):
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

    t = np.load(NAVSIM_LOGS, allow_pickle=True)
    log_by_tok = {str(n): str(l) for n, l in zip(t['names'], t['log'])}   # join by token, never index
    assert set(toks) <= set(log_by_tok)
    logs = np.array([log_by_tok[x] for x in toks])

    M = pd.read_csv(NAVSIM_MAT, index_col=0)
    tok2col = {str(c): i for i, c in enumerate(M.columns)}
    assert set(toks) <= set(tok2col)
    V = M.values.astype(float)[:, [tok2col[x] for x in toks]]
    Y = (V < 0.5).astype(np.float32)                            # fail=1, PDMS<0.5 (repo convention)
    Y[np.isnan(V)] = np.nan
    if target == 'nc':                                          # section 19.2 NC ceiling
        z = np.load(f'{RG}/transfer/zs/navsim_nc_bref.npz', allow_pickle=True)
        row = {str(t): i for i, t in enumerate(z['tokens'])}
        assert set(toks) <= set(row)
        Y = z['Y'][:, [row[x] for x in toks]].astype(np.float32)   # fail=1 iff NC<1, no NaN
        print(f'[navsim] --target nc: response panel swapped to NC<1 from transfer/zs/navsim_nc_bref.npz', flush=True)
    print(f'[navsim] {len(toks)} scenes, {len(set(logs))} logs, panel {Y.shape[0]} planners, '
          f'missing cells {int(np.isnan(Y).sum())}, overall failure rate {np.nanmean(Y):.4f}',
          flush=True)
    print(f'[navsim] one window per scored scene -> 12 unique 2 Hz steps each, no overlap to undo',
          flush=True)
    return X, MK, toks, logs, Y


def run_navsim(a):
    import math
    import torch, torch.nn as nn
    from scipy.stats import spearmanr
    from scirt.encoder import rasch

    dev = 'cuda'
    X, MK, toks, logs, Y = build_navsim(getattr(a, 'target', 'pdms'))
    N = len(toks)
    fail = np.nanmean(Y, 0)
    _, b_ref = rasch(Y, it=800)
    print(f'[navsim] rho(observed failure rate, full-panel Rasch b) '
          f'{spearmanr(fail, b_ref).correlation:+.4f}', flush=True)

    ulogs = np.array(sorted(set(logs)))
    chunks = np.array_split(np.random.default_rng(0).permutation(len(ulogs)), a.nfolds)
    pred = np.full(N, np.nan)
    per_fold = []
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
        th_tr, _ = rasch(Y[:, tr])
        THE = torch.tensor(th_tr, dtype=torch.float32, device=dev)
        mu = X[tr][MK[tr]].mean(0); sd = X[tr][MK[tr]].std(0) + 1e-6
        Xd = torch.tensor(((X - mu) / sd) * MK[..., None], device=dev)
        Mk = torch.tensor(MK, device=dev)
        m = build(torch, nn, a.d).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
        idx_tr = np.where(tr)[0]
        steps = a.epochs * max(len(idx_tr) // a.bs, 1)
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(s / 200 + 1e-2, 0.5 * (1 + math.cos(math.pi * s / max(steps, 1)))))
        shuf = np.random.default_rng(a.seed)

        def fwd(sel):
            s = torch.tensor(sel, device=dev)
            return m(Xd[s], Mk[s])

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
        pred[te] = best[1]
        per_fold.append(spearmanr(best[1], fail[te]).correlation)
        print(f'  [navsim fold {fold}] train {tr.sum()} innerval {iv.sum()} test {te.sum()} '
              f'| best ep {best[2]} nll {best[0]:.4f} | rho_scene '
              f'{spearmanr(best[1], fail[te]).correlation:+.4f} '
              f'rho_ref {spearmanr(best[1], b_ref[te]).correlation:+.4f}', flush=True)

    out = f'{RG}/r0_navsim_s{a.seed}.npz' if getattr(a, 'target', 'pdms') == 'pdms' \
        else f'{RG}/transfer/zs/r0_navsim_nc_s{a.seed}.npz'
    np.savez(out, pred=pred, tokens=toks, logs=logs, fail=fail, b_ref=b_ref, Y=Y)
    k = np.isfinite(pred)
    print(f'R0_NAVSIM seed={a.seed}  rho_scene {spearmanr(pred[k], fail[k]).correlation:+.4f}  '
          f'rho_ref {spearmanr(pred[k], b_ref[k]).correlation:+.4f}  '
          f'(out-of-fold {k.sum()}/{N})  '
          f'per-fold mean {np.mean(per_fold):+.4f} +/- {np.std(per_fold, ddof=1):.4f}', flush=True)
    print(f'WROTE {out}', flush=True)


# ── nuPlan val14 (section 20.2 S2): TARGET only — ego/cmd windows, no responses read here ──
def build_nuplan_target(which='routed'):
    """-> Xw (N,12,NIN) window_ego_feats rows, MKw (N,12) all-True, item ids '<token>_<widx>',
    tokens (N,), widx (N,).  Sorted by (token, widx) as stored; 2 windows per scenario."""
    import transfer_common as tc
    d = np.load(NUPLAN_EGO[which], allow_pickle=True)
    toks = np.array([str(x) for x in d['token']]); widx = d['widx'].astype(int)
    ids = np.array([f'{t}_{w}' for t, w in zip(toks, widx)])
    assert len(set(ids)) == len(ids)
    ego = d['ego'].astype(np.float32); cmd = d['cmd'].astype(np.float32)
    assert ego.shape[1:] == (12, 6) and cmd.shape[1] == 4 and np.allclose(cmd.sum(1), 1)
    assert np.allclose(ego[:, 3, :2], 0) and np.allclose(ego[:, 3, EGO_COS], 1)   # anchor at step 3
    Xw = tc.window_ego_feats(ego, cmd)
    print(f'[nuplan] target ego source {which!r}: {len(ids)} windows / {len(set(toks))} scenarios, '
          f'widx counts {np.bincount(widx).tolist()}, windows with constant future speed '
          f'{int((ego[:, 4:, EGO_SPEED].std(1) < 1e-3).sum())}', flush=True)
    return Xw, np.ones(Xw.shape[:2], bool), ids, toks, widx


# ── nuPlan val14 same-domain OOF (section 20.2 S3) ───────────────────────────
def build_nuplan(which='logged'):
    """S3 loader: item = WINDOW (1,168 = 584 scenarios x widx {0,1}; log-disjoint folds keep both
    windows of a scenario on the same side).  X = per-window ego features from the chosen ego
    tensor (default the LOGGED-ego rebuild — the shipped/routed ego future is synthetic, S0
    caveat), row order asserted equal to the frozen S1 graph tensor.  Y = k11 fail (score <= 0.5,
    NaN kept) with each SCENARIO column repeated for both its windows.  Groups: log (folds) and
    scenario type (cluster bootstrap), both from the S0 map.
    -> X, MK, ids, toks, widx, logs, types, Y"""
    import pandas as pd
    d = np.load(NUPLAN_NPZ, allow_pickle=True)
    gids = np.array([str(x) for x in d['item_id']])
    Xw, MKw, ids, toks, widx = build_nuplan_target(which)
    assert np.array_equal(gids, ids), 'graph rows must be the ego-tensor rows'
    M = pd.read_csv(NUPLAN_K11, index_col=0)
    V = M.values.astype(float)
    Yc = (V <= 0.5).astype(np.float32)                          # fail = 1, k11 rule score <= 0.5
    Yc[np.isnan(V)] = np.nan
    col = {str(c): i for i, c in enumerate(M.columns)}
    meta = pd.read_csv(NUPLAN_META, dtype=str).set_index('scenario')
    assert set(toks) <= set(col) and set(toks) <= set(meta.index)
    Y = Yc[:, [col[t] for t in toks]]
    logs = meta.loc[toks, 'log_name'].values.astype(str)
    types = meta.loc[toks, 'scenario_type'].values.astype(str)
    print(f'[nuplan] OOF item = window: {len(ids)} windows / {len(set(toks))} scenarios, '
          f'{len(set(logs))} logs, {len(set(types))} types, panel {Y.shape[0]} planners, '
          f'missing cells {int(np.isnan(Y).sum())} (scenario columns repeated per window), '
          f'overall failure rate {np.nanmean(Y):.4f}, ego tensor {which!r}', flush=True)
    return Xw, MKw, ids, toks, widx, logs, types, Y


def nuplan_scn_readout(pred_w, toks, widx, mask=None):
    """Window predictions -> per-scenario (w0, w1, mean) over the scenarios whose BOTH windows
    are inside `mask` (all of them, when mask is None).  -> scn tokens, (3, S) readouts."""
    keep = np.ones(len(toks), bool) if mask is None else mask
    w0 = {t: i for i, t in zip(np.where(keep & (widx == 0))[0], toks[keep & (widx == 0)])}
    w1 = {t: i for i, t in zip(np.where(keep & (widx == 1))[0], toks[keep & (widx == 1)])}
    scn = np.array(sorted(set(w0) & set(w1)))
    i0 = np.array([w0[t] for t in scn]); i1 = np.array([w1[t] for t in scn])
    return scn, np.stack([pred_w[i0], pred_w[i1], 0.5 * (pred_w[i0] + pred_w[i1])])


def run_nuplan(a):
    """S3 R0: the NavSim OOF loop verbatim (BCE, wd 0.05, warmup+cosine, inner-val checkpoint)
    on nuPlan val14 windows, folds log-disjoint over the 218 logs.  Writes ONLY to
    transfer/nuplan/, never beside the frozen b2d/navsim outputs."""
    import math
    import torch, torch.nn as nn
    from scipy.stats import spearmanr
    from scirt.encoder import rasch

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, MK, ids, toks, widx, logs, types, Y = build_nuplan(a.nuplan_ego)
    N = len(ids)
    fail = np.nanmean(Y, 0)
    _, b_ref = rasch(Y, it=800)                   # window-duplicated panel; scoring joins by token
    print(f'[nuplan] rho(observed failure rate, full-panel Rasch b) '
          f'{spearmanr(fail, b_ref).correlation:+.4f}', flush=True)

    ulogs = np.array(sorted(set(logs)))
    chunks = np.array_split(np.random.default_rng(0).permutation(len(ulogs)), a.nfolds)
    pred = np.full(N, np.nan)
    fold_of = np.full(N, -1, np.int64)
    per_fold = []
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
        th_tr, _ = rasch(Y[:, tr])
        THE = torch.tensor(th_tr, dtype=torch.float32, device=dev)
        mu = X[tr][MK[tr]].mean(0); sd = X[tr][MK[tr]].std(0) + 1e-6
        Xd = torch.tensor(((X - mu) / sd) * MK[..., None], device=dev)
        Mk = torch.tensor(MK, device=dev)
        m = build(torch, nn, a.d).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
        idx_tr = np.where(tr)[0]
        steps = a.epochs * max(len(idx_tr) // a.bs, 1)
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(s / 200 + 1e-2, 0.5 * (1 + math.cos(math.pi * s / max(steps, 1)))))
        shuf = np.random.default_rng(a.seed)

        def fwd(sel):
            s = torch.tensor(sel, device=dev)
            return m(Xd[s], Mk[s])

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
        pred[te] = best[1]
        fold_of[te] = fold
        scn, ro = nuplan_scn_readout(pred, toks, widx, te)
        b_s = {t: b for t, b in zip(toks, b_ref)}
        bt = np.array([b_s[t] for t in scn])
        per_fold.append(float(spearmanr(ro[0], bt).correlation))
        print(f'  [nuplan fold {fold}] train {tr.sum()} innerval {iv.sum()} test {te.sum()} '
              f'| best ep {best[2]} nll {best[0]:.4f} | rho_scene(w0) {per_fold[-1]:+.4f} '
              f'rho_win {spearmanr(best[1], b_ref[te]).correlation:+.4f}', flush=True)

    os.makedirs(NUPLAN_OOF_DIR, exist_ok=True)
    out = f'{NUPLAN_OOF_DIR}/r0_nuplan_oof_s{a.seed}.npz'
    np.savez(out, pred=pred, item_id=ids, tokens=toks, widx=widx, logs=logs, types=types,
             fold=fold_of, fail=fail, b_ref=b_ref, Y=Y, per_fold=np.array(per_fold),
             nuplan_ego=a.nuplan_ego)
    scn, ro = nuplan_scn_readout(pred, toks, widx)
    b_s = {t: b for t, b in zip(toks, b_ref)}
    bt = np.array([b_s[t] for t in scn])
    print(f'R0_NUPLAN seed={a.seed} ego={a.nuplan_ego}  pooled rho_scene w0 '
          f'{spearmanr(ro[0], bt).correlation:+.4f}  mean {spearmanr(ro[2], bt).correlation:+.4f}  '
          f'w1 {spearmanr(ro[1], bt).correlation:+.4f}  per-fold(w0) {np.mean(per_fold):+.4f} '
          f'+/- {np.std(per_fold, ddof=1):.4f}', flush=True)
    print(f'WROTE {out}', flush=True)


# ── section 18: ONE model on ALL source items, forwarded on the other domain ──────────────
def run_full(a):
    """--full-train --transfer-to DOM [--subsample-logs N].  Additive: the US/UP loops above are
    untouched.  The stage-1/stage-2 schedule (es.TwoStage), e* selection and the per-domain
    recipe come from transfer_common; this function supplies R0's tensors and forward only.
    The target tensor is loaded AFTER training; target responses are never read or saved."""
    import torch, torch.nn as nn
    from scipy.stats import spearmanr
    from scirt.encoder import rasch
    import transfer_common as tc

    dev, devname = tc.device(torch)
    src, tgt = a.domain, a.transfer_to
    tag = f'r0 {tc.src_tag(a)}->{tgt} s{a.seed}'
    sub_logs = np.array([])
    if src == 'b2d':
        X, MK, items, grp, Y, _ = build_b2d()
    else:
        X, MK, items, grp, Y = build_navsim()
        if a.subsample_logs:
            keep, sub_logs = tc.subsample_logs(grp, a.subsample_logs, a.subsample_seed)
            X, MK, items, grp, Y = X[keep], MK[keep], items[keep], grp[keep], Y[:, keep]
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
        th_f, _ = rasch(Y[:, trn])                # theta on this stage's SOURCE columns
        torch.manual_seed(a.seed)                 # proper-init (section 12): re-seed AFTER rasch
        plan.note_theta(stg, th_f)
        mu = X[trn][MK[trn]].mean(0); sd = X[trn][MK[trn]].std(0) + 1e-6
        Xd = torch.tensor(((X - mu) / sd) * MK[..., None], device=dev)
        m = build(torch, nn, a.d).to(dev)
        print(f'  [init] s{stg.no} seed {a.seed} proper_init=True weight-hash {es.init_hash(m)}',
              flush=True)
        THE = torch.tensor(th_f, dtype=torch.float32, device=dev)
        print(plan.head(stg), flush=True)

        def fwd(sel):
            s = torch.tensor(sel, device=dev)
            return m(Xd[s], Mk[s])
        tc.fit_stage(src, torch, nn, m, fwd, stg, plan, Yd, Md, THE, a.bs, a.seed, dev, tag,
                     warmup=a.warmup_steps)
    print(plan.line(0), flush=True)
    pred_src = tc.predict(torch, m, fwd, N)
    rho_src = float(spearmanr(pred_src, b_src).correlation)
    print(f'  [{tag}] e* {plan.best_ep}, stage 2 {plan.stage2_epochs} epochs on {N} items | '
          f'source in-sample rho_ref {rho_src:+.4f} rho_fail '
          f'{spearmanr(pred_src, fail).correlation:+.4f}', flush=True)

    # ── target: loaded only now; normalised with the SOURCE mu/sd (stage 2 = all source items) ──
    out = dict(arm='r0', src=tc.src_tag(a), tgt=tgt, seed=a.seed, device=devname, bs=a.bs,
               epochs=a.epochs, pred_src=pred_src, src_item_id=items, src_groups=grp,
               src_b_ref=b_src, rho_src_insample=rho_src, x_mu=mu, x_sd=sd, sub_logs=sub_logs,
               subsample_seed=a.subsample_seed, warmup_steps=a.warmup_steps,
               **tc.plan_fields(plan))

    def fwd_t(Xt, MKt):
        Xd_t = torch.tensor(((Xt - mu) / sd) * MKt[..., None], device=dev)
        Mk_t = torch.tensor(MKt, device=dev)
        return tc.predict(torch, m, lambda sel: m(Xd_t[torch.tensor(sel, device=dev)],
                                                  Mk_t[torch.tensor(sel, device=dev)]), len(Xt))
    if tgt == 'navsim':
        Xt, MKt, toks, logs, _ = build_navsim()   # its Y is discarded, never saved
        out.update(pred=fwd_t(Xt, MKt), item_id=toks, tgt_groups=logs)
    elif tgt == 'nuplan':                         # section 20.2 S2: no nuPlan response is read
        Xw, MKw, wids, wtok, widx = build_nuplan_target(a.nuplan_ego)
        pred_w = fwd_t(Xw, MKw)
        scn = np.array(sorted(set(wtok)))
        w0 = {t: i for i, t in zip(np.where(widx == 0)[0], wtok[widx == 0])}
        w1 = {t: i for i, t in zip(np.where(widx == 1)[0], wtok[widx == 1])}
        assert set(w0) == set(w1) == set(scn), 'every scenario must have both windows'
        i0 = np.array([w0[t] for t in scn]); i1 = np.array([w1[t] for t in scn])
        # widx 0 (anchor t0+1.5 s, steps span t0+0.0..5.5 s) is the window nearest the scored
        # scenario's initial frame -> primary; the mean of both windows is the secondary readout
        out.update(pred=pred_w, item_id=wids, tgt_groups=wtok, widx=widx, nuplan_ego=a.nuplan_ego,
                   scn_token=scn, pred_scn_w0=pred_w[i0], pred_scn_w1=pred_w[i1],
                   pred_scn_mean=0.5 * (pred_w[i0] + pred_w[i1]))
    else:
        d = np.load(B2D_NPZ, allow_pickle=True)
        wids = np.array([str(x) for x in d['item_id']])
        Xw = tc.window_ego_feats(d['ego'].astype(np.float32), d['command'].astype(np.float32))
        pred_w = fwd_t(Xw, np.ones(Xw.shape[:2], bool))          # window-level, NavSim-style rows
        Xr, MKr, routes, types, _, _ = build_b2d()               # R0's OWN B2D item: one unique
        W = tc.b2d_windows_by_route(wids, routes)                # 2 Hz sequence per route
        out.update(pred=pred_w, item_id=wids, pred_route=fwd_t(Xr, MKr),
                   pred_route_winmean=tc.route_mean(pred_w, W, len(routes)),
                   routes=routes, tgt_groups=types)
    print(f'  [{tag}] target {tgt}: {len(out["pred"])} window preds'
          + (f', {len(out["pred_route"])} route preds' if 'pred_route' in out else '')
          + (f', {len(out["scn_token"])} scenario preds (w0 / w1 / mean)' if 'scn_token' in out else ''),
          flush=True)
    tc.save(tc.out_path('r0', a), **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, choices=['b2d', 'navsim', 'nuplan'])
    ap.add_argument('--gpu', default='2')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--bs', type=int, default=None)
    ap.add_argument('--nfolds', type=int, default=5)
    ap.add_argument('--draws', type=int, default=99)
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
    ap.add_argument('--proper-init', action='store_true',
                    help='B2D two-stage only: re-seed the torch RNG AFTER the rasch call so '
                         'the model initialisation actually varies with --seed. Without this '
                         'flag scirt.encoder.rasch reseeds torch internally and every seed '
                         'gets the identical initial weights, so the seeds differ only in '
                         'minibatch order. Writes to RG/es_pinit/, never over RG/es/. '
                         'Off by default: the existing es/ results stay bit-reproducible.')
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
                         'target, route-level) predictions plus the source in-sample prediction.')
    ap.add_argument('--subsample-logs', type=int, default=None,
                    help='NavSim source only: keep whole logs (fixed seed) until ~N windows '
                         '(section 18.2 T3, N=2656). The kept logs are recorded in the output.')
    ap.add_argument('--subsample-seed', type=int, default=None,
                    help='section 18.6 (a): RNG seed of the --subsample-logs log draw. Default '
                         '3000 (= section 18 T3, file name unchanged); any other value appends '
                         '_sub<seed> to the output name.')
    ap.add_argument('--warmup-steps', type=int, default=None,
                    help='section 18.6 (b)/(c): NavSim-recipe LR warmup length in optimizer '
                         'steps. Default 200 (= the frozen recipe, file name unchanged); any '
                         'other value appends _wu<n> to the output name.')
    ap.add_argument('--nuplan-ego', default='routed', choices=['routed', 'logged'],
                    help='section 20.2 S2, --transfer-to nuplan only: which val14 ego/cmd tensor '
                         'to forward. routed = val14_tensors.npz as shipped (synthetic constant-'
                         'speed ego future; file name unchanged); logged = the logged-ego rebuild '
                         'transfer/nuplan/val14_ego_logged.npz (appends _logged).')
    ap.add_argument('--target', default='pdms', choices=['pdms', 'nc'],
                    help='section 19.2: NavSim OOF only. nc = swap the response panel to the '
                         'collision-only construct (fail = no_at_fault_collisions < 1) read from '
                         'transfer/zs/navsim_nc_bref.npz; b_ref/theta are refitted on that panel '
                         'inside the run. Output goes to transfer/zs/r0_navsim_nc_s<seed>.npz. '
                         'Default pdms = the frozen recipe, file name unchanged.')
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    assert a.target == 'pdms' or (a.domain == 'navsim' and not a.full_train and not a.transfer_to), \
        '--target nc is the NavSim same-domain OOF only'
    if a.domain == 'nuplan':                       # section 20.2 S3: same-domain OOF only
        assert not (a.full_train or a.transfer_to or a.subsample_logs or a.early_stop
                    or a.proper_init), '--domain nuplan is the S3 same-domain OOF only'
        a.epochs = a.epochs or 40; a.bs = a.bs or 256   # frozen NavSim-style recipe
        run_nuplan(a)
        return
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
    assert not (a.proper_init and not a.early_stop), \
        '--proper-init is only defined for the two-stage --early-stop protocol'
    if a.domain == 'b2d':
        a.epochs = a.epochs or 30; a.bs = a.bs or 64
        run_b2d(a)
    else:
        a.epochs = a.epochs or 40; a.bs = a.bs or 256
        run_navsim(a)


if __name__ == '__main__':
    main()
