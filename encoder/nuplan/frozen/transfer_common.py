#!/usr/bin/env python3
"""Shared machinery for the --full-train / --transfer-to / --subsample-logs branches
(PROTOCOL_R.md section 18; recipe sections 5, 12, 12.1).

Everything that is NOT arm-specific lives here so r0_ego / r2_graph / hrel cannot drift:
the inner split, the log subsample, the two-stage plan (es.TwoStage re-used, not re-typed),
the per-source-domain training recipe, and the output location.  The arm supplies only the
model and its forward.

Source-only discipline: nothing in this module reads a response matrix; the arm passes the
SOURCE Y in and loads the target tensor only after plan.stages() is exhausted."""
import os

import numpy as np

import b2d_earlystop as es

RG = '/data2/jeongtae/relgraph'
TRANSFER_DIR = f'{RG}/transfer'
B2D_IV_TYPES = 8          # stage-1 inner-val: 8 of the 44 types (= the outer held-out size H_S)
NAVSIM_IV_FOLDS = 5       # stage-1 inner-val: first chunk of a 5-way log split (28 of 136 logs)
SUBSAMPLE_SEED = 3000     # --subsample-logs: fixed, independent of the model seed and the arm
WARMUP_STEPS = 200        # NavSim recipe LR warmup (optimizer steps); section 18.6 varies it


# ── device ───────────────────────────────────────────────────────────────────
def device(torch):
    """cuda when a device is visible, else cpu with <= 16 threads.  Logged, never silent."""
    if torch.cuda.is_available():
        dev = 'cuda'
        what = (f'cuda (CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r}, '
                f'{torch.cuda.get_device_name(0)})')
    else:
        torch.set_num_threads(min(16, os.cpu_count() or 1))
        dev = 'cpu'
        what = f'cpu ({torch.get_num_threads()} threads)'
    print(f'[transfer] device {what}', flush=True)
    return dev, what


# ── splits ───────────────────────────────────────────────────────────────────
def inner_split(domain, groups):
    """Stage-1 inner-val block, grouped: whole scenario types (B2D) / whole logs (NavSim).
    Seeded by es.IV_RNG_BASE only, so every arm and every model seed selects e* on the
    identical block (the b2d_earlystop rule).  -> (itr, iv) boolean item masks."""
    ug = np.array(sorted(set(groups)))
    rng = np.random.default_rng(es.IV_RNG_BASE)
    if domain == 'b2d':
        pick = rng.choice(len(ug), B2D_IV_TYPES, replace=False)
    else:
        pick = np.array_split(rng.permutation(len(ug)), NAVSIM_IV_FOLDS)[0]
    iv = np.isin(groups, ug[pick])
    itr = ~iv
    assert set(groups[itr]).isdisjoint(set(groups[iv])), 'a group straddles the inner split'
    assert itr.sum() and iv.sum()
    print(f'[transfer] inner split ({domain}): train {int(itr.sum())} items / '
          f'{len(set(groups[itr]))} groups, inner-val {int(iv.sum())} items / {len(pick)} groups, '
          f'fp {es.fingerprint(groups, iv)}', flush=True)
    return itr, iv


def subsample_logs(logs, n_target, seed=SUBSAMPLE_SEED):
    """Keep WHOLE logs in a fixed random order until the window count is as close to n_target
    as a whole-log step allows.  -> (keep mask, kept log names)."""
    ul = np.array(sorted(set(logs)))
    cnt = {l: int((logs == l).sum()) for l in ul}
    order = ul[np.random.default_rng(seed).permutation(len(ul))]
    kept, tot = [], 0
    for l in order:
        if tot + cnt[l] >= n_target:
            if abs(tot + cnt[l] - n_target) < abs(tot - n_target):
                kept.append(l); tot += cnt[l]
            break
        kept.append(l); tot += cnt[l]
    keep = np.isin(logs, kept)
    assert keep.sum() == tot
    print(f'[transfer] --subsample-logs {n_target}: kept {len(kept)} of {len(ul)} logs = '
          f'{tot} windows (seed {seed})', flush=True)
    return keep, np.array(kept)


# ── the plan: es.TwoStage with the inner split supplied from outside ─────────
class FullTrainPlan(es.TwoStage):
    """es.TwoStage driven on ALL source items: stage 1 fits on itr and selects e* by inner-val
    NLL, stage 2 refits on every item for e*+1 epochs under theta refitted on every column.
    Schedule, Selector, theta bookkeeping and the stage-2 > stage-1 asserts are TwoStage's own;
    only the inner block (8-of-44 types / 1-of-5 logs) is substituted."""

    def __init__(self, groups, itr, iv, epochs):
        super().__init__(0, groups, np.ones(len(groups), bool), epochs, True)
        self.itr, self.iv = itr, iv


def plan_fields(plan):
    """What a full-train run records about its selection (Ledger's keys, minus the guard)."""
    return dict(es_curve=plan.picker.curve, es_best_ep=int(plan.best_ep),
                es_stage2_epochs=int(plan.stage2_epochs),
                es_train_items=np.array([plan.n_train[1], plan.n_train[2]]),
                es_theta_delta=float(plan.theta_delta),
                es_theta_hash=np.array([es._vhash(plan.theta[1]), es._vhash(plan.theta[2])]),
                iv_fingerprint=es.fingerprint(plan.types, plan.iv),
                sigma=float(plan.sigma_final), es_sigma_stage1=float(plan.picker.sigma))


# ── the recipe (section 5), one implementation for all arms ─────────────────
def fit_stage(src, torch, nn, m, fwd, stg, plan, Yd, Md, THE, bs, seed, dev, tag, chunk=2048,
              warmup=WARMUP_STEPS):
    """Train `m` for stg.epochs epochs on stg.train under the SOURCE domain's frozen recipe
    (B2D: GH-marginalised cell likelihood with learned sigma, wd 0.1; NavSim: plain cell BCE,
    wd 0.05, warmup+cosine over the frozen 40-epoch horizon) and, when stg.select, score the
    inner-val block after every epoch through plan.update.  Returns the final sigma."""
    import math
    idx = np.where(stg.train)[0]
    iv_c = np.where(stg.iv)[0]
    shuf = np.random.default_rng(seed)
    params = list(m.parameters())
    if src == 'b2d':
        from numpy.polynomial.hermite_e import hermegauss
        gxn, gwn = hermegauss(15); gwn = gwn / gwn.sum()
        gx = torch.tensor(gxn, dtype=torch.float32, device=dev)
        lgw = torch.log(torch.tensor(gwn, dtype=torch.float32, device=dev))
        ls = torch.tensor(-0.5, device=dev, requires_grad=True)
        opt = torch.optim.AdamW(params + [ls], lr=1e-3, weight_decay=0.1)
        sch = None
        starts = lambda: range(0, len(idx), bs)                       # last partial batch kept
    else:
        ls = None
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.05)
        # horizon = the frozen recipe's full epoch count, so stage 2's LR trajectory up to e*
        # is the one stage 1 selected under; stage 2 simply stops after e*+1 epochs
        spe = max(len(idx) // bs, 1)
        steps = plan.epochs * spe
        lr_mult = lambda s: min(s / warmup + 1e-2, 0.5 * (1 + math.cos(math.pi * s / max(steps, 1))))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
        # first step at which the cosine branch is the min = warmup over (section 18.6 bookkeeping)
        wu_end = next((s for s in range(steps)
                       if 0.5 * (1 + math.cos(math.pi * s / max(steps, 1))) <= s / warmup + 1e-2), steps)
        print(f'  [{tag}] s{stg.no} lr schedule: {spe} steps/epoch x {plan.epochs} epochs = {steps} '
              f'steps, warmup {warmup} steps -> ends at step {wu_end} = epoch {wu_end // spe}', flush=True)
        starts = lambda: range(0, len(idx) - bs + 1, bs)              # last partial batch dropped

    def batch_loss(sel):
        bt = fwd(sel)
        s = torch.tensor(sel, device=dev)
        yy = Yd[:, s]; mm = Md[:, s]
        if src == 'b2d':
            sg = torch.exp(ls)
            z = (bt[None, :, None] + sg * gx[None, None, :]) - THE[:, None, None]
            p = torch.sigmoid(z)
            llc = (yy[:, :, None] * torch.log(p + 1e-7)
                   + (1 - yy[:, :, None]) * torch.log(1 - p + 1e-7)) * mm[:, :, None]
            return -torch.logsumexp(llc.sum(0) + lgw[None, :], 1).sum() / mm.sum() \
                + 0.05 * ls.pow(2)
        p = torch.sigmoid(bt[None, :] - THE[:, None])
        return (-(yy * torch.log(p + 1e-7) + (1 - yy) * torch.log(1 - p + 1e-7)) * mm).sum() \
            / mm.sum()

    def iv_nll():
        with torch.no_grad():
            bv = torch.cat([fwd(iv_c[i:i + chunk]) for i in range(0, len(iv_c), chunk)])
            if src == 'b2d':
                return es.cell_nll(torch, bv, iv_c, Yd, Md, THE, gx, lgw, torch.exp(ls).detach())
            s = torch.tensor(iv_c, device=dev)
            p = torch.sigmoid(bv[None, :] - THE[:, None])
            yy = Yd[:, s]; mm = Md[:, s]
            return float((-(yy * torch.log(p + 1e-7)
                            + (1 - yy) * torch.log(1 - p + 1e-7)) * mm).sum() / mm.sum())

    sigma = lambda: float(torch.exp(ls)) if ls is not None else float('nan')
    for ep in range(stg.epochs):
        m.train(); shuf.shuffle(idx); tl = nb = 0
        for i0 in starts():
            loss = batch_loss(idx[i0:i0 + bs])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if sch is not None:
                sch.step()
            tl += float(loss); nb += 1
        if stg.select:
            m.eval(); nll = iv_nll()
            plan.update(stg, ep, tl / max(nb, 1), nll, sigma())
            print(f'  [{tag}] s{stg.no} ep {ep:2d} loss {tl/max(nb,1):.4f} '
                  f'innerval nll {nll:.4f} sigma {sigma():.3f}', flush=True)
        elif ep in (0, 4, stg.epochs - 1):
            print(f'  [{tag}] s{stg.no} ep {ep} loss {tl/max(nb,1):.4f} sigma {sigma():.3f}',
                  flush=True)
    plan.end_stage(stg, sigma())
    return sigma()


def predict(torch, m, fwd, n, chunk=2048):
    m.eval()
    with torch.no_grad():
        return np.concatenate([fwd(np.arange(i, min(i + chunk, n))).cpu().numpy()
                               for i in range(0, n, chunk)])


# ── target-side helpers ──────────────────────────────────────────────────────
def window_ego_feats(ego, cmd):
    """The per-step ego features of r0_ego.build_navsim, applied window-locally
    (= r2_graph.window_ego_feats; re-typed here so r0_ego need not import r2_graph).
    Channels by r0_ego's symbols, never numeric (KEYS.md)."""
    import r0_ego as r0
    sp = ego[:, :, r0.EGO_SPEED]
    psi = np.arctan2(ego[:, :, r0.EGO_SIN], ego[:, :, r0.EGO_COS])
    acc = np.gradient(sp, r0.DT, axis=1)
    yr = np.concatenate([np.zeros((len(sp), 1), np.float32),
                         r0.wrap(np.diff(psi, axis=1)) / r0.DT], 1)
    T = ego.shape[1]
    return np.concatenate([sp[..., None], acc[..., None], yr[..., None],
                           np.abs(acc)[..., None], np.abs(yr)[..., None],
                           np.repeat(cmd[:, None, :], T, 1)], -1).astype(np.float32)


def b2d_windows_by_route(ids, routes):
    """Per-route window rows, sorted by window index (= r2_graph.route_windows)."""
    import re
    mm = [re.match(r'route_(\d+)_(\d+)$', s) for s in ids]
    rid = np.array([m.group(1) for m in mm]); widx = np.array([int(m.group(2)) for m in mm])
    return [np.where(rid == r)[0][np.argsort(widx[rid == r])] for r in routes]


def route_mean(pred_w, W, n_routes):
    return np.array([pred_w[W[i]].mean() for i in range(n_routes)])


# ── output ───────────────────────────────────────────────────────────────────
def src_tag(a):
    return a.domain if not a.subsample_logs else f'{a.domain}sub{a.subsample_logs}'


def variant_suffix(a):
    """'' under the section-18 defaults (files keep their section-18 names); otherwise the
    section-18.6 arm markers, e.g. '_sub3001', '_wu844', '_wu47'."""
    ss = getattr(a, 'subsample_seed', SUBSAMPLE_SEED)
    wu = getattr(a, 'warmup_steps', WARMUP_STEPS)
    return ((f'_sub{ss}' if a.subsample_logs and ss != SUBSAMPLE_SEED else '')
            + (f'_wu{wu}' if wu != WARMUP_STEPS else '')
            + getattr(a, 'zs_suffix', '')           # section 19.1 arm markers (_tcol/_fal/_fk8)
            + ('_logged' if getattr(a, 'nuplan_ego', 'routed') == 'logged' else ''))  # section 20.2 S2


def out_path(arm, a):
    # section 19.1 runs (any zs_suffix) live under transfer/zs/, never beside the section-18 files
    d = f'{TRANSFER_DIR}/zs' if getattr(a, 'zs_suffix', '') else TRANSFER_DIR
    if a.transfer_to == 'nuplan':                 # section 20.2: nuPlan target lives under transfer/nuplan/
        d = f'{TRANSFER_DIR}/nuplan'
    os.makedirs(d, exist_ok=True)
    return f'{d}/{arm}_{src_tag(a)}2{a.transfer_to}_s{a.seed}{variant_suffix(a)}.npz'


def save(path, **kw):
    assert '/frozen30/' not in path and '/es/' not in path and '/es_pinit/' not in path
    np.savez(path, **kw)
    print(f'WROTE {path}', flush=True)


def check_args(a):
    assert bool(a.full_train) == bool(a.transfer_to), \
        '--full-train and --transfer-to go together'
    if a.transfer_to:
        assert a.transfer_to != a.domain, '--transfer-to must name the OTHER domain'
    if a.subsample_logs:
        assert a.full_train and a.domain == 'navsim', \
            '--subsample-logs is a NavSim-source --full-train option'
    if getattr(a, 'subsample_seed', SUBSAMPLE_SEED) != SUBSAMPLE_SEED:
        assert a.subsample_logs, '--subsample-seed needs --subsample-logs'
    if getattr(a, 'warmup_steps', WARMUP_STEPS) != WARMUP_STEPS:
        assert a.full_train and a.domain == 'navsim', \
            '--warmup-steps is a NavSim-source --full-train option (B2D recipe has no warmup)'
