#!/usr/bin/env python3
"""Agent-JEPA (minDrive-JEPA) -- faithful re-implementation of

    Santosh Jaiswal, "Zero-Label Driving Scenario Complexity Detection via
    Joint Embedding Predictive Architecture", arXiv:2606.28383v1 (2026).

There is NO official code release for this paper (checked: no code/data link in
the abstract, the paper body, or the LaTeX source).  Everything below is coded
from the published text; every formula carries the section it comes from and
every departure from the text is listed in DEVIATIONS (and copied verbatim into
the provenance JSON).

What the paper specifies (Sec. 3):
  3.1 scenario tensor [50, 21, 6] = 50 steps @10 Hz (5 s), ego + 20 agents,
      features (x, y, vx, vy, theta, type); x,y / 50 m, v / 10 m/s
  3.2 context = first half, target = second half
  3.3 context encoder  : 4-layer Transformer encoder, d=128, 4 heads, ff=512,
                         flatten 21*6=126 -> Linear -> +sinusoidal PE ->
                         mean-pool over time -> z_ctx in R^128
      predictor        : 2-layer Transformer encoder, learnable horizon
                         embedding, z_ctx -> z_hat
      target encoder   : identical copy, EMA only, reads the target window
      position decoder : small MLP z_hat -> 21x2 future positions (auxiliary)
  3.4 L = ||z_hat - z_tgt||^2 + lambda ||p_hat - p_tgt||^2, lambda = 0.1,
      stop-grad at the target encoder, EMA alpha = 0.996
  3.5 surprise score s = ||z_hat - z_tgt||_2
  3.6 50 epochs, batch 32, lr 3e-4, linear warmup 5 epochs then cosine decay,
      weight decay 1e-4, dropout 0.2, grad clip 1.0, 80/20 train/val split,
      1,289,130 trainable parameters

subcommands: train | extract | provenance
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = '/data1/jeongtae/b2d_jepa'          # read-only: the split the old row used
OUTDIR = Path('/data2/jeongtae/official_baselines/us_features')
CACHE = OUTDIR / 'agentjepa_cache'
NAME = 'agentjepa_official'

# ---- data geometry (Sec. 3.1 / 3.2, adapted to the stored B2D grid) ---------
GRID = 5            # stored agent grid step, in 10 Hz frames -> 0.5 s
T_WIN = 10          # 10 grid steps = 5.0 s scenario  (paper: 50 frames @10 Hz)
CTX = 5             # first half  = 2.5 s
TGT = T_WIN - CTX   # second half = 2.5 s
STRIDE = 1          # sliding-window stride, in grid steps (0.5 s)
NSLOT, FDIM = 21, 6         # ego + 20 agents, (x, y, vx, vy, theta, type)
POS_NORM, VEL_NORM = 50.0, 10.0
TYPE_EGO, TYPE_VEH, TYPE_PED = 3.0, 1.0, 2.0

# ---- model / optimisation (Sec. 3.3 / 3.4 / 3.6) ---------------------------
D = 128
ENC_LAYERS, PRED_LAYERS, HEADS, FF, DROPOUT = 4, 2, 4, 512, 0.2
LAMBDA_POS = 0.1
EMA_ALPHA = 0.996
EPOCHS, BATCH, LR, WD, WARMUP_EPOCHS, CLIP = 50, 32, 3e-4, 1e-4, 5, 1.0
VAL_FRAC = 0.2
SEED = 0
# Table 3A scores, measured with experiments/us_official/score_one.py (same unified
# split, 16 draws, 640 pooled route evaluations, same two-stage Ridge plug-in; the
# shipped anchors re-assert on every run).  Reference rows on the same run:
#   planner-only null            AUROC .699 / MAE .214
#   Kinematics (cmdkin, 25d)     AUROC .752 / MAE .180 (+15.6%) / rho +.497
#   Hand-crafted risk (73d)      AUROC .758 / MAE .175 (+18.0%) / rho +.533
#   RelGraph R2 (3 runs)         AUROC .751 / MAE .192 / rho +.490
SCORES = {
  'shipped row (old in-house replication, 12d, epoch-0 checkpoint)':
      {'auroc': 0.696, 'mae': 0.217, 'mae_gain': '-1.5%', 'rho': +0.001},
  'agentjepa_official (this file: full 50-epoch schedule, epoch 49)':
      {'auroc': 0.699, 'mae': 0.216, 'mae_gain': '-1.0%', 'rho': -0.003},
  'agentjepa_official_bestval (paper-literal best-validation-loss checkpoint, epoch 1/50)':
      {'auroc': 0.711, 'mae': 0.205, 'mae_gain': '+4.1%', 'rho': +0.151},
  'agentjepa_official_randinit (paper Sec. 4.2 Ablation 2, random encoder)':
      {'auroc': 0.698, 'mae': 0.215, 'mae_gain': '-0.6%', 'rho': -0.009},
}


DEVIATIONS = [
  ("temporal resolution: 0.5 s grid (10 steps per 5 s scenario, 5 context + 5 "
   "target) instead of the paper's 10 Hz / 50 steps",
   "forced. The paper's own corpus (nuPlan mini) is not the bank under test; the "
   "training corpus here is the one the shipped row used -- 1000 Bench2Drive v1 "
   "expert clips whose agent tracks survive only as the 0.5 s-grid tensor "
   "/data1/jeongtae/b2d_jepa/gtod_train.npz. The raw 10 Hz anno of those clips is "
   "gone from this machine (/data1/Bench2Drive no longer exists; /data2/bench2drive "
   "holds 253 of the 1000 clips as tar.gz owned by another user). Evaluation routes "
   "are read at the same 0.5 s grid so train and test match. Scenario length (5 s), "
   "the half/half context-target split and the 2.5 s prediction horizon are the "
   "paper's."),
  ("dataset: Bench2Drive v1 expert clips (CARLA) instead of nuPlan mini",
   "required by the protocol -- Table 3A scores descriptors of the 220 Bench2Drive "
   "routes. Self-supervised, label-free, exactly as the paper prescribes."),
  ("agent slots are re-sorted by distance to the ego at every grid step, so a slot "
   "index is not a persistent track",
   "property of the stored tensor (built by b2d_irt/scripts/gtod_pipeline.py:parse_scene, "
   "which keeps the 24 nearest agents within 60 m per step). The paper's nuPlan "
   "tensors are presumably track-consistent. Not fixable without the deleted raw anno."),
  ("agent type is coded 1=vehicle, 2=pedestrian, 3=ego, 0=empty slot as a single "
   "scalar feature",
   "the paper says 'a categorical agent type' as one of the six features but does "
   "not give the coding; Bench2Drive annotations only distinguish vehicle/walker."),
  ("the 21 slots are expressed in the ego frame of the window's FIRST step "
   "(translation and rotation), then normalised by 50 m / 10 m/s",
   "the paper normalises positions by 50 m and expects values near [-1,1], which is "
   "only possible in an ego-relative frame; CARLA world coordinates are O(1e3) m. "
   "Rotation as well as translation follows the standard nuPlan agent-feature "
   "convention and removes absolute-map memorisation."),
  ("the auxiliary position loss is masked to occupied slots",
   "the paper pads to a fixed 21 slots and does not say. Unmasked, the decoder would "
   "spend most of its capacity predicting the constant 0 of empty slots."),
  ("p_tgt is taken at the LAST target step (t = +2.5 s)",
   "the paper writes p_hat in R^{21x2} -- one position per agent -- without saying "
   "which target frame; the end of the target window is the only single frame that "
   "makes the term a genuine forecasting signal."),
  ("both loss terms use mean-reduction MSE",
   "the paper writes squared L2 norms; mean reduction reproduces its reported loss "
   "scale (best val 0.0345) and is scale-free in the latent width."),
  ("the 80/20 train/validation split is taken over CLIPS, not over windows",
   "sliding windows from one clip overlap; a window-level split puts near-duplicates "
   "on both sides and makes the validation loss uninformative. This is the split the "
   "old in-house replication (b2d_irt/scripts/agentjepa.py) got wrong."),
  ("optimiser AdamW; the target encoder is always in eval mode (no dropout)",
   "the paper gives weight decay and dropout but not the optimiser; AdamW is the "
   "standard pairing. A dropout-perturbed EMA target would inject noise into the "
   "regression target."),
  ("the per-route descriptor is [mean, p90, max] of the per-window surprise score",
   "the paper scores one 5 s scenario; a Bench2Drive route is a sequence of them. "
   "The three aggregates mirror the aggregation the shipped row used, so only the "
   "model and the score definition change. Column 0 alone is the paper-literal "
   "score averaged over the route and is reported separately."),
]


# ============================ data =========================================
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def scene_windows(ag, ego):
    """(Tg, 24, 8) world-frame agents + (nf, 5) ego -> (W, T_WIN, 21, 6) float32
    plus (W, T_WIN, 21) validity.  Sec. 3.1 features in the Sec. 3.2 split."""
    Tg = min(len(ag), (len(ego) - 1) // GRID + 1)
    outs, msks = [], []
    for g0 in range(0, Tg - T_WIN + 1, STRIDE):
        fa = g0 * GRID
        if fa >= len(ego) or ego[fa, 4] <= 0:
            continue
        gs = np.arange(g0, g0 + T_WIN)
        fs = gs * GRID
        if fs[-1] >= len(ego) or (ego[fs, 4] <= 0).any():
            continue
        x0, y0, h0 = float(ego[fa, 0]), float(ego[fa, 1]), float(ego[fa, 2])
        c, s = math.cos(h0), math.sin(h0)
        W = np.zeros((T_WIN, NSLOT, FDIM), np.float32)
        M = np.zeros((T_WIN, NSLOT), bool)
        for t, (g, f) in enumerate(zip(gs, fs)):
            # slot 0: ego
            dx, dy = ego[f, 0] - x0, ego[f, 1] - y0
            th = _wrap(ego[f, 2] - h0)
            spd = float(ego[f, 3])
            W[t, 0] = ((dx * c + dy * s) / POS_NORM, (-dx * s + dy * c) / POS_NORM,
                       spd * math.cos(th) / VEL_NORM, spd * math.sin(th) / VEL_NORM,
                       th, TYPE_EGO)
            M[t, 0] = True
            # slots 1..20: the 20 nearest annotated agents at this step
            a = ag[g, :NSLOT - 1]
            v = a[:, 7] > 0
            if not v.any():
                continue
            dx, dy = a[:, 0] - x0, a[:, 1] - y0
            tha = _wrap(a[:, 2] - h0)
            W[t, 1:, 0] = (dx * c + dy * s) / POS_NORM
            W[t, 1:, 1] = (-dx * s + dy * c) / POS_NORM
            W[t, 1:, 2] = a[:, 3] * np.cos(tha) / VEL_NORM
            W[t, 1:, 3] = a[:, 3] * np.sin(tha) / VEL_NORM
            W[t, 1:, 4] = tha
            W[t, 1:, 5] = np.where(a[:, 6] > 0.5, TYPE_VEH, TYPE_PED)
            W[t, 1:] *= v[:, None]
            M[t, 1:] = v
        outs.append(W)
        msks.append(M)
    if not outs:
        return (np.zeros((0, T_WIN, NSLOT, FDIM), np.float32),
                np.zeros((0, T_WIN, NSLOT), bool))
    return np.stack(outs), np.stack(msks)


def build_split(split):
    """Cache (windows, masks, window->scene index, scene names)."""
    f = CACHE / f'{split}.npz'
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return z['W'], z['M'], z['gi'], [str(x) for x in z['names']]
    CACHE.mkdir(parents=True, exist_ok=True)
    Ws, Ms, gis, names = [], [], [], []
    if split == 'train':
        ix = json.load(open(f'{ROOT}/index.json'))
        z = np.load(f'{ROOT}/gtod_train.npz', allow_pickle=True)
        E = np.load(f'{ROOT}/ego.npy', mmap_mode='r')
        items = [(n, np.asarray(z[n]), np.asarray(E[i][:ix['nf'][i]]))
                 for i, n in enumerate(ix['clips']) if n in z.files]
    else:
        z = np.load(f'{ROOT}/gtod_eval.npz', allow_pickle=True)
        items = []
        for n in sorted(z.files):
            items.append((n, np.asarray(z[n]),
                          np.asarray(np.load(f'{ROOT}/eval_feats/{n}.npz')['ego'])))
    for si, (n, ag, e) in enumerate(items):
        W, M = scene_windows(ag, e)
        names.append(n)
        if len(W):
            Ws.append(W)
            Ms.append(M)
            gis.append(np.full(len(W), si, np.int32))
    W = np.concatenate(Ws)
    M = np.concatenate(Ms)
    gi = np.concatenate(gis)
    np.savez_compressed(f, W=W, M=M, gi=gi, names=np.array(names))
    print(f'built {split}: {W.shape} windows over {len(names)} scenes', flush=True)
    return W, M, gi, names


# ============================ model (Sec. 3.3) ==============================
def make_models(torch, nn):
    class Encoder(nn.Module):
        """Sec. 3.3 context/target encoder."""
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(NSLOT * FDIM, D)
            lay = nn.TransformerEncoderLayer(D, HEADS, FF, dropout=DROPOUT,
                                             batch_first=True)
            self.tr = nn.TransformerEncoder(lay, ENC_LAYERS)
            pe = torch.zeros(max(CTX, TGT), D)
            pos = torch.arange(max(CTX, TGT)).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
            self.register_buffer('pe', pe)

        def forward(self, x):                      # (B, T, 21, 6)
            h = self.proj(x.flatten(2)) + self.pe[:x.shape[1]]
            return self.tr(h).mean(1)              # (B, D)

    class Predictor(nn.Module):
        """Sec. 3.3 predictor: 2-layer Transformer + learnable horizon embedding."""
        def __init__(self):
            super().__init__()
            self.hor = nn.Parameter(torch.zeros(1, 1, D))
            nn.init.trunc_normal_(self.hor, std=0.02)
            lay = nn.TransformerEncoderLayer(D, HEADS, FF, dropout=DROPOUT,
                                             batch_first=True)
            self.tr = nn.TransformerEncoder(lay, PRED_LAYERS)

        def forward(self, z):                      # (B, D)
            h = torch.cat([z.unsqueeze(1), self.hor.expand(len(z), 1, D)], 1)
            return self.tr(h)[:, 0]

    class PosDecoder(nn.Module):
        """Sec. 3.3 auxiliary position decoder."""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(D, 256), nn.ReLU(),
                                     nn.Linear(256, NSLOT * 2))

        def forward(self, z):
            return self.net(z).view(-1, NSLOT, 2)

    return Encoder(), Predictor(), PosDecoder(), Encoder()


def losses(torch, F, enc, pred, dec, tgt, W, M):
    zc = enc(W[:, :CTX])
    zh = pred(zc)
    with torch.no_grad():
        zt = tgt(W[:, CTX:])                       # stop-grad (Sec. 3.4)
    l_j = F.mse_loss(zh, zt)
    p_hat = dec(zh)
    p_tgt = W[:, -1, :, :2]
    m = M[:, -1].unsqueeze(-1).float()
    l_p = ((p_hat - p_tgt) ** 2 * m).sum() / m.sum().clamp(min=1) / 2.0
    return l_j, l_p, zc, zh, zt


# ============================ train (Sec. 3.6) ==============================
def cmd_train(a):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    W, M, gi, names = build_split('train')
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(names))                 # 80/20 split over CLIPS
    nval = int(round(VAL_FRAC * len(names)))
    vset = set(order[:nval].tolist())
    isval = np.array([g in vset for g in gi])
    Wtr, Mtr, Wva, Mva = W[~isval], M[~isval], W[isval], M[isval]
    print(f'train windows {len(Wtr)} ({len(names) - nval} clips) | '
          f'val windows {len(Wva)} ({nval} clips)', flush=True)

    enc, pred, dec, tgt = make_models(torch, nn)
    enc, pred, dec, tgt = enc.to(dev), pred.to(dev), dec.to(dev), tgt.to(dev)
    tgt.load_state_dict(enc.state_dict())
    for p in tgt.parameters():
        p.requires_grad_(False)
    torch.save({'enc': enc.state_dict(), 'pred': pred.state_dict(),
                'dec': dec.state_dict(), 'tgt': tgt.state_dict()},
               CACHE / 'agentjepa_random_init.pt')     # paper Ablation 2
    params = list(enc.parameters()) + list(pred.parameters()) + list(dec.parameters())
    nparam = sum(p.numel() for p in params)
    print(f'trainable params {nparam}  (paper: 1,289,130)', flush=True)

    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    spe = max(1, len(Wtr) // BATCH)
    total, warm = EPOCHS * spe, WARMUP_EPOCHS * spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    curve, best, best_ep = [], float('inf'), -1
    for ep in range(EPOCHS):
        enc.train(); pred.train(); dec.train(); tgt.eval()
        perm = rng.permutation(len(Wtr))
        tj = tp = 0.0
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            b = perm[i:i + BATCH]
            w = torch.as_tensor(Wtr[b], device=dev)
            m = torch.as_tensor(Mtr[b], device=dev)
            l_j, l_p, *_ = losses(torch, F, enc, pred, dec, tgt, w, m)
            loss = l_j + LAMBDA_POS * l_p
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            sched.step()
            with torch.no_grad():                      # EMA (Sec. 3.4)
                for q, k in zip(tgt.parameters(), enc.parameters()):
                    q.mul_(EMA_ALPHA).add_(k.detach(), alpha=1 - EMA_ALPHA)
            tj += l_j.item(); tp += l_p.item()
        nb = max(1, (len(perm) - BATCH + 1 + BATCH - 1) // BATCH)
        enc.eval(); pred.eval(); dec.eval()
        vj = vp = 0.0; vn = 0; ZT, ZH = [], []
        with torch.no_grad():
            for i in range(0, len(Wva), 256):
                w = torch.as_tensor(Wva[i:i + 256], device=dev)
                m = torch.as_tensor(Mva[i:i + 256], device=dev)
                l_j, l_p, zc, zh, zt = losses(torch, F, enc, pred, dec, tgt, w, m)
                vj += l_j.item() * len(w); vp += l_p.item() * len(w); vn += len(w)
                ZT.append(zt.cpu().numpy()); ZH.append(zh.cpu().numpy())
        vj, vp = vj / max(vn, 1), vp / max(vn, 1)
        v = vj + LAMBDA_POS * vp
        ZT, ZH = np.concatenate(ZT), np.concatenate(ZH)
        lstd = float(ZT.std(0).mean())
        # scale-free diagnostics: is the predictor better than predicting the mean?
        nmse = float(((ZH - ZT) ** 2).mean() / max(ZT.var(0).mean(), 1e-12))
        cos = float((ZH * ZT).sum(1).mean() /
                    max((np.linalg.norm(ZH, axis=1) * np.linalg.norm(ZT, axis=1)).mean(), 1e-12))
        row = dict(epoch=ep, train_jepa=tj / nb, train_pos=tp / nb,
                   train_total=(tj + LAMBDA_POS * tp) / nb, val_jepa=vj, val_pos=vp,
                   val_total=v, latent_std=lstd, val_nmse=nmse, val_cos=cos,
                   lr=sched.get_last_lr()[0])
        curve.append(row)
        print('ep {epoch:2d} train {train_total:.5f} (j {train_jepa:.5f} p {train_pos:.5f}) | '
              'val {val_total:.5f} (j {val_jepa:.5f} p {val_pos:.5f}) | '
              'latent std {latent_std:.4f} nmse {val_nmse:.4f} cos {val_cos:.4f}'.format(**row),
              flush=True)
        if v < best:
            best, best_ep = v, ep
            torch.save({'enc': enc.state_dict(), 'pred': pred.state_dict(),
                        'dec': dec.state_dict(), 'tgt': tgt.state_dict(),
                        'epoch': ep, 'val_total': v}, CACHE / 'agentjepa_best.pt')
    torch.save({'enc': enc.state_dict(), 'pred': pred.state_dict(),
                'dec': dec.state_dict(), 'tgt': tgt.state_dict(),
                'epoch': EPOCHS - 1}, CACHE / 'agentjepa_final.pt')
    json.dump({'curve': curve, 'best_epoch': best_ep, 'best_val_total': best,
               'n_params': nparam, 'n_train_windows': int(len(Wtr)),
               'n_val_windows': int(len(Wva))},
              open(CACHE / 'train_curve.json', 'w'), indent=1)
    print(f'DONE best val {best:.5f} at epoch {best_ep}', flush=True)


# ============================ extract (Sec. 3.5) ============================
AGG = ['mean', 'p90', 'max']
LAYOUT = ('[mean,p90,max] of the per-window surprise score s = ||z_hat - z_tgt||_2 '
          '(arXiv:2606.28383v1 Sec. 3.5) over the 5 s windows of the route. '
          'Column 0 is the paper-literal score averaged over the route.')


def cmd_extract(a):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    W, M, gi, names = build_split('eval')
    enc, pred, dec, tgt = make_models(torch, nn)
    ck = torch.load(CACHE / f'agentjepa_{a.ckpt}.pt', map_location='cpu')
    for mod, k in ((enc, 'enc'), (pred, 'pred'), (dec, 'dec'), (tgt, 'tgt')):
        mod.load_state_dict(ck[k])
    enc, pred, tgt = enc.to(dev).eval(), pred.to(dev).eval(), tgt.to(dev).eval()
    S = []
    with torch.no_grad():
        for i in range(0, len(W), 512):
            w = torch.as_tensor(W[i:i + 512], device=dev)
            zh = pred(enc(w[:, :CTX]))
            zt = tgt(w[:, CTX:])
            S.append((zh - zt).norm(dim=-1).cpu().numpy())
    S = np.concatenate(S).astype(np.float64)

    stats, kept, nw, extra = [], [], [], []
    for si, n in enumerate(names):
        s = S[gi == si]
        if len(s) < 3:
            print(f'SKIP {n}: only {len(s)} windows', flush=True)
            continue
        stats.append([s.mean(), np.percentile(s, 90), s.max()])
        extra.append([s.std(), np.median(s), s.min()])
        kept.append(n)
        nw.append(len(s))
    st = np.asarray(stats, np.float64)
    out = OUTDIR / f'{a.out}.npz'
    np.savez(out, stats=st, names=np.array(kept),
             layout=LAYOUT, agg=np.array(AGG),
             n_windows=np.array(nw), extra_stats=np.asarray(extra, np.float64),
             extra_names=np.array(['std', 'median', 'min']),
             checkpoint=str(a.ckpt))
    print(f'WROTE {out}  {st.shape}  windows/route min {min(nw)} med '
          f'{int(np.median(nw))} max {max(nw)}', flush=True)
    return out, st, kept, nw


# ============================ provenance ====================================
def contamination_report():
    ix = json.load(open(f'{ROOT}/index.json'))
    clips = ix['clips']
    tr_routes = sorted({int(re.search(r'_Route(\d+)_', c).group(1)) for c in clips})
    tr_types = sorted({c.split('_Town')[0] for c in clips})
    import csv
    rt = dict(csv.reader(open('/home/jeongtae/SC-IRT/data/matrices/b2d_route_types.csv')))
    ev_routes = sorted(int(k) for k in rt)
    ev_types = sorted(set(rt.values()))
    return {
      'train_clips': len(clips),
      'train_route_ids': f'{min(tr_routes)}..{max(tr_routes)} ({len(tr_routes)} distinct)',
      'eval_route_ids': f'{min(ev_routes)}..{max(ev_routes)} ({len(ev_routes)} distinct)',
      'route_id_overlap_train_vs_eval': sorted(set(tr_routes) & set(ev_routes)),
      'n_train_scenario_types': len(tr_types),
      'n_eval_scenario_types': len(ev_types),
      'eval_types_absent_from_train': sorted(set(ev_types) - set(tr_types)),
      'train_types_absent_from_eval': sorted(set(tr_types) - set(ev_types)),
      'verdict': ('No evaluation route appears in the training corpus (route-id sets '
                  'are disjoint: train 1..1636 are Bench2Drive v1 expert clips, eval '
                  '1711..28330 are the 220 held-out routes). 41 of the 44 evaluation '
                  'scenario-type NAMES also occur among the 43 training-clip types '
                  '(the three that do not -- SequentialLaneChange, T_Junction, '
                  'VanillaNonSignalizedTurn -- are mostly spelling variants of '
                  'LaneChange / TJunction). That type-level overlap is inherent to the '
                  'split the shipped row used and is label-free: training touches no '
                  'planner response, no difficulty value and no type label -- the loss '
                  'is self-supervised on agent state only. It does mean the row must '
                  'NOT be described as trained on calibration types only, unlike the '
                  'RelGraph encoder row, which is retrained per draw on the 36 '
                  'calibration types.')}


def score_diagnostics():
    """Scale-free checks on the shipped scores: does the surprise score track
    anything other than route duration, and do the three checkpoints agree?"""
    from scipy.stats import spearmanr
    files = {'final (50-epoch schedule)': NAME, 'bestval (epoch 1/50)': NAME + '_bestval',
             'random encoder (Ablation 2)': NAME + '_randinit'}
    got = {}
    for k, f in files.items():
        p_ = OUTDIR / f'{f}.npz'
        if p_.exists():
            z = np.load(p_, allow_pickle=True)
            got[k] = (z['stats'][:, 0], z['n_windows'].astype(float))
    out = {'rho(route mean surprise, n_windows per route)':
           {k: round(float(spearmanr(v[0], v[1]).correlation), 3) for k, v in got.items()},
           'rho between checkpoints (route mean surprise)': {}}
    ks = list(got)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            out['rho between checkpoints (route mean surprise)'][f'{ks[i]} vs {ks[j]}'] = \
                round(float(spearmanr(got[ks[i]][0], got[ks[j]][0]).correlation), 3)
    out['reading'] = (
        'The epoch-1 checkpoint that the paper\'s own best-validation-loss rule selects is '
        'the only variant with a non-trivial Table 3A rho (+.151) -- but its per-route '
        'score ranks the bank the same way the RANDOM-weight encoder does (rho +0.76) and '
        'is largely a route-duration proxy (rho -0.47 with the number of 5 s windows in '
        'the route). So that +.151 is a random-feature/route-length effect, not learned '
        'complexity. The fully trained model (50-epoch schedule) correlates with nothing '
        '-- not route duration, not the epoch-1 model, not the random encoder -- and lands '
        'exactly on the planner-only null.')
    return out


def cmd_provenance(a):
    curve = json.load(open(CACHE / 'train_curve.json'))
    z = np.load(OUTDIR / f'{a.out}.npz', allow_pickle=True)
    st, nw = z['stats'], z['n_windows']
    prov = {
      'feature_file': str(OUTDIR / f'{a.out}.npz'),
      'built_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
      'family': 'Agent-JEPA',
      'paper': {'id': 'arXiv:2606.28383v1', 'author': 'Santosh Jaiswal',
                'title': 'Zero-Label Driving Scenario Complexity Detection via Joint '
                         'Embedding Predictive Architecture',
                'sections_implemented': {
                    '3.1 Data Representation': 'scenario tensor, ego + 20 agents, '
                        '(x, y, vx, vy, theta, type), /50 m and /10 m/s normalisation',
                    '3.2 Temporal Split': 'context = first half, target = second half',
                    '3.3 Architecture': '4-layer d128 4-head ff512 context encoder with '
                        'sinusoidal PE and time mean-pool; 2-layer transformer predictor '
                        'with learnable horizon embedding; EMA target encoder; MLP '
                        'position decoder to 21x2',
                    '3.4 Training Objective': 'L = MSE(z_hat, z_tgt) + 0.1 * MSE(p_hat, '
                        'p_tgt), stop-grad at the target encoder, EMA alpha 0.996',
                    '3.5 Surprise Score': 's = ||z_hat - z_tgt||_2',
                    '3.6 Training Details': '50 epochs, batch 32, lr 3e-4, 5-epoch linear '
                        'warmup then cosine decay, weight decay 1e-4, dropout 0.2, grad '
                        'clip 1.0, 80/20 train/val split'}},
      'official_code': ('NONE. The paper releases no code and no data: there is no code '
                        'or project link in the abstract, the body or the arXiv LaTeX '
                        'source. Every line of this implementation is written from the '
                        'text of Sec. 3.'),
      'code_run': {'repo': '/home/jeongtae/SC-IRT (ATDrive)',
                   'git_commit': a.commit,
                   'file': 'experiments/us_official/agentjepa.py',
                   'entry_points': 'cmd_train -> cmd_extract -> cmd_provenance',
                   'python': '/home/jeongtae/miniconda3/envs/smart/bin/python',
                   'torch': a.torch_version, 'device': a.device},
      'inputs': {
        'train_split': ('/data1/jeongtae/b2d_jepa/gtod_train.npz + index.json + ego.npy '
                        '-- the SAME split the shipped row used (b2d_irt/scripts/'
                        'agentjepa.py lines 31-39): 1000 Bench2Drive v1 expert clips, '
                        'no outcome-based selection'),
        'eval_routes': ('/data1/jeongtae/b2d_jepa/gtod_eval.npz + eval_feats/*.npz, the '
                        '220-route bank, agents parsed from the PDM-Lite reference '
                        'rollout /data1/jeongtae/b2d_eval_sensors/route_*/anno by '
                        'b2d_irt/scripts/gtod_pipeline.py:parse_scene'),
        'note': ('PDM-Lite is EXCLUDED_PLANNER and is not one of the 16 evaluated '
                 'planners; the RelGraph encoder reads the same rollout. Nothing in this '
                 'pipeline reads the response matrix, planner outcomes or scenario types.')},
      'parameters': {
        'GRID_frames': GRID, 'grid_step_seconds': GRID * 0.1, 'T_WIN_steps': T_WIN,
        'scenario_seconds': T_WIN * GRID * 0.1, 'CTX_steps': CTX, 'TGT_steps': TGT,
        'window_stride_steps': STRIDE, 'NSLOT': NSLOT, 'FDIM': FDIM,
        'pos_norm_m': POS_NORM, 'vel_norm_mps': VEL_NORM,
        'type_codes': {'empty': 0, 'vehicle': TYPE_VEH, 'pedestrian': TYPE_PED,
                       'ego': TYPE_EGO},
        'D': D, 'enc_layers': ENC_LAYERS, 'pred_layers': PRED_LAYERS, 'heads': HEADS,
        'ff': FF, 'dropout': DROPOUT, 'lambda_pos': LAMBDA_POS, 'ema_alpha': EMA_ALPHA,
        'epochs': EPOCHS, 'batch': BATCH, 'lr': LR, 'weight_decay': WD,
        'warmup_epochs': WARMUP_EPOCHS, 'grad_clip': CLIP, 'val_frac': VAL_FRAC,
        'seed': SEED, 'optimizer': 'AdamW',
        'n_trainable_params': curve['n_params'],
        'paper_n_trainable_params': 1289130,
        'checkpoint_used': str(z['checkpoint'])},
      'deviations_from_source': [{'what': w, 'why': y} for w, y in DEVIATIONS],
      'training': {'n_train_windows': curve['n_train_windows'],
                   'n_val_windows': curve['n_val_windows'],
                   'best_epoch': curve['best_epoch'],
                   'best_val_total': curve['best_val_total'],
                   'curve': curve['curve']},
      'table3a_scores': SCORES,
      'score_diagnostics': score_diagnostics(),
      'contamination_audit': contamination_report(),
      'per_route_sanity': {
        'n_routes': int(st.shape[0]), 'd': int(st.shape[1]),
        'columns': [str(x) for x in z['agg']],
        'windows_per_route': {'min': int(nw.min()), 'median': float(np.median(nw)),
                              'max': int(nw.max()), 'total': int(nw.sum())},
        'per_column': {str(z['agg'][j]): {
            'min': float(st[:, j].min()), 'p25': float(np.percentile(st[:, j], 25)),
            'median': float(np.median(st[:, j])), 'p75': float(np.percentile(st[:, j], 75)),
            'max': float(st[:, j].max()), 'mean': float(st[:, j].mean()),
            'std': float(st[:, j].std())} for j in range(st.shape[1])},
        'n_nonfinite': int((~np.isfinite(st)).sum()),
        'n_constant_columns': int(sum(st[:, j].std() < 1e-12 for j in range(st.shape[1])))},
      'notes': a.notes or '',
    }
    p = OUTDIR / f'{a.out}_provenance.json'
    json.dump(prov, open(p, 'w'), indent=1)
    print(f'WROTE {p}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'train', 'extract', 'provenance'])
    ap.add_argument('--ckpt', default='best', choices=['best', 'final', 'random_init'])
    ap.add_argument('--out', default=NAME)
    ap.add_argument('--split', default='eval')
    ap.add_argument('--notes', default='')
    ap.add_argument('--commit', default=subprocess.run(
        ['git', '-C', '/home/jeongtae/SC-IRT', 'rev-parse', 'HEAD'],
        capture_output=True, text=True).stdout.strip() or 'unknown')
    ap.add_argument('--torch_version', default='')
    ap.add_argument('--device', default=os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    a = ap.parse_args()
    if a.cmd == 'build':
        build_split(a.split)
    else:
        globals()[f'cmd_{a.cmd}'](a)
