#!/usr/bin/env python3
"""Train the difficulty-supervised scene encoder on the unified split.

Verbatim port of the research script that produced data/encoder/
unified_enc_pred_d{64,96}s{0,1,2}.npz (the Table 3A encoder rows).
Per draw: theta for the 13 calibration planners from a fold-internal Rasch
fit (`scirt.encoder.rasch`, fail parameterisation), then the encoder is
trained end to end on the calibration block with the cell-level Bernoulli
likelihood  p_ij = sigmoid(b_tilde(x_i) - theta_j)  (fail = 1) — no b-hat
targets, no ensembling — and predicts b_tilde for the held-out-type routes.

    python train/train_encoder_unified.py --seed 0 --d 64
    python experiments/eval_us_predictions.py results/unified_enc_pred_d64s0.npz

Recipe constants (part of the reproduction contract): W_MAX=22 windows per
route, 30 epochs, batch 64, AdamW lr 1e-3 wd 0.1, grad clip 1.0, kin
features standardised on the training routes. Seeding: numpy is reset to
--seed per draw (it drives the batch shuffle order); the torch seed is
effectively pinned to 0 for every run because `scirt.encoder.rasch()`
calls torch.manual_seed(0) internally after the per-draw reset, so model
initialisation and dropout streams are identical across --seed values and
seeds differ only through data order. Route order inside the tensors is
sorted(route id). Verified: a full seed-0 / d64 run reproduces the shipped
artifact bit-for-bit on the development GPUs.
"""
import argparse
import csv
import os
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scirt.splits import unified_split, R_DRAWS   # noqa: E402

W_MAX, EPOCHS, BATCH = 22, 30, 64


def theta_null(torch, Ytr_pass, dev, it=400):
    M = torch.tensor(np.nan_to_num(Ytr_pass), dtype=torch.float32, device=dev)
    W = torch.tensor((~np.isnan(Ytr_pass)).astype(np.float32), device=dev)
    th = torch.zeros(Ytr_pass.shape[0], device=dev, requires_grad=True)
    opt = torch.optim.Adam([th], lr=0.05)
    for _ in range(it):
        p = torch.sigmoid(th[:, None].expand_as(M))
        nll = (-(M * torch.log(p + 1e-7) + (1 - M) * torch.log(1 - p + 1e-7)) * W).sum() / W.sum()
        (nll + 1e-2 * th.pow(2).mean()).backward()
        opt.step()
        opt.zero_grad()
    return th.detach().cpu().numpy()


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--draws', type=int, default=R_DRAWS, help='number of draws (smoke tests)')
    ap.add_argument('--out', default=str(ROOT / 'results'))
    a = ap.parse_args()
    import torch
    import torch.nn as nn
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from scirt.encoder import build, rasch
    dev = a.device

    d = np.load(ROOT / 'data/encoder/b2d_tensors.npz', allow_pickle=True)
    routes = [str(x) for x in d['route']]
    rid2w = {}
    for i, r in enumerate(routes):
        rid2w.setdefault(r.replace('route_', ''), []).append(i)
    rows = list(csv.reader(open(os.environ.get('SCIRT_RESPONSE_CSV', ROOT / 'data/matrices/b2d_e2e16_response_matrix.csv'))))
    rids = rows[0][1:]
    data = [r for r in rows[1:] if r[0] != 'PDM-Lite']
    Yf = np.full((len(data), len(rids)), np.nan)
    for pi, row in enumerate(data):
        for j in range(len(rids)):
            if row[1 + j] != '':
                Yf[pi, j] = 1.0 - float(row[1 + j])          # fail = 1
    types_map = dict(csv.reader(open(ROOT / 'data/matrices/b2d_route_types.csv')))
    keep = [r for r in sorted(rid2w) if r in types_map and r in rids]
    col = {rid: j for j, rid in enumerate(rids)}
    Y = Yf[:, [col[r] for r in keep]]
    types = np.array([types_map[r] for r in keep])
    R, J = len(keep), Y.shape[0]
    AG = np.zeros((R, W_MAX, 48, 12, 8), np.float16)
    AM = np.zeros((R, W_MAX, 48, 12), bool)
    EG = np.zeros((R, W_MAX, 12, 6), np.float16)
    CM = np.zeros((R, W_MAX, 4), np.float16)
    WM = np.zeros((R, W_MAX), bool)
    for ri, r in enumerate(keep):
        ws = rid2w[r][:W_MAX]
        AG[ri, :len(ws)] = d['agents'][ws]
        AM[ri, :len(ws)] = d['amask'][ws]
        EG[ri, :len(ws)] = d['ego'][ws]
        CM[ri, :len(ws)] = d['cmd'][ws]
        WM[ri, :len(ws)] = True
    z = np.load(ROOT / 'data/features/eval_cmdkin_stats.npz', allow_pickle=True)
    nm = {str(x).replace('route_', ''): i for i, x in enumerate(z['names'])}
    CK = np.stack([z['stats'][nm[r]] for r in keep])
    kin_dim = CK.shape[1]
    utypes = sorted(set(types))

    PRED, o = {}, {'p': [], 'y': [], 'p0': [], 'rp': [], 'rp0': [], 'ro': [], 'rho': [], 'bt': [], 'fl': []}
    for r_ in range(a.draws):
        hp, ht = unified_split(r_, utypes, J)
        keepJ = np.array([j for j in range(J) if j not in hp])
        te = np.isin(types, list(ht))
        tr = ~te
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        th_f, _ = rasch(Y[keepJ][:, tr])                       # 13-planner theta (fail param.)
        mu, sd = CK[tr].mean(0), CK[tr].std(0) + 1e-9
        KZ = torch.tensor((CK - mu) / sd, dtype=torch.float32, device=dev)
        m = build(torch, nn, d=a.d, kin_dim=kin_dim).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)
        THE = torch.tensor(th_f, dtype=torch.float32, device=dev)
        idx = np.where(tr)[0]
        Yd = torch.tensor(np.nan_to_num(Y[keepJ]), dtype=torch.float32, device=dev)
        Md = torch.tensor((~np.isnan(Y[keepJ])).astype(np.float32), device=dev)

        def fwd(sel):
            return m(torch.tensor(AG[sel], dtype=torch.float32, device=dev),
                     torch.tensor(AM[sel], device=dev),
                     torch.tensor(EG[sel], dtype=torch.float32, device=dev),
                     torch.tensor(CM[sel], dtype=torch.float32, device=dev),
                     torch.tensor(WM[sel], device=dev),
                     kf=KZ[torch.tensor(sel, device=dev)])

        for _ep in range(a.epochs):
            m.train()
            np.random.shuffle(idx)
            for i0 in range(0, len(idx), BATCH):
                sel = idx[i0:i0 + BATCH]
                bt, _ = fwd(sel)
                p = torch.sigmoid(bt[None, :] - THE[:, None])
                yy = Yd[:, torch.tensor(sel, device=dev)]
                mm = Md[:, torch.tensor(sel, device=dev)]
                loss = (-(yy * torch.log(p + 1e-7) + (1 - yy) * torch.log(1 - p + 1e-7)) * mm).sum() / mm.sum()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
        m.eval()
        with torch.no_grad():
            bt_te, _ = fwd(np.where(te)[0])
        bt_te = bt_te.cpu().numpy()
        th0 = theta_null(torch, 1 - Y[keepJ][:, tr], dev)
        fail_obs = np.nanmean(Y[keepJ][:, te], 0)
        Yk = Y[keepJ]
        for k, ri in enumerate(np.where(te)[0]):
            js = np.where(~np.isnan(Yk[:, ri]))[0]
            if len(js) == 0:
                continue
            ps = sig(th_f[js] - bt_te[k])
            ys = 1 - Yk[js, ri]
            p0 = sig(th0[js])
            o['p'] += ps.tolist(); o['y'] += ys.tolist(); o['p0'] += p0.tolist()
            o['rp'].append(float(ps.mean())); o['rp0'].append(float(p0.mean())); o['ro'].append(float(ys.mean()))
        o['rho'].append(float(spearmanr(bt_te, fail_obs).correlation))
        o['bt'] += bt_te.tolist(); o['fl'] += fail_obs.tolist()
        PRED[f'draw{r_}_bt'] = bt_te
        PRED[f'draw{r_}_rt'] = np.array([keep[i] for i in np.where(te)[0]])
        print(f'draw {r_} done  rho {o["rho"][-1]:+.3f}', flush=True)

    au = roc_auc_score(o['y'], o['p']); a0 = roc_auc_score(o['y'], o['p0'])
    rp, rp0, ro = map(np.array, (o['rp'], o['rp0'], o['ro']))
    mm_, m0 = float(np.mean(np.abs(rp - ro))), float(np.mean(np.abs(rp0 - ro)))
    rh = np.array(o['rho']); pooled = float(spearmanr(o['bt'], o['fl']).correlation)
    print(f'\nencoder d{a.d} seed {a.seed}: AUROC {au:.3f} (null {a0:.3f}) MAE {mm_:.3f} (null {m0:.3f}) '
          f'rho per-draw {rh.mean():+.3f}+-{rh.std(ddof=1) / np.sqrt(len(rh)) if len(rh) > 1 else 0:.3f}  pooled {pooled:+.3f}')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'd{a.d}s{a.seed}' + ('' if a.draws == R_DRAWS and a.epochs == EPOCHS else f'_draws{a.draws}_ep{a.epochs}')
    json.dump({'unified': {'auroc': au, 'auroc0': a0, 'mae': mm_, 'mae0': m0,
                           'rho_perdraw': rh.tolist(), 'rho_pooled': pooled}},
              open(out / f'unified_us_encoder_{tag}.json', 'w'), indent=1)
    np.savez(out / f'unified_enc_pred_{tag}.npz', **PRED)
    print(f'saved {out / f"unified_enc_pred_{tag}.npz"}')


if __name__ == '__main__':
    main()
