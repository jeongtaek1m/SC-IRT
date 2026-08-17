#!/usr/bin/env python3
"""LOTO training of the interaction encoder on the Bench2Drive panel.

One invocation = one (width, epochs, seed) run: 44 scenario-type folds, a fresh
encoder per fold, theta frozen per fold by the a==1 calibration, BCE over the
16-planner response cells, out-of-fold b_tilde written for every route. The
released artifact data/interact/interact_b2d_w2a_final.npz is the seed-ensemble
of six such runs:

  for d,ep in (64,30) (96,60):
    for s in 0 1 2:
      python train/train_encoder_b2d.py --tensors <b2d_tensors.npz> \
          --d $d --epochs $ep --seed $s --out runs/kcat_d${d}e${ep}s${s}.npz
  python train/assemble_ensemble.py runs/kcat_*.npz --out interact_b2d_w2a_final.npz

Reproducibility tier: GPU training. Unlike the CPU-pinned evaluation package,
retraining is NOT bit-reproducible across devices/cuDNN builds; the measured
seed spread of the pooled Spearman is ~0.01 and the shipped npz remains the
reference artifact. Windows are the first W_MAX of each route, as in the runs
that produced the artifact.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scirt import paths                    # noqa: E402
from scirt.encoder import build, rasch     # noqa: E402

W_MAX = 22


def load_panel(matrix_csv, types_csv, tensors):
    d = np.load(tensors, allow_pickle=True)
    routes = [str(x) for x in d["route"]]
    rid2w = {}
    for i, r in enumerate(routes):
        rid2w.setdefault(r.replace("route_", ""), []).append(i)
    rows = list(csv.reader(open(matrix_csv)))
    rids = rows[0][1:]
    data = [r for r in rows[1:] if r[0] != "PDM-Lite"]     # expert never rates itself
    Y = np.full((len(data), len(rids)), np.nan)
    for pi, row in enumerate(data):
        for j in range(len(rids)):
            if row[1 + j] != "":
                Y[pi, j] = 1.0 - float(row[1 + j])          # fail = 1
    types = dict(csv.reader(open(types_csv)))
    keep = [r for r in sorted(rid2w) if r in types and r in rids]
    col = {rid: j for j, rid in enumerate(rids)}
    return (d, rid2w, keep, Y[:, [col[r] for r in keep]],
            np.array([types[r] for r in keep]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors", required=True, help="output of build_tensors_b2d.py")
    ap.add_argument("--matrix", default=os.path.join(paths.MATRICES, "b2d_e2e16_response_matrix.csv"))
    ap.add_argument("--route_types", default=os.path.join(paths.MATRICES, "b2d_route_types.csv"))
    ap.add_argument("--kin_feats", default=os.path.join(paths.FEATURES, "eval_cmdkin_stats.npz"),
                    help="route-level ego-kinematics npz {names, stats}; the kin-embedded input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    import torch.nn as nn
    from scipy.stats import spearmanr

    dev = a.device
    d, rid2w, keep, Y, types = load_panel(a.matrix, a.route_types, a.tensors)
    _, reference = rasch(Y, it=800)
    R = len(keep)
    AG = np.zeros((R, W_MAX, 48, 12, 8), np.float16)
    AM = np.zeros((R, W_MAX, 48, 12), bool)
    EG = np.zeros((R, W_MAX, 12, 6), np.float16)
    CM = np.zeros((R, W_MAX, 4), np.float16)
    WM = np.zeros((R, W_MAX), bool)
    for ri, r in enumerate(keep):
        ws = rid2w[r]
        if len(ws) > W_MAX:
            ws = ws[:W_MAX]
        AG[ri, :len(ws)] = d["agents"][ws]; AM[ri, :len(ws)] = d["amask"][ws]
        EG[ri, :len(ws)] = d["ego"][ws]; CM[ri, :len(ws)] = d["cmd"][ws]
        WM[ri, :len(ws)] = True
    z = np.load(a.kin_feats, allow_pickle=True)
    nm = {str(x).replace("route_", ""): i for i, x in enumerate(z["names"])}
    CK = np.stack([z["stats"][nm[r]] for r in keep])
    kin_dim = CK.shape[1]
    pred = np.full(R, np.nan)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    utypes = sorted(set(types))
    for fi, t in enumerate(utypes):
        te = types == t; tr = ~te
        th_f, _ = rasch(Y[:, tr])
        # NOTE rasch() resets the torch RNG; per-fold init is therefore common
        # across seeds and the seed enters through data order and dropout draws.
        # This mirrors the runs that produced the released artifact exactly.
        mu, sd = CK[tr].mean(0), CK[tr].std(0) + 1e-9
        KZ = torch.tensor((CK - mu) / sd, dtype=torch.float32, device=dev)
        m = build(torch, nn, d=a.d, kin_dim=kin_dim).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)
        THE = torch.tensor(th_f, dtype=torch.float32, device=dev)
        idx = np.where(tr)[0]
        Yd = torch.tensor(np.nan_to_num(Y), dtype=torch.float32, device=dev)
        Md = torch.tensor((~np.isnan(Y)).astype(np.float32), device=dev)

        def fwd(sel):
            return m(torch.tensor(AG[sel], dtype=torch.float32, device=dev),
                     torch.tensor(AM[sel], device=dev),
                     torch.tensor(EG[sel], dtype=torch.float32, device=dev),
                     torch.tensor(CM[sel], dtype=torch.float32, device=dev),
                     torch.tensor(WM[sel], device=dev),
                     kf=KZ[torch.tensor(sel, device=dev)])

        for _ep in range(a.epochs):
            m.train(); np.random.shuffle(idx)
            for i0 in range(0, len(idx), a.batch):
                sel = idx[i0:i0 + a.batch]
                bt, _ = fwd(sel)
                p = torch.sigmoid(bt[None, :] - THE[:, None])
                yy = Yd[:, torch.tensor(sel, device=dev)]
                mm = Md[:, torch.tensor(sel, device=dev)]
                loss = (-(yy * torch.log(p + 1e-7)
                          + (1 - yy) * torch.log(1 - p + 1e-7)) * mm).sum() / mm.sum()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
        m.eval()
        with torch.no_grad():
            bt, _ = fwd(np.where(te)[0])
            pred[te] = bt.cpu().numpy()
        if (fi + 1) % 8 == 0:
            print(f"  ...fold {fi+1}/{len(utypes)}", flush=True)
    ok = ~np.isnan(pred)
    print(f"RESULT d={a.d} ep={a.epochs} seed={a.seed} "
          f"rho {spearmanr(pred[ok], reference[ok]).correlation:+.4f}", flush=True)
    # route ids carry the route_ prefix, matching the released artifact layout.
    np.savez(a.out, pred_m=pred, reference=reference,
             routes=np.array(["route_" + r for r in keep]), types=types)


if __name__ == "__main__":
    main()
