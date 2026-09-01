#!/usr/bin/env python3
"""Diagnostic: how much do theta-hat and b-hat move between the full
22 x 220 panel fit and the per-draw calibration block (16 planners x ~180
routes)? Compared on the shared routes / planners. Both fits are
theta-mean-centred on *different* planner sets, so a constant offset is an
identification convention, not estimation change: raw and mean-aligned
differences are both reported, and b shifts are scaled by each route's own
posterior SD s_i.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirt.b2d import Panel
from scirt.splits import unified_split, R_DRAWS
from scirt.calibration import calibrate
from scirt.curves import posterior_sd

np.random.seed(0)
torch.manual_seed(0)
OUT = Path(os.environ.get('SCIRT_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))


def main():
    panel = Panel()
    allc = list(range(panel.J))
    full = calibrate(panel.Y, panel.allr, allc, mode='1pl')
    idx_full = {r: i for i, r in enumerate(panel.allr)}
    B, T = [], []
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        fA = calibrate(panel.Y, calR, cols, mode='1pl')
        bF = np.array([full['b'][idx_full[r]] for r in calR])
        bA, sA = fA['b'], posterior_sd(fA['W'])
        sFull = posterior_sd(full['W'])
        sF = np.array([sFull[idx_full[r]] for r in calR])
        off_b = float((bA - bF).mean())
        d = bA - bF - off_b
        B.append(dict(pearson=float(np.corrcoef(bA, bF)[0, 1]),
                      spearman=float(spearmanr(bA, bF).correlation),
                      offset=off_b, rmse_raw=float(np.sqrt(((bA - bF) ** 2).mean())),
                      rmse_aligned=float(np.sqrt((d ** 2).mean())),
                      max_aligned=float(np.abs(d).max()),
                      frac_gt_s=float((np.abs(d) > sA).mean()),
                      sd_b=float(bF.std()), mean_s_A=float(sA.mean()), mean_s_F=float(sF.mean())))
        tF = full['th'][cols]
        tA = fA['th']
        off_t = float((tA - tF).mean())
        dt = tA - tF - off_t
        T.append(dict(pearson=float(np.corrcoef(tA, tF)[0, 1]),
                      spearman=float(spearmanr(tA, tF).correlation),
                      offset=off_t, rmse_raw=float(np.sqrt(((tA - tF) ** 2).mean())),
                      rmse_aligned=float(np.sqrt((dt ** 2).mean())),
                      max_aligned=float(np.abs(dt).max()), sd_t=float(tF.std())))
        print(f'seed {seed} done', flush=True)

    def summ(L, k):
        v = np.array([x[k] for x in L])
        return f'{v.mean():+.3f} (min {v.min():+.3f}, max {v.max():+.3f})'

    print('\n===== b-hat: full 22x220 vs A-block 16x~180, shared routes (16 draws) =====')
    for k in ('pearson', 'spearman', 'offset', 'rmse_raw', 'rmse_aligned', 'max_aligned', 'frac_gt_s'):
        print(f'  {k:13s} {summ(B, k)}')
    print(f'  scale: sd(b-hat) {summ(B, "sd_b")}; mean s_i  A-block {summ(B, "mean_s_A")}  full {summ(B, "mean_s_F")}')
    print('\n===== theta-hat: full vs A-block, shared 16 planners =====')
    for k in ('pearson', 'spearman', 'offset', 'rmse_raw', 'rmse_aligned', 'max_aligned'):
        print(f'  {k:13s} {summ(T, k)}')
    print(f'  scale: sd(theta-hat) {summ(T, "sd_t")}')
    OUT.mkdir(exist_ok=True)
    json.dump({'b': B, 'theta': T}, open(OUT / 'calibration_stability.json', 'w'))


if __name__ == '__main__':
    main()
