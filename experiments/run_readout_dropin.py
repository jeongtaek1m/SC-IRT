#!/usr/bin/env python3
"""Analysis — the Rasch readout as a drop-in for any selector (contribution C1).

Every published selector's subsets are re-scored with SC-IRT's readout
(exact difficulty posteriors, testlet, posterior median) instead of the selector's
native estimator; compared with the native numbers of `run_up_frontier.py`
this isolates what the uncertainty-aware inference layer contributes,
independently of how scenes were chosen. The gain is largest under
calibration scarcity (K_cal = 7).

    python experiments/run_readout_dropin.py            # ~40 min, GPU
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirt.b2d import Panel
from scirt.splits import unified_split, R_DRAWS
from scirt.calibration import calibrate
from scirt.bayes import bank_from_fit, readout
from scirt.baselines import (fluid_order, total_fisher_order, metabench_order, kmeans_anchors,
                             anchorpoints_select)

OUT = Path(os.environ.get('SCIRT_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
KCALS = tuple(int(x) for x in os.environ.get('SCIRT_KCALS', '7,10,16').split(','))
BG = (30, 55, 110)
SELS = ('Fluid', 'Total-Fisher', 'metabench', 'tinyBenchmarks', 'AnchorPoints', 'Random')


def subsample(cols, seed, Jc):
    if Jc >= len(cols):
        return list(cols)
    rs = np.random.RandomState(9000 + seed * 100 + Jc * 10 + 0)
    return sorted(np.array(cols)[rs.choice(len(cols), Jc, replace=False)].tolist())


def anchor_set(Rb, budget):
    return anchorpoints_select(Rb, budget)[0]


def main():
    panel = Panel()
    ERR = {s: {J: {B: [] for B in BG} for J in KCALS} for s in SELS}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        typ = np.array([panel.sn[r] for r in calR])
        for Jc in KCALS:
            cs = subsample(cols, seed, Jc)
            f1 = calibrate(panel.Y, calR, cs, mode='1pl', types=typ)
            f2 = calibrate(panel.Y, calR, cs, mode='2pl', sigma_b=f1['sigma_b'])
            R = np.full((len(calR), len(cs)), np.nan)
            for a_, rid in enumerate(calR):
                for b_, pi in enumerate(cs):
                    if (rid, pi) in panel.Y:
                        R[a_, b_] = panel.Y[(rid, pi)]
            pbar = np.nanmean(R, 1)
            pbar = np.where(np.isnan(pbar), np.nanmean(R), pbar)
            Rf = np.where(np.isnan(R), pbar[:, None], R)
            for js in hp:
                bi, yy = panel.bank_rows(calR, js)
                n = len(bi)
                SR = yy.mean()
                bank = bank_from_fit(f1, bi, typ)
                a2, b2 = f2['a'][bi], f2['b'][bi]
                fl = fluid_order(a2, b2, yy, max(BG))
                tf = total_fisher_order(a2, b2, f2['th'])
                perm = list(np.random.RandomState(100 + seed * panel.J + js).permutation(n))
                for B in BG:
                    sets = {'Fluid': fl[:B], 'Total-Fisher': tf[:B], 'metabench': metabench_order(a2, b2, B, n),
                            'tinyBenchmarks': kmeans_anchors(a2, b2, B, n), 'AnchorPoints': anchor_set(Rf[bi], B),
                            'Random': perm[:B]}
                    for s in SELS:
                        ERR[s][Jc][B].append(abs(readout(bank, yy, sets[s]) - SR))
        print(f'seed {seed} done', flush=True)
    nat = {}
    fp = OUT / 'up_frontier.json'
    if fp.exists():
        recs = json.load(open(fp))
        for s in SELS:
            key = {'Random': 'Random + IRT'}.get(s, s)
            nat[s] = {J: {B: np.mean([r['err'][key][str(B)] for r in recs if r['K'] == J]) for B in BG} for J in KCALS}
    print('\n===== selector subsets re-scored with the SC-IRT readout (native readout in parentheses) =====')
    for Jc in KCALS:
        print(f'-- K_cal = {Jc} --      ' + ' '.join(f'{B:>15d}' for B in BG))
        for s in SELS:
            cells = []
            for B in BG:
                v = np.mean(ERR[s][Jc][B])
                cells.append(f'{v:.4f} ({nat[s][Jc][B]:.4f})' if nat else f'{v:.4f}')
            print(f'   {s:15s} ' + ' '.join(f'{c:>15s}' for c in cells))
    OUT.mkdir(exist_ok=True)
    json.dump({s: {str(J): {str(B): [float(x) for x in ERR[s][J][B]] for B in BG} for J in KCALS} for s in SELS},
              open(OUT / 'readout_dropin.json', 'w'))
    for sel, J, B, v in (('Fluid', 7, 30, .0493), ('AnchorPoints', 16, 110, .0207), ('Random', 10, 110, .0171),
                         ('Fluid', 16, 55, .0345)):
        m = float(np.mean(ERR[sel][J][B]))
        assert abs(m - v) < .0003, (sel, J, B, m)
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
