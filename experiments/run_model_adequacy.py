#!/usr/bin/env python3
"""Psychometric model adequacy on the calibration block (ATLAS reports M2 /
RMSEA; with 13 raters we use what is identifiable): held-out cell
negative log-likelihood of 1PL vs 2PL vs 3PL, and the split-half
reliability of log-discrimination across two random halves of the
calibration planners. If 2PL/3PL do not predict held-out responses better
than the Rasch model and a-hat is unreliable, the 1PL is the adequate
model, not a simplification.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirt.b2d import Panel
from scirt.splits import unified_split, R_DRAWS
from scirt.calibration import calibrate
from scirt.curves import sig
from scirt.metrics import mean_se

np.random.seed(0)
torch.manual_seed(0)
OUT = Path(__file__).resolve().parents[1] / 'results'


def cell_nll(f, mode, i, k, y):
    base = sig(f['a'][i] * (f['th'][k] - f['b'][i]))
    p = f['cc'][i] + (1 - f['cc'][i]) * base if mode == '3pl' else base
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    panel = Panel()
    NLL = {m: [] for m in ('1pl', '2pl', '3pl')}
    REL = []
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        cells = [(r, c) for r in calR for c in cols if (r, c) in panel.Y]
        rng = np.random.RandomState(2000 + seed)
        held = set(rng.choice(len(cells), int(0.1 * len(cells)), replace=False).tolist())
        Ytr = {cells[i]: panel.Y[cells[i]] for i in range(len(cells)) if i not in held}
        ridx = {r: i for i, r in enumerate(calR)}
        cidx = {c: k for k, c in enumerate(cols)}
        for mode in NLL:
            f = calibrate(Ytr, calR, cols, mode=mode)
            NLL[mode].append(float(np.mean([cell_nll(f, mode, ridx[cells[i][0]], cidx[cells[i][1]],
                                                     panel.Y[cells[i]]) for i in held])))
        # split-half reliability of log a-hat (two random halves of the panel)
        half = rng.permutation(len(cols))
        h1 = sorted(np.array(cols)[half[:len(cols) // 2]].tolist())
        h2 = sorted(np.array(cols)[half[len(cols) // 2:]].tolist())
        g1 = calibrate(panel.Y, calR, h1, mode='2pl')
        g2 = calibrate(panel.Y, calR, h2, mode='2pl')
        REL.append(float(np.corrcoef(np.log(g1['a'] + 1e-9), np.log(g2['a'] + 1e-9))[0, 1]))
        print(f'seed {seed} done', flush=True)
    print('\n===== held-out cell NLL on the calibration block (10% cells, 16 draws) =====')
    for m in NLL:
        v, se = mean_se(NLL[m])
        print(f'  {m}: {v:.4f} +- {se:.4f}')
    d = np.array(NLL['2pl']) - np.array(NLL['1pl'])
    print(f'  2PL - 1PL: {d.mean():+.4f} +- {d.std(ddof=1) / 4:.4f}   '
          f'3PL - 1PL: {(np.array(NLL["3pl"]) - np.array(NLL["1pl"])).mean():+.4f}')
    r, se = mean_se(REL)
    print(f'\nsplit-half reliability of log a-hat (6/7-planner halves): {r:+.3f} +- {se:.3f}')
    OUT.mkdir(exist_ok=True)
    json.dump({'nll': {m: list(map(float, NLL[m])) for m in NLL}, 'rel_loga': list(map(float, REL))},
              open(OUT / 'model_adequacy.json', 'w'))


if __name__ == '__main__':
    main()
