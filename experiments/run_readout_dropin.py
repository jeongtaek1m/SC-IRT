#!/usr/bin/env python3
"""Analysis — the Rasch readout as a drop-in for any selector (contribution C1).

Every published selector's subsets are re-scored with ATDrive's readout
(exact difficulty posteriors, testlet, posterior median) instead of the selector's
native estimator; compared with the native numbers of `run_up_frontier.py`
this isolates what the uncertainty-aware inference layer contributes,
independently of how scenes were chosen. The gain is largest under
calibration scarcity (K_cal = 4).

ATDRIVE_OFFICIAL_ORDERS=1 takes the selector subsets from the METHODS' OWN code
(experiments/official/orders.py): Fluid's and catR's Total-Fisher orders, and
metabench's / tinyBenchmarks' / AnchorPoints' per-budget subsets -- those three
select a different set per budget and have no order, which is why the API hands
back subsets for them. Only Random stays ours (it is the protocol's control, not
a published method). The readout, the bank and the budgets are untouched; the
result goes to results/readout_dropin_official.json and the native-readout
comparison then reads results/up_official.json instead of Table 1's.

    python experiments/run_readout_dropin.py            # ~40 min, GPU
    ATDRIVE_OFFICIAL_ORDERS=1 python experiments/run_readout_dropin.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atdrive.b2d import Panel
from atdrive.splits import up_split, R_DRAWS
from atdrive.calibration import calibrate
from atdrive.bayes import bank_from_fit, readout
from atdrive.baselines import (fluid_order, total_fisher_order, metabench_order, kmeans_anchors,
                             anchorpoints_select)

OFFICIAL = os.environ.get('ATDRIVE_OFFICIAL_ORDERS', '0') == '1'   # subsets from the official code
TAG = '_official' if OFFICIAL else ''
if OFFICIAL:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from official.orders import order as official_order, subsets as official_subsets, padded, slot_of

OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
KCALS = tuple(int(x) for x in os.environ.get('ATDRIVE_KCALS', '4,8,12').split(','))
BG = (30, 55, 110, 165)
SELS = ('Fluid', 'Total-Fisher', 'metabench', 'tinyBenchmarks', 'AnchorPoints', 'Random')
# SELS -> the official-code method that owns that selector (Random has none: it is the protocol's control).
# 'fluid' / 'total_fisher' are prefix orders; the other three select a budget-specific subset.
OFFICIAL_OF = {'Fluid': 'fluid', 'Total-Fisher': 'total_fisher', 'metabench': 'metabench',
               'tinyBenchmarks': 'tinybench', 'AnchorPoints': 'anchorpoints'}


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
        hp, ht = up_split(seed, panel.utypes, panel.J)
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
                if OFFICIAL:
                    p_ = slot_of(seed, js)
                    fl = padded(official_order('fluid', seed, Jc, p_, n_bank=n)['order'], n)
                    tf = padded(official_order('total_fisher', seed, Jc, p_, n_bank=n)['order'], n)
                    SUB = {s: official_subsets(OFFICIAL_OF[s], seed, Jc, p_, BG, n_bank=n)
                           for s in ('metabench', 'tinyBenchmarks', 'AnchorPoints')}
                else:
                    fl = fluid_order(a2, b2, yy, max(BG))
                    tf = total_fisher_order(a2, b2, f2['th'])
                perm = list(np.random.RandomState(100 + seed * panel.J + js).permutation(n))
                for B in BG:
                    sets = ({'Fluid': fl[:B], 'Total-Fisher': tf[:B], 'Random': perm[:B],
                             **{s: SUB[s][B] for s in SUB}} if OFFICIAL else
                            {'Fluid': fl[:B], 'Total-Fisher': tf[:B], 'metabench': metabench_order(a2, b2, f2['th'], B, n),
                             'tinyBenchmarks': kmeans_anchors(a2, b2, B, n), 'AnchorPoints': anchor_set(Rf[bi], B),
                             'Random': perm[:B]})
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
    if OFFICIAL:                # the native number of a swapped row is the official code's own, not Table 1's
        off = json.load(open(OUT / 'up_official.json'))['methods']
        for s, m in OFFICIAL_OF.items():
            mm = 'catr' if m in ('total_fisher', 'marginal_fisher') else m
            nat[s] = {J: {B: off[mm]['cells'][f'K{J}_B{B}']['sr_mae'] for B in BG} for J in KCALS}
    print('\n===== selector subsets re-scored with the ATDrive readout (native readout in parentheses) =====')
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
              open(OUT / f'readout_dropin{TAG}.json', 'w'))
    for sel, J, B, v in (('Fluid', 4, 30, 0.0536), ('AnchorPoints', 12, 110, 0.0217), ('AnchorPoints', 12, 55, 0.0396),
                         ('Random', 8, 110, 0.0239), ('Fluid', 12, 55, 0.0290), ('tinyBenchmarks', 4, 165, 0.0139)):
        m = float(np.mean(ERR[sel][J][B]))
        if OFFICIAL and sel != 'Random':     # only the Random row keeps its subsets in the official-order arm
            print(f'   official-subset arm: {sel} K{J} B{B} = {m:.4f} (was {v:.4f} on our subsets)')
            continue
        assert abs(m - v) < .0003, (sel, J, B, m)
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
