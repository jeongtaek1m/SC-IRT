#!/usr/bin/env python3
"""Table 1 — UP fixed-budget accuracy on the primary protocol.

22-planner panel, 16 calibration : 6 evaluation planners per draw (random,
RandomState(1000 + draw)), budgets B = number of routes rolled out,
5 x {6, 11, 22} = {30, 55, 110} (so the type-stratified baseline can execute
whole scenario types), calibration-panel sizes K_cal in {7, 10, 16}
(K_cal < 16: RandomState(9000 + 100 draw + 10 K_cal) subsample, bank
re-calibrated from those planners only). 96 evaluations per cell.

Baselines use their native readouts; SC-IRT selects by Delta-R1 (the same
posterior L1 risk the stopping rule uses) and reads out with the Rasch MAP
fill (PROTOCOL sections 2-5).

    python experiments/run_up_frontier.py                  # all 16 draws (GPU)
    python experiments/run_up_frontier.py --seeds 0 4      # shard
    python experiments/run_up_frontier.py --merge          # merge, tables, anchors
"""
import argparse
import glob
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
from scirt.acquisition import r1_traj
from scirt.baselines import (fluid_order, total_fisher_order, marginal_fisher_order, disco_order,
                             kmeans_anchors, metabench_order, anchorpoints_estimate,
                             stratified_order, pirt)
from scirt.metrics import paired_cluster_boot

OUT = Path(os.environ.get('SCIRT_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
KCALS = tuple(int(x) for x in os.environ.get('SCIRT_KCALS', '7,10,16').split(','))
BGRID = [30, 55, 110]
T = max(BGRID)
METHODS = ['Random (IRT-free)', 'Random + IRT', 'Random-strat + IRT', 'DISCO', 'AnchorPoints',
           'Total-Fisher', 'Marginal-Fisher', 'tinyBenchmarks', 'metabench', 'Fluid', 'SC-IRT']


def subsample(cols, seed, Kc):
    if Kc >= len(cols):
        return list(cols)
    rs = np.random.RandomState(9000 + seed * 100 + Kc * 10 + 0)
    return sorted(np.array(cols)[rs.choice(len(cols), Kc, replace=False)].tolist())




def run(seeds):
    panel = Panel()
    recs = []
    for seed in seeds:
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        typ = np.array([panel.sn[r] for r in calR])
        for Kc in KCALS:
            cs = subsample(cols, seed, Kc)
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
                SR = float(yy.mean())
                b1 = f1['b'][bi]
                a2, b2 = f2['a'][bi], f2['b'][bi]
                bank = bank_from_fit(f1, bi, typ)
                ones = np.ones(n)
                rng = np.random.RandomState(100 + seed * 20 + js)
                perm = list(rng.permutation(n))
                strat = stratified_order(typ[bi], rng)
                orders = {'disco': disco_order(pbar[bi]), 'tf': total_fisher_order(a2, b2, f2['th']),
                          'mf': marginal_fisher_order(a2, b2), 'fluid': fluid_order(a2, b2, yy, T),
                          'ours': r1_traj(bank, yy, T)}
                err = {m: {} for m in METHODS}
                for B in BGRID:
                    est = {
                        'Random (IRT-free)': yy[perm[:B]].mean(),
                        'Random + IRT': pirt(b1, ones, yy, perm[:B]),
                        'Random-strat + IRT': pirt(b1, ones, yy, strat[:B]),
                        'DISCO': pirt(b2, a2, yy, orders['disco'][:B]),
                        'AnchorPoints': anchorpoints_estimate(Rf[bi], yy, B),
                        'Total-Fisher': pirt(b2, a2, yy, orders['tf'][:B]),
                        'Marginal-Fisher': pirt(b2, a2, yy, orders['mf'][:B]),
                        'tinyBenchmarks': pirt(b2, a2, yy, kmeans_anchors(a2, b2, B, n)),
                        'metabench': pirt(b2, a2, yy, metabench_order(a2, b2, B, n)),
                        'Fluid': pirt(b2, a2, yy, orders['fluid'][:B]),
                        'SC-IRT': readout(bank, yy, orders['ours'][:B]),
                    }
                    for m in METHODS:
                        err[m][B] = abs(float(est[m]) - SR)
                recs.append({'seed': seed, 'K': Kc, 'js': int(js), 'SR': SR,
                             'sel': [calR[bi[i]] for i in orders['ours']],
                             'sigma_b': f1['sigma_b'], 'sigma_g': f1['sigma_g'],
                             'err': {m: {str(B): err[m][B] for B in BGRID} for m in METHODS}})
            print(f'seed {seed} K{Kc} done', flush=True)
    return recs


def report(recs):
    recs = sorted(recs, key=lambda r: (r['seed'], r['K'], r['js']))
    E = {K: {m: {B: [r['err'][m][str(B)] for r in recs if r['K'] == K] for B in BGRID} for m in METHODS}
         for K in KCALS}
    J = {K: [r['js'] for r in recs if r['K'] == K] for K in KCALS}
    cells = [(K, B) for K in KCALS for B in BGRID]
    print(f'\n{len(recs)} planner evaluations ({len(recs) // len(KCALS)} per K_cal, 96 per cell)')
    print('\n===== Table 1: SR-MAE, K_cal x B; * = paired 95% CI vs SC-IRT excludes 0 =====')
    print(f'{"method":20s} ' + ' '.join(f'K{K}B{B:<3d}' for K, B in cells) + '   macro')
    for m in METHODS:
        row = []
        for K, B in cells:
            v = np.mean(E[K][m][B])
            star = ''
            if m != 'SC-IRT':
                d, lo, hi = paired_cluster_boot(E[K][m][B], E[K]['SC-IRT'][B], J[K])
                star = '*' if (lo > 0 or hi < 0) else ' '
            row.append(f'{v:.4f}{star}')
        print(f'{m:20s} ' + ' '.join(f'{c:>8s}' for c in row)
              + f'   {np.mean([np.mean(E[K][m][B]) for K, B in cells]):.4f}')
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs=2, type=int, default=None)
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.merge:
        recs = sum([json.load(open(f)) for f in sorted(glob.glob(str(OUT / 'up_frontier_*_*.json')))], [])
        json.dump(recs, open(OUT / 'up_frontier.json', 'w'))
    elif args.seeds:
        lo, hi = args.seeds
        recs = run(range(lo, hi))
        json.dump(recs, open(OUT / f'up_frontier_{lo}_{hi}.json', 'w'))
        print('shard saved; run with --merge after all shards')
        return
    else:
        recs = run(range(R_DRAWS))
        json.dump(recs, open(OUT / 'up_frontier.json', 'w'))
    E = report(recs)
    assert len(recs) == len(KCALS) * 96
    macro = np.mean([np.mean(E[K]['SC-IRT'][B]) for K in KCALS for B in BGRID])
    for (K, B, ref) in ((7, 30, .0492), (7, 55, .0296), (10, 110, .0137), (16, 55, .0317)):
        assert abs(np.mean(E[K]['SC-IRT'][B]) - ref) < .002, (K, B, np.mean(E[K]['SC-IRT'][B]))
    assert abs(np.mean(E[7]['Fluid'][30]) - .0515) < .002
    assert abs(np.mean(E[16]['Random-strat + IRT'][55]) - .0352) < .002
    assert abs(np.mean(E[7]['Random (IRT-free)'][30]) - .0670) < .002
    macro = np.mean([np.mean(E[K]['SC-IRT'][B]) for K in KCALS for B in BGRID])
    assert abs(macro - .0307) < .0005, macro
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
