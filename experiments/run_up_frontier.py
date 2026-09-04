#!/usr/bin/env python3
"""Table 1 — UP fixed-budget accuracy on the primary protocol.

16-planner panel, 12 calibration : 4 evaluation planners per draw (random,
RandomState(1000 + draw)); the bank is the whole 220-route benchmark (no
scenario-type hold-out — US / UPS keep the 36:8 type split). Budgets
B = number of routes rolled out, 5 x {6, 11, 22, 33} = {30, 55, 110, 165}
(14 / 25 / 50 / 75% of the benchmark), calibration-panel sizes K_cal in
{4, 8, 12} (RandomState(9000 + 100 draw + 10 K_cal) subsample, bank
re-calibrated from those planners only). 64 evaluations per cell (4 evaluation planners x 16 draws).

Baselines use their native readouts; DriveAT selects by Delta-R1 (the same
posterior L1 risk the stopping rule uses) and reads out the posterior median
of the full-bank SR (driveat.bayes.readout; PROTOCOL sections 2-5).

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
from driveat.b2d import Panel
from driveat.splits import up_split, R_DRAWS
from driveat.calibration import calibrate
from driveat.bayes import bank_from_fit, readout
from driveat.acquisition import r1_traj
from driveat.baselines import (fluid_order, total_fisher_order, marginal_fisher_order, disco_order,
                             kmeans_anchors, metabench_order, anchorpoints_estimate,
                             stratified_order, pirt)
from driveat.metrics import paired_cluster_boot

OUT = Path(os.environ.get('DRIVEAT_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
KCALS = tuple(int(x) for x in os.environ.get('DRIVEAT_KCALS', '4,8,12').split(','))
BGRID = [30, 55, 110, 165]
NREP = 5          # random-policy rows: expected |error| over NREP independent orders per evaluation
T = max(BGRID)
METHODS = ['Random (IRT-free)', 'Random + IRT', 'Random-strat + IRT', 'DISCO', 'AnchorPoints',
           'Total-Fisher', 'Marginal-Fisher', 'tinyBenchmarks', 'metabench', 'Fluid', 'DriveAT']


def subsample(cols, seed, Kc):
    if Kc >= len(cols):
        return list(cols)
    rs = np.random.RandomState(9000 + seed * 100 + Kc * 10 + 0)
    return sorted(np.array(cols)[rs.choice(len(cols), Kc, replace=False)].tolist())




def run(seeds):
    panel = Panel()
    recs = []
    for seed in seeds:
        hp, ht = up_split(seed, panel.utypes, panel.J)
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
                RNG = [np.random.RandomState(100 + seed * panel.J + js + 100000 * rep) for rep in range(NREP)]
                perms = [list(r_.permutation(n)) for r_ in RNG]
                strats = [stratified_order(typ[bi], r_) for r_ in RNG]
                orders = {'disco': disco_order(pbar[bi]), 'tf': total_fisher_order(a2, b2, f2['th']),
                          'mf': marginal_fisher_order(a2, b2), 'fluid': fluid_order(a2, b2, yy, T),
                          'ours': r1_traj(bank, yy, T)}
                err = {m: {} for m in METHODS}
                for B in BGRID:
                    est = {
                        'Random (IRT-free)': np.mean([yy[pm[:B]].mean() for pm in perms]),
                        'Random + IRT': np.mean([pirt(b1, ones, yy, pm[:B]) for pm in perms]),
                        'Random-strat + IRT': np.mean([pirt(b1, ones, yy, st[:B]) for st in strats]),
                        'DISCO': pirt(b2, a2, yy, orders['disco'][:B]),
                        'AnchorPoints': anchorpoints_estimate(Rf[bi], yy, B),
                        'Total-Fisher': pirt(b2, a2, yy, orders['tf'][:B]),
                        'Marginal-Fisher': pirt(b2, a2, yy, orders['mf'][:B]),
                        'tinyBenchmarks': pirt(b2, a2, yy, kmeans_anchors(a2, b2, B, n)),
                        'metabench': pirt(b2, a2, yy, metabench_order(a2, b2, B, n)),
                        'Fluid': pirt(b2, a2, yy, orders['fluid'][:B]),
                        'DriveAT': readout(bank, yy, orders['ours'][:B]),
                    }
                    for m in METHODS:
                        err[m][B] = abs(float(est[m]) - SR)
                    for m, ests in (('Random (IRT-free)', [yy[pm[:B]].mean() for pm in perms]),
                                    ('Random + IRT', [pirt(b1, ones, yy, pm[:B]) for pm in perms]),
                                    ('Random-strat + IRT', [pirt(b1, ones, yy, st[:B]) for st in strats])):
                        err[m][B] = float(np.mean([abs(e - SR) for e in ests]))   # expected error of the random policy
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
    print(f'\n{len(recs)} planner evaluations ({len(recs) // len(KCALS)} per K_cal = per cell)')
    print('\n===== Table 1: SR-MAE, K_cal x B; * = paired 95% CI vs DriveAT excludes 0 =====')
    print(f'{"method":20s} ' + ' '.join(f'K{K}B{B:<3d}' for K, B in cells) + '   macro')
    for m in METHODS:
        row = []
        for K, B in cells:
            v = np.mean(E[K][m][B])
            star = ''
            if m != 'DriveAT':
                d, lo, hi = paired_cluster_boot(E[K][m][B], E[K]['DriveAT'][B], J[K])
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
    assert len(recs) == len(KCALS) * 64
    macro = np.mean([np.mean(E[K]['DriveAT'][B]) for K in KCALS for B in BGRID])
    for (K, B, ref) in ((4, 30, .0450), (4, 55, .0332), (8, 110, .0202), (12, 55, .0231), (12, 165, .0081)):
        assert abs(np.mean(E[K]['DriveAT'][B]) - ref) < .002, (K, B, np.mean(E[K]['DriveAT'][B]))
    assert abs(np.mean(E[4]['Fluid'][30]) - .0653) < .002
    assert abs(np.mean(E[12]['Random-strat + IRT'][55]) - .0332) < .002
    assert abs(np.mean(E[4]['Random (IRT-free)'][30]) - .0623) < .002
    macro = np.mean([np.mean(E[K]['DriveAT'][B]) for K in KCALS for B in BGRID])
    assert abs(macro - .0262) < .0005, macro
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
