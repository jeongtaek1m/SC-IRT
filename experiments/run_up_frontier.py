#!/usr/bin/env python3
"""Table 1 (+ full budget grids, Figure data) — UP fixed-budget accuracy.

Every method is calibrated from the same J_cal planners and scored at the
same rollout budgets B on the same bank; baselines use their native
readouts, SC-IRT the Rasch MAP-fill readout (PROTOCOL sections 2-4).

  J_cal in {4, 7, 10, 13}   (J_cal < 13: RandomState(9000 + 100*draw + 10*J_cal) subsample,
                             the bank is re-calibrated from those planners only)
  B     in {10, 20, 30, 40, 60, 80, 100, 120}

Primary protocol (Table 1): J_cal in {7, 10, 13} x B in {30, 60}, SR-MAE,
macro = mean of the six cells. The full grids feed the cost-error figure
and the J_cal x B map (`make_figures.py`).

    python experiments/run_up_frontier.py                  # all 16 draws (~1 h, GPU)
    python experiments/run_up_frontier.py --seeds 0 4      # shard -> results/up_frontier_0_4.json
    python experiments/run_up_frontier.py --merge          # merge shards, print tables, assert anchors
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
from scirt.curves import marginal_curves
from scirt.bayes import map_fill
from scirt.acquisition import localize_cover, K_LOCALIZE
from scirt.baselines import (fluid_order, total_fisher_order, marginal_fisher_order, disco_order,
                             kmeans_anchors, metabench_order, anchorpoints_estimate,
                             stratified_order, pirt)
from scirt.metrics import paired_seed_boot

OUT = Path(os.environ.get('SCIRT_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
JCALS = tuple(int(x) for x in os.environ.get('SCIRT_JCALS', '4, 7, 10, 13').split(','))
BGRID = [10, 20, 30, 40, 60, 80, 100, 120]
TMAX = max(BGRID)
METHODS = ['Random (IRT-free)', 'Random + IRT', 'Random-strat + IRT', 'DISCO', 'AnchorPoints',
           'Total-Fisher', 'Marginal-Fisher', 'tinyBenchmarks', 'metabench', 'Fluid', 'SC-IRT']
CELLS = [(J, B) for J in JCALS if J >= 7 for B in (30, 60)]


def subsample(cols, seed, Jc):
    if Jc >= len(cols):
        return list(cols)
    rs = np.random.RandomState(9000 + seed * 100 + Jc * 10 + 0)
    return sorted(np.array(cols)[rs.choice(len(cols), Jc, replace=False)].tolist())


def run(seeds):
    panel = Panel()
    recs = []
    for seed in seeds:
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        typ = np.array([panel.sn[r] for r in calR])
        for Jc in JCALS:
            cs = subsample(cols, seed, Jc)
            f1 = calibrate(panel.Y, calR, cs, mode='1pl')
            f2 = calibrate(panel.Y, calR, cs, mode='2pl')
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
                b1, s1 = f1['b'][bi], f1['s'][bi]
                a2, b2 = f2['a'][bi], f2['b'][bi]
                M1 = marginal_curves(b1, s1)
                ones = np.ones(n)
                rng = np.random.RandomState(100 + seed * 20 + js)
                perm = list(rng.permutation(n))
                strat = stratified_order(typ[bi], rng)
                orders = {'disco': disco_order(pbar[bi]), 'tf': total_fisher_order(a2, b2, f2['th']),
                          'mf': marginal_fisher_order(a2, b2), 'fluid': fluid_order(a2, b2, yy, TMAX),
                          'ours': localize_cover(a2, b2, f2['th'], yy, K=K_LOCALIZE, T=TMAX)}
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
                        'SC-IRT': map_fill(M1, yy, orders['ours'][:B]),
                    }
                    for m in METHODS:
                        err[m][B] = abs(float(est[m]) - SR)
                recs.append({'seed': seed, 'J': Jc, 'js': int(js), 'SR': SR,
                             'err': {m: {str(B): err[m][B] for B in BGRID} for m in METHODS}})
            print(f'seed {seed} J{Jc} done', flush=True)
    return recs


def report(recs):
    recs = sorted(recs, key=lambda r: (r['seed'], r['J'], r['js']))
    E = {J: {m: {B: [r['err'][m][str(B)] for r in recs if r['J'] == J] for B in BGRID} for m in METHODS}
         for J in JCALS}
    print(f'\n{len(recs)} planner evaluations ({len(recs) // len(JCALS)} per J_cal)')
    print('\n===== Table 1: UP fixed-budget SR-MAE, J_cal in {7,10,13} x B in {30,60}; * = paired 95% CI vs SC-IRT excludes 0 =====')
    print(f'{"method":20s} ' + ' '.join(f'J{J}B{B:<3d}' for J, B in CELLS) + '   macro')
    for m in METHODS:
        cells = []
        for J, B in CELLS:
            v = np.mean(E[J][m][B])
            star = ''
            if m != 'SC-IRT':
                d, lo, hi = paired_seed_boot(E[J][m][B], E[J]['SC-IRT'][B])
                star = '*' if (lo > 0 or hi < 0) else ' '
            cells.append(f'{v:.4f}{star}')
        print(f'{m:20s} ' + ' '.join(f'{c:>8s}' for c in cells)
              + f'   {np.mean([np.mean(E[J][m][B]) for J, B in CELLS]):.4f}')
    for J in JCALS:
        print(f'\n-- full grid J_cal = {J} --   ' + ' '.join(f'{B:>6d}' for B in BGRID))
        for m in METHODS:
            print(f'{m:20s} ' + ' '.join(f'{np.mean(E[J][m][B]):6.4f}' for B in BGRID))
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
    assert len(recs) == 4 * 48
    for (J, B, ref) in ((13, 30, .0388), (13, 40, .0342), (13, 60, .0213), (13, 80, .0222),
                        (7, 30, .0423), (10, 60, .0260)):
        assert abs(np.mean(E[J]['SC-IRT'][B]) - ref) < .002, (J, B, np.mean(E[J]['SC-IRT'][B]))
    assert abs(np.mean(E[13]['Fluid'][30]) - .0379) < .002
    assert abs(np.mean(E[13]['metabench'][80]) - .0217) < .002
    assert abs(np.mean(E[13]['Random + IRT'][60]) - .0345) < .002
    macro = np.mean([np.mean(E[J]['SC-IRT'][B]) for J, B in CELLS])
    assert abs(macro - .0354) < .002, macro
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
