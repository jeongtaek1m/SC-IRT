#!/usr/bin/env python3
"""Table 2 + cost-error figure — adaptive evaluation under a common stopping machine.

Four bank orders (SC-IRT localize->cover, Fluid, metabench, Random) are run
with the *same* Rasch readout and the *same* posterior-risk stopping rule
R1(D_t) = E|S - S_hat_t| <= tau (PROTOCOL section 5); every step's readout
and risk is recorded so any budget or threshold can be scored afterwards.

Thresholds are never selected on held-out planners: `run_tau_calibration.py`
fixes tau per (draw, J_cal, method, target budget) by leave-one-planner-out
simulation on the calibration panel and writes results/tau_hat.json, which
this script applies (design A: each method at its own tau for a common
target budget; design B: SC-IRT's tau applied to all).

    python experiments/run_adaptive.py                 # all draws (~1.5 h, GPU)
    python experiments/run_adaptive.py --seeds 0 4     # shard
    python experiments/run_adaptive.py --merge         # merge, Table 2, figure data, anchors
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirt.b2d import Panel
from scirt.splits import unified_split, R_DRAWS
from scirt.calibration import calibrate
from scirt.curves import marginal_curves
from scirt.bayes import track, stop_at
from scirt.acquisition import localize_cover, K_LOCALIZE
from scirt.baselines import fluid_order, metabench_order
from scirt.metrics import paired_seed_boot, ies

OUT = Path(__file__).resolve().parents[1] / 'results'
JCALS = (4, 7, 10, 13)
ORD = ('SC-IRT', 'Fluid', 'metabench', 'Random')
BGRID = [10, 20, 30, 40, 60, 80, 100, 120]
TMAX = max(BGRID)
TAU_SWEEP = (0.06, 0.05, 0.04, 0.03, 0.02)
TARGETS = (30, 60)


def subsample(cols, seed, Jc):
    if Jc == 13:
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
        for Jc in JCALS:
            cs = subsample(cols, seed, Jc)
            f1 = calibrate(panel.Y, calR, cs, mode='1pl')
            f2 = calibrate(panel.Y, calR, cs, mode='2pl')
            for js in hp:
                bi, yy = panel.bank_rows(calR, js)
                n = len(bi)
                T = min(TMAX, n)
                M1 = marginal_curves(f1['b'][bi], f1['s'][bi])
                a, b = f2['a'][bi], f2['b'][bi]
                orders = {'SC-IRT': localize_cover(a, b, f2['th'], yy, K=K_LOCALIZE, T=T),
                          'Fluid': fluid_order(a, b, yy, T),
                          'metabench': [int(i) for i in metabench_order(a, b, T, n)],
                          'Random': [int(i) for i in np.random.RandomState(100 + seed * 20 + js).permutation(n)[:T]]}
                rec = {'seed': seed, 'J': Jc, 'js': int(js), 'SR': float(yy.mean())}
                for k, o in orders.items():
                    Sh, R1 = track(M1, yy, o)
                    rec[k] = {'Shat': [float(x) for x in Sh], 'R1': [float(x) for x in R1]}
                recs.append(rec)
            print(f'seed {seed} J{Jc} done', flush=True)
    return recs


def fixed_err(r, o, B):
    return abs(r[o]['Shat'][B - 1] - r['SR'])


def stopped(r, o, tau):
    t = stop_at(r[o]['R1'], tau)
    return t, abs(r[o]['Shat'][t - 1] - r['SR'])


def report(recs):
    recs = sorted(recs, key=lambda r: (r['seed'], r['J'], r['js']))
    FX = {J: {o: {B: [fixed_err(r, o, B) for r in recs if r['J'] == J] for B in BGRID} for o in ORD} for J in JCALS}
    print(f'\n{len(recs)} planner evaluations')
    print('\n===== fixed budgets under the common Rasch readout (figure data), SR-MAE =====')
    for J in JCALS:
        print(f'-- J_cal = {J} --   ' + ' '.join(f'{B:>6d}' for B in BGRID))
        for o in ORD:
            print(f'   {o:10s} ' + ' '.join(f'{np.mean(FX[J][o][B]):6.4f}' for B in BGRID))
    print('\n===== semantic risk sweep: mean rollouts / SR-MAE / MAE:tau =====')
    for J in JCALS:
        print(f'-- J_cal = {J} --')
        for tau in TAU_SWEEP:
            row = []
            for o in ORD:
                st = [stopped(r, o, tau) for r in recs if r['J'] == J]
                Bs, es = np.mean([s[0] for s in st]), np.mean([s[1] for s in st])
                row.append(f'{o}: {Bs:5.1f}/{es:.4f}/{es / tau:4.2f}')
            print(f'   tau {tau:.3f}  ' + '   '.join(row))
    tau_path = OUT / 'tau_hat.json'
    T2 = {}
    if tau_path.exists():
        TAU = json.load(open(tau_path))
        for design in ('A', 'B'):
            print(f'\n===== Table 2 (design {design}: '
                  + ('each method at its own calibration-fixed tau, common target budget'
                     if design == 'A' else "SC-IRT's calibration-fixed tau applied to every method") + ') =====')
            for J in (7, 10, 13):
                ref = np.mean(FX[J]['Random'][60])
                print(f'-- J_cal = {J} --  (IES reference: Random @ fixed 60 = {ref:.4f})')
                for tg in TARGETS:
                    res = {}
                    for o in ORD:
                        Bs, es = [], []
                        for r in [r for r in recs if r['J'] == J]:
                            key = f"{r['seed']}|{J}|{o if design == 'A' else 'SC-IRT'}|{tg}"
                            t, e = stopped(r, o, TAU[key])
                            Bs.append(t)
                            es.append(e)
                        res[o] = (np.array(Bs), np.array(es))
                    for o in ORD:
                        Bs, es = res[o]
                        d, lo, hi = paired_seed_boot(es, res['SC-IRT'][1]) if o != 'SC-IRT' else (0, 0, 0)
                        print(f'   target {tg:2d} {o:10s} rollouts {Bs.mean():5.1f}  SR-MAE {es.mean():.4f}  '
                              f'IES {ies(es.mean(), Bs.mean(), ref):.2f}'
                              + ('' if o == 'SC-IRT' else f'   d vs SC-IRT {d:+.4f} [{lo:+.4f},{hi:+.4f}]'))
                        T2[(design, J, tg, o)] = (Bs.mean(), es.mean())
    else:
        print('\n(results/tau_hat.json not found: run run_tau_calibration.py for Table 2)')
    return FX, T2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs=2, type=int, default=None)
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.merge:
        recs = sum([json.load(open(f)) for f in sorted(glob.glob(str(OUT / 'adaptive_*_*.json')))], [])
        json.dump(recs, open(OUT / 'adaptive.json', 'w'))
    elif args.seeds:
        lo, hi = args.seeds
        recs = run(range(lo, hi))
        json.dump(recs, open(OUT / f'adaptive_{lo}_{hi}.json', 'w'))
        print('shard saved; run with --merge after all shards')
        return
    else:
        recs = run(range(R_DRAWS))
        json.dump(recs, open(OUT / 'adaptive.json', 'w'))
    FX, T2 = report(recs)
    assert len(recs) == 4 * 48
    for (B, ref) in ((30, .0388), (40, .0342), (60, .0213), (80, .0222)):
        assert abs(np.mean(FX[13]['SC-IRT'][B]) - ref) < .002
    if T2:
        assert abs(T2[('A', 13, 30, 'SC-IRT')][1] - T2_ANCHOR[0]) < .003 and abs(T2[('A', 13, 60, 'SC-IRT')][1] - T2_ANCHOR[1]) < .003
    print('anchors OK')


T2_ANCHOR = (0.0384, 0.0228)   # SC-IRT design-A SR-MAE at target budgets 30 / 60, J_cal = 13

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
