#!/usr/bin/env python3
"""Table 2 + cost-error data — adaptive evaluation under the unified machine.

Four bank orders (SC-IRT's Delta-R1 selection, Fluid, metabench, Random) run
with the same Rasch readout and the same stopping risk R1(D_t) <= tau; every
step's readout and risk is recorded so any budget or threshold can be scored.
Thresholds come from `run_tau_calibration.py` (calibration-panel LOO, target
mean budgets 30 / 55) and are never selected on evaluation planners.

    python experiments/run_adaptive.py --seeds 0 4     # shard (GPU)
    python experiments/run_adaptive.py --merge         # Table 2, sweeps, anchors
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
from scirt.curves import marginal_curves, PRIOR
from scirt.bayes import post_from, track, stop_at
from scirt.acquisition import r1_pick
from scirt.baselines import fluid_order, metabench_order
from scirt.metrics import paired_seed_boot, ies

OUT = Path(os.environ.get('SCIRT_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
KCALS = tuple(int(x) for x in os.environ.get('SCIRT_KCALS', '7,10,16').split(','))
ORD = ('SC-IRT', 'Fluid', 'metabench', 'Random')
BGRID = [30, 55, 110]
T = 110
TAU_SWEEP = (0.05, 0.04, 0.035, 0.03)
TARGETS = (30, 55)


def subsample(cols, seed, Kc):
    if Kc >= len(cols):
        return list(cols)
    rs = np.random.RandomState(9000 + seed * 100 + Kc * 10 + 0)
    return sorted(np.array(cols)[rs.choice(len(cols), Kc, replace=False)].tolist())


def r1_traj(M, y, n):
    S, q = [], PRIOR.copy()
    for _ in range(min(T, n)):
        rem = [i for i in range(n) if i not in S]
        S.append(r1_pick(M, y, S, q, rem))
        q = post_from(M, y, S)
    return S


def orders_for(f1, f2, bi, yy, seed, js):
    n = len(bi)
    M1 = marginal_curves(f1['b'][bi], f1['s'][bi])
    return M1, {'SC-IRT': r1_traj(M1, yy, n),
                'Fluid': fluid_order(f2['a'][bi], f2['b'][bi], yy, T),
                'metabench': [int(i) for i in metabench_order(f2['a'][bi], f2['b'][bi], T, n)],
                'Random': [int(i) for i in np.random.RandomState(100 + seed * 20 + js).permutation(n)[:T]]}


def run(seeds):
    panel = Panel()
    recs = []
    for seed in seeds:
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        for Kc in KCALS:
            cs = subsample(cols, seed, Kc)
            f1 = calibrate(panel.Y, calR, cs, mode='1pl')
            f2 = calibrate(panel.Y, calR, cs, mode='2pl')
            for js in hp:
                bi, yy = panel.bank_rows(calR, js)
                M1, od = orders_for(f1, f2, bi, yy, seed, js)
                rec = {'seed': seed, 'K': Kc, 'js': int(js), 'SR': float(yy.mean())}
                for k, o in od.items():
                    Sh, R1 = track(M1, yy, o)
                    rec[k] = {'Shat': [float(x) for x in Sh], 'R1': [float(x) for x in R1]}
                recs.append(rec)
            print(f'seed {seed} K{Kc} done', flush=True)
    return recs


def stopped(r, o, tau):
    t = stop_at(r[o]['R1'], tau)
    return t, abs(r[o]['Shat'][t - 1] - r['SR'])


def report(recs):
    recs = sorted(recs, key=lambda r: (r['seed'], r['K'], r['js']))
    FX = {K: {o: {B: [abs(r[o]['Shat'][B - 1] - r['SR']) for r in recs if r['K'] == K] for B in BGRID}
              for o in ORD} for K in KCALS}
    print(f'\n{len(recs)} planner evaluations')
    print('\n===== fixed budgets under the common Rasch readout, SR-MAE =====')
    for K in KCALS:
        print(f'-- K_cal = {K} --   ' + ' '.join(f'{B:>6d}' for B in BGRID))
        for o in ORD:
            print(f'   {o:10s} ' + ' '.join(f'{np.mean(FX[K][o][B]):6.4f}' for B in BGRID))
    print('\n===== SC-IRT: adaptive stop vs its own fixed-budget curve at matched mean rollouts =====')
    xs = np.array([10, 20, 30, 40, 55, 60, 80, 110], float)
    for K in KCALS:
        rs = [r for r in recs if r['K'] == K]
        ys = np.array([np.mean([abs(r['SC-IRT']['Shat'][int(B) - 1] - r['SR']) for r in rs]) for B in xs])
        row = []
        for tau in TAU_SWEEP:
            st = [stopped(r, 'SC-IRT', tau) for r in rs]
            Bm, em = np.mean([s[0] for s in st]), np.mean([s[1] for s in st])
            row.append(f'tau {tau:.3f}: {Bm:5.1f} roll, {em:.4f} vs fixed {float(np.interp(Bm, xs, ys)):.4f} ({em - float(np.interp(Bm, xs, ys)):+.4f})')
        print(f'-- K_cal = {K} --\n   ' + '\n   '.join(row))
    tau_path = OUT / 'tau_hat.json'
    T2 = {}
    if tau_path.exists():
        TAU = json.load(open(tau_path))
        print('\n===== Table 2 (each method at its own calibration-fixed tau; IES ref = Random at fixed 55) =====')
        for K in KCALS:
            rs = [r for r in recs if r['K'] == K]
            ref = np.mean(FX[K]['Random'][55])
            print(f'-- K_cal = {K} --  (ref {ref:.4f})')
            for tg in TARGETS:
                res = {}
                for o in ORD:
                    Bs, es = [], []
                    for r in rs:
                        t, e = stopped(r, o, TAU[f"{r['seed']}|{K}|{o}|{tg}"])
                        Bs.append(t)
                        es.append(e)
                    res[o] = (np.array(Bs), np.array(es))
                for o in ORD:
                    Bs, es = res[o]
                    d, lo, hi = paired_seed_boot(es, res['SC-IRT'][1], n_seeds=16, per_seed=6) if o != 'SC-IRT' else (0, 0, 0)
                    print(f'   target {tg:2d} {o:10s} rollouts {Bs.mean():5.1f}  SR-MAE {es.mean():.4f}  '
                          f'IES {ies(es.mean(), Bs.mean(), ref):.2f}'
                          + ('' if o == 'SC-IRT' else f'   d {d:+.4f} [{lo:+.4f},{hi:+.4f}]'))
                    T2[(K, tg, o)] = es.mean()
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
    assert len(recs) == len(KCALS) * 96
    for (K, B, ref) in ((7, 30, .0497), (10, 110, .0189), (16, 55, .0419)):
        assert abs(np.mean(FX[K]['SC-IRT'][B]) - ref) < .002
    if T2:
        assert abs(T2[(7, 55, 'SC-IRT')] - .0342) < .003 and abs(T2[(16, 30, 'SC-IRT')] - .0543) < .003
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
