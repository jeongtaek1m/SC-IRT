#!/usr/bin/env python3
"""Table 2 + cost-error data — adaptive evaluation under the common readout and stopping rule.

Four bank orders (ATDrive's Delta-R1 selection, Fluid, metabench, Random) run
with the same Rasch readout and the same stopping risk R1(D_t) <= tau; every
step's readout and risk is recorded so any budget or threshold can be scored.
Table 2 is the risk-target rule: stop at the first t with c * R1_t <= epsilon,
epsilon in {.03, .05}, with the risk scale c fixed by `run_tau_calibration.py`
(calibration-panel LOO, results/risk_cal.json); it reports rollouts, SR-MAE,
the calibration gap mean|err| - mean c*R1 at
the stop. The matched-cost rule (tau_hat at target mean budgets 30 / 55,
results/tau_hat.json) is kept as the appendix table. Nothing is selected on
evaluation planners.

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
from atdrive.b2d import Panel
from atdrive.splits import up_split, R_DRAWS
from atdrive.calibration import calibrate
from atdrive.curves import marginal_curves
from atdrive.bayes import Bank, bank_from_fit, track, stop_at
from atdrive.acquisition import r1_traj
from atdrive.baselines import fluid_order, metabench_order, stratified_order
from atdrive.metrics import paired_cluster_boot, ies

OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
KCALS = tuple(int(x) for x in os.environ.get('ATDRIVE_KCALS', '4,8,12').split(','))
ORD = ('ATDrive', 'Fluid', 'metabench', 'Random', 'Random-strat')
BGRID = [30, 55, 110, 165]        # 5 x {6, 11, 22, 33} = 14 / 25 / 50 / 75% of the 220-route benchmark
NROUTES = 220                     # routes in the benchmark; the bank is all of them
T = None          # trajectory length = the whole bank (was 110): every order runs until the rule stops it or the bank is exhausted
TAU_SWEEP = (0.05, 0.04, 0.035, 0.03)
TARGETS = (30, 55)
EPS = (0.03, 0.05)
RISK_T0 = 10          # the risk scale c is fitted on t >= 10 (run_tau_calibration T0); the stop honours the same window, never binding here
NO_TESTLET = os.environ.get('ATDRIVE_NO_TESTLET', '0') == '1'   # ablation: sigma_g = 0 (independent items)
POINT_CURVES = os.environ.get('ATDRIVE_POINT_CURVES', '0') == '1'   # ablation: point curves at b_hat (difficulty posterior collapsed)
if POINT_CURVES:
    ORD = ('ATDrive',)   # the point-curve arm is only compared against ATDrive, so no baseline order is needed


def subsample(cols, seed, Kc):
    if Kc >= len(cols):
        return list(cols)
    rs = np.random.RandomState(9000 + seed * 100 + Kc * 10 + 0)
    return sorted(np.array(cols)[rs.choice(len(cols), Kc, replace=False)].tolist())




PJ = 16   # planner count, set from the panel in run()


def orders_for(f1, f2, bi, yy, seed, js, typ):
    n = len(bi)
    bank = (Bank(marginal_curves(f1['b'][bi], np.full(n, 1e-9)), typ[bi], f1['sigma_g']) if POINT_CURVES
            else bank_from_fit(f1, bi, typ, sigma_g=0.0 if NO_TESTLET else None))
    Tn = n if T is None else min(T, n)
    return bank, {'ATDrive': r1_traj(bank, yy, Tn),
                'Fluid': fluid_order(f2['a'][bi], f2['b'][bi], yy, Tn),
                'metabench': [int(i) for i in metabench_order(f2['a'][bi], f2['b'][bi], Tn, n)],
                'Random': [int(i) for i in np.random.RandomState(100 + seed * PJ + js).permutation(n)[:Tn]],
                'Random-strat': [int(i) for i in stratified_order(typ[bi], np.random.RandomState(100 + seed * PJ + js))[:Tn]]}


def run(seeds):
    global PJ
    panel = Panel()
    PJ = panel.J
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
            for js in hp:
                bi, yy = panel.bank_rows(calR, js)
                bank, od = orders_for(f1, f2, bi, yy, seed, js, typ)
                rec = {'seed': seed, 'K': Kc, 'js': int(js), 'SR': float(yy.mean()), 'sigma_g': f1['sigma_g']}
                for k, o in od.items():
                    Sh, R1 = track(bank, yy, o)
                    rec[k] = {'Shat': [float(x) for x in Sh], 'R1': [float(x) for x in R1]}
                recs.append(rec)
            print(f'seed {seed} K{Kc} done', flush=True)
    return recs


def stopped(r, o, tau):
    t = stop_at(r[o]['R1'], tau)
    return t, abs(r[o]['Shat'][t - 1] - r['SR'])


def risk_stopped(r, o, c, eps):
    """(rollouts, |err|, c*R1, hit-cap) at the first t with c * R1_t <= eps (the bank size if never)."""
    t = stop_at(c * np.array(r[o]['R1']), eps, tmin=RISK_T0)
    return t, abs(r[o]['Shat'][t - 1] - r['SR']), c * r[o]['R1'][t - 1], float(t == len(r[o]['R1']))


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
    print('\n===== ATDrive: adaptive stop vs its own fixed-budget curve at matched mean rollouts =====')
    xs = np.array([10, 20, 30, 40, 55, 60, 80, 110, 165, 200], float)
    for K in KCALS:
        rs = [r for r in recs if r['K'] == K]
        ys = np.array([np.mean([abs(r['ATDrive']['Shat'][int(B) - 1] - r['SR']) for r in rs]) for B in xs])
        row = []
        for tau in TAU_SWEEP:
            st = [stopped(r, 'ATDrive', tau) for r in rs]
            Bm, em = np.mean([s[0] for s in st]), np.mean([s[1] for s in st])
            row.append(f'tau {tau:.3f}: {Bm:5.1f} roll, {em:.4f} vs fixed {float(np.interp(Bm, xs, ys)):.4f} ({em - float(np.interp(Bm, xs, ys)):+.4f})')
        print(f'-- K_cal = {K} --\n   ' + '\n   '.join(row))
    tau_path = OUT / 'tau_hat.json'
    T2 = {}
    if tau_path.exists() and not POINT_CURVES:      # the matched-cost table's IES reference is the Random order
        TAU = json.load(open(tau_path))
        print('\n===== Table 2 (matched-cost, appendix): each method at its own calibration-fixed tau; IES ref = Random at fixed 55 =====')
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
                    d, lo, hi = paired_cluster_boot(es, res['ATDrive'][1], [r['js'] for r in rs]) if o != 'ATDrive' else (0, 0, 0)
                    print(f'   target {tg:2d} {o:10s} rollouts {Bs.mean():5.1f}  SR-MAE {es.mean():.4f}  '
                          f'IES {ies(es.mean(), Bs.mean(), ref):.2f}'
                          + ('' if o == 'ATDrive' else f'   d {d:+.4f} [{lo:+.4f},{hi:+.4f}]'))
                    T2[(K, tg, o)] = es.mean()
    else:
        print('\n(matched-cost table skipped: no tau_hat.json, or a single-order run with no Random reference)')
    cal_path = OUT / 'risk_cal.json'
    if cal_path.exists():
        C = json.load(open(cal_path))
        print('\n===== Table 2 — risk-target stopping: first t with c*R1_t <= eps, c = LOO 90th pct |err|/R1 (risk_cal.json) =====')
        for K in KCALS:
            rs = [r for r in recs if r['K'] == K]
            js = [r['js'] for r in rs]
            print(f'-- K_cal = {K} --')
            for eps in EPS:
                res = {o: np.array([risk_stopped(r, o, C[f"{r['seed']}|{K}|{o}"], eps) for r in rs]) for o in ORD}
                for o in ORD:
                    Bs, es, cr, cap = res[o].T
                    d, lo, hi = paired_cluster_boot(es, res['ATDrive'][:, 1], js) if o != 'ATDrive' else (0, 0, 0)
                    print(f'   eps {eps:.2f} {o:12s} rollouts {Bs.mean():5.1f} ({Bs.mean() / NROUTES:3.0%} of {NROUTES}, cap {cap.mean():.0%})  '
                          f'SR-MAE {es.mean():.4f}  gap {es.mean() - cr.mean():+.4f}'
                          + ('' if o == 'ATDrive' else f'   d {d:+.4f} [{lo:+.4f},{hi:+.4f}]'))
    else:
        print('\n(results/risk_cal.json not found: run run_tau_calibration.py --merge for the risk-target table)')
    return FX, T2


def budget_table(recs):
    """RESULTS.md 'What each budget buys': over the 64 evaluations of a cell, the
    ATDrive estimate's SR-MAE, the share within 1 / 2 / 3 SR points of the
    truth, and the within-draw pairwise ranking of the four evaluation
    planners (the 6 pairs of ESTIMATES ordered as their true SRs)."""
    out = {}
    print('\n===== what each budget buys (ATDrive): SR-MAE, P(|err| <= 1 / 2 / 3 pt), within-draw pairwise rank correct =====')
    for K in KCALS:
        rs = [r for r in recs if r['K'] == K]
        for B in BGRID:
            err = np.array([abs(r['ATDrive']['Shat'][B - 1] - r['SR']) for r in rs])
            pairs = []
            for s in sorted(set(r['seed'] for r in rs)):
                d = [r for r in rs if r['seed'] == s]
                for i in range(len(d)):
                    for j in range(i + 1, len(d)):
                        a, b = d[i], d[j]
                        pairs.append((a['ATDrive']['Shat'][B - 1] > b['ATDrive']['Shat'][B - 1]) == (a['SR'] > b['SR']))
            out[(K, B)] = (float(err.mean()), float(np.mean(err <= .01)), float(np.mean(err <= .02)),
                           float(np.mean(err <= .03)), float(np.mean(pairs)))
            print(f'   K_cal {K:2d} B {B:3d}: SR-MAE {err.mean():.4f}  within 1 pt {np.mean(err <= .01):3.0%}  '
                  f'2 pt {np.mean(err <= .02):3.0%}  3 pt {np.mean(err <= .03):3.0%}  pairwise rank correct '
                  f'{100 * np.mean(pairs):.1f}%  ({int(np.sum(pairs))} of {len(pairs)} pairs)')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs=2, type=int, default=None)
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.merge:
        recs = sum([json.load(open(f)) for f in sorted(glob.glob(str(OUT / 'adaptive_*_*.json')))], [])
        if recs:
            json.dump(recs, open(OUT / 'adaptive.json', 'w'))
        else:                                   # no shards (a clone): score the results of record
            recs = json.load(open(OUT / 'adaptive.json'))
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
    assert len(recs) == len(KCALS) * 64
    if NO_TESTLET or POINT_CURVES:
        return                                    # the anchors pin the paper's panel, not the switched arms
    BT = budget_table(recs)
    for K, B, v in ((4, 30, (.0450, .20, .27, .38, .917)), (8, 55, (.0337, .19, .36, .56, .906)),
                    (12, 30, (.0477, .11, .22, .36, .896)), (12, 165, (.0081, .72, .94, .98, 1.0))):
        assert all(abs(a - b) < t for a, b, t in zip(BT[(K, B)], v, (.002, .01, .01, .01, .002))), (K, B, BT[(K, B)])
    fx = lambda K, o, t: np.mean([abs(r[o]['Shat'][t - 1] - r['SR']) for r in recs if r['K'] == K])
    for K, o, t, v in ((4, 'ATDrive', 30, .0450), (4, 'ATDrive', 55, .0332), (8, 'ATDrive', 110, .0202),
                       (12, 'ATDrive', 165, .0081), (4, 'Random', 55, .0405), (12, 'Fluid', 55, .0290)):
        assert abs(fx(K, o, t) - v) < .002, (K, o, t, fx(K, o, t))
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
