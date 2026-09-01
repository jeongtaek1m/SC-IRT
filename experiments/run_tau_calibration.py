#!/usr/bin/env python3
"""Stopping thresholds fixed on the calibration panel (never on evaluation planners).

For each draw and K_cal, every calibration planner j is held out in turn, the
bank is re-calibrated from the other K_cal - 1 planners, and the four bank
orders of `run_adaptive.py` are run on j with the same readout and the same
posterior L1 risk R1. For a target mean budget B* in {30, 55} the threshold
is

    tau_hat(draw, K_cal, method, B*) = argmin_tau | mean_j rollouts_j(tau) - B* |

over a 0.001 grid — a cost target, not an accuracy target, so held-out
SR-MAE and IES are measured, not selected. Output: results/tau_hat.json,
consumed by `run_adaptive.py --merge` (matched-cost table, appendix).

The same LOO tracks also fix the risk scale of the stopping rule: per
(draw, K_cal, order), c = 90th percentile of |Shat_t - SR| / R1_t over the
left-out planners and t in [10, 110] (results/risk_cal.json), so that
`run_adaptive.py --merge` can stop at c * R1_t <= epsilon — an error target
rather than a cost target. The merge also prints a reliability diagnostic of
raw vs scaled R1 against the realised error, by deciles of raw R1.

    python experiments/run_tau_calibration.py --seeds 0 4   # shard (~25 min each, GPU)
    python experiments/run_tau_calibration.py --merge       # tau_hat.json + summary
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
from scirt.bayes import bank_from_fit, track, stop_at
from scirt.acquisition import r1_traj
from scirt.baselines import fluid_order, metabench_order

OUT = Path(os.environ.get('SCIRT_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
KCALS = tuple(int(x) for x in os.environ.get('SCIRT_KCALS', '7,10,16').split(','))
ORD = ('SC-IRT', 'Fluid', 'metabench', 'Random')
TMAX = 110
TARGETS = (30, 55)
TAUS = np.round(np.arange(0.010, 0.0801, 0.001), 3)
RISK_T0 = 10


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
        for Jc in KCALS:
            cs = subsample(cols, seed, Jc)
            f0 = calibrate(panel.Y, calR, cs, mode='1pl', types=typ)      # sigma_b / sigma_g of the panel
            for j in cs:
                csl = [c for c in cs if c != j]
                f1 = calibrate(panel.Y, calR, csl, mode='1pl', sigma_b=f0['sigma_b'])
                f2 = calibrate(panel.Y, calR, csl, mode='2pl', sigma_b=f0['sigma_b'])
                bi, yy = panel.bank_rows(calR, j)
                n = len(bi)
                T = min(TMAX, n)
                bank = bank_from_fit(f1, bi, typ, sigma_g=f0['sigma_g'])
                a, b = f2['a'][bi], f2['b'][bi]
                orders = {'SC-IRT': r1_traj(bank, yy, T),
                          'Fluid': fluid_order(a, b, yy, T),
                          'metabench': [int(i) for i in metabench_order(a, b, T, n)],
                          'Random': [int(i) for i in np.random.RandomState(700 + seed * 20 + j).permutation(n)[:T]]}
                rec = {'seed': seed, 'J': Jc, 'j': int(j), 'SR': float(yy.mean()), 'sigma_g': f0['sigma_g']}
                for k, o in orders.items():
                    Sh, R1 = track(bank, yy, o)
                    rec[k] = {'Shat': [float(x) for x in Sh], 'R1': [float(x) for x in R1]}
                recs.append(rec)
            print(f'seed {seed} K{Jc} done', flush=True)
    return recs


def select(recs):
    TAU, summary = {}, {}
    for seed in sorted(set(r['seed'] for r in recs)):
        for J in KCALS:
            rj = [r for r in recs if r['seed'] == seed and r['J'] == J]
            for o in ORD:
                for tg in TARGETS:
                    mb = np.array([np.mean([stop_at(r[o]['R1'], t) for r in rj]) for t in TAUS])
                    tau = float(TAUS[int(np.argmin(np.abs(mb - tg)))])
                    TAU[f'{seed}|{J}|{o}|{tg}'] = tau
                    summary.setdefault((J, o, tg), []).append(tau)
    print('\n===== calibration-fixed thresholds tau_hat: median [IQR] over draws =====')
    for J in KCALS:
        for tg in TARGETS:
            print(f'K_cal {J:2d} target {tg}: ' + '  '.join(
                f'{o} {np.median(summary[(J, o, tg)]):.3f} [{np.percentile(summary[(J, o, tg)], 25):.3f},'
                f'{np.percentile(summary[(J, o, tg)], 75):.3f}]' for o in ORD))
    return TAU, summary


def risk_scale(recs):
    """c(draw, K_cal, order) = 90th percentile of |Shat_t - SR| / R1_t over the
    left-out planners and t in [RISK_T0, TMAX] of the LOO tracks (R1_t <= 1e-6
    skipped); prints the reliability of raw and scaled R1 pooled over draws."""
    C, cs, pool = {}, {}, {}
    for seed in sorted(set(r['seed'] for r in recs)):
        for J in KCALS:
            rj = [r for r in recs if r['seed'] == seed and r['J'] == J]
            for o in ORD:
                raw = np.concatenate([r[o]['R1'][RISK_T0 - 1:TMAX] for r in rj])
                act = np.concatenate([np.abs(np.array(r[o]['Shat'][RISK_T0 - 1:TMAX]) - r['SR']) for r in rj])
                ok = raw > 1e-6
                c = float(np.percentile(act[ok] / raw[ok], 90))
                C[f'{seed}|{J}|{o}'] = c
                cs.setdefault((J, o), []).append(c)
                pool.setdefault((J, o), []).append(np.stack([raw[ok], act[ok], c * raw[ok]]))
    print(f'\n===== reliability of R1 on the LOO tracks (t in [{RISK_T0},{TMAX}], pooled over draws; rows = deciles of raw R1) =====')
    for J in KCALS:
        print(f'-- K_cal = {J} --  c median [IQR]: ' + '  '.join(
            f'{o} {np.median(cs[(J, o)]):.2f} [{np.percentile(cs[(J, o)], 25):.2f},{np.percentile(cs[(J, o)], 75):.2f}]'
            for o in ORD))
        print('   decile ' + ' | '.join(f'{o:^22s}' for o in ORD))
        print('          ' + ' | '.join('raw R1   |err|    c*R1' for o in ORD))
        P = {o: np.concatenate(pool[(J, o)], 1) for o in ORD}
        D = {o: np.digitize(P[o][0], np.percentile(P[o][0], np.arange(10, 100, 10))) for o in ORD}
        for b in range(10):
            print(f'   {b + 1:6d} ' + ' | '.join('  '.join(f'{v:.4f}' for v in P[o][:, D[o] == b].mean(1)) for o in ORD))
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs=2, type=int, default=None)
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.merge:
        recs = sum([json.load(open(f)) for f in sorted(glob.glob(str(OUT / 'tau_loo_*_*.json')))], [])
    elif args.seeds:
        lo, hi = args.seeds
        recs = run(range(lo, hi))
        json.dump(recs, open(OUT / f'tau_loo_{lo}_{hi}.json', 'w'))
        print('shard saved; run with --merge after all shards')
        return
    else:
        recs = run(range(R_DRAWS))
        json.dump(recs, open(OUT / 'tau_loo_0_16.json', 'w'))
    TAU, summary = select(recs)
    json.dump(TAU, open(OUT / 'tau_hat.json', 'w'))
    assert len(set(r['seed'] for r in recs)) == R_DRAWS
    print('tau_hat.json written')
    C = risk_scale(recs)
    json.dump(C, open(OUT / 'risk_cal.json', 'w'))
    print('risk_cal.json written')
    for J, v in ((7, 1.79), (10, 1.97), (16, 2.10)):
        cm = float(np.median([C[f'{s}|{J}|SC-IRT'] for s in sorted(set(r['seed'] for r in recs))]))
        assert abs(cm - v) < .03, (J, cm)
    assert abs(float(np.median(summary[(16, 'SC-IRT', 55)])) - .029) < .0015
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
