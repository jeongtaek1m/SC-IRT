#!/usr/bin/env python3
"""Stopping thresholds fixed on the calibration panel (never on held-out planners).

For each draw and J_cal, every calibration planner j is held out in turn, the
bank is re-calibrated from the other J_cal - 1 planners, and the four bank
orders of `run_adaptive.py` are run on j with the same readout and the same
posterior L1 risk R1. For a target mean budget B* in {30, 60} the threshold
is

    tau_hat(draw, J_cal, method, B*) = argmin_tau | mean_j rollouts_j(tau) - B* |

over a 0.001 grid — a cost target, not an accuracy target, so held-out
SR-MAE and IES are measured, not selected. Output: results/tau_hat.json,
consumed by `run_adaptive.py --merge`.

    python experiments/run_tau_calibration.py --seeds 0 4   # shard (~25 min each, GPU)
    python experiments/run_tau_calibration.py --merge       # tau_hat.json + summary
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

OUT = Path(__file__).resolve().parents[1] / 'results'
JCALS = (7, 10, 13)
ORD = ('SC-IRT', 'Fluid', 'metabench', 'Random')
TMAX = 120
TARGETS = (30, 60)
TAUS = np.round(np.arange(0.010, 0.0801, 0.001), 3)


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
            for j in cs:
                csl = [c for c in cs if c != j]
                f1 = calibrate(panel.Y, calR, csl, mode='1pl')
                f2 = calibrate(panel.Y, calR, csl, mode='2pl')
                bi, yy = panel.bank_rows(calR, j)
                n = len(bi)
                T = min(TMAX, n)
                M1 = marginal_curves(f1['b'][bi], f1['s'][bi])
                a, b = f2['a'][bi], f2['b'][bi]
                orders = {'SC-IRT': localize_cover(a, b, f2['th'], yy, K=K_LOCALIZE, T=T),
                          'Fluid': fluid_order(a, b, yy, T),
                          'metabench': [int(i) for i in metabench_order(a, b, T, n)],
                          'Random': [int(i) for i in np.random.RandomState(700 + seed * 20 + j).permutation(n)[:T]]}
                rec = {'seed': seed, 'J': Jc, 'j': int(j), 'SR': float(yy.mean())}
                for k, o in orders.items():
                    Sh, R1 = track(M1, yy, o)
                    rec[k] = {'Shat': [float(x) for x in Sh], 'R1': [float(x) for x in R1]}
                recs.append(rec)
            print(f'seed {seed} J{Jc} done', flush=True)
    return recs


def select(recs):
    TAU, summary = {}, {}
    for seed in sorted(set(r['seed'] for r in recs)):
        for J in JCALS:
            rj = [r for r in recs if r['seed'] == seed and r['J'] == J]
            for o in ORD:
                for tg in TARGETS:
                    mb = np.array([np.mean([stop_at(r[o]['R1'], t) for r in rj]) for t in TAUS])
                    tau = float(TAUS[int(np.argmin(np.abs(mb - tg)))])
                    TAU[f'{seed}|{J}|{o}|{tg}'] = tau
                    summary.setdefault((J, o, tg), []).append(tau)
    print('\n===== calibration-fixed thresholds tau_hat: median [IQR] over draws =====')
    for J in JCALS:
        for tg in TARGETS:
            print(f'J_cal {J:2d} target {tg}: ' + '  '.join(
                f'{o} {np.median(summary[(J, o, tg)]):.3f} [{np.percentile(summary[(J, o, tg)], 25):.3f},'
                f'{np.percentile(summary[(J, o, tg)], 75):.3f}]' for o in ORD))
    return TAU, summary


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
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
