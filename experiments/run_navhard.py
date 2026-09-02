#!/usr/bin/env python3
"""Table 4 — UP on the two-stage NAVSIM navhard panel (87 unique leaderboard
submissions x 225 scored units, pass = EPDMS >= 0.5).

Per draw: 6 evaluation planners (RandomState(1000+draw)); K_cal in {7,10,16,81}
subsampled from the remaining 81 (RandomState(9000+100*draw+10*K)).  Budgets
B in {30,55,110} scored units (no scenario-type structure on this panel, so
budgets are unit counts and Random has no stratified variant).

Usage: --seeds lo hi to run a shard; --merge to pool shards and print the table.
"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirt.calibration import calibrate
from scirt.curves import curves_from_posterior
from scirt.bayes import Bank, readout
from scirt.acquisition import r1_traj
from scirt.baselines import (fluid_order, total_fisher_order, marginal_fisher_order,
                             disco_order, kmeans_anchors, metabench_order,
                             anchorpoints_estimate, pirt)
from scirt.metrics import paired_cluster_boot

np.random.seed(0); torch.manual_seed(0)
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('SCIRT_RESULTS_DIR', ROOT / 'results')); OUT.mkdir(exist_ok=True)
KC = (7, 10, 16, 81); BB = (30, 55, 110); T = 110
NREP = 5          # random-policy rows: expected |error| over NREP independent orders per evaluation
METHODS = ['Random (IRT-free)', 'Random + IRT', 'DISCO', 'AnchorPoints', 'Total-Fisher',
           'Marginal-Fisher', 'tinyBenchmarks', 'metabench', 'Fluid', 'SC-IRT']

d = np.load(ROOT / 'data' / 'navhard' / 'navhard_binary_panel.npz', allow_pickle=True)
Yb = d['Y']; units = [str(x) for x in d['cols']]; P = Yb.shape[0]; U = len(units)
Ydict = {(units[u], k): float(Yb[k, u]) for k in range(P) for u in range(U)}




def run(lo, hi):
    recs = []
    for seed in range(lo, hi):
        rng = np.random.RandomState(1000 + seed); hp = sorted(rng.choice(P, 6, replace=False).tolist())
        pool = [k for k in range(P) if k not in hp]
        for Kc in KC:
            cs = pool if Kc >= len(pool) else sorted(np.array(pool)[
                np.random.RandomState(9000 + seed * 100 + Kc * 10).choice(len(pool), Kc, replace=False)].tolist())
            f1 = calibrate(Ydict, units, cs, mode='1pl'); f2 = calibrate(Ydict, units, cs, mode='2pl', sigma_b=f1['sigma_b'])
            bank = Bank(curves_from_posterior(f1['W']), np.zeros(U, int), 0.0)   # no scenario-type structure on this panel
            Rb = Yb[cs].T; pbar = Rb.mean(1)
            for js in hp:
                yy = Yb[js]; n = U; SR = float(yy.mean())
                b1 = f1['b']; a2, b2 = f2['a'], f2['b']; ones = np.ones(n)
                perms = [[int(i) for i in np.random.RandomState(100 + seed * P + js + 100000 * rep).permutation(n)] for rep in range(NREP)]
                od = {'disco': disco_order(pbar), 'tf': total_fisher_order(a2, b2, f2['th']),
                      'mf': marginal_fisher_order(a2, b2), 'fluid': fluid_order(a2, b2, yy, T),
                      'ours': r1_traj(bank, yy, T)}
                err = {m: {} for m in METHODS}
                for B in BB:
                    est = {'Random (IRT-free)': yy[perms[0][:B]].mean(),
                           'Random + IRT': pirt(b1, ones, yy, perms[0][:B]),
                           'DISCO': pirt(b2, a2, yy, od['disco'][:B]),
                           'AnchorPoints': anchorpoints_estimate(Rb, yy, B),
                           'Total-Fisher': pirt(b2, a2, yy, od['tf'][:B]),
                           'Marginal-Fisher': pirt(b2, a2, yy, od['mf'][:B]),
                           'tinyBenchmarks': pirt(b2, a2, yy, kmeans_anchors(a2, b2, B, n)),
                           'metabench': pirt(b2, a2, yy, metabench_order(a2, b2, B, n)),
                           'Fluid': pirt(b2, a2, yy, od['fluid'][:B]),
                           'SC-IRT': readout(bank, yy, od['ours'][:B])}
                    for m in METHODS: err[m][str(B)] = abs(float(est[m]) - SR)
                    err['Random (IRT-free)'][str(B)] = float(np.mean([abs(yy[pm[:B]].mean() - SR) for pm in perms]))
                    err['Random + IRT'][str(B)] = float(np.mean([abs(pirt(b1, ones, yy, pm[:B]) - SR) for pm in perms]))
                recs.append({'seed': seed, 'K': Kc, 'js': int(js), 'SR': SR, 'err': err})
            print(f'seed {seed} K{Kc} done', flush=True)
    json.dump(recs, open(OUT / f'navhard_{lo}_{hi}.json', 'w'))
    print('shard saved; run with --merge after all shards')


def merge():
    recs = [r for f in sorted(glob.glob(str(OUT / 'navhard_*_*.json'))) for r in json.load(open(f))]
    assert len(recs) == 16 * len(KC) * 6, f'expected {16 * len(KC) * 6} recs, got {len(recs)}'
    print('===== Table 4 — UP on the two-stage navhard panel, SR-MAE (96 evals/cell) =====')
    for Kc in KC:
        sub = [r for r in recs if r['K'] == Kc]
        print(f'\n-- K_cal = {Kc} --')
        print(f"{'method':18s} " + ' '.join(f'{B:>7d}' for B in BB))
        for m in METHODS:
            row = f'{m:18s} '
            for B in BB:
                e = [r['err'][m][str(B)] for r in sub]
                if m == 'SC-IRT':
                    row += f'{np.mean(e):7.4f} '
                else:
                    d_, lo_, hi_ = paired_cluster_boot(e, [r["err"]["SC-IRT"][str(B)] for r in sub], [r['js'] for r in sub])
                    row += f'{np.mean(e):7.4f}{"*" if (lo_ > 0 or hi_ < 0) else " "}'
            print(row)
    json.dump({str(K): {m: {str(B): float(np.mean([r['err'][m][str(B)] for r in recs if r['K'] == K]))
                            for B in BB} for m in METHODS} for K in KC},
              open(OUT / 'navhard.json', 'w'))
    mean = lambda K, m, B: np.mean([r['err'][m][str(B)] for r in recs if r['K'] == K])
    for K, m, B, v in ((7, 'SC-IRT', 55, .0347), (16, 'SC-IRT', 110, .0191), (81, 'SC-IRT', 110, .0169),
                       (16, 'SC-IRT', 30, .0488), (81, 'Fluid', 30, .0304), (7, 'Random (IRT-free)', 30, .0599)):
        assert abs(mean(K, m, B) - v) < .0003, (K, m, B, mean(K, m, B))
    print('anchors OK')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs=2, type=int, default=None)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    merge() if a.merge else run(*(a.seeds or (0, 16)))
