#!/usr/bin/env python3
"""Component ablation — switch OUR pieces off, one at a time.

Same readout and the same bank for every row; only the marked component
changes. Primary protocol, B in {30, 55}.

  ATDrive (full)            exact difficulty posteriors + testlet + Delta-R1 acquisition
  w/o b-uncertainty        point curves sigmoid(theta - b_hat + u) (difficulty posterior collapsed)
  w/o testlet              sigma_g = 0 (independent items), posteriors and acquisition unchanged
  w/o risk acquisition     random scene order, inference unchanged

    python experiments/run_ablation.py --seeds 0 4   # shard (GPU)
    python experiments/run_ablation.py --merge
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
from atdrive.bayes import Bank, bank_from_fit, readout
from atdrive.acquisition import r1_traj
from atdrive.metrics import paired_cluster_boot

OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
KCALS = tuple(int(x) for x in os.environ.get('ATDRIVE_KCALS', '4,8,12').split(','))
BGRID = [30, 55, 110, 165]
T = max(BGRID)
ARMS = ['ATDrive (full)', 'w/o b-uncertainty', 'w/o testlet', 'w/o risk acquisition']


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
            for js in hp:
                bi, yy = panel.bank_rows(calR, js)
                n = len(bi)
                SR = float(yy.mean())
                bank = bank_from_fit(f1, bi, typ)
                bank_p = Bank(marginal_curves(f1['b'][bi], np.full(n, 1e-9)), typ[bi], f1['sigma_g'])
                bank_t = bank_from_fit(f1, bi, typ, sigma_g=0.0)
                perm = [int(i) for i in np.random.RandomState(100 + seed * panel.J + js).permutation(n)[:T]]
                sets = {'ATDrive (full)': (bank, r1_traj(bank, yy, T)),
                        'w/o b-uncertainty': (bank_p, r1_traj(bank_p, yy, T)),
                        'w/o testlet': (bank_t, r1_traj(bank_t, yy, T)),
                        'w/o risk acquisition': (bank, perm)}
                recs.append({'seed': seed, 'K': Kc, 'js': int(js), 'SR': SR, 'sigma_g': f1['sigma_g'],
                             'err': {a: {str(B): abs(readout(M, yy, o[:B]) - SR) for B in BGRID}
                                     for a, (M, o) in sets.items()}})
            print(f'seed {seed} K{Kc} done', flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs=2, type=int, default=None)
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.merge:
        recs = sum([json.load(open(f)) for f in sorted(glob.glob(str(OUT / 'ablation_*_*.json')))], [])
        if recs:
            json.dump(recs, open(OUT / 'ablation.json', 'w'))
        else:                                   # no shards (a clone): score the results of record
            recs = json.load(open(OUT / 'ablation.json'))
    elif args.seeds:
        lo, hi = args.seeds
        recs = run(range(lo, hi))
        json.dump(recs, open(OUT / f'ablation_{lo}_{hi}.json', 'w'))
        print('shard saved; run with --merge after all shards')
        return
    else:
        recs = run(range(R_DRAWS))
        json.dump(recs, open(OUT / 'ablation.json', 'w'))
    recs = sorted(recs, key=lambda r: (r['seed'], r['K'], r['js']))
    print('\n===== component ablation, SR-MAE; d = paired vs full =====')
    for K in KCALS:
        rs = [r for r in recs if r['K'] == K]
        for B in BGRID:
            row = []
            for a in ARMS:
                v = np.mean([r['err'][a][str(B)] for r in rs])
                if a == 'ATDrive (full)':
                    row.append(f'{a}: {v:.4f}')
                else:
                    d, lo, hi = paired_cluster_boot([r['err'][a][str(B)] for r in rs],
                                                    [r['err']['ATDrive (full)'][str(B)] for r in rs],
                                                    [r['js'] for r in rs])
                    row.append(f'{a}: {v:.4f} ({d:+.4f} [{lo:+.4f},{hi:+.4f}])')
            print(f'  K_cal={K:2d} B={B:>3d}  ' + '  '.join(row))
    assert len(recs) == len(KCALS) * 64
    m = lambda a, K, B: np.mean([r['err'][a][str(B)] for r in recs if r['K'] == K])
    for a, K, B, v in (('ATDrive (full)', 4, 30, .0450), ('ATDrive (full)', 8, 55, .0337), ('ATDrive (full)', 12, 165, .0081),
                       ('w/o risk acquisition', 8, 30, .0567), ('w/o testlet', 8, 55, .0383),
                       ('w/o b-uncertainty', 12, 55, .0270)):
        assert abs(m(a, K, B) - v) < .0003, (a, K, B, m(a, K, B))
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
