#!/usr/bin/env python3
"""Component ablation — switch OUR pieces off, one at a time (2 x 2).

Same readout and the same bank for every row; only the marked component
changes. Primary protocol, B in {30, 55}.

  SC-IRT (full)            difficulty-marginalised curves + Delta-R1 acquisition
  w/o b-uncertainty        plug-in curves sigmoid(theta - b_hat) everywhere (s_s = 0)
  w/o risk acquisition     random scene order, marginalised inference unchanged
  w/o both                 plug-in curves + random order

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
from scirt.b2d import Panel
from scirt.splits import unified_split, R_DRAWS
from scirt.calibration import calibrate
from scirt.curves import marginal_curves, PRIOR
from scirt.bayes import post_from, map_fill
from scirt.acquisition import r1_pick
from scirt.metrics import paired_seed_boot

OUT = Path(os.environ.get('SCIRT_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
KCALS = tuple(int(x) for x in os.environ.get('SCIRT_KCALS', '7,10,16').split(','))
BGRID = [30, 55]
T = max(BGRID)
ARMS = ('SC-IRT (full)', 'w/o b-uncertainty', 'w/o risk acquisition', 'w/o both')


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
            for js in hp:
                bi, yy = panel.bank_rows(calR, js)
                n = len(bi)
                SR = float(yy.mean())
                Mm = marginal_curves(f1['b'][bi], f1['s'][bi])
                Mp = marginal_curves(f1['b'][bi], np.full(n, 1e-9))
                perm = [int(i) for i in np.random.RandomState(100 + seed * 20 + js).permutation(n)[:T]]
                sets = {'SC-IRT (full)': (Mm, r1_traj(Mm, yy, n)),
                        'w/o b-uncertainty': (Mp, r1_traj(Mp, yy, n)),
                        'w/o risk acquisition': (Mm, perm),
                        'w/o both': (Mp, perm)}
                recs.append({'seed': seed, 'K': Kc, 'js': int(js),
                             'err': {a: {str(B): abs(map_fill(M, yy, o[:B]) - SR) for B in BGRID}
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
        json.dump(recs, open(OUT / 'ablation.json', 'w'))
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
                if a == 'SC-IRT (full)':
                    row.append(f'{a}: {v:.4f}')
                else:
                    d, lo, hi = paired_seed_boot([r['err'][a][str(B)] for r in rs],
                                                 [r['err']['SC-IRT (full)'][str(B)] for r in rs],
                                                 n_seeds=16, per_seed=6)
                    row.append(f'{a}: {v:.4f} ({d:+.4f} [{lo:+.4f},{hi:+.4f}])')
            print(f'  K_cal={K:2d} B={B:>3d}  ' + '  '.join(row))
    assert len(recs) == len(KCALS) * 96
    m = lambda a, K, B: np.mean([r['err'][a][str(B)] for r in recs if r['K'] == K])
    for a, K, B, v in (('SC-IRT (full)', 10, 30, .0453), ('w/o risk acquisition', 10, 30, .0586),
                       ('SC-IRT (full)', 7, 30, .0497), ('w/o risk acquisition', 7, 30, .0617),
                       ('SC-IRT (full)', 16, 55, .0419), ('w/o both', 16, 55, .0375)):
        assert abs(m(a, K, B) - v) < .0003, (a, K, B, m(a, K, B))
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
