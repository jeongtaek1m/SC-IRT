#!/usr/bin/env python3
"""The localize budget K is a bank constant, estimated on the calibration panel.

Leave-one-planner-out simulation: each calibration planner is evaluated with
the bank re-calibrated from the others, running localize(K) -> cover for
K in {0, 5, 10, 15, 20, 25, 30, 40, 60} and scoring SR-MAE at B in {30, 40, 60, 80}. On
Bench2Drive the loss is flat for K in [15, 30] and degrades for K >= 40
(never switching); K = 20 is used everywhere (`scirt.acquisition.K_LOCALIZE`).

    python experiments/run_k_calibration.py --seeds 0 4   # shard (~25 min, GPU)
    python experiments/run_k_calibration.py --merge       # LOO curves + anchors
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
from scirt.acquisition import localize_cover

OUT = Path(os.environ.get('SCIRT_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
JCALS = tuple(int(x) for x in os.environ.get('SCIRT_JCALS', '7, 10, 13').split(','))
KG = (0, 5, 10, 15, 20, 25, 30, 40, 60)
BG = (30, 40, 60, 80)


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
        for Jc in JCALS:
            cs = subsample(cols, seed, Jc)
            for j in cs:
                csl = [c for c in cs if c != j]
                f1 = calibrate(panel.Y, calR, csl, mode='1pl')
                f2 = calibrate(panel.Y, calR, csl, mode='2pl')
                bi, yy = panel.bank_rows(calR, j)
                M1 = marginal_curves(f1['b'][bi], f1['s'][bi])
                a, b = f2['a'][bi], f2['b'][bi]
                SR = float(yy.mean())
                err = {}
                for K in KG:
                    o = localize_cover(a, b, f2['th'], yy, K=K, T=max(BG))
                    err[str(K)] = {str(B): abs(map_fill(M1, yy, o[:B]) - SR) for B in BG}
                recs.append({'seed': seed, 'J': Jc, 'j': int(j), 'err': err})
            print(f'seed {seed} J{Jc} done', flush=True)
    return recs


def report(recs):
    print('\n===== LOO calibration-panel SR-MAE by localize budget K (mean over B in {30,40,60,80}) =====')
    print(f'{"J_cal":>5s} ' + ' '.join(f'K={K:<4d}' for K in KG) + '   flat[15,30] range   K>=40 penalty')
    stats = {}
    for J in JCALS:
        m = [np.mean([np.mean([r['err'][str(K)][str(B)] for B in BG]) for r in recs if r['J'] == J]) for K in KG]
        flat = [m[KG.index(K)] for K in (15, 20, 25, 30)]
        pen = m[KG.index(40)] - min(flat)
        stats[J] = (max(flat) - min(flat), pen, m)
        print(f'{J:5d} ' + ' '.join(f'{x:.4f}' for x in m) + f'   {max(flat) - min(flat):.4f}            {pen:+.4f}')
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs=2, type=int, default=None)
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.merge:
        recs = sum([json.load(open(f)) for f in sorted(glob.glob(str(OUT / 'k_loo_*_*.json')))], [])
        json.dump(recs, open(OUT / 'k_calibration.json', 'w'))
    elif args.seeds:
        lo, hi = args.seeds
        recs = run(range(lo, hi))
        json.dump(recs, open(OUT / f'k_loo_{lo}_{hi}.json', 'w'))
        print('shard saved; run with --merge after all shards')
        return
    else:
        recs = run(range(R_DRAWS))
        json.dump(recs, open(OUT / 'k_calibration.json', 'w'))
    stats = report(recs)
    assert len(set(r['seed'] for r in recs)) == R_DRAWS
    assert stats[13][0] < .0015 and stats[13][1] > .002          # flat plateau, cliff at K >= 40
    assert all(stats[J][0] < .005 for J in JCALS)
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
