#!/usr/bin/env python3
"""Full-system ablation — the IRT side and the CAT side, at a fixed budget and at a risk target.

Merge-only: no new trajectories. DriveAT is decomposed into two IRT pieces
(the exact difficulty posterior, the planner x type testlet) and two CAT
pieces (the Delta-R1 acquisition, the LOO-calibrated risk scale c), and each
is switched off alone. Every arm is scored twice — at the fixed budget
B = 55 (SR-MAE, paired planner-cluster delta vs full) and under the
risk-target rule c * R1_t <= eps (mean rollouts, SR-MAE at the stop,
fraction stopped only by the bank running out). Coverage P(|err| <= eps) is
still written to the json but no longer printed.

  arm                     fixed budget                stopping trajectories / c
  DriveAT (full)           ablation 'DriveAT (full)'    adaptive + risk_cal, DriveAT order
  w/o b posterior         ablation 'w/o b-uncertainty'  pointcurves/{adaptive,risk_cal}, DriveAT order
  w/o testlet             ablation 'w/o testlet'      notestlet/{adaptive,risk_cal}, DriveAT order
  w/o Delta-R1 acq        ablation 'w/o risk acq.'    adaptive + risk_cal, Random order
  w/o LOO c               = full (c is a stopping-only knob)   adaptive, c = 1

The two ablation result trees come from the same scripts with
`DRIVEAT_NO_TESTLET=1` / `DRIVEAT_POINT_CURVES=1` and `DRIVEAT_RESULTS_DIR`.

    python experiments/run_system_ablation.py      # the table, results/system_ablation.json, anchors
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from driveat.metrics import paired_cluster_boot
from run_adaptive import risk_stopped

OUT = Path(os.environ.get('DRIVEAT_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
KCALS = tuple(int(x) for x in os.environ.get('DRIVEAT_KCALS', '4,8,12').split(','))
BGRID = [30, 55, 110, 165]
BMAIN = 55
EPS = (0.05, 0.03)
# arm -> (ablation.json arm for the fixed budgets, results subdir for the stopping tracks, bank order, c)
ARMS = {'DriveAT (full)':    ('DriveAT (full)', '', 'DriveAT', None),
        'w/o b posterior':  ('w/o b-uncertainty', 'pointcurves', 'DriveAT', None),
        'w/o testlet':      ('w/o testlet', 'notestlet', 'DriveAT', None),
        'w/o Delta-R1 acq': ('w/o risk acquisition', '', 'Random', None),
        'w/o LOO c':        ('DriveAT (full)', '', 'DriveAT', 1.0)}


def load(sub):
    """(trajectory records keyed by (seed, K, js), risk scales) of one results tree."""
    d = OUT / sub if sub else OUT
    recs = json.load(open(d / 'adaptive.json'))
    assert len(recs) == len(KCALS) * 64, (sub, len(recs))
    return {(r['seed'], r['K'], r['js']): r for r in recs}, json.load(open(d / 'risk_cal.json'))


def main():
    abl = sorted(json.load(open(OUT / 'ablation.json')), key=lambda r: (r['seed'], r['K'], r['js']))
    assert len(abl) == len(KCALS) * 64, len(abl)
    trees = {sub: load(sub) for sub in sorted({a[1] for a in ARMS.values()})}
    res = {}
    print('\n===== full-system ablation: SR-MAE at B = 55 and under the risk target c*R1 <= eps '
          '(d = paired planner-cluster delta vs full) =====')
    for K in KCALS:
        rs = [r for r in abl if r['K'] == K]
        js = [r['js'] for r in rs]
        print(f'-- K_cal = {K} --      B={BMAIN} SR-MAE            '
              + '   '.join(f'eps={e:.2f}: roll (cap)  MAE' for e in EPS))
        for arm, (a, sub, o, c0) in ARMS.items():
            fixed = {str(B): float(np.mean([r['err'][a][str(B)] for r in rs])) for B in BGRID}
            d, lo, hi = ((0.0, 0.0, 0.0) if arm == 'DriveAT (full)' else
                         paired_cluster_boot([r['err'][a][str(BMAIN)] for r in rs],
                                             [r['err']['DriveAT (full)'][str(BMAIN)] for r in rs], js))
            recs, C = trees[sub]
            st = {}
            for eps in EPS:
                v = np.array([risk_stopped(recs[(r['seed'], K, r['js'])], o,
                                           C[f"{r['seed']}|{K}|{o}"] if c0 is None else c0, eps) for r in rs])
                B_, e_, cr_, cap_ = v.T
                st[eps] = {'rollouts': float(B_.mean()), 'mae': float(e_.mean()),
                           'coverage': float(np.mean(e_ <= eps)), 'cap': float(cap_.mean()),
                           'gap': float(e_.mean() - cr_.mean())}
            res[f'K{K}|{arm}'] = {'fixed': fixed, 'd55': [float(d), float(lo), float(hi)], 'stop':
                                  {f'{e:.2f}': st[e] for e in EPS}}
            dtxt = '   —                    ' if arm == 'DriveAT (full)' else f'({d:+.4f} [{lo:+.4f},{hi:+.4f}])'
            print(f'   {arm:17s} {fixed[str(BMAIN)]:.4f} {dtxt}  '
                  + '  '.join(f'{st[e]["rollouts"]:5.1f} ({st[e]["cap"]:3.0%})  {st[e]["mae"]:.4f}'
                              for e in EPS))
    json.dump(res, open(OUT / 'system_ablation.json', 'w'), indent=1)
    print(f'\nwritten: {OUT / "system_ablation.json"}')
    for k, f, v, tol in (('K4|DriveAT (full)', lambda x: x['fixed']['55'], .0332, .0003),
                         ('K8|w/o b posterior', lambda x: x['fixed']['55'], .0344, .0003),
                         ('K12|w/o testlet', lambda x: x['stop']['0.05']['rollouts'], 94.4, .5),
                         ('K8|w/o Delta-R1 acq', lambda x: x['stop']['0.05']['mae'], .0260, .0005),
                         ('K4|w/o LOO c', lambda x: x['stop']['0.05']['rollouts'], 29.0, .5),
                         ('K8|w/o b posterior', lambda x: x['stop']['0.03']['rollouts'], 123.3, .5)):
        assert abs(f(res[k]) - v) < tol, (k, f(res[k]))
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    main()
