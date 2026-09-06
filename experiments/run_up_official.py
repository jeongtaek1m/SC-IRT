#!/usr/bin/env python3
"""Table 1 (UP) with the published baselines run through their OFFICIAL
implementations (experiments/official/*.py; contract in official/base.py).

Same protocol as run_up_frontier.py — 16 draws x K_cal {4, 8, 12} x 4
evaluation planners = 64 evaluations per cell, budgets {30, 55, 110, 165}, the
bank = the evaluation planner's recorded routes — but every baseline row is
produced by the method's own code: its item-parameter fit, its item selection
and its native readout. The ATDrive row and the Random rows are the ones of
results/up_frontier.json (unchanged; identical draws, so every comparison is
paired at the evaluation level).

Runs in the official environment (CPU; py-irt / mirt / catR are the slow parts):

    P=/data2/jeongtae/envs/atdrive_official/bin/python
    $P experiments/run_up_official.py --methods tinybench fluid --seeds 0 4   # shard (draws 0-3)
    $P experiments/run_up_official.py --merge                                  # tables + json of record

Per-cell records are written incrementally to
results/up_official_<methods>_<a>_<b>.json (the method tag is part of the name:
two shards of different methods over the same seed range would otherwise write
the same file and the second would overwrite the first), so a killed shard
resumes; a method that raises in a cell is recorded with its
error string and the cell is excluded from that method's mean (the count is
printed — a row is only reported when every cell ran).
"""
import argparse
import glob
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official.base import provenance, write_json                        # noqa: E402
from official.data import BGRID, KCALS, NDRAWS, protocol_cell            # noqa: E402

OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
SCRATCH = Path(os.environ.get('ATDRIVE_OFFICIAL_SCRATCH', '/data2/jeongtae/official_baselines/runs'))
ALL_METHODS = ['tinybench', 'fluid', 'anchorpoints', 'disco', 'metabench', 'atlas', 'catr']


def load_method(name):
    mod = importlib.import_module(f'official.{name}')
    return mod.METHOD


def cell_seed(seed, Kc, slot):
    return 100000 + 1000 * seed + 10 * Kc + slot


def run(methods, seeds, out_path):
    recs = json.load(open(out_path)) if out_path.exists() else []
    done = {(r['seed'], r['K'], r['slot'], r['method']) for r in recs}
    M = {m: load_method(m) for m in methods}
    for seed in seeds:
        for Kc in KCALS:
            for slot in range(4):
                R, y, bi, SR = protocol_cell(seed, Kc, slot)
                for name, m in M.items():
                    if (seed, Kc, slot, name) in done:
                        continue
                    rec = {'seed': seed, 'K': Kc, 'slot': slot, 'method': name, 'SR': SR, 'n_bank': len(bi)}
                    wd = SCRATCH / name / f's{seed}_K{Kc}_p{slot}'
                    wd.mkdir(parents=True, exist_ok=True)
                    t0 = time.time()
                    try:
                        model = m.fit(R, cell_seed(seed, Kc, slot), wd)
                        t1 = time.time()
                        est = m.estimate(model, y, list(BGRID), cell_seed(seed, Kc, slot))
                        stops = m.stop(model, y, cell_seed(seed, Kc, slot)) or {}
                        rec['fit_s'] = t1 - t0
                        rec['est_s'] = time.time() - t1
                        rec['budgets'] = {str(B): {'est': float(v['est']), 'err': abs(float(v['est']) - SR),
                                                   'n_items': len(v['items']), 'items': [int(i) for i in v['items']],
                                                   'variants': {k: float(x) for k, x in (v.get('variants') or {}).items()
                                         if not (k.endswith('_theta') or k.endswith('_se') or k == 'ability')},
                                                   'note': v.get('note', '')}
                                          for B, v in est.items()}
                        rec['stops'] = {str(k): {'est': float(v['est']), 'err': abs(float(v['est']) - SR),
                                                 'n_items': len(v['items']), 'items': [int(i) for i in v['items']],
                                                 'variants': {kk: float(x) for kk, x in (v.get('variants') or {}).items()
                                                              if not (kk.endswith('_theta') or kk.endswith('_se') or kk == 'ability')}}
                                        for k, v in stops.items()}
                        rec['ability_variants'] = {str(B): {k: float(x) for k, x in (v.get('variants') or {}).items()
                                                            if k.endswith('_theta') or k.endswith('_se') or k == 'ability'}
                                                   for B, v in est.items()}
                        rec['model_info'] = {k: v for k, v in (model.get('info') or {}).items()} if isinstance(model, dict) else {}
                    except Exception as e:                                     # recorded, never fatal for the shard
                        rec['error'] = f'{type(e).__name__}: {e}'
                        rec['traceback'] = traceback.format_exc()[-3000:]
                        rec['fit_s'] = time.time() - t0
                    recs.append(rec)
                    write_json(out_path, recs)
                    msg = rec.get('error', ' '.join(f"B{B}:{v['err']:.4f}" for B, v in rec.get('budgets', {}).items()))
                    print(f'seed {seed} K{Kc} slot {slot} {name:12} {rec["fit_s"]:6.1f}s  {msg[:120]}', flush=True)
    return recs


def merge():
    recs = []
    for f in sorted(glob.glob(str(OUT / 'up_official_*_[0-9]*_[0-9]*.json'))
                    + glob.glob(str(OUT / 'up_official_[0-9]*_[0-9]*.json'))):
        recs += json.load(open(f))
    ref = json.load(open(OUT / 'up_frontier.json'))
    refE = {(r['seed'], r['K'], r['js']): r for r in ref}
    # slot -> js map via the draw
    from official.data import draw
    js_of = {(s, k): draw(s)[0] for s in range(NDRAWS) for k in KCALS}
    table = {}
    for r in recs:
        if 'error' in r:
            table.setdefault(r['method'], {}).setdefault('errors', []).append((r['seed'], r['K'], r['slot'], r['error'][:80]))
            continue
        key = (r['seed'], r['K'], js_of[(r['seed'], r['K'])][r['slot']])
        for B, v in r['budgets'].items():
            t = table.setdefault(r['method'], {}).setdefault('cells', {}).setdefault((r['K'], int(B)), {'err': [], 'n': [], 'd_atdrive': [], 'var': {}})
            t['err'].append(v['err'])
            t['n'].append(v['n_items'])
            t['d_atdrive'].append(v['err'] - refE[key]['err']['ATDrive'][B])
            for vn, x in v['variants'].items():
                t['var'].setdefault(vn, []).append(abs(x - r['SR']))
        for rule, v in r.get('stops', {}).items():
            t = table.setdefault(r['method'], {}).setdefault('stops', {}).setdefault((r['K'], rule), {'err': [], 'n': []})
            t['err'].append(v['err'])
            t['n'].append(v['n_items'])
    out = {'provenance': provenance(), 'methods': {}}
    print(f"{'method':14}{'K':>3}{'B':>5}{'n_eval':>7}{'SR-MAE':>8}{'items':>7}{'d vs ATDrive':>14}   variants")
    for m, T in table.items():
        out['methods'][m] = {'errors': T.get('errors', []), 'cells': {}, 'stops': {}}
        for (K, B), t in sorted(T.get('cells', {}).items()):
            row = {'n_eval': len(t['err']), 'sr_mae': float(np.mean(t['err'])), 'items_mean': float(np.mean(t['n'])),
                   'd_atdrive': float(np.mean(t['d_atdrive'])),
                   'variants': {vn: float(np.mean(x)) for vn, x in t['var'].items()}}
            out['methods'][m]['cells'][f'K{K}_B{B}'] = row
            print(f"{m:14}{K:3d}{B:5d}{row['n_eval']:7d}{row['sr_mae']:8.4f}{row['items_mean']:7.1f}{row['d_atdrive']:+14.4f}   "
                  + ' '.join(f'{k}={v:.4f}' for k, v in row['variants'].items()))
        for (K, rule), t in sorted(T.get('stops', {}).items()):
            row = {'n_eval': len(t['err']), 'sr_mae': float(np.mean(t['err'])), 'items_mean': float(np.mean(t['n']))}
            out['methods'][m]['stops'][f'K{K}_{rule}'] = row
            print(f"{m:14}{K:3d}{rule:>5}{row['n_eval']:7d}{row['sr_mae']:8.4f}{row['items_mean']:7.1f}   (own stopping rule)")
        if T.get('errors'):
            print(f"{m:14} {len(T['errors'])} cells FAILED, e.g. {T['errors'][0]}")
    write_json(OUT / 'up_official.json', out)
    print(f"wrote {OUT / 'up_official.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS)
    ap.add_argument('--seeds', nargs=2, type=int, default=[0, NDRAWS], help='draw range [a, b)')
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    if a.merge:
        return merge()
    OUT.mkdir(exist_ok=True)
    tag = '-'.join(sorted(a.methods))
    run(a.methods, range(a.seeds[0], a.seeds[1]), OUT / f'up_official_{tag}_{a.seeds[0]}_{a.seeds[1]}.json')


if __name__ == '__main__':
    main()
