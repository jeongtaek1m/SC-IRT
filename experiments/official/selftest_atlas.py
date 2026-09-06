#!/usr/bin/env python
"""Self-test of experiments/official/atlas.py (ATLAS, Li et al. 2025).

    /data2/jeongtae/envs/atdrive_official/bin/python experiments/official/selftest_atlas.py

Cells (seed, K_cal, slot) = (0, 12, 0) and (0, 4, 0) of the protocol:
  * fit once, then estimate for every budget; est, |est - SR|, n_items, seconds;
  * items are unique bank indices, len == B unless the fitted bank is smaller
    (ADAPTATION 7: items on which all calibration planners agree are dropped by
    01_fit_irt.r's constant-column cleaning and have no item parameters);
  * LEAK: estimate again with every item that was never administered flipped ->
    identical items and identical est;
  * DETERMINISM: same seed twice -> identical items and est;
  * PREFIX: the B = 30 items are the first 30 of the B = 165 items;
  * stop(): their SE rule, tau in {0.1, 0.2, 0.3}.

The mirt 3PL fit is the slow part (their technical = list(NCYCLES = 100000), see
the timing lines below); the CAT stage is a few seconds per cell.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official.atlas import METHOD                            # noqa: E402
from official.data import BGRID, protocol_cell               # noqa: E402

SCRATCH = Path('/data2/jeongtae/official_baselines/runs/atlas/selftest')
CELLS = [(0, 12, 0), (0, 4, 0)]


def cell_seed(seed, Kc, slot):
    return 100000 + 1000 * seed + 10 * Kc + slot              # run_up_official.cell_seed


def check_cell(seed, Kc, slot):
    R, y, bi, SR = protocol_cell(seed, Kc, slot)
    n_bank = len(bi)
    cs = cell_seed(seed, Kc, slot)
    wd = SCRATCH / f's{seed}_K{Kc}_p{slot}'
    wd.mkdir(parents=True, exist_ok=True)
    print(f'\n=== cell seed {seed} K_cal {Kc} slot {slot}: R {R.shape}, bank {n_bank}, SR {SR:.4f} ===')

    t0 = time.time()
    model = METHOD.fit(R, cs, wd)
    fit_s = time.time() - t0
    info = model['info']
    n_fit = len(model['item_ids'])
    print(f'fit   {fit_s:8.1f}s  mirt 3PL: {info["n_rows_fit"]}x{info["n_items_fit"]} '
          f'(dropped {n_bank - info["n_items_fit"]} constant items, {info["dropped_constant_rows"]} constant planners), '
          f'converged={info["converged"]} EM iterations={info["em_iterations"]}')
    print(f'      a1 [{info["a1"]["min"]:.3g}, {info["a1"]["max"]:.3g}] median {info["a1"]["median"]:.3g}, '
          f'{info["a1"]["n_nonpositive"]} non-positive; b median {info["b"]["median"]:.3g}, '
          f'{info["b"]["n_abs_gt_5"]} with |b| > 5; g [{info["g"]["min"]:.3g}, {info["g"]["max"]:.3g}]')
    print(f'      mirt warnings: {info["warnings"] if info["warnings"] else "none"}')

    t0 = time.time()
    res = METHOD.estimate(model, y, list(BGRID), cs)
    est_s = time.time() - t0
    print(f'est   {est_s:8.1f}s  ({est_s / len(BGRID):.1f}s per budget)')
    for B in BGRID:
        v = res[B]
        it = v['items']
        assert len(set(it)) == len(it), f'B={B}: repeated item'
        assert all(0 <= i < n_bank for i in it), f'B={B}: item outside the bank'
        assert set(it) <= set(model['item_ids']), f'B={B}: item outside the fitted bank'
        assert len(it) == min(B, n_fit), f'B={B}: {len(it)} items, expected {min(B, n_fit)}'
        note = v.get('note', '')
        if len(it) < B:
            assert note, f'B={B}: short run without a note'
        print(f'  B{B:<4d} est {v["est"]:.4f}  |est-SR| {abs(v["est"] - SR):.4f}  n_items {len(it):3d}  '
              + '  '.join(f'{k}={x:.4f}' for k, x in v['variants'].items()) + ('  ' + note[:70] if note else ''))

    # PREFIX (adaptive method: the budget grid must be nested)
    for B in BGRID[:-1]:
        k = min(B, n_fit)
        assert res[B]['items'] == res[BGRID[-1]]['items'][:k], f'B={B} is not a prefix of B={BGRID[-1]}'
    print(f'  prefix   B30/B55/B110 are prefixes of B{BGRID[-1]}: OK')

    # DETERMINISM
    t0 = time.time()
    res2 = METHOD.estimate(model, y, list(BGRID), cs)
    det_s = time.time() - t0
    for B in BGRID:
        assert res2[B]['items'] == res[B]['items'], f'B={B}: items differ across identical runs'
        assert res2[B]['est'] == res[B]['est'], f'B={B}: est differs across identical runs'
    print(f'  determ   identical items and est on a second run with the same seed: OK ({det_s:.1f}s)')

    # LEAK: flip every item that was never administered
    used = sorted(set().union(*[set(res[B]['items']) for B in BGRID]))
    y2 = np.asarray(y, float).copy()
    mask = np.ones(n_bank, bool)
    mask[used] = False
    y2[mask] = 1 - y2[mask]
    res3 = METHOD.estimate(model, y2, list(BGRID), cs)
    for B in BGRID:
        assert res3[B]['items'] == res[B]['items'], f'B={B}: items changed when unread outcomes were flipped'
        assert res3[B]['est'] == res[B]['est'], f'B={B}: est changed when unread outcomes were flipped'
    print(f'  leak     {int(mask.sum())} never-administered outcomes flipped, items and est unchanged: OK')

    # stop(): their SE rule
    t0 = time.time()
    st = METHOD.stop(model, y, cs)
    stop_s = time.time() - t0
    for k, v in st.items():
        it = v['items']
        assert len(set(it)) == len(it) and all(0 <= i < n_bank for i in it)
        assert len(it) >= min(30, n_fit), f'{k}: stopped before min_items = 30'
        # the stopping run may run past the largest budget (max_items = bank size);
        # the two trajectories must agree wherever both exist
        common = min(len(it), len(res[BGRID[-1]]['items']))
        assert it[:common] == res[BGRID[-1]]['items'][:common], f'{k}: diverges from the B={BGRID[-1]} run'
        print(f'  {k:<8s} est {v["est"]:.4f}  |est-SR| {abs(v["est"] - SR):.4f}  n_items {len(it):3d}  '
              f'final SE {v["variants"]["theta_se"]:.3f}')
    print(f'  stop  {stop_s:8.1f}s  (3 rules, prefixes of the fixed-budget run: OK)')
    return {'fit_s': fit_s, 'est_s': est_s, 'stop_s': stop_s}


def main():
    t0 = time.time()
    times = {}
    for seed, Kc, slot in CELLS:
        times[(seed, Kc, slot)] = check_cell(seed, Kc, slot)
    print('\ntimings (the harness runs 192 cells = 16 draws x 3 K_cal x 4 planners)')
    for c, t in times.items():
        print(f'  cell {c}: fit {t["fit_s"]:.1f}s, estimate (4 budgets) {t["est_s"]:.1f}s, stop (3 rules) {t["stop_s"]:.1f}s')
    print(f'total {time.time() - t0:.1f}s')
    print('SELFTEST OK')


if __name__ == '__main__':
    main()
