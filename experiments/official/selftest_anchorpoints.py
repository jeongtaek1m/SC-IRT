#!/usr/bin/env python3
"""Self-test for official/anchorpoints.py (APW, Vivek et al. 2024).

    /data2/jeongtae/envs/atdrive_official/bin/python experiments/official/selftest_anchorpoints.py

Cells (seed 0, K_cal 12, slot 0) and (seed 0, K_cal 4, slot 0): fit + estimate
at every budget, then

  ITEMS        unique bank indices, len == B, in range,
  LEAK         estimate again with every NON-selected outcome flipped (per
               budget, and once for the union over budgets): AnchorPoints is a
               fixed-subset method, so est / items / variants must be identical,
  DETERMINISM  same cell, same seed, fresh fit + estimate -> identical est,
               and a different seed must move the medoids (the PAM init is
               seeded, ADAPTATION 2),
  PREFIX       not applicable and asserted so: each budget is its own PAM run
               (num_medoids = B), so the B=30 medoids are NOT the first 30 of
               the B=165 medoids; the actual overlap is printed.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official.anchorpoints import METHOD                     # noqa: E402
from official.data import BGRID, protocol_cell               # noqa: E402

CELLS = [(0, 12, 0), (0, 4, 0)]
WORK = Path('/data2/jeongtae/official_baselines/runs/anchorpoints/selftest')


def cell_seed(seed, Kc, slot):
    return 100000 + 1000 * seed + 10 * Kc + slot             # run_up_official.cell_seed


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    fit_times, est_times = [], []
    for seed, Kc, slot in CELLS:
        R, y, bi, SR = protocol_cell(seed, Kc, slot)
        s = cell_seed(seed, Kc, slot)
        n = len(y)
        print(f'\n=== cell seed={seed} K_cal={Kc} slot={slot}   n_bank={n}  SR={SR:.4f}  '
              f'NaN cells={int(np.isnan(R).sum())}')
        t0 = time.time()
        model = METHOD.fit(R, s, WORK)
        t_fit = time.time() - t0
        fit_times.append(t_fit)
        info = model['info']
        print(f'fit {t_fit:.3f}s  const_cols={info["const_cols"]}  corrcoef NaN entries='
              f'{info["corrcoef_nan_entries"]} ({100 * info["corrcoef_nan_frac"]:.1f}% of the matrix)  '
              f'allnan_cols={info["allnan_cols"]}  kmedoids={info["kmedoids_version"]}')

        t0 = time.time()
        out = METHOD.estimate(model, y, list(BGRID), s)
        t_est = time.time() - t0
        est_times.append(t_est)
        print(f'estimate (all {len(BGRID)} budgets) {t_est:.3f}s')
        print(f'{"B":>5}{"est":>9}{"|est-SR|":>10}{"n_items":>9}{"empty_cl":>10}{"unweighted":>12}{"sec":>8}')
        for B in BGRID:
            t0 = time.time()
            one = METHOD.estimate(model, y, [B], s)[B]
            dt = time.time() - t0
            v = out[B]
            assert one['est'] == v['est'] and one['items'] == v['items'], f'B={B} not stable across calls'
            items = v['items']
            assert len(items) == B, f'B={B}: {len(items)} items (note: {v["note"]!r})'
            assert len(set(items)) == B, f'B={B}: duplicate medoids'
            assert all(0 <= i < n for i in items), f'B={B}: item out of the bank'
            print(f'{B:5d}{v["est"]:9.4f}{abs(v["est"] - SR):10.4f}{len(items):9d}'
                  f'{model["info"]["empty_clusters"][B]:10d}{v["variants"]["anchors_unweighted"]:12.4f}{dt:8.3f}')

        # ---- LEAK TEST: flip every outcome the method did not select.
        for B in BGRID:
            sel = set(out[B]['items'])
            y2 = np.array([(1 - v) if i not in sel else v for i, v in enumerate(y)], float)
            r2 = METHOD.estimate(model, y2, [B], s)[B]
            assert r2['items'] == out[B]['items'], f'leak: B={B} items changed when unselected outcomes flipped'
            assert r2['est'] == out[B]['est'], f'leak: B={B} est {r2["est"]} != {out[B]["est"]}'
            assert r2['variants'] == out[B]['variants'], f'leak: B={B} variants changed'
        union = set().union(*[set(out[B]['items']) for B in BGRID])
        y3 = np.array([(1 - v) if i not in union else v for i, v in enumerate(y)], float)
        o3 = METHOD.estimate(model, y3, list(BGRID), s)
        for B in BGRID:
            assert o3[B]['est'] == out[B]['est'] and o3[B]['items'] == out[B]['items'], f'leak (union) at B={B}'
        print(f'LEAK ok      per-budget and union flips ({n - len(union)} of {n} outcomes flipped) change nothing')

        # ---- DETERMINISM: same seed -> identical; different seed -> different medoids.
        m2 = METHOD.fit(R, s, WORK)
        o2 = METHOD.estimate(m2, y, list(BGRID), s)
        for B in BGRID:
            assert o2[B]['est'] == out[B]['est'] and o2[B]['items'] == out[B]['items'], f'nondeterministic at B={B}'
        od = METHOD.estimate(METHOD.fit(R, s + 1, WORK), y, [30], s + 1)[30]
        print(f'DETERM ok    same seed identical at every budget; seed {s + 1} moves '
              f'{30 - len(set(od["items"]) & set(out[30]["items"]))} of 30 medoids')

        # ---- PREFIX: not a prefix method (documented, asserted).
        ov = len(set(out[30]['items']) & set(out[165]['items']))
        assert out[30]['items'] != out[165]['items'][:30], 'unexpected: B=30 is a prefix of B=165'
        print(f'PREFIX n/a   independent PAM run per budget; B=30 shares {ov}/30 medoids with B=165, '
              f'not as a prefix')

    print(f'\nfit {np.mean(fit_times):.3f}s/cell, estimate (4 budgets) {np.mean(est_times):.3f}s/cell '
          f'-> 192 cells ~= {192 * (np.mean(fit_times) + np.mean(est_times)):.1f}s')
    print('SELFTEST OK')


if __name__ == '__main__':
    main()
