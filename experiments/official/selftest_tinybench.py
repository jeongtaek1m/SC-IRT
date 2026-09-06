#!/usr/bin/env python3
"""Self-test of the official tinyBenchmarks wrapper (experiments/official/tinybench.py).

    /data2/jeongtae/envs/atdrive_official/bin/python experiments/official/selftest_tinybench.py

Cells (0, 12, 0) and (0, 4, 0) of the protocol; every budget; prints the
estimate, |est - SR|, the number of items and the seconds spent in fit and in
estimate separately (the harness runs 192 cells).

  BUDGET     items are unique bank indices and there are B of them, unless the
             official KMeans/argmin returned the same bank index for two
             clusters -- then the shortfall must equal the duplicate count the
             wrapper reports in 'note'/'diag' (ADAPTATION 9 of tinybench.py).
  LEAK       estimate() is called again on y2 = y with every item OUTSIDE the
             selected set flipped. tinyBenchmarks is a fixed-subset method, so
             every readout and every item must be bit-identical.
  DETERMIN.  the same cell twice -> identical estimates (estimate() for both
             cells; a full re-fit, i.e. three more py-irt runs, for the K = 4
             cell, which is the cheap one).
  NESTING    reported, not asserted: tinyBenchmarks is not adaptive and each
             budget runs its own KMeans, so the B = 30 anchors are NOT the
             first 30 of the B = 165 anchors. The prefix requirement applies to
             adaptive/prefix methods only; the overlap is printed so that the
             non-nesting is on the record.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official.data import BGRID, protocol_cell            # noqa: E402
from official.tinybench import METHOD                     # noqa: E402

SCRATCH = Path('/data2/jeongtae/official_baselines/runs/selftest_tinybench')
CELLS = [(0, 12, 0), (0, 4, 0)]


def cell_seed(seed, Kc, slot):
    return 100000 + 1000 * seed + 10 * Kc + slot


def variants_equal(a, b):
    return set(a) == set(b) and all(a[k] == b[k] for k in a)


def main():
    t_start = time.time()
    for (seed, Kc, slot) in CELLS:
        R, y, bi, SR = protocol_cell(seed, Kc, slot)
        cs = cell_seed(seed, Kc, slot)
        wd = SCRATCH / f's{seed}_K{Kc}_p{slot}'
        wd.mkdir(parents=True, exist_ok=True)
        print(f'\n=== cell seed={seed} K_cal={Kc} slot={slot}  bank={len(bi)}  '
              f'SR={SR:.4f}  NaN cells={int(np.isnan(R).sum())}')

        t0 = time.time()
        model = METHOD.fit(R, cs, wd)
        fit_s = time.time() - t0
        info = model['info']
        print(f'fit: {fit_s:.1f}s   D={info["D"]} (val errors {["%.4f" % e for e in info["val_errors"]]}), '
              f'b={info["b_val_error"]:.4f}, v={info["v_within_planner_var"]:.4f}, '
              f'py-irt {info["fit_seconds"]}')

        t0 = time.time()
        out = METHOD.estimate(model, y, list(BGRID), cs)
        est_s = time.time() - t0
        print(f'estimate: {est_s:.1f}s for {len(BGRID)} budgets '
              f'({est_s / len(BGRID):.1f}s per budget)')
        print(f'{"B":>5}{"est(gp-IRT)":>13}{"|est-SR|":>10}{"n_items":>9}'
              f'{"anchor":>9}{"p-IRT":>9}{"lambda":>8}   note')
        for B in BGRID:
            v = out[B]
            print(f'{B:5d}{v["est"]:13.4f}{abs(v["est"] - SR):10.4f}{len(v["items"]):9d}'
                  f'{v["variants"]["anchor"]:9.4f}{v["variants"]["pirt"]:9.4f}'
                  f'{v["diag"]["lambda"]:8.3f}   {v["note"][:60]}')

        # ---- budget / uniqueness -----------------------------------------
        for B in BGRID:
            items = out[B]['items']
            assert len(set(items)) == len(items), f'B={B}: duplicate items reported'
            assert all(0 <= i < len(bi) for i in items), f'B={B}: item outside the bank'
            if len(items) != B:
                assert len(items) < B and out[B]['note'], f'B={B}: {len(items)} items and no note'
                assert out[B]['diag']['n_unique_anchors'] == len(items)
                print(f'  B={B}: {len(items)} < {B} items, documented: {out[B]["note"][:80]}')
        print('  BUDGET OK')

        # ---- leak test ----------------------------------------------------
        for B in BGRID:
            items = out[B]['items']
            y2 = np.asarray(y, float).copy()
            mask = np.ones(len(y2), bool)
            mask[items] = False
            y2[mask] = 1 - y2[mask]                       # flip every NON-selected item
            o2 = METHOD.estimate(model, y2, [B], cs)[B]
            assert o2['items'] == items, f'B={B}: items changed when unselected outcomes flipped'
            assert o2['est'] == out[B]['est'], f'B={B}: est changed ({o2["est"]} vs {out[B]["est"]})'
            assert variants_equal(o2['variants'], out[B]['variants']), f'B={B}: a variant changed'
        print(f'  LEAK OK (all {len(BGRID)} budgets identical under flipped unselected outcomes)')

        # ---- determinism ---------------------------------------------------
        out_b = METHOD.estimate(model, y, list(BGRID), cs)
        for B in BGRID:
            assert out_b[B]['items'] == out[B]['items'], f'B={B}: items not deterministic'
            assert out_b[B]['est'] == out[B]['est'], f'B={B}: est not deterministic'
            assert variants_equal(out_b[B]['variants'], out[B]['variants'])
        print('  DETERMINISM(estimate) OK')
        if Kc == 4:                                       # the cheap cell: re-run the three py-irt fits
            t0 = time.time()
            m2 = METHOD.fit(R, cs, SCRATCH / f's{seed}_K{Kc}_p{slot}_refit')
            o3 = METHOD.estimate(m2, y, list(BGRID), cs)
            assert m2['info']['D'] == info['D'], 'D not deterministic'
            assert np.allclose(m2['A'], model['A'], atol=0, rtol=0), 'discriminations not deterministic'
            assert np.allclose(m2['B'], model['B'], atol=0, rtol=0), 'difficulties not deterministic'
            for B in BGRID:
                assert o3[B]['items'] == out[B]['items'], f'B={B}: items not deterministic across fits'
                assert o3[B]['est'] == out[B]['est'], f'B={B}: est not deterministic across fits'
            print(f'  DETERMINISM(fit) OK, re-fit in {time.time() - t0:.1f}s')

        # ---- nesting (reported, not required) --------------------------------
        big = out[max(BGRID)]['items']
        small = out[min(BGRID)]['items']
        pref = big[:len(small)]
        print(f'  NESTING: B={min(BGRID)} items are the first {len(small)} of B={max(BGRID)}: '
              f'{small == pref}; overlap {len(set(small) & set(big))}/{len(small)} '
              f'(not required: fixed-subset method, one KMeans per budget)')

        assert METHOD.stop(model, y, cs) == {}, 'tinyBenchmarks publishes no stopping rule'

    print(f'\ntotal {time.time() - t_start:.1f}s')
    print('SELFTEST OK')


if __name__ == '__main__':
    main()
