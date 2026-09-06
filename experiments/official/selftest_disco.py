#!/usr/bin/env python
"""Self-test for experiments/official/disco.py (DISCO through the official code).

    /data2/jeongtae/envs/atdrive_official/bin/python \
        experiments/official/selftest_disco.py

Cells (seed 0, K_cal 12, slot 0) and (seed 0, K_cal 4, slot 0); budgets
{30, 55, 110, 165}.  Checks, per the wrapper contract:
  * fit + estimate run, printing est, |est - SR|, n_items and separate fit /
    estimate timings (the harness runs 192 cells per method);
  * every 'items' list is a set of unique, in-range bank indices with len == B;
  * PREFIX: the B = 30 items are the first 30 of the B = 165 items, for the pds
    order (which is 'items') and for the jsd order behind the jsd_* variants --
    both are one stable descending sort of a fixed score, so every budget is a
    prefix of the next;
  * LEAK 1: re-estimating with every route outside the union of the read sets
    flipped gives bit-identical items, est and variants (fixed-subset method);
  * LEAK 2, per budget: flipping everything outside
    items_pds(B) u items_jsd(B) leaves that budget's entry bit-identical (the
    other budgets legitimately move, they read flipped routes).  estimate() is
    always called with the full budget list so that the global RNG that the
    official sklearn estimators draw from is in the same state;
  * LEAK 3, per budget: flipping only the routes that the jsd order reads at
    that budget and the pds order does not leaves 'est' and every 'pds_*'
    variant of that budget bit-identical, while the jsd_* variants move (the jsd_* readouts are a separate B-item evaluation;
    see the VARIANTS section of disco.py);
  * DETERMINISM (pipeline): fit + estimate again with the same seed and the same
    workdir -> identical info, items, est and variants;
  * DETERMINISM (py-irt): retraining the chosen final IRT model into a FRESH
    directory reproduces the discrimination and difficulty parameters bit for
    bit (irt.py:78 passes `--seed 42 --deterministic`).  This is split out
    because a full fresh fit costs ~2 minutes per cell (5-6 py-irt runs), which
    would put the self-test far over its time budget.

The two primary fits reuse a persistent scratch directory, so the first run of
this file pays for the py-irt training (~2 min per cell) and later runs do not.
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                     # experiments/
sys.path.insert(0, str(HERE.parents[1]))                 # repo root

from official.data import BGRID, protocol_cell           # noqa: E402
from official.disco import METHOD, _train_and_load                 # noqa: E402

SCRATCH = Path('/data2/jeongtae/official_baselines/runs/disco/selftest')
CELLS = [(0, 12, 0), (0, 4, 0)]


def cell_seed(seed, Kc, slot):
    return 100000 + 1000 * seed + 10 * Kc + slot          # run_up_official.cell_seed


def jsd_items(res, B):
    """The jsd order is not returned in 'items'; recover it from the wrapper."""
    return res[B]['diag']['jsd_items']


def main():
    fit_times, est_times = [], []
    for seed, Kc, slot in CELLS:
        R, y, bi, SR = protocol_cell(seed, Kc, slot)
        n = len(bi)
        cs = cell_seed(seed, Kc, slot)
        tag = f's{seed}_K{Kc}_p{slot}'
        wd = SCRATCH / tag
        wd.mkdir(parents=True, exist_ok=True)
        print(f'\ncell {tag}: bank {n}, K_cal {Kc}, SR {SR:.4f}')

        t0 = time.time()
        model = METHOD.fit(R, cs, wd)
        t_fit = time.time() - t0
        t0 = time.time()
        res = METHOD.estimate(model, y, list(BGRID), cs)
        t_est = time.time() - t0
        fit_times.append(t_fit)
        est_times.append(t_est)
        info = model['info']
        print(f'  fit {t_fit:.1f}s  estimate {t_est:.1f}s   D={info["D"]} '
              f'(candidates {info["D_candidates"]}, diverged {sorted(info["D_diverged"])} / '
              f'final {sorted(info["D_final_diverged"])}), pds disagreeing routes '
              f'{info["pds_n_disagreeing"]}/{n}, pds distinct scores {info["pds_distinct_values"]}')
        for B in BGRID:
            v = res[B]
            print(f'  B={B:4d}  est {v["est"]:.4f}  |est-SR| {abs(v["est"] - SR):.4f}  '
                  f'n_items {len(v["items"])}  read(pds u jsd) {v["diag"]["n_read_union"]}  '
                  f'pds_pirt {v["variants"]["pds_pirt"]:.4f}  pds_naive {v["variants"]["pds_naive"]:.4f}  '
                  f'mean_train_score {v["variants"]["mean_train_score"]:.4f}')

        # --- items are unique, in range, and fill the budget ------------------
        for B in BGRID:
            it = res[B]['items']
            assert len(it) == B, f'B{B}: len(items) = {len(it)} != {B}'
            assert len(set(it)) == B, f'B{B}: repeated item'
            assert all(0 <= i < n for i in it), f'B{B}: item out of the bank'
            ji = jsd_items(res, B)
            assert len(ji) == len(set(ji)) == B and all(0 <= i < n for i in ji), f'B{B}: bad jsd items'
        print('  items: unique, in range, len == B (both orders)  OK')

        # --- PREFIX ----------------------------------------------------------
        big = max(BGRID)
        for B in BGRID:
            assert res[B]['items'] == res[big]['items'][:B], f'B{B}: pds items are not a prefix'
            assert jsd_items(res, B) == jsd_items(res, big)[:B], f'B{B}: jsd items are not a prefix'
        print(f'  prefix: B=30 items are the first 30 of B={big} (pds and jsd)  OK')

        # --- LEAK 1: flip every route outside the union of the read sets ------
        read = set(res[big]['items']) | set(jsd_items(res, big))
        outside = [j for j in range(n) if j not in read]
        y2 = np.asarray(y, float).copy()
        y2[outside] = 1 - y2[outside]
        res2 = METHOD.estimate(model, y2, list(BGRID), cs)
        for B in BGRID:
            assert res2[B]['items'] == res[B]['items'], f'B{B}: items moved under the leak flip'
            assert res2[B]['est'] == res[B]['est'], f'B{B}: est moved under the leak flip'
            assert res2[B]['variants'] == res[B]['variants'], f'B{B}: a variant moved under the leak flip'
        print(f'  leak 1: {len(outside)} unread routes flipped -> identical items/est/variants  OK')

        # --- LEAK 2: per budget, flip everything outside that budget's reads --
        for B in BGRID:
            keep = set(res[B]['items']) | set(jsd_items(res, B))
            y3 = np.asarray(y, float).copy()
            out3 = [j for j in range(n) if j not in keep]
            y3[out3] = 1 - y3[out3]
            r3 = METHOD.estimate(model, y3, list(BGRID), cs)[B]
            assert r3['items'] == res[B]['items'], f'B{B}: items moved (per-budget leak)'
            assert r3['est'] == res[B]['est'], f'B{B}: est moved (per-budget leak)'
            assert r3['variants'] == res[B]['variants'], f'B{B}: a variant moved (per-budget leak)'
        print('  leak 2: per budget, flipping everything outside that budget\'s reads '
              'leaves that budget identical  OK')

        # --- LEAK 3: the pds readouts must not see the jsd-only routes --------
        n_only, moved = 0, 0
        for B in BGRID:
            only_j = sorted(set(jsd_items(res, B)) - set(res[B]['items']))
            n_only += len(only_j)
            if not only_j:
                continue
            y4 = np.asarray(y, float).copy()
            y4[only_j] = 1 - y4[only_j]
            r4 = METHOD.estimate(model, y4, list(BGRID), cs)[B]
            assert r4['est'] == res[B]['est'], f'B{B}: est saw a jsd-only route'
            for k, v in res[B]['variants'].items():
                if k.startswith('pds_') or k == 'mean_train_score':
                    assert r4['variants'][k] == v, f'B{B}: {k} saw a jsd-only route'
            moved += sum(r4['variants'][k] != v for k, v in res[B]['variants'].items()
                         if k.startswith('jsd_'))
        print(f'  leak 3: {n_only} jsd-only routes flipped over the four budgets -> est and every '
              f'pds_* variant identical, {moved} jsd_* readouts moved  OK')

        # --- DETERMINISM (pipeline) ------------------------------------------
        t0 = time.time()
        model_b = METHOD.fit(R, cs, wd)
        t_fit_b = time.time() - t0
        res_b = METHOD.estimate(model_b, y, list(BGRID), cs)
        assert model_b['info'] == info, 'info not reproducible'
        assert np.array_equal(model_b['A'], model['A']) and np.array_equal(model_b['B'], model['B']), \
            'IRT parameters not reproducible'
        for B in BGRID:
            assert res_b[B]['items'] == res[B]['items'], f'B{B}: items not reproducible'
            assert res_b[B]['est'] == res[B]['est'], f'B{B}: est not reproducible'
            assert res_b[B]['variants'] == res[B]['variants'], f'B{B}: variants not reproducible'
        print(f'  determinism (pipeline): same seed, same workdir -> identical est/items/variants  '
              f'OK (refit {t_fit_b:.1f}s)')

        # --- DETERMINISM (py-irt) --------------------------------------------
        fresh = SCRATCH / (tag + '_fresh')
        shutil.rmtree(fresh, ignore_errors=True)
        fresh.mkdir(parents=True, exist_ok=True)
        recorded = ~np.isnan(np.asarray(R, float))
        responses = np.where(recorded, (np.asarray(R, float) > info['threshold_c']).astype(float), np.nan)
        t0 = time.time()
        A2, B2, _ = _train_and_load(responses, fresh / f'irt_D{info["D"]}_final', info['D'], n)
        t_irt = time.time() - t0
        assert np.array_equal(A2, model['A']), 'py-irt discriminations not reproducible'
        assert np.array_equal(B2, model['B']), 'py-irt difficulties not reproducible'
        shutil.rmtree(fresh, ignore_errors=True)
        print(f'  determinism (py-irt): fresh retrain at D={info["D"]} reproduces a and b bit for bit  '
              f'OK ({t_irt:.1f}s)')

    print(f'\ntiming per cell: fit {np.mean(fit_times):.1f}s (max {np.max(fit_times):.1f}), '
          f'estimate {np.mean(est_times):.1f}s (max {np.max(est_times):.1f}) -> '
          f'192 cells ~= {192 * (np.mean(fit_times) + np.mean(est_times)) / 3600:.1f} h single-process. '
          f'NOTE: this is only the harness cost if the scratch dir was cold; a cold fit trains 5-6 '
          f'py-irt models (~20 s each, ~120 s per cell -> ~6.5 h for 192 cells), a warm one reloads '
          f'them and only re-runs the Ds that diverge (~15 s).  Wall time of this file: '
          f'~6.5 min cold, ~4 min warm.')
    print('\nSELFTEST OK')


if __name__ == '__main__':
    main()
