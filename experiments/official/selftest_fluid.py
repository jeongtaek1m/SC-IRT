#!/usr/bin/env python3
"""Self-test of official/fluid.py (Fluid Benchmarking through allenai/fluid-benchmarking).

    /data2/jeongtae/envs/atdrive_official/bin/python experiments/official/selftest_fluid.py

Cells (seed 0, K_cal 12, slot 0) and (seed 0, K_cal 4, slot 0). Per cell:
  * fit + estimate for every budget; prints est, |est - SR|, n_items and the timings
    of fit and estimate separately (the harness runs 192 cells);
  * items are unique bank indices and len(items) == B;
  * LEAK TEST -- estimate again on y2 = y with every item that was never selected
    (never administered in the n_max = 165 run) flipped to 1 - y: identical items and
    identical est at every budget. Fluid is adaptive, so the budget-B readout may only
    depend on the first B administered responses; the test therefore also flips, for
    each B, every item outside items[:B] and requires the B-readout to be unchanged;
  * DETERMINISM TEST -- the whole fit + estimate path twice with the same seed: bitwise
    identical item parameters, items and est;
  * PREFIX TEST -- items(B=30) == items(B=165)[:30], and a fresh engine run with
    n_max = 30 reproduces those items and that ability exactly (MFI at the running MAP
    ability is deterministic given the administered responses).
Prints SELFTEST OK when everything passes. Runtime ~1 min (4 fits at ~10 s).
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from official.data import BGRID, protocol_cell          # noqa: E402
from official.fluid import METHOD, _Responses           # noqa: E402
from fluid_benchmarking import config as fluid_config   # noqa: E402
from fluid_benchmarking import engine                   # noqa: E402

WORK = Path('/data2/jeongtae/official_baselines/runs/fluid_selftest')
CELLS = [(0, 12, 0), (0, 4, 0)]


def cell_seed(seed, Kc, slot):
    return 100000 + 1000 * seed + 10 * Kc + slot                 # run_up_official.cell_seed


def run_cell(seed, Kc, slot):
    R, y, bi, SR = protocol_cell(seed, Kc, slot)
    n = len(y)
    cs = cell_seed(seed, Kc, slot)
    wd = WORK / f's{seed}_K{Kc}_p{slot}'
    print(f'\n=== cell seed {seed} K_cal {Kc} slot {slot}: n_bank {n}, '
          f'NaN cells {int(np.isnan(R).sum())}, SR {SR:.4f}')

    t0 = time.time()
    model = METHOD.fit(R, cs, wd)
    fit_s = time.time() - t0
    t0 = time.time()
    res = METHOD.estimate(model, y, list(BGRID), cs)
    est_s = time.time() - t0
    info = model['info']
    print(f'  fit {fit_s:.2f}s (py-irt {info["epochs"]} epochs, seed {info["fit_seed"]}, '
          f'{info["n_obs"]} responses)   estimate {est_s:.3f}s (one n_max={max(BGRID)} run, all budgets)')
    print(f'  a in [{info["a_min"]:.3f}, {info["a_max"]:.3f}] mean {info["a_mean"]:.3f} | '
          f'b mean {info["b_mean"]:.3f} sd {info["b_sd"]:.3f} | start_ability {info["start_ability"]:.4f}')
    print(f'  {"B":>5}{"est":>10}{"|est-SR|":>10}{"n_items":>9}{"ability":>10}{"sample_mean":>13}')
    for B in BGRID:
        v = res[B]
        print(f'  {B:5d}{v["est"]:10.4f}{abs(v["est"] - SR):10.4f}{len(v["items"]):9d}'
              f'{v["variants"]["ability"]:10.4f}{v["variants"]["sample_mean"]:13.4f}'
              + (f'  note: {v["note"]}' if v.get('note') else ''))

    # --- budget / item sanity
    for B in BGRID:
        it = res[B]['items']
        assert len(it) == B, f'B={B}: {len(it)} items (bank has {n}, so the budget must be fillable)'
        assert len(set(it)) == B, f'B={B}: repeated item'
        assert all(isinstance(i, int) and 0 <= i < n for i in it), f'B={B}: item out of bank range'

    # --- PREFIX TEST
    full = res[max(BGRID)]['items']
    for B in BGRID:
        assert res[B]['items'] == full[:B], f'B={B} items are not the prefix of the B={max(BGRID)} run'
    fresh = engine.run_fluid_benchmarking(
        lm_responses=_Responses(y), irt_model=np.asarray(model['irt_model'], float),
        start_ability=model['start_ability'], n_max=30, method=fluid_config.ESTIMATION_METHOD_IRT)
    assert [int(i) for i in fresh['items_fb']] == res[30]['items'], 'n_max=30 run differs from the prefix'
    assert float(fresh['abilities_fb'][-1]) == res[30]['variants']['ability'], 'n_max=30 ability differs'
    print(f'  prefix   OK (B=30 items are the first 30 of B=165 and equal a fresh n_max=30 run)')

    # --- LEAK TEST
    administered = set(full)
    y_flip = np.where(np.isin(np.arange(n), sorted(administered)), y, 1.0 - y)
    res2 = METHOD.estimate(model, y_flip, list(BGRID), cs)
    for B in BGRID:
        assert res2[B]['items'] == res[B]['items'], f'B={B}: items changed when never-selected items were flipped'
        assert res2[B]['est'] == res[B]['est'], f'B={B}: est changed when never-selected items were flipped'
        assert res2[B]['variants'] == res[B]['variants'], f'B={B}: variants changed'
    print(f'  leak     OK ({n - len(administered)} never-administered items flipped: items and est unchanged)')
    for B in BGRID:                       # per-budget: only the first B administered responses may matter
        keep = set(res[B]['items'])
        yb = np.where(np.isin(np.arange(n), sorted(keep)), y, 1.0 - y)
        rb = METHOD.estimate(model, yb, [B], cs)[B]
        assert rb['items'] == res[B]['items'] and rb['est'] == res[B]['est'], \
            f'B={B}: readout depends on items outside its own budget'
    print(f'  leak/B   OK (for every B, flipping everything outside items[:B] leaves that budget unchanged)')

    # --- DETERMINISM TEST (whole path, same seed)
    t0 = time.time()
    model2 = METHOD.fit(R, cs, wd)
    fit2_s = time.time() - t0
    assert np.array_equal(np.asarray(model2['irt_model']), np.asarray(model['irt_model'])), 'refit changed [a, b]'
    assert model2['start_ability'] == model['start_ability'], 'refit changed start_ability'
    res3 = METHOD.estimate(model2, y, list(BGRID), cs)
    for B in BGRID:
        assert res3[B]['items'] == res[B]['items'] and res3[B]['est'] == res[B]['est'], f'B={B}: not deterministic'
    print(f'  determ   OK (refit {fit2_s:.2f}s: identical item parameters, items and est)')
    return fit_s, est_s, [(B, res[B]['est'], abs(res[B]['est'] - SR)) for B in BGRID]


def main():
    assert METHOD.stop({}, None, 0) == {}, 'Fluid publishes no stopping rule; stop() must be {}'
    tot = []
    for c in CELLS:
        tot.append(run_cell(*c))
    print(f'\nstop() = {{}} (the released code has no stopping rule; run_fluid_benchmarking stops at n_max)')
    print(f'timing per cell: fit {np.mean([t[0] for t in tot]):.2f}s, estimate {np.mean([t[1] for t in tot]):.3f}s '
          f'-> 192 cells about {192 * (np.mean([t[0] for t in tot]) + np.mean([t[1] for t in tot])) / 60:.0f} min')
    print('SELFTEST OK')


if __name__ == '__main__':
    main()
