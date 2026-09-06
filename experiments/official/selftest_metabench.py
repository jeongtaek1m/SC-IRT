"""Self-test for experiments/official/metabench.py.

    /data2/jeongtae/envs/atdrive_official/bin/python experiments/official/selftest_metabench.py

Cells (0, 12, 0) and (0, 4, 0): fit + estimate at every budget, then the leak,
determinism and prefix checks required of an official wrapper.  It also checks the two
attribution-sensitive facts of the ADAPTATIONS block: the linear readout is reported as
'wrapper_lin' (metabench's own linear baseline is meta.R:284 lm(grand ~ .), a different
model and not runnable here) and the GAM never falls back, so info['readout'] is 'gam'.
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official.data import BGRID, protocol_cell                      # noqa: E402
from official.metabench import METHOD                               # noqa: E402

SCRATCH = Path('/data2/jeongtae/official_baselines/runs/selftest_metabench')


def run_cell(seed, Kc, slot):
    R, y, bi, SR = protocol_cell(seed, Kc, slot)
    n = R.shape[1]
    print(f'\n=== cell (seed {seed}, K_cal {Kc}, slot {slot})  bank {n} routes  SR {SR:.4f} '
          f'({int(np.isnan(R).sum())} NaN cells, {int(np.isnan(R).all(0).sum())} routes with no record)')
    wd = SCRATCH / f's{seed}_K{Kc}_p{slot}'
    shutil.rmtree(wd, ignore_errors=True)

    t0 = time.time()
    model = METHOD.fit(R, 100000 + 1000 * seed + 10 * Kc + slot, wd)
    fit_s = time.time() - t0
    t0 = time.time()
    est = METHOD.estimate(model, y, list(BGRID), 100000 + 1000 * seed + 10 * Kc + slot)
    est_s = time.time() - t0

    info = model['info']
    print(f'  fit {fit_s:.1f}s   estimate (all {len(BGRID)} budgets, one Rscript call) {est_s:.1f}s')
    print(f'  mirt: {info["n_fitted"]}/{info["n_bank"]} routes eligible '
          f'(sd > 0.01), a in [{info["a_min"]:.2f}, {info["a_max"]:.2f}], readout {info["readout"]}')
    print(f'  ADAPTATION 7(b): {info["n_a_nonpositive"]}/{info["n_fitted"]} fitted routes have '
          f'a1 <= 0 -> ATLAS compute_3pl_prob returns p = 0 (g = 0) for them when unobserved')
    for w in info['warnings']:
        print(f'    warning: {w.splitlines()[0][:110]}')
    print(f'  {"B":>5}{"est":>9}{"|est-SR|":>10}{"n_items":>9}   variants')
    for B in BGRID:
        v = est[B]
        items = v['items']
        assert len(set(items)) == len(items), f'B={B}: repeated items'
        assert all(0 <= i < n for i in items), f'B={B}: item index outside the bank'
        if len(items) < B:                                        # ADAPTATION 3/5
            assert v.get('note'), f'B={B}: short subset without a note'
            assert len(items) == info['n_fitted'], f'B={B}: short for an unexplained reason'
        else:
            assert len(items) == B, f'B={B}: {len(items)} items'
        # the linear readout is wrapper-defined, not metabench's meta.R:284 lm(grand ~ .)
        assert 'wrapper_lin' in v['variants'] and 'mod_lin' not in v['variants']
        imp = v['pirt_impute']                                    # ADAPTATION 7
        assert imp['p_zero'] + imp['p_half'] <= n - len(items)
        print(f'  {B:5d}{v["est"]:9.4f}{abs(v["est"] - SR):10.4f}{len(items):9d}   '
              + ' '.join(f'{k}={x:.4f}' for k, x in v['variants'].items())
              + f'   pirt imputed p=0:{imp["p_zero"]} p=0.5:{imp["p_half"]}'
              + ('   ' + v['note'] if v.get('note') else ''))

    # ---- LEAK TEST: metabench is a fixed-subset method, so flipping every route it
    #      never selected (the union over the budgets) must change nothing.
    sel = sorted({i for B in BGRID for i in est[B]['items']})
    y2 = 1.0 - np.asarray(y, float)
    y2[sel] = np.asarray(y, float)[sel]
    est2 = METHOD.estimate(model, y2, list(BGRID), 100000 + 1000 * seed + 10 * Kc + slot)
    for B in BGRID:
        assert est[B]['items'] == est2[B]['items'], f'leak: items changed at B={B}'
        assert est[B]['est'] == est2[B]['est'], f'leak: est changed at B={B}'
        assert est[B]['variants'] == est2[B]['variants'], f'leak: a variant changed at B={B}'
        assert est[B]['pirt_impute'] == est2[B]['pirt_impute'], f'leak: pirt_impute changed at B={B}'
    print(f'  LEAK TEST ok ({n - len(sel)} unselected routes flipped, all budgets identical)')

    # ---- DETERMINISM: same seed, fresh workdir, identical numbers.
    wd2 = SCRATCH / f's{seed}_K{Kc}_p{slot}_rerun'
    shutil.rmtree(wd2, ignore_errors=True)
    m2 = METHOD.fit(R, 100000 + 1000 * seed + 10 * Kc + slot, wd2)
    e2 = METHOD.estimate(m2, y, list(BGRID), 100000 + 1000 * seed + 10 * Kc + slot)
    for B in BGRID:
        assert e2[B]['items'] == est[B]['items'], f'determinism: items differ at B={B}'
        assert e2[B]['est'] == est[B]['est'], f'determinism: est differs at B={B}'
        assert e2[B]['variants'] == est[B]['variants'], f'determinism: variants differ at B={B}'
        assert e2[B]['pirt_impute'] == est[B]['pirt_impute'], f'determinism: pirt_impute differs at B={B}'
    print('  DETERMINISM TEST ok (refit from scratch, identical items / est / variants)')

    # ---- PREFIX: not applicable.  metabench selects one route per quantile bin and the
    #      bins are re-drawn for every subset size (ADAPTATION 3), so it is neither
    #      adaptive nor a prefix order; the check is that the wrapper does not claim to be.
    assert METHOD.adaptive is False
    # ADAPTATION 6: the reduce.R:105 GAM is the readout; the wrapper fallback never fires.
    assert set(info['readout'].values()) == {'gam'}, info['readout']
    ov = len(set(est[30]['items']) & set(est[max(BGRID)]['items']))
    is_prefix = est[max(BGRID)]['items'][:30] == est[30]['items']
    print(f'  PREFIX CHECK n/a: fixed-subset method (adaptive=False); B=30 shares {ov}/30 routes '
          f'with B={max(BGRID)}, B=165 prefix equals B=30 subset: {is_prefix}')
    return fit_s, est_s


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    t = time.time()
    times = [run_cell(0, 12, 0), run_cell(0, 4, 0)]
    print(f'\ntiming per cell: fit {[round(a, 1) for a, _ in times]}s, '
          f'estimate {[round(b, 1) for _, b in times]}s  '
          f'(harness = 192 cells, ~{192 * np.mean([a + b for a, b in times]) / 60:.0f} min single-process)')
    print(f'total self-test {time.time() - t:.0f}s')
    print('SELFTEST OK')


if __name__ == '__main__':
    main()
