#!/usr/bin/env python
"""Self-test for experiments/official/catr.py (catR static information orders).

    /data2/jeongtae/envs/atdrive_official/bin/python \
        experiments/official/selftest_catr.py

Cells (seed 0, K_cal 12, slot 0) and (seed 0, K_cal 4, slot 0); budgets
{30, 55, 110, 165}.  Checks, per the wrapper contract:
  * fit + estimate run, printing est, |est - SR|, n_items and separate fit /
    estimate timings (the harness runs 192 cells);
  * PRIOR: the intercept-prior sd is the empirical-Bayes argmax over ATDrive's
    SIGMA_B_GRID (ADAPTATION 1) -- the selected value, the profile and the
    induced difficulty-prior sd are printed and re-checked against the profile;
  * every 'items' list is a set of unique, in-range bank indices with len == B,
    and the marginal_fisher selection rule costs the same B routes (ADAPTATION 7);
  * PREFIX: the B = 30 items are the first 30 of the B = 165 items; the
    marginal rule reports no items list, so its prefix property is checked
    functionally (forcing the routes it reads only at B = 165 leaves the B = 30
    marginal readout bit-identical and moves the B = 165 one);
  * LEAK: re-estimating with every route OUTSIDE the read set flipped gives
    bit-identical items, est and variants (fixed-subset method).  A second,
    sharper pass perturbs only the routes the marginal_fisher order reads and
    total_fisher does not: 'est' and the total_fisher variants must be
    bit-identical (ADAPTATION 7) while marginal_fisher's own readout moves
    whenever the perturbation changes a response it reads;
  * DETERMINISM: fit + estimate twice with the same seed -> identical est.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                     # experiments/
sys.path.insert(0, str(HERE.parents[1]))                 # repo root

from official.catr import METHOD                          # noqa: E402
from official.data import BGRID, protocol_cell            # noqa: E402
from atdrive.calibration import SIGMA_B_GRID              # noqa: E402  (the EB grid, ADAPTATION 1)

SCRATCH = Path('/data2/jeongtae/official_baselines/runs/catr/selftest')
CELLS = [(0, 12, 0), (0, 4, 0)]
TF_KEYS = ('total_fisher_theta', 'total_fisher_se', 'total_fisher_mean')


def cell_seed(seed, Kc, slot):
    return 100000 + 1000 * seed + 10 * Kc + slot          # run_up_official.cell_seed


def run(tag, R, y, cs, seed_dir):
    wd = SCRATCH / seed_dir / tag
    wd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = METHOD.fit(R, cs, wd)
    t1 = time.time()
    est = METHOD.estimate(model, y, list(BGRID), cs)
    t2 = time.time()
    return model, est, t1 - t0, t2 - t1


def main():
    fit_times, est_times = [], []
    for seed, Kc, slot in CELLS:
        R, y, bi, SR = protocol_cell(seed, Kc, slot)
        n = len(bi)
        cs = cell_seed(seed, Kc, slot)
        tag = f's{seed}_K{Kc}_p{slot}'
        print(f'\n=== cell seed {seed} K_cal {Kc} slot {slot}: bank {n}, SR {SR:.4f}, '
              f'cell_seed {cs} ===')

        model, est, t_fit, t_est = run('main', R, y, cs, tag)
        fit_times.append(t_fit)
        est_times.append(t_est)
        i = model['info']
        print(f'fit {t_fit:.2f}s  estimate {t_est:.2f}s   '
              f'(mirt converged {i["converged"]}, {i["em_iterations"]} EM cycles, '
              f'{i["n_constant_items"]} single-category routes, '
              f'{i["n_ties_total"]}/{i["n_ties_marginal"]} tied total/marginal info values, '
              f'{i["n_warnings"]} muffled warnings)')

        # --- empirical-Bayes prior (ADAPTATION 1) ---------------------------
        grid, eb = list(i['prior_grid']), list(i['prior_eb_loglik'])
        assert len(eb) == len(grid) == len(SIGMA_B_GRID), 'prior grid is not ATDrive SIGMA_B_GRID'
        assert grid == [float(g) for g in SIGMA_B_GRID], f'prior grid {grid} != {list(SIGMA_B_GRID)}'
        assert i['prior_sd_d'] == grid[int(np.argmax(eb))], 'prior sd is not the EB argmax'
        print(f'  prior: sd_d {i["prior_sd_d"]:g} = EB argmax over {grid} '
              f'(loglik ' + ' '.join(f'{v:.1f}' for v in eb) + ')  OK')
        print(f'         induced b-prior sd {i["prior_sd_b_induced_min"]:.2f}-'
              f'{i["prior_sd_b_induced_max"]:.2f}, {i["n_prior_fits"]} mirt fits')
        print(f'  {"B":>4} {"est":>8} {"|est-SR|":>9} {"n_items":>8} '
              f'{"marginal":>9} {"|mar-SR|":>9} {"theta":>7} {"se":>6}')
        for B in BGRID:
            v = est[B]
            print(f'  {B:4d} {v["est"]:8.4f} {abs(v["est"] - SR):9.4f} {len(v["items"]):8d} '
                  f'{v["variants"]["marginal_fisher"]:9.4f} '
                  f'{abs(v["variants"]["marginal_fisher"] - SR):9.4f} '
                  f'{v["variants"]["total_fisher_theta"]:7.3f} '
                  f'{v["variants"]["total_fisher_se"]:6.3f}')

        # --- items are unique in-range bank indices, len == B --------------
        for B in BGRID:
            it = est[B]['items']
            assert len(it) == len(set(it)), f'B{B}: duplicate items'
            assert all(0 <= j < n for j in it), f'B{B}: item out of range'
            assert len(it) == B, f'B{B}: {len(it)} items (bank {n})'
        ord_t = model['info']['order_total_fisher']
        ord_m = model['info']['order_marginal_fisher']
        assert sorted(ord_t) == list(range(n)) and sorted(ord_m) == list(range(n)), 'order is not a permutation'
        for B in BGRID:                     # both rules cost B routes, never their union (ADAPTATION 7)
            assert f'own {B} routes' in est[B]['note'], f'B{B}: marginal_fisher cost is not B'
        print('  items: unique, in range, len == B for every budget, both rules cost B  OK')

        # --- prefix property ------------------------------------------------
        # total_fisher publishes its subset as 'items', so its prefix property is
        # read off directly.  The marginal rule publishes no items list
        # (ADAPTATION 7), and asserting ord_m[:30] == ord_m[:165][:30] would be a
        # tautology, so its prefix property is checked FUNCTIONALLY: forcing every
        # route it reads at B = 165 but not at B = 30 must leave the B = 30
        # marginal readout bit-identical and must move the B = 165 one.
        assert est[30]['items'] == est[165]['items'][:30], 'B30 is not the prefix of B165'
        for B in BGRID:
            assert est[B]['items'] == ord_t[:B], f'B{B}: items != total_fisher order prefix'
        tail_m = list(ord_m[30:165])
        y_t = y.copy()
        y_t[tail_m] = 1.0
        est_t = METHOD.estimate(model, y_t, list(BGRID), cs)
        changed_m = [j for j in tail_m if y_t[j] != y[j]]
        assert est_t[30]['variants']['marginal_fisher'] == est[30]['variants']['marginal_fisher'], \
            'B30 marginal readout saw a route only the B165 marginal subset reads'
        assert (est_t[165]['variants']['marginal_fisher'] != est[165]['variants']['marginal_fisher']
                or not changed_m), \
            f'B165 marginal readout ignored {len(changed_m)} of its own routes'
        print(f'  prefix: B30 items == first 30 of B165 items; forcing the {len(tail_m)} marginal '
              f'routes B30 does not read ({len(changed_m)} changed) leaves the B30 marginal readout '
              f'identical and moves B165  OK')

        # --- LEAK TEST 1: flip everything outside the read set ---------------
        read = set(ord_t[:max(BGRID)]) | set(ord_m[:max(BGRID)])
        y2 = y.copy()
        outside = [j for j in range(n) if j not in read]
        y2[outside] = 1 - y2[outside]
        est2 = METHOD.estimate(model, y2, list(BGRID), cs)
        for B in BGRID:
            assert est2[B]['items'] == est[B]['items'], f'B{B}: items changed under the leak flip'
            assert est2[B]['est'] == est[B]['est'], f'B{B}: est changed under the leak flip'
            assert est2[B]['variants'] == est[B]['variants'], f'B{B}: a variant changed under the leak flip'
        print(f'  leak: {len(outside)} unread routes flipped -> identical items/est/variants  OK')

        # --- LEAK TEST 2: total_fisher must not see the marginal-only routes -
        # Two perturbations of the marginal-only routes.  The flip is the sharp
        # one for total_fisher; it can leave marginal_fisher itself unmoved,
        # because tied routes carry IDENTICAL (a, b) and a pass/fail-swapping
        # flip over one such group is exactly likelihood-invariant.  Forcing
        # those routes to 1 changes the group's pass count, so marginal_fisher
        # MUST move wherever it reads one -- a positive control that the readout
        # really is a function of its own subset.
        only_m = sorted(set(ord_m[:max(BGRID)]) - set(ord_t[:max(BGRID)]))
        moved, forced = 0, 0
        if only_m:
            for y_pert, is_flip in ((np.where(np.isin(np.arange(n), only_m), 1 - y, y), True),
                                    (np.where(np.isin(np.arange(n), only_m), 1.0, y), False)):
                est3 = METHOD.estimate(model, y_pert, list(BGRID), cs)
                for B in BGRID:
                    assert est3[B]['est'] == est[B]['est'], f'B{B}: est saw a marginal-only route'
                    for k in TF_KEYS:
                        assert est3[B]['variants'][k] == est[B]['variants'][k], f'B{B}: {k} saw a marginal-only route'
                    m_moved = est3[B]['variants']['marginal_fisher'] != est[B]['variants']['marginal_fisher']
                    if is_flip:
                        moved += m_moved
                    else:
                        changed = [j for j in ord_m[:B] if y_pert[j] != y[j]]
                        assert m_moved or not changed, f'B{B}: marginal_fisher ignored {len(changed)} of its own routes'
                        forced += bool(changed)
        print(f'  leak: {len(only_m)} marginal-only routes perturbed -> est/total_fisher variants '
              f'identical; marginal_fisher moved in {moved}/{len(BGRID)} budgets under the flip and '
              f'in every one of the {forced}/{len(BGRID)} budgets that read a forced route  OK')

        # --- DETERMINISM ------------------------------------------------------
        model_b, est_b, t_fit_b, t_est_b = run('rerun', R, y, cs, tag)
        fit_times.append(t_fit_b)
        est_times.append(t_est_b)
        assert model_b['itemBank'] == model['itemBank'], 'itemBank not reproducible'
        assert model_b['orders'] == model['orders'], 'orders not reproducible'
        for B in BGRID:
            assert est_b[B]['est'] == est[B]['est'], f'B{B}: est not reproducible'
            assert est_b[B]['items'] == est[B]['items'], f'B{B}: items not reproducible'
            assert est_b[B]['variants'] == est[B]['variants'], f'B{B}: variants not reproducible'
        print(f'  determinism: same seed twice -> identical bank, orders, items, est  OK '
              f'(fit {t_fit_b:.2f}s, estimate {t_est_b:.2f}s)')

    print(f'\ntiming per cell: fit {np.mean(fit_times):.2f}s +- {np.std(fit_times):.2f}, '
          f'estimate {np.mean(est_times):.2f}s +- {np.std(est_times):.2f} '
          f'-> 192 cells ~ {192 * (np.mean(fit_times) + np.mean(est_times)) / 60:.1f} min')
    print(json.dumps({'fit_s_mean': float(np.mean(fit_times)),
                      'est_s_mean': float(np.mean(est_times))}))
    print('SELFTEST OK')


if __name__ == '__main__':
    main()
