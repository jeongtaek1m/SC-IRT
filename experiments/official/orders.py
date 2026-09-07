#!/usr/bin/env python3
"""The item ORDERS the official implementations select, cached so that the
experiments outside Table 1 can be run on them instead of on the
re-implementations of `atdrive/baselines.py`.

`run_up_official.py` runs every published baseline through its own code, but
only at the four fixed budgets of Table 1. The other experiments
(`run_adaptive.py`, `run_tau_calibration.py`, `run_route_discrimination.py`,
`run_readout_dropin.py`, `run_ups.py`) need either the whole bank order or a
subset at their own budget. This module produces both, from the SAME official
wrappers (`official/fluid.py`, `official/metabench.py`, `official/catr.py`,
`official/atlas.py`), and caches them under

    results/up_official_orders/<method>_s<seed>_K<K>_p<slot>.json          (protocol cell)
    results/up_official_orders/<method>_loo_s<seed>_K<K>_j<j>.json         (LOO cell)
    results/up_official_orders/<method>_ups_s<seed>_j<js>.json             (UPS cell)

so a second run is free. Nothing here re-implements a selection rule: every
order is what the method's own code returns.

API
---
    order(method, seed, K, slot)          full-bank order of a PREFIX method
    loo_order(method, seed, K, j)         the same for a leave-one-planner-out cell
                                          (the cells `run_tau_calibration.py` runs:
                                          calibration = the K_cal panel minus j,
                                          evaluation planner = j)
    ups_order(method, seed, js)           the same for a UPS cell (`run_ups.py`: all 12
                                          calibration planners, bank = the
                                          calibration-type routes)
    subsets(method, seed, K, slot, budgets)   {B: items} of a BUDGET-SPECIFIC method
    padded(order, n)                      the order extended to the whole bank
    slot_of(seed, js)                     protocol slot of an evaluation planner id

Three cell families, because the experiments run three: the 192 protocol cells of
Table 1 (`--seeds`), the 384 leave-one-planner-out cells of the threshold
calibration (`--loo`) and the 64 UPS cells (`--ups`). Each is cached separately and
each has its own seed range, so no two families can collide in an official method's
RNG.

WHICH METHODS HAVE AN ORDER AND WHICH DO NOT
--------------------------------------------
ORDER_METHODS -- one run at the bank size gives every prefix:

  'fluid'           `engine.run_fluid_benchmarking` with n_max = bank size. MFI at
                    the running MAP ability is deterministic given the responses,
                    so items_fb[:B] is the B-budget run (fluid.py ADAPTATION 6).
  'total_fisher'    catR's two static information orders, `model['orders']` of
  'marginal_fisher' `catr.py` -- the whole-bank ranking is what the fit returns, so
                    order[:B] IS the budget-B subset by construction. Both come out
                    of one catR fit; asking for either caches both.
  'atlas'           `run_atlas` at max_items = the fitted bank, se_theta_stop = 0
                    (their stop can then never fire), min_items = 30 as in their
                    pipeline. Uses `Atlas._run_cat` directly rather than
                    `Atlas.estimate`, so that the order is still produced for the two
                    cells whose official p-IRT readout comes back NA.
  'metabench'       their information-quantile selection with B = bank size. NOTE:
                    metabench is NOT a prefix method -- the quantile bins are a
                    function of B, so the B = 30 subset is not the first 30 of the
                    B = 165 one (metabench.py, "Not adaptive"). The full-bank order
                    is therefore metabench's own bank-size ranking, NOT a
                    reconstruction of its per-budget subsets; where a per-budget
                    subset is what the experiment needs, use `subsets()`.
                    `order(...)['prefix']` is False for it and `--check` prints by
                    how much its prefixes differ from the recorded budgets.

SUBSET_METHODS -- no order exists; the method selects a different set per budget:

  'tinybench'       tinyBenchmarks' KMeans anchors: K = B clusters, so the B = 30
                    anchors are not a subset of the B = 55 anchors.
  'anchorpoints'    AnchorPoints' K-medoids: same, K = B medoids.
  'metabench'       (also available as a subset method, see above)

  `subsets()` serves these from the per-cell records of the run of record
  (results/up_official_*.json), which is where those subsets were computed by the
  official code; it therefore only serves the budgets that run was priced at
  (30 / 55 / 110 / 165) and raises for any other, rather than silently
  recomputing something the numbers of record do not cover.

Not covered: 'Random' and 'Random-strat' of run_adaptive.py are the protocol's own
controls, not published methods, and 'disco' selects by disagreement at a fixed
budget and is not used by any of the bridged experiments. Both keep the
re-implementation.

Cost (MEASURED here, from the 'seconds' field of every cached record; one process,
per cell, OMP_NUM_THREADS=1)

    method                    per cell   192 protocol cells   384 LOO cells
    fluid                        5.5 s        18 min               31 min
    atlas                        3.4 s        11 min                 --
    total_fisher (+ marginal)   15.1 s        48 min                 --
    metabench                   18.3 s        58 min              103 min

  atlas is 3.4 s because the run of record's 3PL fit is REUSED (`_atlas_model`);
  refitting it is 20 s at K_cal = 4 and 671 s at K_cal = 12 -- 32 h over the grid --
  which is why refitting is behind ATDRIVE_OFFICIAL_ATLAS_REFIT=1. metabench costs
  9 s at K_cal = 4 and 22-24 s at K_cal >= 8; catR is 14-16 s throughout (6 mirt
  fits for its empirical-Bayes prior). Only fluid and metabench are needed for the
  LOO cells (run_tau_calibration runs no other published order) and only fluid for
  the 64 UPS cells (5.2 s each, 5.5 min).
  Sharded 8 ways by seed, as below, all four methods' protocol cells finished in
  ~25 min of wall clock together. `--seeds a b` writes each cell as it is computed
  and a second run skips every cached cell, so a killed shard resumes.

    P=/data2/jeongtae/envs/atdrive_official/bin/python
    $P experiments/official/orders.py --methods fluid --seeds 0 4     # shard
    $P experiments/official/orders.py --check --methods fluid catr atlas metabench
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', Path(__file__).resolve().parents[2] / 'results'))
CACHE = OUT / 'up_official_orders'
SCRATCH = Path(os.environ.get('ATDRIVE_OFFICIAL_ORDERS_SCRATCH',
                              '/data2/jeongtae/official_baselines/order_runs'))
RECORD_SCRATCH = Path(os.environ.get('ATDRIVE_OFFICIAL_SCRATCH',
                                     '/data2/jeongtae/official_baselines/runs'))

ORDER_METHODS = ('fluid', 'total_fisher', 'marginal_fisher', 'atlas', 'metabench')
SUBSET_METHODS = ('tinybench', 'anchorpoints', 'metabench')
PREFIX = {'fluid': True, 'total_fisher': True, 'marginal_fisher': True, 'atlas': True,
          'metabench': False}


# --------------------------------------------------------------------------- keys
def cell_seed(seed, K, slot):
    """run_up_official.cell_seed -- the same seed the run of record used, so that
    ATLAS's random first item and randomesque draws are the recorded ones."""
    return 100000 + 1000 * seed + 10 * K + slot


def loo_cell_seed(seed, K, j):
    """Seed of a leave-one-planner-out cell; disjoint from cell_seed's range."""
    return 200000 + 1000 * seed + 10 * K + j


def ups_cell_seed(seed, js):
    """Seed of a UPS cell (run_ups.py); disjoint from the other two ranges."""
    return 300000 + 1000 * seed + js


def slot_of(seed, js):
    """Protocol slot (0-3) of evaluation planner `js` in draw `seed`."""
    from official.data import draw
    return list(draw(seed)[0]).index(int(js))


def cache_path(method, seed, K, slot=None, j=None):
    if j is None:
        return CACHE / f'{method}_s{seed}_K{K}_p{slot}.json'
    return CACHE / f'{method}_loo_s{seed}_K{K}_j{j}.json'


def ups_cache_path(method, seed, js):
    return CACHE / f'{method}_ups_s{seed}_j{js}.json'


# --------------------------------------------------------------------------- read
def _read(path):
    if not path.exists():
        return None
    rec = json.load(open(path))
    return rec if 'order' in rec else None


def order(method, seed, K, slot, n_bank=None, compute=None):
    """Full-bank order of `method` on protocol cell (seed, K, slot).

    Returns the record dict: {'order', 'n_bank', 'n_selected', 'prefix', ...}.
    `n_bank`, when given, is asserted against the cached bank size (the guard that
    the caller's bank indexing is the protocol's). `compute` (default: the
    ATDRIVE_OFFICIAL_ORDERS_COMPUTE env var) allows a cache miss to be filled;
    experiments leave it off so a missing cell is an error, not a 13-minute stall.
    """
    _check_method(method, ORDER_METHODS)
    rec = _read(cache_path(method, seed, K, slot))
    if rec is None:
        if not (compute if compute is not None else os.environ.get('ATDRIVE_OFFICIAL_ORDERS_COMPUTE') == '1'):
            raise FileNotFoundError(
                f'no cached official order for {method} cell (seed {seed}, K {K}, slot {slot}); '
                f'run experiments/official/orders.py --methods {method} --seeds {seed} {seed + 1}')
        rec = compute_cell(method, seed, K, slot)
    if n_bank is not None and rec['n_bank'] != n_bank:
        raise ValueError(f'{method} cell (seed {seed}, K {K}, slot {slot}): cached bank {rec["n_bank"]} '
                         f'!= caller bank {n_bank}')
    return rec


def loo_order(method, seed, K, j, n_bank=None, compute=None):
    """Full-bank order of `method` on a leave-one-planner-out cell: the calibration
    panel of (seed, K) minus planner j, evaluated on planner j."""
    _check_method(method, ORDER_METHODS)
    rec = _read(cache_path(method, seed, K, j=j))
    if rec is None:
        if not (compute if compute is not None else os.environ.get('ATDRIVE_OFFICIAL_ORDERS_COMPUTE') == '1'):
            raise FileNotFoundError(
                f'no cached official order for {method} LOO cell (seed {seed}, K {K}, planner {j}); '
                f'run experiments/official/orders.py --loo --methods {method} --seeds {seed} {seed + 1}')
        rec = compute_loo_cell(method, seed, K, j)
    if n_bank is not None and rec['n_bank'] != n_bank:
        raise ValueError(f'{method} LOO cell (seed {seed}, K {K}, j {j}): cached bank {rec["n_bank"]} '
                         f'!= caller bank {n_bank}')
    return rec


def ups_order(method, seed, js, n_bank=None, compute=None):
    """Full-bank order of `method` on a UPS cell (run_ups.py): the 12 calibration
    planners of draw `seed` calibrate on the UPS bank (the routes whose scenario
    type is not held out) and planner `js` is the evaluation planner. A different
    cell family from the UP protocol cells: no K_cal subsample, and the bank is the
    calibration-type routes only."""
    _check_method(method, ORDER_METHODS)
    rec = _read(ups_cache_path(method, seed, js))
    if rec is None:
        if not (compute if compute is not None else os.environ.get('ATDRIVE_OFFICIAL_ORDERS_COMPUTE') == '1'):
            raise FileNotFoundError(
                f'no cached official order for {method} UPS cell (seed {seed}, planner {js}); '
                f'run experiments/official/orders.py --ups --methods {method} --seeds {seed} {seed + 1}')
        rec = compute_ups_cell(method, seed, js)
    if n_bank is not None and rec['n_bank'] != n_bank:
        raise ValueError(f'{method} UPS cell (seed {seed}, planner {js}): cached bank {rec["n_bank"]} '
                         f'!= caller bank {n_bank}')
    return rec


def subsets(method, seed, K, slot, budgets, n_bank=None):
    """{B: items} for a method that selects a budget-specific set with NO order.

    Served from the per-cell records of the run of record, i.e. from the official
    code's own selection at that budget; only the budgets that run was priced at
    (30 / 55 / 110 / 165) exist.
    """
    _check_method(method, SUBSET_METHODS)
    rec = _records().get((method, int(seed), int(K), int(slot)))
    if rec is None:
        raise FileNotFoundError(f'no run-of-record cell for {method} (seed {seed}, K {K}, slot {slot}); '
                                f'run experiments/run_up_official.py --methods {method}')
    if 'error' in rec:
        raise RuntimeError(f'{method} cell (seed {seed}, K {K}, slot {slot}) failed in the run of record: '
                           f'{rec["error"]}')
    if n_bank is not None and rec['n_bank'] != n_bank:
        raise ValueError(f'{method} cell (seed {seed}, K {K}, slot {slot}): recorded bank {rec["n_bank"]} '
                         f'!= caller bank {n_bank}')
    out = {}
    for B in budgets:
        B = int(B)
        if str(B) not in rec['budgets']:
            raise KeyError(f'{method} has no official subset at B = {B}: the run of record covers '
                           f'{sorted(int(b) for b in rec["budgets"])} only')
        out[B] = [int(i) for i in rec['budgets'][str(B)]['items']]
    return out


def padded(order_, n):
    """The order extended to the whole bank with the routes it never selected, in
    ascending bank index. OURS, not the method's: metabench's quantile bins and
    ATLAS's constant-column drop leave part of the bank unselectable, and an
    experiment that runs a trajectory through the bank needs the full length. The
    pad point is `n_selected` in the order record."""
    seen = set(int(i) for i in order_)
    return [int(i) for i in order_] + [i for i in range(n) if i not in seen]


def _check_method(method, allowed):
    if method not in allowed:
        raise KeyError(f'{method!r} is not one of {allowed}')


_REC = None


def _records():
    """Per-cell records of the run of record, keyed by (method, seed, K, slot)."""
    global _REC
    if _REC is None:
        import glob
        recs = []
        for f in sorted(glob.glob(str(OUT / 'up_official_*_[0-9]*_[0-9]*.json'))
                        + glob.glob(str(OUT / 'up_official_[0-9]*_[0-9]*.json'))):
            recs += json.load(open(f))
        _REC = {(r['method'], r['seed'], r['K'], r['slot']): r for r in recs}
    return _REC


# --------------------------------------------------------------------------- compute
def _workdir(method, tag):
    wd = SCRATCH / method / tag
    wd.mkdir(parents=True, exist_ok=True)
    return wd


def _write(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(path, 'w'), indent=1)
    return rec


def _order_fluid(R, y, seed, wd):
    """official/fluid.py at n_max = the bank size (its ADAPTATION 6 prefix run)."""
    from official import fluid
    n = R.shape[1]
    model = fluid.METHOD.fit(R, seed, wd)
    res = fluid.METHOD.estimate(model, np.asarray(y, float), [n], seed)
    return {'fluid': [int(i) for i in res[n]['items']]}


def _order_catr(R, y, seed, wd):
    """official/catr.py: both static information orders come out of the fit."""
    from official import catr
    model = catr.METHOD.fit(R, seed, wd)
    return {name: [int(i) for i in model['orders'][name]] for name in ('total_fisher', 'marginal_fisher')}


def _order_metabench(R, y, seed, wd):
    """official/metabench.py with the budget set to the bank size.

    The wrapper takes its budget grid from its module constant BUDGETS (it is the
    Table 1 grid there); it is rebound for the duration of this call so that the
    official R pipeline is asked for one selection of B = bank-size quantile bins.
    No code of the wrapper or of metabench itself is changed."""
    from official import metabench
    n = R.shape[1]
    old = metabench.BUDGETS
    metabench.BUDGETS = (n,)
    try:
        model = metabench.METHOD.fit(R, seed, wd)
    finally:
        metabench.BUDGETS = old
    return {'metabench': [int(i) for i in model['budgets'][n]['items']]}


def _atlas_model(R, seed, wd, record_dir=None):
    """ATLAS's fitted bank for this cell.

    Their 3PL EM is the expensive part (671 s at K_cal = 12, and it runs to the
    100000-cycle cap; atlas.py "Cost"). When the run of record already fitted this
    cell, its two fit artifacts -- the item-parameter csv that 03/04 read and
    fit_info.json -- are copied into our workdir and reused; the fit is a pure
    function of the calibration matrix, which the protocol fixes. Otherwise the
    official fit is run."""
    from official import atlas
    if record_dir is not None and (record_dir / 'irt_item_parameters_combined.csv').exists() \
            and (record_dir / 'fit_info.json').exists():
        for f in ('irt_item_parameters_combined.csv', 'fit_info.json', 'calibration.csv'):
            if (record_dir / f).exists():
                shutil.copy(record_dir / f, wd / f)
        info = json.load(open(wd / 'fit_info.json'))
        if int(info['n_bank']) != R.shape[1]:
            raise ValueError(f'reused ATLAS fit has n_bank {info["n_bank"]}, cell has {R.shape[1]}')
        item_ids = [int(i) for i in atlas._nums(info['item_ids'])]
        dropped = [int(i) for i in atlas._nums(info.get('dropped_constant_items'))]
        dvals = atlas._nums(info.get('dropped_constant_values'))
        info.pop('item_ids', None)
        info.update({'K_cal': int(R.shape[0]), 'n_bank': int(R.shape[1]), 'reused_fit': str(record_dir)})
        return {'workdir': str(wd), 'n_bank': int(R.shape[1]), 'item_ids': item_ids,
                'dropped_items': dropped, 'dropped_values': dvals, 'info': info}
    if os.environ.get('ATDRIVE_OFFICIAL_ATLAS_REFIT') != '1':
        raise FileNotFoundError(
            f'no ATLAS fit to reuse in {record_dir}; their 3PL EM is 20-670 s per cell, so refitting is '
            f'opt-in: set ATDRIVE_OFFICIAL_ATLAS_REFIT=1 (or re-run run_up_official.py --methods atlas)')
    return atlas.METHOD.fit(R, seed, wd)


def _order_atlas(R, y, seed, wd, record_dir=None):
    """official/atlas.py: one run_atlas call at max_items = the fitted bank with
    se_theta_stop = 0, so their stop can never fire and the run is their selection
    order out to the end of the bank they can score. `_run_cat` is called instead of
    `estimate` because `estimate`'s readout raises when the official p-IRT comes
    back NA (2 of the 192 cells), and the ORDER is well defined there too."""
    from official import atlas
    model = _atlas_model(R, seed, wd, record_dir)
    n_fit = len(model['item_ids'])
    cfg = [{'name': 'FULL', 'max_items': n_fit, 'min_items': atlas.MIN_ITEMS, 'se_theta_stop': 0}]
    out = atlas.METHOD._run_cat(model, np.asarray(y, float), cfg, seed)
    return {'atlas': [int(i) for i in atlas._nums(out['FULL']['items'])]}


def _compute(method, R, y, seed, wd, record_dir=None):
    if method == 'fluid':
        return _order_fluid(R, y, seed, wd)
    if method in ('total_fisher', 'marginal_fisher'):
        return _order_catr(R, y, seed, wd)
    if method == 'metabench':
        return _order_metabench(R, y, seed, wd)
    if method == 'atlas':
        return _order_atlas(R, y, seed, wd, record_dir)
    raise KeyError(method)


def _record(method, cell, R, orders, seconds, seed, extra):
    """One cache record per produced order (catR produces two)."""
    n = int(R.shape[1])
    out = {}
    for name, o in orders.items():
        rec = {'method': name, 'kind': 'order', 'prefix': PREFIX[name], 'n_bank': n,
               'n_selected': len(o), 'order': o, 'cell': cell, 'cell_seed': int(seed),
               'K_cal': int(R.shape[0]), 'seconds': float(seconds), 'created': time.time()}
        rec.update(extra)
        out[name] = rec
    return out


def compute_cell(method, seed, K, slot):
    """Compute (and cache) the official order of `method` on protocol cell
    (seed, K, slot). catR caches both of its orders."""
    from official.data import protocol_cell
    R, y, bi, SR = protocol_cell(seed, K, slot)
    cs = cell_seed(seed, K, slot)
    tag = f's{seed}_K{K}_p{slot}'
    wd = _workdir(method, tag)
    t0 = time.time()
    orders = _compute(method, R, y, cs, wd, record_dir=RECORD_SCRATCH / 'atlas' / tag)
    recs = _record(method, {'seed': int(seed), 'K': int(K), 'slot': int(slot)}, R, orders,
                   time.time() - t0, cs, {'SR': float(SR)})
    for name, rec in recs.items():
        _write(cache_path(name, seed, K, slot), rec)
    return recs[method]


def compute_loo_cell(method, seed, K, j):
    """Compute (and cache) the official order on the leave-one-planner-out cell
    run_tau_calibration.py runs: calibration = the (seed, K) panel minus planner j,
    evaluation planner = j, bank = j's recorded routes."""
    R, y, bi = loo_cell(seed, K, j)
    cs = loo_cell_seed(seed, K, j)
    wd = _workdir(method, f'loo_s{seed}_K{K}_j{j}')
    t0 = time.time()
    orders = _compute(method, R, y, cs, wd, record_dir=None)      # LOO fits are not in the run of record
    recs = _record(method, {'seed': int(seed), 'K': int(K), 'j': int(j)}, R, orders,
                   time.time() - t0, cs, {'SR': float(np.mean(y))})
    for name, rec in recs.items():
        _write(cache_path(name, seed, K, j=j), rec)
    return recs[method]


def compute_ups_cell(method, seed, js):
    """Compute (and cache) the official order on a UPS cell (run_ups.py)."""
    R, y, bi = ups_cell(seed, js)
    cs = ups_cell_seed(seed, js)
    wd = _workdir(method, f'ups_s{seed}_j{js}')
    t0 = time.time()
    orders = _compute(method, R, y, cs, wd, record_dir=None)
    recs = _record(method, {'seed': int(seed), 'js': int(js), 'protocol': 'ups'}, R, orders,
                   time.time() - t0, cs, {'SR': float(np.mean(y))})
    for name, rec in recs.items():
        _write(ups_cache_path(name, seed, js), rec)
    return recs[method]


def ups_cell(seed, js):
    """(R, y, bank rows) of a UPS cell, built exactly as run_ups.run does: all 12
    calibration planners of the draw, bank = the calibration-type routes recorded
    for planner js (`unified_split`, not `up_split`)."""
    from official.data import panel
    from atdrive.splits import unified_split
    p = panel()
    hp, ht = unified_split(seed, p.utypes, p.J)
    cols = [c for c in range(p.J) if c not in hp]
    calR, _ = p.split_routes(ht)
    if int(js) not in hp:
        raise KeyError(f'planner {js} is not an evaluation planner of draw {seed}: {hp}')
    bi, yy = p.bank_rows(calR, int(js))
    R = np.full((len(cols), len(bi)), np.nan)
    for a_, b_ in enumerate(bi):
        rid = calR[b_]
        for k, pi in enumerate(cols):
            if (rid, pi) in p.Y:
                R[k, a_] = p.Y[(rid, pi)]
    return R, np.asarray(yy, float), [int(b) for b in bi]


def loo_cell(seed, K, j):
    """(R, y, bank rows) of a leave-one-planner-out cell, built exactly as
    run_tau_calibration.run does: the K_cal subsample of the draw minus planner j
    calibrates, planner j is evaluated on its own recorded routes."""
    from official.data import draw, panel, subsample
    p = panel()
    hp, cols, calR = draw(seed)
    cs = subsample(cols, seed, K)
    if int(j) not in cs:
        raise KeyError(f'planner {j} is not in the K_cal = {K} subsample of draw {seed}: {cs}')
    csl = [c for c in cs if c != int(j)]
    bi, yy = p.bank_rows(calR, int(j))
    R = np.full((len(csl), len(bi)), np.nan)
    for a_, b_ in enumerate(bi):
        rid = calR[b_]
        for k, pi in enumerate(csl):
            if (rid, pi) in p.Y:
                R[k, a_] = p.Y[(rid, pi)]
    return R, np.asarray(yy, float), [int(b) for b in bi]


# --------------------------------------------------------------------------- check
def check(methods, cells=None):
    """Verify the cached orders against the run of record.

    Two comparisons, whichever the record supports:
      * BUDGET PREFIXES -- the first 30 / 55 / 110 / 165 items of the cached order
        must BE the 'items' the recorded budgets read. This must hold exactly for a
        prefix method; where it does not, the difference is printed (set overlap and
        the first differing position) rather than papered over.
        Skipped for 'marginal_fisher': the catR record's 'items' is the total_fisher
        subset (catr.py ADAPTATION 7), so it is not this order's cost.
      * FULL ORDER -- catR writes both whole-bank rankings into the record's
        model_info, so the cached order is compared against them at full length.
    Returns {method: (n_checked, n_exact, rows)}."""
    R = _records()
    out = {}
    for m in methods:
        rows = []
        for (mm, seed, K, slot) in sorted(R):
            if mm != _record_method(m):
                continue
            if cells is not None and (seed, K, slot) not in cells:
                continue
            rec = _read(cache_path(m, seed, K, slot))
            if rec is None:
                continue
            r = R[(mm, seed, K, slot)]
            if 'error' in r:
                rows.append((seed, K, slot, None, 'record failed: ' + r['error'][:60]))
                continue
            det, ok = [], True
            full = r.get('model_info', {}).get(f'order_{m}')
            if full is not None:
                same = rec['order'] == [int(i) for i in full]
                det.append(f'full order ({len(full)}) ' + ('exact' if same else 'DIFFERS'))
                ok &= same
            if not (m == 'marginal_fisher' and full is not None):
                for B in sorted(int(b) for b in r['budgets']):
                    rec_items = [int(i) for i in r['budgets'][str(B)]['items']]
                    pref = rec['order'][:len(rec_items)]
                    if pref == rec_items:
                        det.append(f'B{B} exact')
                    else:
                        ov = len(set(pref) & set(rec_items))
                        first = next((i for i, (a, b) in enumerate(zip(pref, rec_items)) if a != b), None)
                        det.append(f'B{B} DIFFERS (shares {ov}/{len(rec_items)}, first at {first})')
                        ok = False
            rows.append((seed, K, slot, ok, '; '.join(det)))
        n = len([r for r in rows if r[3] is not None])
        nok = len([r for r in rows if r[3]])
        out[m] = (n, nok, rows)
        print(f'\n== {m}: {nok} of {n} cached cells reproduce the run of record exactly '
              f'(prefix method: {PREFIX[m]})')
        for seed, K, slot, o, det in rows:
            print(f'   s{seed} K{K} p{slot}  {"OK " if o else "-- "}{det}')
    return out


def _record_method(m):
    """The run-of-record method name a cached order belongs to."""
    return 'catr' if m in ('total_fisher', 'marginal_fisher') else m


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=list(ORDER_METHODS))
    ap.add_argument('--seeds', nargs=2, type=int, default=[0, 16], help='draw range [a, b)')
    ap.add_argument('--kcals', nargs='+', type=int, default=[4, 8, 12])
    ap.add_argument('--loo', action='store_true', help='the leave-one-planner-out cells instead')
    ap.add_argument('--ups', action='store_true', help='the UPS cells of run_ups.py instead')
    ap.add_argument('--check', action='store_true', help='verify cached orders against the run of record')
    ap.add_argument('--refresh', action='store_true', help='recompute even when cached')
    a = ap.parse_args()
    if a.check:
        cells = {(s, K, p) for s in range(a.seeds[0], a.seeds[1]) for K in a.kcals for p in range(4)}
        check([m for m in a.methods], cells)
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    todo, done, failed = 0, 0, 0
    if a.ups:
        from official.data import panel
        p = panel()
        from atdrive.splits import unified_split
        for seed in range(a.seeds[0], a.seeds[1]):
            for js in unified_split(seed, p.utypes, p.J)[0]:
                for m in a.methods:
                    path = ups_cache_path(m, seed, js)
                    if path.exists() and not a.refresh:
                        continue
                    todo += 1
                    t0 = time.time()
                    try:
                        rec = compute_ups_cell(m, seed, js)
                        done += 1
                        print(f'{m:15} ups s{seed} j{js}  {time.time() - t0:6.1f}s  '
                              f'{rec["n_selected"]}/{rec["n_bank"]} items', flush=True)
                    except Exception as e:
                        failed += 1
                        _write(path.with_suffix('.error.json'),
                               {'method': m, 'seed': seed, 'js': js, 'error': f'{type(e).__name__}: {e}',
                                'traceback': traceback.format_exc()[-3000:]})
                        print(f'{m:15} ups s{seed} j{js}  FAILED {type(e).__name__}: {e}', flush=True)
        print(f'\n{done} computed, {failed} failed, {todo} attempted (cached cells skipped)')
        return
    for seed in range(a.seeds[0], a.seeds[1]):
        for K in a.kcals:
            if a.loo:
                from official.data import draw, subsample
                units = [('j', j) for j in subsample(draw(seed)[1], seed, K)]
            else:
                units = [('p', p) for p in range(4)]
            for kind, u in units:
                for m in a.methods:
                    p = cache_path(m, seed, K, slot=u) if kind == 'p' else cache_path(m, seed, K, j=u)
                    if p.exists() and not a.refresh:
                        continue
                    todo += 1
                    t0 = time.time()
                    try:
                        rec = (compute_cell(m, seed, K, u) if kind == 'p' else compute_loo_cell(m, seed, K, u))
                        done += 1
                        print(f'{m:15} s{seed} K{K} {kind}{u}  {time.time() - t0:6.1f}s  '
                              f'{rec["n_selected"]}/{rec["n_bank"]} items', flush=True)
                    except Exception as e:
                        failed += 1
                        _write(p.with_suffix('.error.json'),
                               {'method': m, 'seed': seed, 'K': K, kind: u, 'error': f'{type(e).__name__}: {e}',
                                'traceback': traceback.format_exc()[-3000:]})
                        print(f'{m:15} s{seed} K{K} {kind}{u}  FAILED {type(e).__name__}: {e}', flush=True)
    print(f'\n{done} computed, {failed} failed, {todo} attempted (cached cells skipped)')


if __name__ == '__main__':
    main()
