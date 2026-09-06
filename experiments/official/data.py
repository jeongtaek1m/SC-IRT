"""Protocol cells for the official-code baselines — the SAME draws, calibration
subsamples and banks as experiments/run_up_frontier.py (Table 1).

    R, y, bi, SR = protocol_cell(seed, Kc, slot)

seed : draw 0-15 (up_split(seed, ...) -> 4 evaluation planners, no type hold-out)
Kc   : calibration-panel size in {4, 8, 12}; RandomState(9000 + 100 seed + 10 Kc)
       subsample of the 12 calibration planners, identical to run_up_frontier.subsample
slot : 0-3, which of the draw's 4 evaluation planners
R    : (Kc, n_bank) calibration outcomes on the evaluation planner's bank (1 pass,
       0 fail, NaN no record); y : (n_bank,) the evaluation planner's outcomes;
       bi : bank row ids into the 220-route list; SR = y.mean() (the target).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from atdrive.b2d import Panel                      # noqa: E402
from atdrive.splits import up_split                # noqa: E402

KCALS = (4, 8, 12)
BGRID = (30, 55, 110, 165)
NDRAWS = 16
_PANEL = None


def panel():
    global _PANEL
    if _PANEL is None:
        _PANEL = Panel()
    return _PANEL


def subsample(cols, seed, Kc):
    """Verbatim run_up_frontier.subsample."""
    if Kc >= len(cols):
        return list(cols)
    rs = np.random.RandomState(9000 + seed * 100 + Kc * 10 + 0)
    return sorted(np.array(cols)[rs.choice(len(cols), Kc, replace=False)].tolist())


def draw(seed):
    """(held-out evaluation planners, calibration planners, calibration routes) of one draw."""
    p = panel()
    hp, ht = up_split(seed, p.utypes, p.J)
    cols = [c for c in range(p.J) if c not in hp]
    calR, _ = p.split_routes(ht)
    return list(hp), cols, calR


def protocol_cell(seed, Kc, slot):
    p = panel()
    hp, cols, calR = draw(seed)
    cs = subsample(cols, seed, Kc)
    js = hp[slot]
    bi, yy = p.bank_rows(calR, js)
    R = np.full((len(cs), len(bi)), np.nan)
    for a_, b_ in enumerate(bi):
        rid = calR[b_]
        for k, pi in enumerate(cs):
            if (rid, pi) in p.Y:
                R[k, a_] = p.Y[(rid, pi)]
    y = np.asarray(yy, float)
    return R, y, [int(b) for b in bi], float(y.mean())


def route_types(bi):
    """Scenario type of every bank row (for stratified variants)."""
    p = panel()
    _, _, calR = draw(0)
    return [p.sn[calR[b]] for b in bi]
