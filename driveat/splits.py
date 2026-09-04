"""The unified evaluation split — the single source of truth for every experiment.

12/4 planners x 36/8 scene types (16-planner panel), R = 16 Monte-Carlo cross-validation draws.
Within one draw every regime uses the same planner split:

                     calibration planners (12)  evaluation planners (4)
  calibration types (36)  A: calibration block    UP evaluation
  evaluation types (8)    C: US evaluation        D: UPS target (0 rollouts)

UP does not hold scenario types out: it calibrates the bank from the 12
calibration planners over all 220 routes and evaluates the 4 held-out
planners on that same 220-route bank (`up_split`). US and UPS keep the 36/8
type partition, so the C and D blocks are unchanged.

Verbatim port of the research script `b2d_splits.py`; the RandomState seed
convention (1000 + draw index) is part of the protocol and must not change.
"""
import numpy as np

R_DRAWS = 16
H_P, H_S = 4, 8


def unified_split(seed, utypes, n_planners=16):
    """Return (held_out_planner_ids, held_out_type_set) for one draw.

    `utypes` must be the sorted list of unique scenario types (44 for B2D);
    passing it explicitly keeps the split independent of any data-loading
    order except the canonical sorted type list.
    """
    rng = np.random.RandomState(1000 + seed)
    hp = sorted(rng.choice(n_planners, H_P, replace=False).tolist())
    ht = set(np.array(sorted(utypes))[rng.choice(len(utypes), H_S, replace=False)].tolist())
    return hp, ht


def up_split(seed, utypes, n_planners=16):
    """The UP-side split of one draw: the same held-out planners, no held-out
    scenario types, so the bank is the whole 220-route benchmark."""
    return unified_split(seed, utypes, n_planners)[0], set()
