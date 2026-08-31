"""The unified evaluation split — the single source of truth for every experiment.

16/6 planners x 36/8 scene types (22-planner panel), R = 16 Monte-Carlo cross-validation draws.
Within one draw the three regimes (US / UP / UPS) share the same partition:

                     calibration planners (16)  evaluation planners (6)
  calibration types (36)  A: calibration block    UP evaluation
  evaluation types (8)    C: US evaluation        D: UPS target (0 rollouts)

Verbatim port of the research script `b2d_splits.py`; the RandomState seed
convention (1000 + draw index) is part of the protocol and must not change.
"""
import numpy as np

R_DRAWS = 16
H_P, H_S = 6, 8


def unified_split(seed, utypes, n_planners=22):
    """Return (held_out_planner_ids, held_out_type_set) for one draw.

    `utypes` must be the sorted list of unique scenario types (44 for B2D);
    passing it explicitly keeps the split independent of any data-loading
    order except the canonical sorted type list.
    """
    rng = np.random.RandomState(1000 + seed)
    hp = sorted(rng.choice(n_planners, H_P, replace=False).tolist())
    ht = set(np.array(sorted(utypes))[rng.choice(len(utypes), H_S, replace=False)].tolist())
    return hp, ht
