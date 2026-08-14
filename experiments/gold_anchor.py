#!/usr/bin/env python3
"""Generate the frozen gold difficulty anchor. Run this first.

Two artifacts, both consumed by every other experiment:

* **gold** — a single 2PL calibration on the complete panel (219 routes x 16
  planners, 800 Adam steps), frozen and never refitted. It is the evaluation
  anchor only: no model is ever trained against it, so a predictor correlating
  with it is being scored, not fitted.
* **bt** — out-of-fold predicted difficulty from the two-stage explanatory
  baseline, under 44-type leave-one-type-out. Each fold recalibrates on the
  training types alone and fits a ridge from standardised scene features to the
  resulting difficulty, so no held-out scenario type contributes to its own
  prediction.

The printed rho values are the hand-crafted rows of Table IV: kin-only ridge
(+0.418) and the ground-truth stack (+0.529).

Note on the route universe: routes must appear in the type map *and* in both
feature files. That intersection is what defines the canonical 219-route bank,
so dropping a feature file here silently changes the anchor.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scirt import data, features, irt, paths, runtime, stage2  # noqa: E402

runtime.configure()
runtime.set_global_seeds(0)
paths.ensure_results()

CK = features.load_st("eval_cmdkin_stats")
CRF = features.load_st("eval_camrisk")
GTR = features.load_st("eval_gtrisk")

CONFIGS = {
    "ck": CK,
    "ck+camrisk-full": features.concat(CK, CRF),
    "GT:ck+gtrisk": features.concat(CK, GTR),
}

panel = data.read_response_panel()
route_types = data.read_route_types()
J = panel.n_planners

allr = data.route_universe(panel.route_ids, route_types, CK, CRF)
data.assert_canonical_universe(allr)
types = sorted(set(route_types[r] for r in allr))

ALL_PLANNERS = [True] * J


def calibrate(routes, it):
    fit, _ = irt.calibrate_panel(panel, routes, planner_mask=ALL_PLANNERS,
                                 model="2pl", it=it)
    return {r: float(b) for r, b in zip(routes, irt.center_b(fit))}


GOLD = calibrate(allr, it=800)

BT = {name: {} for name in CONFIGS}
for t in types:
    train = [r for r in allr if route_types[r] != t]
    test = [r for r in allr if route_types[r] == t]
    bh = calibrate(train, it=400)
    for name, feat in CONFIGS.items():
        BT[name].update(
            stage2.ridge_b(
                feat, train, [bh[r] for r in train], test,
                alpha=100.0, predict="per_row",
            )
        )

print(f"=== {len(allr)} route · 44-type LOTO · pooled Spearman vs frozen gold ===")
for name in CONFIGS:
    rho = spearmanr(
        [GOLD[r] for r in allr], [BT[name][r] for r in allr]
    ).correlation
    print(f"{name:26s} ρ={rho:+.3f}")

json.dump(
    {"gold": GOLD, "bt": BT},
    open(paths.result("gold_anchor.json"), "w"),
)
print("saved results/gold_anchor.json")
