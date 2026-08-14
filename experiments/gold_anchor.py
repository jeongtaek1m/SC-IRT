#!/usr/bin/env python3
"""Frozen gold difficulty anchor.

Fits the full-panel 2PL calibration once (800 Adam steps) on the 219-route
universe and freezes it as `results/gold_anchor.json`. Every later step scores
against this anchor; it is never refit.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scirt import data, features, irt, paths, runtime  # noqa: E402

runtime.configure()
runtime.set_global_seeds(0)
paths.ensure_results()

CK = features.load_st("eval_cmdkin_stats")
panel = data.read_response_panel()
route_types = data.read_route_types()

allr = data.route_universe(panel.route_ids, route_types, CK)
data.assert_canonical_universe(allr)

fit, _ = irt.calibrate_panel(panel, allr, planner_mask=[True] * panel.n_planners,
                             model="2pl", it=800)
GOLD = {r: float(b) for r, b in zip(allr, irt.center_b(fit))}

print(f"=== frozen 2PL gold anchor · {len(allr)} routes ===")
json.dump({"gold": GOLD}, open(paths.result("gold_anchor.json"), "w"))
print("saved results/gold_anchor.json")
