#!/usr/bin/env python3
"""Table I: how far hand-crafted scene descriptors get toward true difficulty.

Runs the two-stage explanatory IRT baseline for each descriptor family under two
generalisation regimes:

* **US (unseen scenario)** — 44-type leave-one-type-out. Every held-out scenario
  type is absent from calibration, so the descriptor must extrapolate to a kind of
  scenario it has never seen. This is the regime Table I reports.
* **UP (unseen planner)** — leave-one-planner-out with a fixed 20-route
  stratified probe, reported in the appendix. Its success-rate MAE saturates
  around 0.05 for every descriptor, which is why difficulty recovery (rho), not
  MAE, is the discriminating metric.

Three metrics, deliberately not interchangeable: AUROC ranks responses but says
nothing about absolute level; MAE is reported at the aggregate level only, since
per-cell |p - y| is improper; rho measures recovery of the difficulty ordering
against the frozen gold anchor and is the headline.

Descriptors are an explicit registry (see scirt/features.py) rather than the
substring-filtered sweep the exploratory version used.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scirt import data, features, irt, paths, runtime, selection, stage2  # noqa: E402
from scirt.theta import map_theta, sig  # noqa: E402

runtime.configure()
runtime.set_global_seeds(0)
paths.ensure_results()

panel = data.read_response_panel()
route_types = data.read_route_types()
J = panel.n_planners

FEATS = features.build_descriptors()
ALL_PLANNERS = [True] * J


def calibrate(routes, planner_mask, it):
    """2PL fit returning theta (by planner index), difficulty and discrimination."""
    fit, keep = irt.calibrate_panel(panel, routes, planner_mask=planner_mask,
                                    model="2pl", it=it)
    theta, b = irt.center_both(fit)
    return (
        {keep[i]: float(theta[i]) for i in range(len(keep))},
        dict(zip(routes, (float(x) for x in b))),
        dict(zip(routes, (float(x) for x in fit.a))),
    )


def explanatory_fit(feat, all_routes, train_routes, planner_mask):
    """Stage 1 on the training routes, then Stage 2 from features to (b, log a)."""
    theta, bh, ah = calibrate(list(train_routes), planner_mask, it=400)
    b_pred, a_pred = stage2.ridge_explanatory_2pl(
        feat, list(train_routes), bh, ah, all_routes
    )
    return theta, b_pred, a_pred


# Frozen gold anchor. The route universe is gated on the Min-TTC descriptor
# because it is the one built from the traffic CSV, which covers the collection.
allr0 = [r for r in panel.route_ids if r in route_types and r in FEATS["minTTC(1)"]]
_, GOLD, _ = calibrate(allr0, ALL_PLANNERS, it=800)

print(
    f"{'feature':20s} | {'UP mAUC':>7s} {'UP srMAE':>8s} | "
    f"{'US AUC':>7s} {'US prMAE':>8s} {'ρ(gold)':>8s}"
)
print("-" * 72)

OUT = {}
for name, feat in FEATS.items():
    allr = [r for r in panel.route_ids if r in feat and r in route_types]
    types = sorted(set(route_types[r] for r in allr))

    # --- UP: leave one planner out, 20-route stratified probe -----------------
    up_auc, up_mae = [], []
    for js in range(J):
        _, bd, ad = explanatory_fit(feat, allr, allr, [pi != js for pi in range(J)])
        seen = [r for r in allr if panel.observed(r, js)]
        probe = selection.probe_strat(seen, route_types)
        probe_set = set(probe)
        held = [r for r in seen if r not in probe_set]

        th = map_theta(
            np.array([bd[r] for r in probe]),
            np.array([panel.y[(r, js)] for r in probe], float),
            np.array([ad.get(r, 1.0) for r in probe]),
            it=40,
        )
        y_held = [panel.y[(r, js)] for r in held]
        p_held = [sig(ad.get(r, 1.0) * (th - bd[r])) for r in held]
        if len(set(y_held)) == 2:
            up_auc.append(roc_auc_score(np.array(y_held, float), p_held))

        # p-IRT success-rate reconstruction (tinyBenchmarks convention): observed
        # responses enter as themselves, unadministered ones as their predicted
        # probability, and the error is against the whole-bank success rate.
        y_probe = [panel.y[(r, js)] for r in probe]
        n_bank = len(y_probe) + len(y_held)
        up_mae.append(
            abs((sum(y_probe) + sum(p_held)) / n_bank
                - (sum(y_probe) + sum(y_held)) / n_bank)
        )

    # --- US: leave one scenario type out --------------------------------------
    y_all, p_all, route_err, bt_all = [], [], [], {}
    for t in types:
        train = [r for r in allr if route_types[r] != t]
        test = [r for r in allr if route_types[r] == t]
        theta, bd, ad = explanatory_fit(feat, allr, train, ALL_PLANNERS)
        for r in test:
            bt_all[r] = bd[r]
            resp = panel.responses_for(r)
            if not resp:
                continue
            ps = [sig(ad.get(r, 1.0) * (theta.get(pi, 0) - bd[r])) for pi, _ in resp]
            ys = [y for _, y in resp]
            route_err.append(abs(np.mean(ps) - np.mean(ys)))
            y_all += ys
            p_all += ps

    us_auc = roc_auc_score(np.array(y_all, float), p_all)
    scored = [r for r in bt_all if r in GOLD]
    rho = spearmanr([GOLD[r] for r in scored], [bt_all[r] for r in scored]).correlation

    OUT[name] = (np.mean(up_auc), np.mean(up_mae), us_auc, np.mean(route_err), rho)
    print(
        f"{name:20s} |  {np.mean(up_auc):.3f}  {np.mean(up_mae):.3f}   |  "
        f"{us_auc:.3f}  {np.mean(route_err):.3f}   {rho:+.3f}"
    )

json.dump(
    {k: [float(x) for x in v] for k, v in OUT.items()},
    open(paths.result("descriptor_table.json"), "w"),
    indent=1,
)
print("\nsaved results/descriptor_table.json")
print(
    "UP mAUC=macro-AUROC, srMAE=|predSR−SR| (planner mean) | "
    "US AUC=micro, prMAE=route pass-rate error, ρ=LOTO b̃ vs gold (pooled)"
)
