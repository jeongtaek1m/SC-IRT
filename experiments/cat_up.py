#!/usr/bin/env python3
"""Table II(a): adaptive evaluation on a calibrated bank (unseen planner).

A genuine computerised adaptive test. A new planner is evaluated one route at a
time; after each response the ability estimate and its standard error are
updated, and testing stops as soon as SE(theta) drops below tau. The number of
routes needed is therefore not fixed — it is the result.

This is the regime where the bank is already calibrated: difficulty and
discrimination come from the responses of the other 15 planners (leave-one-planner-out),
so classical Fisher-information selection is available. Table II(b) covers the
harder case where the scenarios are new as well and no calibration exists.

Efficiency is reported as IES = bank size / mean routes administered. Selecting
by 2PL information reaches the same precision as random selection in roughly half
the routes.

Reproducibility note: the three strategies that order the bank by *fitted*
difficulty (fisher-1PL, bhat-spread, tinyAnchor) are chaotic. On a 1PL bank the
difficulties take only about thirty distinct values across 219 routes, so ties are
resolved at float noise, and a 1e-7 change flips an early choice and diverges the
whole trajectory. See REPRODUCIBILITY.md.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scirt import data, features, irt, paths, runtime, selection, stage2, stats  # noqa: E402
from scirt.theta import map_theta, sig  # noqa: E402

runtime.configure()
runtime.set_global_seeds(0)
paths.ensure_results()

CK = features.load_st("eval_cmdkin_stats")
SPZ = features.load_st("eval_scenparamz")
FEAT = features.concat(CK, SPZ)

panel = data.read_response_panel()
route_types = data.read_route_types()
J = panel.n_planners
planners = panel.planners

allr = data.route_universe(panel.route_ids, route_types, FEAT)

TAUS = [0.40, 0.35, 0.30]
BMAX = 120
STRAT = ["fisher-2PL", "fisher-1PL", "random-strat", "bhat-spread",
         "btilde-spread", "tinyAnchor"]


def calibrate(routes, planner_mask, *, model, it):
    fit, _ = irt.calibrate_panel(panel, routes, planner_mask=planner_mask,
                                 model=model, it=it)
    b = irt.center_b(fit)
    a = fit.a if model == "2pl" else np.ones(len(routes))
    return (dict(zip(routes, (float(x) for x in b))),
            dict(zip(routes, (float(x) for x in a))))


agg = {s: {t: {"n": [], "e": [], "sr": []} for t in TAUS} for s in STRAT}
per_planner = {}

for js in range(J):
    pmask = [pi != js for pi in range(J)]
    # Fit order is preserved: 1PL then 2PL. Reordering changes the last bits of
    # both fits through allocation order, which the b-hat-ordered strategies amplify.
    bh, _ = calibrate(allr, pmask, model="1pl", it=800)
    bh2, ah2 = calibrate(allr, pmask, model="2pl", it=800)

    bt = stage2.ridge_b(FEAT, allr, [bh[r] for r in allr], allr,
                        alpha=100.0, predict="per_row")

    jr = [r for r in allr if panel.observed(r, js)]
    yy = {r: panel.y[(r, js)] for r in jr}

    # Reference ability from the planner's complete response set.
    th_full = map_theta(
        np.array([bh[r] for r in jr]),
        np.array([yy[r] for r in jr], float),
        np.ones(len(jr)), it=50,
    )

    orders = {
        "random-strat": selection.order_strat_random(jr, route_types),
        "bhat-spread": selection.order_spread(jr, bh),
        "btilde-spread": selection.order_spread(jr, bt),
        "tinyAnchor": selection.order_anchor(jr, bh),
    }

    for st in STRAT:
        administered, th = [], 0.0
        done = {t: False for t in TAUS}
        B_ = bh2 if st == "fisher-2PL" else bh
        A_ = ah2 if st == "fisher-2PL" else {r: 1.0 for r in jr}

        for step in range(min(BMAX, len(jr))):
            if st == "fisher-1PL":
                rem = [r for r in jr if r not in administered]
                p = np.array([sig(th - B_[r]) for r in rem])
                administered.append(rem[int(np.argmax(p * (1 - p)))])
            elif st == "fisher-2PL":
                rem = [r for r in jr if r not in administered]
                info = np.array([
                    selection.fisher_information(th, B_[r], A_[r]) for r in rem
                ])
                administered.append(rem[int(np.argmax(info))])
            else:
                administered.append([r for r in orders[st] if r not in administered][0])

            th = map_theta(
                np.array([B_[r] for r in administered]),
                np.array([yy[r] for r in administered], float),
                np.array([A_[r] for r in administered]), it=50,
            )
            p = np.array([sig(A_[r] * (th - B_[r])) for r in administered])
            info = (np.array([A_[r] ** 2 for r in administered]) * p * (1 - p)).sum()
            se = 1.0 / np.sqrt(info + 1.0)

            for t in TAUS:
                if not done[t] and se < t:
                    done[t] = True
                    rest = [r for r in jr if r not in set(administered)]
                    # p-IRT whole-bank success rate: administered routes contribute
                    # their observed response, the rest their predicted probability.
                    # The observed terms cancel, leaving |sum(p) - sum(y)| / N.
                    sr = (
                        abs(sum(sig(A_[r] * (th - B_[r])) for r in rest)
                            - sum(yy[r] for r in rest)) / len(jr)
                        if rest else 0.0
                    )
                    agg[st][t]["n"].append(step + 1)
                    agg[st][t]["e"].append(abs(th - th_full))
                    agg[st][t]["sr"].append(sr)
                    if st == "fisher-2PL" and t == 0.35:
                        per_planner[js] = (step + 1, abs(th - th_full))

        for t in TAUS:
            if not done[t]:
                agg[st][t]["n"].append(len(administered))
                agg[st][t]["e"].append(abs(th - th_full))
                agg[st][t]["sr"].append(0.0)

print("UP DYNAMIC CAT - items (mean routes administered) + MAE±SE  "
      "[feature=cmdkin+scenparamz, LOPO 16]")
for t in TAUS:
    print(f"\n### τ={t:.2f}")
    print(f"{'strategy':16s} {'items(n)':>14s} {'srMAE±SE':>16s} {'θerr±SE':>16s}")
    for st in STRAT:
        d = agg[st][t]
        n_m, n_se = stats.mean_se(d["n"])
        s_m, s_se = stats.mean_se(d["sr"])
        e_m, e_se = stats.mean_se(d["e"])
        print(f"{st:16s} {n_m:7.1f}±{n_se:<5.1f} {s_m:9.4f}±{s_se:<5.4f} "
              f"{e_m:9.3f}±{e_se:<5.3f}")

print("\nfisher-2PL, τ=0.35 — per-planner stopping point:")
for js in sorted(per_planner, key=lambda x: per_planner[x][0]):
    print(f"  {planners[js]:20s} n={per_planner[js][0]:>3d}  θerr={per_planner[js][1]:.3f}")
