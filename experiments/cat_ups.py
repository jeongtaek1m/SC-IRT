#!/usr/bin/env python3
"""Table II(b): adaptive evaluation when the scenarios are new too.

The regime the method exists for. Both the planner and the scenario type are
unseen, so the held-out routes have no calibrated difficulty at all — classical
Fisher-information selection is not merely worse here, it is undefined. Only
predicted difficulty b-tilde makes adaptive selection possible.

The naive move, greedy Fisher information on b-tilde, fails: it preferentially
picks routes whose difficulty happened to be over-predicted, buying prediction
error instead of information. Discounting the information by the Stage-2 residual
variance (`selection.shrunk_information`) removes that bias, and is what closes
most of the gap to the oracle.

The oracle row selects on the true calibrated difficulty. It is unattainable by
construction — it needs the responses the experiment is trying to avoid
collecting — and is shown only as a ceiling. IES is undefined for it.

Design: three seeds, each splitting the 44 scenario types in half into a
calibration bank and a new-type bank, crossed with leave-one-planner-out.
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

allr = data.route_universe(panel.route_ids, route_types, FEAT)
types = sorted(set(route_types[r] for r in allr))

TAUS = [0.40, 0.35]
BMAX = 100
SEEDS = [0, 1, 2]
STRAT = ["random", "btilde-spread", "btilde-matched", "btilde-hybrid",
         "btilde-fisher-sh", "ORACLE-fisher(b̂)"]
HYBRID_SWITCH_STEP = 8  # spread first, matched thereafter

agg = {s: {t: {"n": [], "e": [], "sr": []} for t in TAUS} for s in STRAT}


def calibrate_1pl(routes, planner_mask, it=600):
    fit, _ = irt.calibrate_panel(panel, routes, planner_mask=planner_mask,
                                 model="1pl", it=it)
    return dict(zip(routes, (float(x) for x in irt.center_b(fit))))


for seed in SEEDS:
    # One generator per seed drives the type split and every subsequent draw.
    # The number of draws depends on the strategy list, BMAX and the band rule,
    # so none of those may change without shifting the stream for every later fold.
    rng = np.random.RandomState(seed)
    tp = rng.permutation(len(types))
    cal_types = set(types[i] for i in tp[: len(types) // 2])
    calR = [r for r in allr if route_types[r] in cal_types]
    newR = [r for r in allr if route_types[r] not in cal_types]

    for js in range(J):
        pmask = [pi != js for pi in range(J)]

        bh_cal = calibrate_1pl(calR, pmask)
        cal_targets = [bh_cal[r] for r in calR]
        rg = stage2.fit_ridge_b(FEAT, calR, cal_targets, alpha=100.0)
        bt = rg.predict(FEAT, newR, predict="per_row")
        bh_all = calibrate_1pl(allr, pmask)  # oracle ceiling only

        # Prediction noise of b-tilde, estimated in-fold from the same ridge.
        s2 = rg.residual_var(FEAT, calR, cal_targets)

        jr = [r for r in newR if panel.observed(r, js)]
        if len(jr) < 40:
            continue
        yy = {r: panel.y[(r, js)] for r in jr}

        th_full = map_theta(
            np.array([bh_all[r] for r in jr]),
            np.array([yy[r] for r in jr], float),
            np.ones(len(jr)), it=50,
        )

        order_sp = selection.order_spread(jr, bt)
        order_rd = list(rng.permutation(jr))

        for st in STRAT:
            bd = bh_all if st.startswith("ORACLE") else bt
            administered, th = [], 0.0
            done = {t: False for t in TAUS}

            for step in range(min(BMAX, len(jr))):
                rem = [r for r in jr if r not in administered]
                if st == "random":
                    pick = [r for r in order_rd if r not in administered][0]
                elif st == "btilde-spread":
                    pick = [r for r in order_sp if r not in administered][0]
                elif st == "btilde-matched" or (
                    st == "btilde-hybrid" and step >= HYBRID_SWITCH_STEP
                ):
                    # Sample from a band near the current estimate rather than
                    # taking its argmax: randomising within the band is what
                    # avoids the winner's curse of greedy selection.
                    dist = np.array([abs(bd[r] - th) for r in rem])
                    k = max(3, len(rem) // 10)
                    band = [rem[i] for i in np.argsort(dist)[:k]]
                    pick = band[rng.randint(len(band))]
                elif st == "btilde-hybrid":
                    pick = [r for r in order_sp if r not in administered][0]
                elif st == "btilde-fisher-sh":
                    info = selection.shrunk_information(
                        th, [bd[r] for r in rem], s2
                    )
                    pick = rem[int(np.argmax(info))]
                else:  # ORACLE-fisher, on true calibrated difficulty
                    p = np.array([sig(th - bd[r]) for r in rem])
                    pick = rem[int(np.argmax(p * (1 - p)))]

                administered.append(pick)
                th = map_theta(
                    np.array([bd[r] for r in administered]),
                    np.array([yy[r] for r in administered], float),
                    np.ones(len(administered)), it=50,
                )
                p = np.array([sig(th - bd[r]) for r in administered])
                se = 1.0 / np.sqrt((p * (1 - p)).sum() + 1.0)

                for t in TAUS:
                    if not done[t] and se < t:
                        done[t] = True
                        rest = [r for r in jr if r not in set(administered)]
                        sr = (
                            abs(sum(sig(th - bd[r]) for r in rest)
                                - sum(yy[r] for r in rest)) / len(jr)
                            if rest else 0.0
                        )
                        agg[st][t]["n"].append(step + 1)
                        agg[st][t]["e"].append(abs(th - th_full))
                        agg[st][t]["sr"].append(sr)
                # No early exit once both taus have latched: the loop runs to BMAX
                # and its remaining rng.randint draws are part of the stream.

            for t in TAUS:
                if not done[t]:
                    agg[st][t]["n"].append(len(administered))
                    agg[st][t]["e"].append(abs(th - th_full))
                    agg[st][t]["sr"].append(0.0)

print("UPS DYNAMIC CAT - new planner x new scene type "
      "(no calibrated b available; only predicted b_tilde)")
print(f"feature=cmdkin+scenparamz, half the types calibrated / half new, "
      f"{len(SEEDS)} seeds x LOPO {J}")
for t in TAUS:
    print(f"\n### τ={t:.2f}")
    print(f"{'strategy':20s} {'items(n)':>14s} {'srMAE±SE':>16s} {'θerr±SE':>16s}")
    for st in STRAT:
        d = agg[st][t]
        if not d["n"]:
            continue
        n_m, n_se = stats.mean_se(d["n"])
        s_m, s_se = stats.mean_se(d["sr"])
        e_m, e_se = stats.mean_se(d["e"])
        print(f"{st:20s} {n_m:7.1f}±{n_se:<5.1f} {s_m:9.4f}±{s_se:<5.4f} "
              f"{e_m:9.3f}±{e_se:<5.3f}")

print("\nORACLE-fisher uses the true calibrated difficulty and is unattainable "
      "in the UPS regime (ceiling only). Every other rule selects from features alone.")
