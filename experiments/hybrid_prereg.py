#!/usr/bin/env python3
"""Table IV, rank-fusion row: does combining the encoder with a hand-crafted stack help?

The encoder and the feature stacks tie on this bank (see encoder_verify.py), but a
tie in aggregate does not mean they are redundant — they can be right about
different scenarios. Averaging their *ranks* tests that directly.

Fusion is on ranks rather than raw values because the two arms are on different
scales: the encoder emits a difficulty in logits, the ridge a regression output
shrunk toward the training mean. Rank averaging is scale-free and needs no
recalibration.

Pre-registration note: gates G1 and G2 were fixed before the comparison was run,
and both fail — their intervals include zero. The only interval excluding zero is
the one this script labels a reference comparison. That distinction is deliberate
and is preserved verbatim.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scirt import data, features, irt, paths, resample, runtime, stage2, stats  # noqa: E402

runtime.configure()
runtime.set_global_seeds(0)
paths.ensure_results()

CK = features.load_st("eval_cmdkin_stats")
SPG = features.load_st("eval_scenparamz_glob")
SPZCK = features.concat(CK, SPG)

panel = data.read_response_panel()
route_types = data.read_route_types()
J = panel.n_planners

anchor = json.load(open(paths.result("gold_anchor.json")))
GOLD, BT = anchor["gold"], anchor["bt"]

allr = [r for r in GOLD if r in SPZCK and r in route_types]
types = sorted(set(route_types[r] for r in allr))
ALL_PLANNERS = [True] * J

# The scenario-parameter stack is the strongest hand-crafted comparator and is
# recomputed here rather than read from the anchor, since it is a comparison
# baseline for this gate only.
SPZBT = {}
for t in types:
    train = [r for r in allr if route_types[r] != t]
    test = [r for r in allr if route_types[r] == t]
    fit, _ = irt.calibrate_panel(panel, train, planner_mask=ALL_PLANNERS,
                                 model="2pl", it=400)
    bh = dict(zip(train, irt.center_b(fit)))
    SPZBT.update(
        stage2.ridge_b(SPZCK, train, [bh[r] for r in train], test,
                       alpha=100.0, predict="per_row")
    )

w2 = np.load(f"{paths.INTERACT}/interact_b2d_w2a_final.npz", allow_pickle=True)
w2r = [str(r).replace("route_", "") for r in w2["routes"]]
ens6 = {w2r[i]: float(np.array(w2["ens6"])[i]) for i in range(len(w2r))}

keep = [r for r in allr if r in ens6 and r in BT["ck+camrisk-full"]]
gv = np.array([GOLD[r] for r in keep])


def vec(d):
    return np.array([d[r] for r in keep])


E, CAM = vec(ens6), vec(BT["ck+camrisk-full"])
GTR, SPZ = vec(BT["GT:ck+gtrisk"]), vec(SPZBT)

hyb_cam = (rankdata(E) + rankdata(CAM)) / 2
hyb_gt = (rankdata(E) + rankdata(GTR)) / 2

ARMS = {
    "hybrid-cam": hyb_cam,
    "hybrid-gt": hyb_gt,
    "ens6": E,
    "ck+camrisk-full": CAM,
    "GT:ck+gtrisk": GTR,
    "ck+spzglob": SPZ,
}

print("=== point estimates (pooled ρ) ===")
for k, v in ARMS.items():
    print(f"  {k:16s} ρ={spearmanr(gv, v).correlation:+.4f}")

cluster_labels = [route_types[r] for r in keep]
unique_types = sorted(set(cluster_labels))
clusters = [
    np.array([i for i, x in enumerate(cluster_labels) if x == t]) for t in unique_types
]

print("\n=== pre-registered gates (44-type cluster paired bootstrap, 10k) ===")
for lab, a, b in [
    ("G1: hybrid-cam − ck+spzglob", hyb_cam, SPZ),
    ("G2: hybrid-cam − GT:ck+gtrisk", hyb_cam, GTR),
    ("(sec) hybrid-gt − ck+spzglob", hyb_gt, SPZ),
    ("(sec) hybrid-gt − GT:ck+gtrisk", hyb_gt, GTR),
    ("(ref) hybrid-cam − ck+camrisk-full", hyb_cam, CAM),
]:
    delta = spearmanr(gv, a).correlation - spearmanr(gv, b).correlation
    ci, p = resample.paired_delta_rho(gv, a, b, clusters)
    print(f"  {lab:36s} Δρ={delta:+.3f}  CI[{ci[0]:+.3f},{ci[1]:+.3f}]  P(Δ>0)={p:.3f}")

print("\n=== between/within decomposition ===")
for k in ["hybrid-cam", "hybrid-gt", "ens6", "ck+camrisk-full"]:
    between, within = stats.between_within(gv, ARMS[k], cluster_labels)
    print(f"  {k:16s} between ρ={between:+.3f} | mean within ρ={within:+.3f}")

json.dump(
    {
        "spzglob_bt": SPZBT,
        "hybrid_cam": {keep[i]: float(hyb_cam[i]) for i in range(len(keep))},
    },
    open(paths.result("hybrid_prereg.json"), "w"),
)
print("\nsaved results/hybrid_prereg.json")
