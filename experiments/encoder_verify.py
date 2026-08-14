#!/usr/bin/env python3
"""Table IV: encoder ablation and the tie against the hand-crafted stack.

Scores the frozen encoder predictions against the canonical gold difficulty and
asks whether the encoder beats the best hand-crafted feature stack on the 219-route
B2D bank. It does not: the paired interval straddles zero. That tie is the honest
result at this scale, and it is what makes the NavSim scale-up the load-bearing
comparison.

Also decomposes each arm into between-type and within-type rank agreement. A
descriptor can look strong purely by ordering scenario *types* correctly while
being blind to route-level variation within a type; separating the two makes that
visible.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scirt import paths, resample, stats  # noqa: E402

d = np.load(f"{paths.INTERACT}/interact_b2d_w2a_final.npz", allow_pickle=True)
routes = [str(r).replace("route_", "") for r in d["routes"]]
types = [str(t) for t in d["types"]]

anchor = json.load(open(paths.result("gold_anchor.json")))
GOLD, BT = anchor["gold"], anchor["bt"]

keep = [i for i, r in enumerate(routes) if r in GOLD]
gv = np.array([GOLD[routes[i]] for i in keep])
print(
    f"matched {len(keep)}/{len(routes)} | gold agreement "
    f"ρ={spearmanr(np.array(d['gold'])[keep], gv).correlation:.4f}"
)

arms = (
    ["ens6", "ens_d64"]
    + [f"d64_s{i}" for i in range(3)]
    + [f"d96_s{i}" for i in range(3)]
)
ARMS = {k: np.array(d[k])[keep] for k in arms if k in d.files}

print("\n=== 1) rescored against the canonical 2PL gold ===")
for k in arms:
    if k in ARMS:
        print(f"  {k:10s} ρ={spearmanr(gv, ARMS[k]).correlation:+.4f}")
per_seed = [spearmanr(gv, ARMS[k]).correlation for k in arms[2:] if k in ARMS]
print(f"  individual 6 seeds: mean {np.mean(per_seed):+.4f}  std {np.std(per_seed, ddof=1):.4f}")

BASELINES = {k: np.array([BT[k][routes[i]] for i in keep]) for k in BT}

cluster_labels = [types[i] for i in keep]
unique_types = sorted(set(cluster_labels))
clusters = [
    np.array([j for j, x in enumerate(cluster_labels) if x == t]) for t in unique_types
]

print(f"\n=== 2) {len(unique_types)}-type cluster paired bootstrap, 10k ===")
for arm, ref in [
    ("ens6", "GT:ck+gtrisk"),
    ("ens6", "ck"),
    ("ens_d64", "GT:ck+gtrisk"),
    ("ens6", "ck+camrisk-full"),
]:
    delta = (
        spearmanr(gv, ARMS[arm]).correlation
        - spearmanr(gv, BASELINES[ref]).correlation
    )
    ci, p = resample.paired_delta_rho(gv, ARMS[arm], BASELINES[ref], clusters)
    print(
        f"  {arm:8s} − {ref:16s}: Δρ={delta:+.3f}  "
        f"CI[{ci[0]:+.3f},{ci[1]:+.3f}]  P(Δ>0)={p:.3f}"
    )

print("\n=== 3) between/within decomposition ===")
for k in ["ens6", "ens_d64"]:
    between, within = stats.between_within(gv, ARMS[k], cluster_labels)
    print(f"  {k:8s} between ρ={between:+.3f} | mean within ρ={within:+.3f}")
