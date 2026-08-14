#!/usr/bin/env python3
"""Encoder verification: seed stability and rank decomposition.

Scores the frozen out-of-fold encoder predictions against the canonical gold
difficulty: per-run pooled Spearman, the 6-run mean/sd, and the ensembles. Then
decomposes rank agreement into between-type and within-type components — a
descriptor can look strong purely by ordering scenario *types* correctly while
being blind to route-level variation within a type.

Baseline comparisons (hand-crafted stacks, rank fusion, adaptive testing) are
experiments of the paper, not of this method release; their code lives on the
`full-reproduction` branch.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scirt import paths, stats  # noqa: E402

d = np.load(f"{paths.INTERACT}/interact_b2d_w2a_final.npz", allow_pickle=True)
routes = [str(r).replace("route_", "") for r in d["routes"]]
types = [str(t) for t in d["types"]]

GOLD = json.load(open(paths.result("gold_anchor.json")))["gold"]

keep = [i for i, r in enumerate(routes) if r in GOLD]
gv = np.array([GOLD[routes[i]] for i in keep])
print(
    f"matched {len(keep)}/{len(routes)} | training-side gold agreement "
    f"ρ={spearmanr(np.array(d['gold'])[keep], gv).correlation:.4f}"
)

arms = ["ens6", "ens_d64"] + [f"d64_s{i}" for i in range(3)] + [f"d96_s{i}" for i in range(3)]
ARMS = {k: np.array(d[k])[keep] for k in arms if k in d.files}

print("\n=== rescored against the canonical 2PL gold ===")
for k in arms:
    if k in ARMS:
        print(f"  {k:10s} ρ={spearmanr(gv, ARMS[k]).correlation:+.4f}")
per_seed = [spearmanr(gv, ARMS[k]).correlation for k in arms[2:] if k in ARMS]
print(f"  individual 6 runs: mean {np.mean(per_seed):+.4f}  std {np.std(per_seed, ddof=1):.4f}")

print("\n=== between/within decomposition ===")
cluster_labels = [types[i] for i in keep]
for k in ["ens6", "ens_d64"]:
    b, w = stats.between_within(gv, ARMS[k], cluster_labels)
    print(f"  {k:8s} between ρ={b:+.3f} | mean within ρ={w:+.3f}")
