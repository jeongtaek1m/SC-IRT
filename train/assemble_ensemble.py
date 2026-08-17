#!/usr/bin/env python3
"""Assemble per-run out-of-fold predictions into the released-artifact format.

Takes the six run files written by train_encoder_b2d.py and writes an npz with
the individual runs plus their rank ensembles, matching the layout of
data/interact/interact_b2d_w2a_final.npz.
"""

import argparse

import numpy as np
from scipy.stats import rankdata, spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run npz files (d64 s0..2, d96 s0..2 order)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    zs = [np.load(f, allow_pickle=True) for f in a.runs]
    reference, routes, types = zs[0]["reference"], zs[0]["routes"], zs[0]["types"]
    preds = [z["pred_m"] for z in zs]
    ens = np.mean([rankdata(p) for p in preds], 0)
    out = {"reference": reference, "routes": routes, "types": types, "ens6": ens}
    names = ["d64_s0", "d64_s1", "d64_s2", "d96_s0", "d96_s1", "d96_s2"]
    for n, p in zip(names, preds):
        out[n] = p
    if len(preds) >= 3:
        out["ens_d64"] = np.mean([rankdata(p) for p in preds[:3]], 0)
    for k in ("ens6", "ens_d64"):
        if k in out:
            print(f"{k}: rho {spearmanr(out[k], reference).correlation:+.4f}")
    np.savez(a.out, **out)


if __name__ == "__main__":
    main()
