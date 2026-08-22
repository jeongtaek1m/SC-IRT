#!/usr/bin/env python3
"""Bundle per-run out-of-fold predictions into the released-artifact format.

Takes the six run files written by train_encoder_b2d.py and writes an npz with
the individual runs. No ensemble keys: prediction averaging is banned.
"""

import argparse

import numpy as np
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run npz files (d64 s0..2, d96 s0..2 order)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    zs = [np.load(f, allow_pickle=True) for f in a.runs]
    reference, routes, types = zs[0]["reference"], zs[0]["routes"], zs[0]["types"]
    preds = [z["pred_m"] for z in zs]
    # Runs are bundled individually. No ensembling: prediction averaging is
    # banned project-wide (2026-08-19); report per-run metrics and their spread.
    out = {"reference": reference, "routes": routes, "types": types}
    names = ["d64_s0", "d64_s1", "d64_s2", "d96_s0", "d96_s1", "d96_s2"]
    for n, p in zip(names, preds):
        out[n] = p
        print(f"{n}: rho {spearmanr(p, reference).correlation:+.4f}")
    np.savez(a.out, **out)


if __name__ == "__main__":
    main()
