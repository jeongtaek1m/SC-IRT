#!/usr/bin/env python3
"""Reslice baseline_feats.npz down to the columns the released code actually uses.

The original archive carries four blocks; only `kin` (10d ego kinematics) and
`den` (8d agent density) feed a reported row. `dino` (1024d frozen features) fed
an exploratory descriptor that is not in any table, and `cmd` was never read.
Dropping them takes the file from ~840 KB to ~14 KB with no effect on any number.

Verifies value equality on the surviving columns before writing.

usage: python tools/reslice_baseline_feats.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scirt import paths  # noqa: E402

SRC = f"{paths.FEATURES}/baseline_feats.npz"
DST = f"{paths.FEATURES}/baseline_kin_den.npz"

src = np.load(SRC, allow_pickle=True)
np.savez_compressed(
    DST,
    kin_names=src["kin_names"],
    kin=src["kin"],
    den_names=src["den_names"],
    den=src["den"],
)

check = np.load(DST, allow_pickle=True)
for key in ("kin_names", "kin", "den_names", "den"):
    assert np.array_equal(np.asarray(src[key]), np.asarray(check[key])), key

print(
    f"{os.path.getsize(SRC):,} B -> {os.path.getsize(DST):,} B "
    f"(kin {src['kin'].shape}, den {src['den'].shape}); values verified identical"
)
