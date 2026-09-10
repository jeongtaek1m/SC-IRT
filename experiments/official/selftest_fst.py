#!/usr/bin/env python3
"""Self-test of experiments/official/fst.py (the checks of av_common.selftest: BUDGET, LEAK, DETERMINISM).
    CUDA_VISIBLE_DEVICES=3 python experiments/official/selftest_fst.py [method ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from official import fst                                  # noqa: E402
from official.av_common import selftest                      # noqa: E402

if __name__ == '__main__':
    for name in (sys.argv[1:] or list(fst.METHODS)):
        selftest(fst.METHODS[name], name, adaptive=False, allow_repeats=False)
