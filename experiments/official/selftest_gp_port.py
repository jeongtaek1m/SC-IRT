#!/usr/bin/env python3
"""Self-test of experiments/official/gp_port.py (the checks of av_common.selftest: BUDGET, LEAK, DETERMINISM, PREFIX).
    CUDA_VISIBLE_DEVICES=3 python experiments/official/selftest_gp_port.py [method ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from official import gp_port                                  # noqa: E402
from official.av_common import selftest                      # noqa: E402

if __name__ == '__main__':
    for name in (sys.argv[1:] or list(gp_port.METHODS)):
        selftest(gp_port.METHODS[name], name, adaptive=True, allow_repeats=False)
