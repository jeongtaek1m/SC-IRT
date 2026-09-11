#!/usr/bin/env python3
"""Self-test of the Sim2Val wrapper (official control-variates estimator, random i.i.d. route draws):
BUDGET / LEAK / DETERMINISM on two protocol cells.  python experiments/official/selftest_sim2val.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from official.av_common import selftest          # noqa: E402
from official import sim2val                      # noqa: E402

if __name__ == '__main__':
    for name, m in sim2val.METHODS.items():
        selftest(m, name, adaptive=False)
