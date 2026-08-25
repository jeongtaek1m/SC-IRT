#!/usr/bin/env python3
"""IRT-free baseline: uniform random rollouts + finite-population proportion CI.

Is any IRT structure needed to certify a full-bank success rate? Sample
routes uniformly without replacement, estimate S by the running mean, and
stop when a Wilson interval with finite-population correction has width
<= 2 eps. Same random orders as the Random+IRT arm (paired), same eps
targets as ours. Together with Random+IRT (40.7 / 98.9) and SRVar (29.0 /
69.1) this gives the ladder: no IRT -> IRT posterior -> IRT + aligned
selection.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirt.b2d import Panel
from scirt.splits import unified_split, R_DRAWS
from scirt.metrics import mean_se, coverage_str

EPSS = [0.10, 0.05]
Z = 1.959964
OUT = Path(__file__).resolve().parents[1] / 'results'


def wilson_fpc(k_succ, k, N):
    """Wilson score interval for a proportion, scaled by the finite-population
    correction sqrt((N-k)/(N-1)) — the textbook interval for a sample mean
    of a finite binary population."""
    p = k_succ / k
    fpc = np.sqrt((N - k) / (N - 1)) if k < N else 0.0
    denom = 1 + Z ** 2 / k
    centre = (p + Z ** 2 / (2 * k)) / denom
    half = Z * np.sqrt(p * (1 - p) / k + Z ** 2 / (4 * k ** 2)) / denom * fpc
    return centre - half, centre + half, p


def main():
    panel = Panel()
    RES = {e: {'n': [], 'err': [], 'cov': [], 'hw': []} for e in EPSS}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        calR, _ = panel.split_routes(ht)
        for js in hp:
            bi, yy = panel.bank_rows(calR, js)
            n = len(bi)
            SR = yy.mean()
            order = list(np.random.RandomState(300 + seed * 20 + js).permutation(n))
            done = {}
            for k in range(2, n + 1):
                ys = yy[order[:k]]
                lo, hi, p = wilson_fpc(ys.sum(), k, n)
                est = (ys.sum() + (n - k) * p) / n
                for e in EPSS:
                    if e not in done and (hi - lo <= 2 * e or k == n):
                        done[e] = (k, abs(est - SR), 1.0 if lo <= SR <= hi else 0.0, (hi - lo) / 2)
                if len(done) == len(EPSS):
                    break
            for e in EPSS:
                for key, v in zip(('n', 'err', 'cov', 'hw'), done[e]):
                    RES[e][key].append(v)
    print('===== IRT-free: uniform random + Wilson/FPC stop (48 samples) =====')
    for e in EPSS:
        d = RES[e]
        n_, ns = mean_se(d['n'])
        hw = np.array(d['hw'])
        print(f'  +-{e:.0%}: {n_:6.1f}+-{ns:.1f} rollouts (bank {np.mean(d["n"]) / 180:.0%})  '
              f'MAE {np.mean(d["err"]):.4f}  cov {coverage_str(d["cov"])}  '
              f'half-width {hw.mean():.3f}+-{hw.std(ddof=1):.3f}')
    OUT.mkdir(exist_ok=True)
    json.dump({str(e): {k: list(map(float, RES[e][k])) for k in RES[e]} for e in EPSS},
              open(OUT / 'random_fpc.json', 'w'))


if __name__ == '__main__':
    main()
