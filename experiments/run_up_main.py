#!/usr/bin/env python3
"""Table 3 — UP main operating points (all-1PL + SR-variance acquisition).

Certify a held-out planner's full-bank success rate to +-eps on the
response-calibrated bank (approx. 180 routes, 13 calibration planners), with
the acquisition and the stopping rule optimising the same quantity.

Arms: Random+CI / ours-EIG (acquisition ablation) / ours-SRVar (canonical),
plus SRVar at fixed budgets B in {29, 69} for the equal-cost panels.

Anchors (paper Table 3 / Table 4): SRVar +-10% -> 29.0 rollouts / MAE .0463 /
coverage 48/48; +-5% -> 69.1 / .0294 / 40/48; Random 40.7 / 98.9; EIG 28.7 /
69.0; SRVar fixed .0443 / .0297.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirt.b2d import Panel
from scirt.splits import unified_split, R_DRAWS
from scirt.calibration import calibrate
from scirt.curves import marginal_curves
from scirt.bayes import post_from, sr_ci
from scirt.acquisition import srvar_pick, eig_pick
from scirt.metrics import mean_se, coverage_str

np.random.seed(0)
torch.manual_seed(0)
EPSS = [0.10, 0.05]
BFIX = [29, 69]
OUT = Path(__file__).resolve().parents[1] / 'results'


def cat_adaptive(Mf, yy, pick, rng_draw):
    """SR +-eps certification loop (posterior-predictive quantile stopping)."""
    n = Mf.shape[1]
    SR = yy.mean()
    S, q, out = [], None, {}
    from scirt.curves import PRIOR
    q = PRIOR.copy()
    for _ in range(min(120, n)):
        rem = [i for i in range(n) if i not in S]
        S.append(pick(q, rem))
        q = post_from(Mf, yy, S)
        lo, hi, m = sr_ci(Mf, yy, S, q, rng_draw)
        for e in EPSS:
            if e not in out and (hi - lo <= 2 * e or len(S) >= min(120, n)):
                out[e] = (len(S), abs(m - SR), 1.0 if lo <= SR <= hi else 0.0)
        if len(out) == len(EPSS):
            break
    return out


def cat_fixed(Mf, yy, pick):
    """Fixed-budget arm: same selection, posterior-mean scoring at B."""
    n = Mf.shape[1]
    SR = yy.mean()
    from scirt.curves import PRIOR
    from scirt.bayes import posterior_mean_sr
    S, q, out = [], PRIOR.copy(), {}
    for _ in range(max(BFIX)):
        rem = [i for i in range(n) if i not in S]
        S.append(pick(q, rem))
        q = post_from(Mf, yy, S)
        if len(S) in BFIX:
            out[len(S)] = abs(posterior_mean_sr(Mf, yy, S, q) - SR)
        if len(out) == len(BFIX):
            break
    return out


def main():
    panel = Panel()
    ARMS = ('Random+CI', 'ours-EIG', 'ours-SRVar')
    RES = {m: {e: {'n': [], 'err': [], 'cov': []} for e in EPSS} for m in ARMS}
    FIX = {B: [] for B in BFIX}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        f1 = calibrate(panel.Y, calR, cols, mode='1pl')
        for js in hp:
            bi, yy = panel.bank_rows(calR, js)
            Mf = marginal_curves(f1['b'][bi], f1['s'][bi])
            rngr = np.random.RandomState(300 + seed * 20 + js)
            order = list(rngr.permutation(len(bi)))

            def pk_rand(q, rem, _o=order):
                return [i for i in _o if i in rem][0]

            def pk_eig(q, rem, _M=Mf):
                return eig_pick(q, _M, rem)

            def pk_sv(q, rem, _M=Mf):
                return srvar_pick(_M, q, rem)

            for m, pick, rd in (('Random+CI', pk_rand, rngr),
                                ('ours-EIG', pk_eig, np.random.RandomState(7)),
                                ('ours-SRVar', pk_sv, np.random.RandomState(13))):
                o = cat_adaptive(Mf, yy, pick, rd)
                for e in EPSS:
                    RES[m][e]['n'].append(o[e][0])
                    RES[m][e]['err'].append(o[e][1])
                    RES[m][e]['cov'].append(o[e][2])
            of = cat_fixed(Mf, yy, pk_sv)
            for B in BFIX:
                FIX[B].append(of[B])
        print(f'seed {seed} done', flush=True)

    print('\n===== Table 3 — UP main (unified split, 48 samples) =====')
    for e in EPSS:
        print(f'--- +-{e:.0%} ---')
        for m in ARMS:
            d = RES[m][e]
            n_, ns = mean_se(d['n'])
            print(f'  {m:12s} {n_:5.1f}+-{ns:.1f} rollouts  '
                  f'MAE {np.mean(d["err"]):.4f}  cov {coverage_str(d["cov"])}')
    for B in BFIX:
        e_, se = mean_se(FIX[B])
        print(f'SRVar fixed B={B}: SR-MAE {e_:.4f}+-{se:.4f}')

    OUT.mkdir(exist_ok=True)
    json.dump({'ad': {m: {str(e): {k: list(map(float, RES[m][e][k])) for k in RES[m][e]}
                          for e in EPSS} for m in RES},
               'fix': {str(B): list(map(float, FIX[B])) for B in BFIX}},
              open(OUT / 'up_main.json', 'w'))

    sv, rd = RES['ours-SRVar'], RES['Random+CI']
    assert abs(np.mean(sv[0.10]['n']) - 29.0) < 0.2, np.mean(sv[0.10]['n'])
    assert abs(np.mean(sv[0.05]['n']) - 69.1) < 0.3, np.mean(sv[0.05]['n'])
    assert abs(np.mean(sv[0.10]['err']) - 0.0463) < 0.002
    assert int(sum(sv[0.10]['cov'])) == 48
    assert abs(np.mean(rd[0.10]['n']) - 40.7) < 0.2
    assert abs(np.mean(RES['ours-EIG'][0.10]['n']) - 28.7) < 0.2
    assert abs(np.mean(FIX[29]) - 0.0443) < 0.002
    print('anchors OK')


if __name__ == '__main__':
    main()
