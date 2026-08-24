#!/usr/bin/env python3
"""The 2x2 factorial — {theta-EIG, SRVar} acquisition x {theta-SE, SR-CI} stop.

Every cell shares the same 1PL calibration, the same marginalised curves and
the same grid posterior; only the acquisition rule and the stopping quantity
change. Because the selection trajectory does not depend on the stopping
rule, one trajectory per acquisition serves every stopping criterion: we
record the first crossing of each theta-SE threshold tau in
{0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2} and of each SR half-width target
eps in {0.10, 0.05}, measuring at each stop the rollouts, SR-MAE, the
95% SR-CI coverage, and the delivered SR half-width (mean +- SD — a
theta-scale stop delivers heterogeneous SR precision; the SR-CI stop
delivers the contract).

Cells: A = EIG + theta-SE, B = EIG + SR-CI, C = SRVar + theta-SE,
D = SRVar + SR-CI (ours). B and D must reproduce Table 3 exactly (anchors).
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
from scirt.curves import marginal_curves, PRIOR
from scirt.bayes import post_from, sr_ci, theta_sd
from scirt.acquisition import srvar_pick, eig_pick
from scirt.metrics import mean_se, coverage_str

np.random.seed(0)
torch.manual_seed(0)
TAUS = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2]
EPSS = [0.10, 0.05]
OUT = Path(__file__).resolve().parents[1] / 'results'


def trajectory(Mf, yy, pick, rng_draw, max_steps=120):
    """One selection trajectory; record every tau and eps first-crossing."""
    n = Mf.shape[1]
    SR = yy.mean()
    S, q = [], PRIOR.copy()
    tau_rec, eps_rec = {}, {}
    for _ in range(min(max_steps, n)):
        rem = [i for i in range(n) if i not in S]
        S.append(pick(q, rem))
        q = post_from(Mf, yy, S)
        lo, hi, m = sr_ci(Mf, yy, S, q, rng_draw)
        sd = theta_sd(q)
        exhausted = len(S) >= min(max_steps, n)
        rec = (len(S), abs(m - SR), 1.0 if lo <= SR <= hi else 0.0, (hi - lo) / 2)
        for t in TAUS:
            if t not in tau_rec and (sd <= t or exhausted):
                tau_rec[t] = rec
        for e in EPSS:
            if e not in eps_rec and (hi - lo <= 2 * e or exhausted):
                eps_rec[e] = rec
        if len(tau_rec) == len(TAUS) and len(eps_rec) == len(EPSS):
            break
    return tau_rec, eps_rec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-steps', type=int, default=120,
                    help='trajectory cap; pass 999 to let the bank itself be '
                         'the only cap (separates truncation from intrinsic '
                         'theta-SE failure at tight tau)')
    args = ap.parse_args()
    panel = Panel()
    ACQ = ('EIG', 'SRVar')
    TAUR = {a: {t: {'n': [], 'err': [], 'cov': [], 'hw': []} for t in TAUS} for a in ACQ}
    EPSR = {a: {e: {'n': [], 'err': [], 'cov': [], 'hw': []} for e in EPSS} for a in ACQ}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        f1 = calibrate(panel.Y, calR, cols, mode='1pl')
        for js in hp:
            bi, yy = panel.bank_rows(calR, js)
            Mf = marginal_curves(f1['b'][bi], f1['s'][bi])

            def pk_eig(q, rem, _M=Mf):
                return eig_pick(q, _M, rem)

            def pk_sv(q, rem, _M=Mf):
                return srvar_pick(_M, q, rem)

            for a, pick, rs in (('EIG', pk_eig, 7), ('SRVar', pk_sv, 13)):
                tr, er = trajectory(Mf, yy, pick, np.random.RandomState(rs),
                                    max_steps=args.max_steps)
                for t in TAUS:
                    for k, v in zip(('n', 'err', 'cov', 'hw'), tr[t]):
                        TAUR[a][t][k].append(v)
                for e in EPSS:
                    for k, v in zip(('n', 'err', 'cov', 'hw'), er[e]):
                        EPSR[a][e][k].append(v)
        print(f'seed {seed} done', flush=True)

    print('\n===== 2x2 factorial (same 1PL + marginalisation + posterior) =====')
    print(f'{"acq":>6s} {"stop":>10s} {"Rollouts":>12s} {"SR-MAE":>8s} '
          f'{"Coverage":>14s} {"SR half-width":>16s}')
    for a in ACQ:
        for t in TAUS:
            d = TAUR[a][t]
            n_, ns = mean_se(d['n'])
            hw = np.array(d['hw'])
            print(f'{a:>6s} {"SE<=" + str(t):>10s} {n_:7.1f}+-{ns:<4.1f} '
                  f'{np.mean(d["err"]):8.4f} {coverage_str(d["cov"]):>14s} '
                  f'{hw.mean():8.3f}+-{hw.std(ddof=1):.3f}')
        for e in EPSS:
            d = EPSR[a][e]
            n_, ns = mean_se(d['n'])
            hw = np.array(d['hw'])
            print(f'{a:>6s} {"SR+-" + f"{e:.0%}":>10s} {n_:7.1f}+-{ns:<4.1f} '
                  f'{np.mean(d["err"]):8.4f} {coverage_str(d["cov"]):>14s} '
                  f'{hw.mean():8.3f}+-{hw.std(ddof=1):.3f}')

    OUT.mkdir(exist_ok=True)
    json.dump({'tau': {a: {str(t): {k: list(map(float, TAUR[a][t][k])) for k in TAUR[a][t]}
                           for t in TAUS} for a in ACQ},
               'eps': {a: {str(e): {k: list(map(float, EPSR[a][e][k])) for k in EPSR[a][e]}
                           for e in EPSS} for a in ACQ}},
              open(OUT / ('factorial_2x2.json' if args.max_steps == 120
                          else f'factorial_2x2_cap{args.max_steps}.json'), 'w'))

    assert abs(np.mean(EPSR['SRVar'][0.10]['n']) - 29.0) < 0.2   # cell D
    assert abs(np.mean(EPSR['EIG'][0.10]['n']) - 28.7) < 0.2     # cell B
    assert abs(np.mean(EPSR['SRVar'][0.10]['err']) - 0.0463) < 0.002
    assert int(sum(EPSR['SRVar'][0.10]['cov'])) == 48
    print('anchors OK')


if __name__ == '__main__':
    main()
