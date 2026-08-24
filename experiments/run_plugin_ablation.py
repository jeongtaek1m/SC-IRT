#!/usr/bin/env python3
"""The causal ablation for uncertainty propagation — plug-in vs marginalised.

Same 1PL calibration, same SR-variance acquisition, same stopping rule; the
ONLY difference is whether the item curves propagate the difficulty
posterior SD:

    plug-in       m_i(theta) = sigmoid(theta - b_hat_i)
    marginalised  m_i(theta) = int sigmoid(theta - b) N(b; b_hat_i, s_i^2) db

swept over calibration-panel sizes J_cal in {4, 7, 10, 13} (same subsample
streams as run_scarcity.py, so rows pair with Table 5). If the coverage /
MAE gap widens as J_cal shrinks, calibration uncertainty is not merely
diagnostic — explicitly propagating it is what makes adaptive evaluation
survive small panels.

Anchors: the marginalised arm must reproduce the Table 5 ours rows exactly
(J13 29.0/.0463/48-48; J4 25.5/.0630/38-48).
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
from scirt.curves import marginal_curves, sig, THG, PRIOR
from scirt.bayes import post_from, sr_ci
from scirt.acquisition import srvar_pick
from scirt.metrics import mean_se, coverage_str, paired_seed_boot

np.random.seed(0)
torch.manual_seed(0)
EPS = 0.10
JCALS = (4, 7, 10, 13)
OUT = Path(__file__).resolve().parents[1] / 'results'


def cat10(M, yy, rng_draw):
    """SRVar + the +-10% stop; identical loop for both arms."""
    n = M.shape[1]
    SR = yy.mean()
    S, q = [], PRIOR.copy()
    for _ in range(min(120, n)):
        rem = [i for i in range(n) if i not in S]
        S.append(srvar_pick(M, q, rem))
        q = post_from(M, yy, S)
        lo, hi, m = sr_ci(M, yy, S, q, rng_draw)
        if hi - lo <= 2 * EPS or len(S) >= min(120, n):
            return len(S), abs(m - SR), 1.0 if lo <= SR <= hi else 0.0


def main():
    panel = Panel()
    ARMS = ('plug-in', 'marginalised')
    RES = {m: {k: {'n': [], 'err': [], 'cov': []} for k in JCALS} for m in ARMS}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        for Jc in JCALS:
            if Jc == 13:
                cs = cols
            else:
                rs = np.random.RandomState(9000 + seed * 100 + Jc * 10 + 0)
                cs = sorted(np.array(cols)[rs.choice(len(cols), Jc, replace=False)].tolist())
            f1 = calibrate(panel.Y, calR, cs, mode='1pl')
            for js in hp:
                bi, yy = panel.bank_rows(calR, js)
                b1, s1 = f1['b'][bi], f1['s'][bi]
                M_plug = sig(THG[:, None] - b1[None, :])
                M_marg = marginal_curves(b1, s1)
                for m, M, rs_draw in (('plug-in', M_plug, 19),
                                      ('marginalised', M_marg, 13)):
                    n_, err, cov = cat10(M, yy, np.random.RandomState(rs_draw))
                    RES[m][Jc]['n'].append(n_)
                    RES[m][Jc]['err'].append(err)
                    RES[m][Jc]['cov'].append(cov)
        print(f'seed {seed} done', flush=True)

    print('\n===== plug-in vs marginalised (1PL + SRVar + same stop, +-10%) =====')
    print(f'{"J_cal":>5s} {"arm":>13s} {"Rollouts":>12s} {"SR-MAE":>8s} {"Coverage":>14s}')
    for Jc in JCALS:
        for m in ARMS:
            d = RES[m][Jc]
            n_, ns = mean_se(d['n'])
            print(f'{Jc:5d} {m:>13s} {n_:7.1f}+-{ns:<4.1f} {np.mean(d["err"]):8.4f} '
                  f'{coverage_str(d["cov"]):>14s}')
    print('\npaired deltas (marginalised - plug-in), seed-cluster bootstrap:')
    for Jc in JCALS:
        dm, lo, hi = paired_seed_boot(RES['marginalised'][Jc]['err'], RES['plug-in'][Jc]['err'])
        dc, lc, hc = paired_seed_boot(RES['marginalised'][Jc]['cov'], RES['plug-in'][Jc]['cov'])
        dn, ln, hn = paired_seed_boot(RES['marginalised'][Jc]['n'], RES['plug-in'][Jc]['n'])
        print(f'  J{Jc:2d}: dMAE {dm:+.4f} [{lo:+.4f},{hi:+.4f}]  '
              f'dcov {dc:+.3f} [{lc:+.3f},{hc:+.3f}]  droll {dn:+.1f} [{ln:+.1f},{hn:+.1f}]')

    OUT.mkdir(exist_ok=True)
    json.dump({m: {str(k): {kk: list(map(float, RES[m][k][kk])) for kk in RES[m][k]}
                   for k in JCALS} for m in RES},
              open(OUT / 'plugin_ablation.json', 'w'))

    mg = RES['marginalised']
    assert abs(np.mean(mg[13]['n']) - 29.0) < 0.2
    assert abs(np.mean(mg[13]['err']) - 0.0463) < 0.002
    assert int(sum(mg[13]['cov'])) == 48
    assert abs(np.mean(mg[4]['err']) - 0.0630) < 0.004
    print('anchors OK')


if __name__ == '__main__':
    main()
