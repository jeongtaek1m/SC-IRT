#!/usr/bin/env python3
"""Table 5 — CAT under calibration scarcity (J_cal sweep, +-10% target).

Subsample k in {4, 7, 10, 13} of the 13 calibration planners and re-calibrate
the bank from scratch with only those responses (1PL for ours, 2PL for the
Fluid-style arm, 3PL for the ATLAS-style arm). Held-out planners are fixed
across J_cal (paired). Each arm runs its native item model end to end —
selection, posterior, stopping and scoring — inside the common SR +-eps
stopping frame; only ours marginalises difficulty uncertainty.

J_cal < 13 runs two subsample replicates: Jaccard(S30) between them is the
selection-stability metric, and the replicate-to-replicate correlation of
log a-hat vs b-hat is the mechanism panel (a^2-weighted acquisition amplifies
a-hat noise; Rasch has no such amplification).

Anchors: J13 SRVar 29.0 / .0463 / 48-48 and Random 40.7; J4 Fluid MAE ~ .0956
coverage ~ 0.56 vs ours .0630 / 0.79.
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
from scirt.curves import marginal_curves, point_curves_2pl, point_curves_3pl, PRIOR
from scirt.bayes import post_from, sr_ci
from scirt.acquisition import srvar_pick, fisher_2pl_pick, atlas_3pl_pick
from scirt.metrics import mean_se, coverage_str

np.random.seed(0)
torch.manual_seed(0)
EPS = 0.10
JCALS = (4, 7, 10, 13)
OUT = Path(__file__).resolve().parents[1] / 'results'


def cat_rule(M, yy, pick, rng_draw):
    """+-10% stop recorded, then selection continues to S30 (for Jaccard)."""
    n = M.shape[1]
    SR = yy.mean()
    S, q, done = [], PRIOR.copy(), None
    for _ in range(min(120, n)):
        rem = [i for i in range(n) if i not in S]
        S.append(pick(q, rem))
        q = post_from(M, yy, S)
        if done is None:
            lo, hi, m = sr_ci(M, yy, S, q, rng_draw)
            if hi - lo <= 2 * EPS or len(S) >= min(120, n):
                done = (len(S), abs(m - SR), 1.0 if lo <= SR <= hi else 0.0)
        if done is not None and len(S) >= 30:
            break
    return done, set(S[:30])


def main():
    panel = Panel()
    ARMS = ('Random(1PL)', 'Fluid-Fisher(2PL)', 'ATLAS-lite(3PL)', 'ours-SRVar(1PL)')
    RES = {m: {k: {'n': [], 'err': [], 'cov': []} for k in JCALS} for m in ARMS}
    JAC = {m: {k: [] for k in JCALS if k < 13} for m in ARMS}
    REL = {'a2': {k: [] for k in JCALS if k < 13}, 'b1': {k: [] for k in JCALS if k < 13}}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        for Jc in JCALS:
            reps = (0,) if Jc == 13 else (0, 1)
            fits, S30 = {}, {m: {} for m in ARMS}
            for rep in reps:
                if Jc == 13:
                    cs = cols
                else:
                    rs = np.random.RandomState(9000 + seed * 100 + Jc * 10 + rep)
                    cs = sorted(np.array(cols)[rs.choice(len(cols), Jc, replace=False)].tolist())
                f1 = calibrate(panel.Y, calR, cs, mode='1pl')
                f2 = calibrate(panel.Y, calR, cs, mode='2pl')
                f3 = calibrate(panel.Y, calR, cs, mode='3pl')
                fits[rep] = (f1, f2, f3)
                for js in hp:
                    bi, yy = panel.bank_rows(calR, js)
                    Mf1 = marginal_curves(f1['b'][bi], f1['s'][bi])
                    a2, b2 = f2['a'][bi], f2['b'][bi]
                    M2 = point_curves_2pl(a2, b2)
                    a3, b3, c3 = f3['a'][bi], f3['b'][bi], f3['cc'][bi]
                    M3 = point_curves_3pl(a3, b3, c3)
                    rngr = np.random.RandomState(300 + seed * 20 + js + 100000 * rep)
                    order = list(rngr.permutation(len(bi)))
                    rngA = np.random.RandomState(1)

                    def pk_rand(q, rem, _o=order):
                        return [i for i in _o if i in rem][0]

                    def pk_sv(q, rem, _M=Mf1):
                        return srvar_pick(_M, q, rem)

                    def pk_fish(q, rem, _a=a2, _b=b2):
                        return fisher_2pl_pick(q, _a, _b, rem)

                    def pk_atlas(q, rem, _a=a3, _b=b3, _c=c3, _r=rngA):
                        return atlas_3pl_pick(q, _a, _b, _c, rem, _r)

                    for m, M, pick, rd in (
                            ('Random(1PL)', Mf1, pk_rand, rngr),
                            ('Fluid-Fisher(2PL)', M2, pk_fish, np.random.RandomState(11)),
                            ('ATLAS-lite(3PL)', M3, pk_atlas, np.random.RandomState(11)),
                            ('ours-SRVar(1PL)', Mf1, pk_sv, np.random.RandomState(13))):
                        done, s30 = cat_rule(M, yy, pick, rd)
                        S30[m][(rep, js)] = s30
                        if rep == 0:
                            RES[m][Jc]['n'].append(done[0])
                            RES[m][Jc]['err'].append(done[1])
                            RES[m][Jc]['cov'].append(done[2])
            if Jc < 13:
                REL['a2'][Jc].append(float(np.corrcoef(
                    np.log(fits[0][1]['a'] + 1e-9), np.log(fits[1][1]['a'] + 1e-9))[0, 1]))
                REL['b1'][Jc].append(float(np.corrcoef(fits[0][0]['b'], fits[1][0]['b'])[0, 1]))
                for m in ARMS:
                    for js in hp:
                        A, B = S30[m][(0, js)], S30[m][(1, js)]
                        JAC[m][Jc].append(len(A & B) / len(A | B))
        print(f'seed {seed} done', flush=True)

    print('\n===== Table 5 — CAT under calibration scarcity (+-10%, rep0, 48 samples) =====')
    for Jc in JCALS:
        print(f'--- J_cal={Jc} ---')
        for m in ARMS:
            d = RES[m][Jc]
            n_, ns = mean_se(d['n'])
            print(f'  {m:20s} {n_:5.1f}+-{ns:.1f} rollouts  '
                  f'MAE {np.mean(d["err"]):.4f}  cov {coverage_str(d["cov"])}')
    print('\n[selection stability — Jaccard(S30) across replicates]')
    for m in ARMS:
        print(f'  {m:20s} ' + '  '.join(f'J{k}: {np.mean(JAC[m][k]):.3f}' for k in JAC[m]))
    print('\n[calibration reliability across replicates]')
    for k in (4, 7, 10):
        print(f'  J_cal={k}: corr(log a-hat)={np.mean(REL["a2"][k]):+.3f}  '
              f'corr(b-hat 1PL)={np.mean(REL["b1"][k]):+.3f}')

    OUT.mkdir(exist_ok=True)
    json.dump({'res': {m: {str(k): {kk: list(map(float, RES[m][k][kk])) for kk in RES[m][k]}
                           for k in JCALS} for m in RES},
               'jac': {m: {str(k): list(map(float, JAC[m][k])) for k in JAC[m]} for m in JAC},
               'rel': {t: {str(k): list(map(float, REL[t][k])) for k in REL[t]} for t in REL}},
              open(OUT / 'cal_scarcity.json', 'w'))

    mo = np.mean(RES['ours-SRVar(1PL)'][13]['n'])
    mr = np.mean(RES['Random(1PL)'][13]['n'])
    assert abs(mo - 29.0) < 0.2, mo
    assert abs(mr - 40.7) < 0.2, mr
    assert abs(np.mean(RES['Fluid-Fisher(2PL)'][4]['err']) - 0.0956) < 0.004
    assert abs(np.mean(RES['ours-SRVar(1PL)'][4]['err']) - 0.0630) < 0.004
    print('anchors OK')


if __name__ == '__main__':
    main()
