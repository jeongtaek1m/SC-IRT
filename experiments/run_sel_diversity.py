#!/usr/bin/env python3
"""Adaptivity diagnostic — do different planners get different subsets?

Within one draw (one bank) the three held-out planners run the same
acquisition; pairwise Jaccard of their first-30 selections measures how
planner-specific the selection is (chance for two 30-subsets of ~180 is
about 0.09; a static anchor method would sit at 1.0). Overlap should be
governed by ability proximity — the definitional signature of adaptive
testing.

Anchors: SRVar Jaccard(S30) 0.141; Spearman(|theta gap|, Jaccard) -0.771.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirt.b2d import Panel
from scirt.splits import unified_split, R_DRAWS
from scirt.calibration import calibrate
from scirt.curves import marginal_curves, point_curves_2pl, THG, PRIOR
from scirt.bayes import post_from, sr_ci
from scirt.acquisition import srvar_pick, eig_pick, fisher_2pl_pick

np.random.seed(0)
torch.manual_seed(0)
OUT = Path(__file__).resolve().parents[1] / 'results'


def run_sel(pick, M, yy, rngseed):
    n = M.shape[1]
    S, q, stop10 = [], PRIOR.copy(), None
    rng = np.random.RandomState(rngseed)
    for _ in range(min(120, n)):
        rem = [i for i in range(n) if i not in S]
        S.append(pick(q, rem))
        q = post_from(M, yy, S)
        if stop10 is None:
            lo, hi, _ = sr_ci(M, yy, S, q, rng)
            if hi - lo <= 0.2 or len(S) >= min(120, n):
                stop10 = len(S)
        if stop10 is not None and len(S) >= 30:
            break
    return set(S[:30]), set(S[:stop10]), float((THG * q).sum()), stop10


def main():
    panel = Panel()
    JC = {'SRVar': [], 'EIG': [], 'Fluid': []}
    JS = {'SRVar': [], 'EIG': [], 'Fluid': []}
    TH, NS = [], []
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        f1 = calibrate(panel.Y, calR, cols, mode='1pl')
        f2 = calibrate(panel.Y, calR, cols, mode='2pl')
        out = {m: {} for m in JC}
        ths = {}
        for js in hp:
            bi, yy = panel.bank_rows(calR, js)
            Mf = marginal_curves(f1['b'][bi], f1['s'][bi])
            a2, b2 = f2['a'][bi], f2['b'][bi]
            M2 = point_curves_2pl(a2, b2)

            def pk_sv(q, rem, _M=Mf):
                return srvar_pick(_M, q, rem)

            def pk_eig(q, rem, _M=Mf):
                return eig_pick(q, _M, rem)

            def pk_fl(q, rem, _a=a2, _b=b2):
                return fisher_2pl_pick(q, _a, _b, rem)

            out['SRVar'][js] = run_sel(pk_sv, Mf, yy, 13)
            out['EIG'][js] = run_sel(pk_eig, Mf, yy, 7)
            out['Fluid'][js] = run_sel(pk_fl, M2, yy, 11)
            ths[js] = out['SRVar'][js][2]
            NS.append(out['SRVar'][js][3])
        for m in JC:
            for j1, j2 in combinations(hp, 2):
                A30, As, _, _ = out[m][j1]
                B30, Bs, _, _ = out[m][j2]
                JC[m].append(len(A30 & B30) / len(A30 | B30))
                JS[m].append(len(As & Bs) / max(len(As | Bs), 1))
                if m == 'SRVar':
                    TH.append(abs(ths[j1] - ths[j2]))
        print(f'seed {seed} done', flush=True)

    print('\n===== cross-planner selection overlap, same bank (48 pairs) =====')
    for m in JC:
        print(f'  {m:6s} Jaccard(S30) {np.mean(JC[m]):.3f}   Jaccard(S_stop) {np.mean(JS[m]):.3f}')
    print('  chance for two 30-subsets of ~180: ~0.09 / identical: 1.00')
    r, p = st.spearmanr(TH, JC['SRVar'])
    print(f'  SRVar: |theta gap| vs Jaccard  Spearman {r:+.3f} (p={p:.3f})')

    OUT.mkdir(exist_ok=True)
    json.dump({'jc': {m: list(map(float, JC[m])) for m in JC},
               'js': {m: list(map(float, JS[m])) for m in JC},
               'dth': list(map(float, TH))},
              open(OUT / 'sel_diversity.json', 'w'))

    assert abs(np.mean(NS) - 29.0) < 0.2
    assert abs(np.mean(JC['SRVar']) - 0.141) < 0.01
    assert r < -0.7
    print('anchors OK')


if __name__ == '__main__':
    main()
