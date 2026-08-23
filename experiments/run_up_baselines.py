#!/usr/bin/env python3
"""Table 4 (a)(b)(c) — published-baseline comparison on the unified split.

(a) Adaptive selection-rule comparison: the stopping / scoring / posterior
    machine (all-1PL, marginalised, SR +-eps quantile stop) is held fixed and
    only the next-item rule is swapped — Random order, metabench-greedy
    order, Fluid-style 2PL Fisher, ATLAS-style 3PL top-5 randomesque, and
    ours-EIG. (ours-SRVar comes from run_up_main.py on identical samples.)
(b) Fixed budgets B in {29, 69} (matched to ours' realised rollouts), each
    baseline run as published: its own item model and point-parameter p-IRT
    scoring (tinyBenchmarks K-means anchors, metabench-lite info-grid,
    Fluid Fisher, ATLAS 3PL randomesque, Random / type-stratified Random).
(c) Random-100 reference for the ATLAS-style IES panel.

Anchors: (a) Random 40.7 / Fluid 27.3 / metabench-greedy 38.9;
(b) Random@29 .0584, Fluid@29 .0375, metabench@69 .0285; (c) Random-100 .0217.
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
from scirt.curves import marginal_curves, sig, PRIOR
from scirt.bayes import post_from, sr_ci
from scirt.acquisition import (srvar_pick, eig_pick, fisher_2pl_pick, atlas_3pl_pick,
                               kmeans_anchors, metabench_order, stratified_order)
from scirt.metrics import mean_se, coverage_str

np.random.seed(0)
torch.manual_seed(0)
EPSS = [0.10, 0.05]
BFIX = [29, 69]
OUT = Path(__file__).resolve().parents[1] / 'results'


# --- as-published p-IRT scoring helpers (point parameters) ------------------
def theta_map(bs, ys, aa=None, cc=None, it=50):
    aa = np.ones(len(bs)) if aa is None else aa
    t = 0.0
    for _ in range(it):
        base = sig(aa * (t - bs))
        p = base if cc is None else cc + (1 - cc) * base
        w = aa * (base * (1 - base)) / np.maximum(p * (1 - p), 1e-6) if cc is not None else aa
        g = (w * (ys - p)).sum() - t
        hh = -((aa * w) * base * (1 - base)).sum() - 1.0
        t -= g / hh
    return float(np.clip(t, -6, 6))


def pirt(t, bs, aa, ys_seen, seen, n):
    un = [i for i in range(n) if i not in seen]
    return (ys_seen.sum() + sig(aa[un] * (t - bs[un])).sum()) / n


def cat_common_stop(Mf, yy, pick, rng_draw):
    """The common stopping machine of panel (a)."""
    n = Mf.shape[1]
    SR = yy.mean()
    S, q, out = [], PRIOR.copy(), {}
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


def main():
    panel = Panel()
    FIXED = ('Random', 'Random-strat', 'tinyBenchmarks', 'metabench-lite',
             'Fluid(2PL-Fisher)', 'ATLAS-lite(3PL)', 'ours(EIG,1PL)')
    ADAPT = ('Random+CI', 'metabench-greedy', 'Fluid-Fisher(2PL)',
             'ATLAS-lite(3PL)', 'ours(EIG,1PL)')
    UPFIX = {m: {B: [] for B in BFIX} for m in FIXED}
    UPAD = {m: {e: {'n': [], 'err': [], 'cov': []} for e in EPSS} for m in ADAPT}
    R100 = {29: [], 100: []}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        f1 = calibrate(panel.Y, calR, cols, mode='1pl')
        f2 = calibrate(panel.Y, calR, cols, mode='2pl')
        f3 = calibrate(panel.Y, calR, cols, mode='3pl')
        typ = np.array([panel.sn[r] for r in calR])
        for js in hp:
            bi, yy = panel.bank_rows(calR, js)
            n = len(bi)
            SR = yy.mean()
            b1, s1 = f1['b'][bi], f1['s'][bi]
            a2, b2 = f2['a'][bi], f2['b'][bi]
            a3, b3, c3 = f3['a'][bi], f3['b'][bi], f3['cc'][bi]
            ty = typ[bi]
            Mf = marginal_curves(b1, s1)

            # ---- (b) fixed budgets, as published --------------------------
            for B in BFIX:
                rng = np.random.RandomState(100 + seed * 20 + js)
                sel = {}
                sel['Random'] = list(rng.permutation(n)[:B])
                sel['Random-strat'] = stratified_order(ty, rng)[:B]
                sel['tinyBenchmarks'] = kmeans_anchors(a2, b2, B, n)
                sel['metabench-lite'] = metabench_order(a2, b2, B, n)
                S, t0 = [], 0.0
                for _ in range(B):
                    rem = [i for i in range(n) if i not in S]
                    p = sig(a2[rem] * (t0 - b2[rem]))
                    S.append(rem[int(np.argmax((a2[rem] ** 2) * p * (1 - p)))])
                    t0 = theta_map(b2[np.array(S)], yy[np.array(S)], a2[np.array(S)])
                sel['Fluid(2PL-Fisher)'] = S
                S, rngA, tA = [], np.random.RandomState(1), 0.0
                for _ in range(B):
                    rem = [i for i in range(n) if i not in S]
                    base = sig(a3[rem] * (tA - b3[rem]))
                    p = c3[rem] + (1 - c3[rem]) * base
                    info = (a3[rem] ** 2) * ((1 - c3[rem]) ** 2) * (base * (1 - base)) ** 2 \
                        / np.maximum(p * (1 - p), 1e-6)
                    top = np.argsort(-info)[:5]
                    S.append(rem[int(top[rngA.randint(len(top))])])
                    tA = theta_map(b3[np.array(S)], yy[np.array(S)], a3[np.array(S)], c3[np.array(S)])
                sel['ATLAS-lite(3PL)'] = S
                q, S = PRIOR.copy(), []
                for _ in range(B):
                    rem = [i for i in range(n) if i not in S]
                    S.append(eig_pick(q, Mf, rem))
                    q = post_from(Mf, yy, S)
                sel['ours(EIG,1PL)'] = S
                for mname, S_ in sel.items():
                    S_ = list(dict.fromkeys(S_))[:B]
                    if mname == 'ours(EIG,1PL)':
                        un = [i for i in range(n) if i not in S_]
                        est = (yy[S_].sum() + (Mf * q[:, None]).sum(0)[un].sum()) / n if un \
                            else yy[S_].sum() / n
                        UPFIX[mname][B].append(abs(est - SR))
                        continue
                    aa = a3 if '3PL' in mname else (
                        a2 if ('Fluid' in mname or 'tiny' in mname or 'meta' in mname)
                        else np.ones(n))
                    bb_ = b3 if '3PL' in mname else (
                        b2 if ('Fluid' in mname or 'tiny' in mname or 'meta' in mname) else b1)
                    cc = c3 if '3PL' in mname else None
                    t = theta_map(bb_[np.array(S_)], yy[np.array(S_)], aa[np.array(S_)],
                                  None if cc is None else cc[np.array(S_)])
                    UPFIX[mname][B].append(abs(pirt(t, bb_, aa, yy[np.array(S_)], S_, n) - SR))

            # ---- (a) adaptive, common stopping machine --------------------
            rngr = np.random.RandomState(300 + seed * 20 + js)
            order_rand = list(rngr.permutation(n))
            mb = metabench_order(a2, b2, min(120, n), n)
            rngA = np.random.RandomState(1)

            def pk_rand(q, rem, _o=order_rand):
                return [i for i in _o if i in rem][0]

            def pk_meta(q, rem, _o=mb):
                return [i for i in _o if i in rem][0]

            def pk_fish(q, rem, _a=a2, _b=b2):
                return fisher_2pl_pick(q, _a, _b, rem)

            def pk_atlas(q, rem, _a=a3, _b=b3, _c=c3, _r=rngA):
                return atlas_3pl_pick(q, _a, _b, _c, rem, _r)

            def pk_eig(q, rem, _M=Mf):
                return eig_pick(q, _M, rem)

            for m, pick, rd in (('Random+CI', pk_rand, rngr),
                                ('metabench-greedy', pk_meta, np.random.RandomState(11)),
                                ('Fluid-Fisher(2PL)', pk_fish, np.random.RandomState(11)),
                                ('ATLAS-lite(3PL)', pk_atlas, np.random.RandomState(11)),
                                ('ours(EIG,1PL)', pk_eig, np.random.RandomState(7))):
                o = cat_common_stop(Mf, yy, pick, rd)
                for e in EPSS:
                    UPAD[m][e]['n'].append(o[e][0])
                    UPAD[m][e]['err'].append(o[e][1])
                    UPAD[m][e]['cov'].append(o[e][2])

            # ---- (c) Random fixed 29 / 100 (IES reference) ----------------
            rng = np.random.RandomState(100 + seed * 20 + js)
            perm = list(rng.permutation(n))
            for B in (29, 100):
                S_ = perm[:B]
                t = theta_map(b1[np.array(S_)], yy[np.array(S_)])
                R100[B].append(abs(pirt(t, b1, np.ones(n), yy[np.array(S_)], S_, n) - SR))
        print(f'seed {seed} done', flush=True)

    print('\n===== Table 4(b) — fixed budgets, as published (SR-MAE, 48 samples) =====')
    print(f'{"method":22s} {"B=29":>16s} {"B=69":>16s}')
    for m in FIXED:
        cells = []
        for B in BFIX:
            e_, se = mean_se(UPFIX[m][B])
            cells.append(f'{e_:.4f}+-{se:.4f}')
        print(f'{m:22s} {cells[0]:>16s} {cells[1]:>16s}')
    print('\n===== Table 4(a) — adaptive, common stopping machine =====')
    for e in EPSS:
        print(f'--- +-{e:.0%} ---')
        for m in ADAPT:
            d = UPAD[m][e]
            n_, ns = mean_se(d['n'])
            print(f'  {m:20s} {n_:5.1f}+-{ns:.1f} rollouts  '
                  f'MAE {np.mean(d["err"]):.4f}  cov {coverage_str(d["cov"])}')
    m100, se100 = mean_se(R100[100])
    print(f'\nRandom-100 reference (IES): MAE {m100:.4f}+-{se100:.4f}')

    OUT.mkdir(exist_ok=True)
    json.dump({'upfix': {m: {str(B): list(map(float, UPFIX[m][B])) for B in BFIX} for m in UPFIX},
               'upad': {m: {str(e): {k: list(map(float, UPAD[m][e][k])) for k in UPAD[m][e]}
                            for e in EPSS} for m in UPAD},
               'r100': list(map(float, R100[100]))},
              open(OUT / 'up_baselines.json', 'w'))

    assert abs(np.mean(UPAD['Random+CI'][0.10]['n']) - 40.7) < 0.2
    assert abs(np.mean(UPAD['Fluid-Fisher(2PL)'][0.10]['n']) - 27.3) < 0.2
    assert abs(np.mean(UPAD['metabench-greedy'][0.10]['n']) - 38.9) < 0.2
    assert abs(np.mean(UPFIX['Random'][29]) - 0.0584) < 0.002
    assert abs(np.mean(UPFIX['Fluid(2PL-Fisher)'][29]) - 0.0375) < 0.002
    assert abs(np.mean(UPFIX['metabench-lite'][69]) - 0.0285) < 0.002
    assert abs(m100 - 0.0217) < 0.001
    print('anchors OK')


if __name__ == '__main__':
    main()
