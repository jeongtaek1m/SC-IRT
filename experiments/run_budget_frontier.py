#!/usr/bin/env python3
"""Efficiency frontier — every method on one predefined budget grid.

Fixed-budget accuracy is a curve, not a point: all methods are scored at
B in {10, 20, 29, 30, 40, 60, 69, 80, 100, 120} (29/69 kept so the earlier
matched-budget anchors still hold). Static methods build one order per
bank; adaptive fixed-budget arms run one trajectory per planner and are
scored at every prefix. Scoring is each method's native estimator.

Methods (label suffixes mark adaptations from the LLM-benchmark originals):
  Random (IRT-free mean)     plain running mean — no model at all
  Random + IRT               random order, 1PL p-IRT fill
  Random-strat + IRT         scenario-type-stratified order, 1PL p-IRT
  DISCO-adapted              inter-planner disagreement p(1-p) top-B, 2PL p-IRT
  AnchorPoints-adapted       K-means on item response vectors, cluster-
                             weighted anchor mean (the Anchor Points estimator)
  Total-Fisher static        sum_j I_i(theta_j) over calibration planners, 2PL p-IRT
  Marginal-Fisher static     E_{theta~N(0,1)} I_i(theta), 2PL p-IRT
  tinyBenchmarks             K-means(a,b) anchors, K=B, 2PL p-IRT
  metabench-lite             information-grid greedy, 2PL p-IRT
  Fluid-style (adaptive)     2PL Fisher argmax, p-IRT
  ours-EIG (adaptive)        theta-EIG, marginalised posterior mean
  ours-SRVar (adaptive)      SR-variance, marginalised posterior mean

Also reports pooled rank agreement (Spearman of estimated vs true SR over
the 48 evaluations) per budget — with the caveat that the panel's SR spread
(.135-.777) saturates rank metrics.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirt.b2d import Panel
from scirt.splits import unified_split, R_DRAWS
from scirt.calibration import calibrate
from scirt.curves import marginal_curves, sig, THG, PRIOR
from scirt.bayes import post_from, posterior_mean_sr
from scirt.acquisition import (srvar_pick, eig_pick, fisher_2pl_pick,
                               kmeans_anchors, metabench_order, stratified_order)
from scirt.metrics import mean_se

np.random.seed(0)
torch.manual_seed(0)
BGRID = [10, 20, 29, 30, 40, 60, 69, 80, 100, 120]
OUT = Path(__file__).resolve().parents[1] / 'results'


def theta_map(bs, ys, aa, it=50):
    t = 0.0
    for _ in range(it):
        p = sig(aa * (t - bs))
        g = (aa * (ys - p)).sum() - t
        hh = -((aa ** 2) * p * (1 - p)).sum() - 1.0
        t -= g / hh
    return float(np.clip(t, -6, 6))


def pirt(bs, aa, yy, S):
    n = len(bs)
    S = np.array(S)
    t = theta_map(bs[S], yy[S], aa[S])
    un = [i for i in range(n) if i not in set(S.tolist())]
    return (yy[S].sum() + sig(aa[un] * (t - bs[un])).sum()) / n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='', help='comma budgets; overrides the default grid and skips anchors')
    args = ap.parse_args()
    global BGRID
    custom = bool(args.grid)
    if custom:
        BGRID = [int(x) for x in args.grid.split(',')]
    panel = Panel()
    METHODS = ['Random (IRT-free mean)', 'Random + IRT', 'Random-strat + IRT',
               'DISCO-adapted', 'AnchorPoints-adapted', 'Total-Fisher static',
               'Marginal-Fisher static', 'tinyBenchmarks', 'metabench-lite',
               'Fluid-style', 'ours-EIG', 'ours-SRVar']
    ERR = {m: {B: [] for B in BGRID} for m in METHODS}
    EST = {m: {B: [] for B in BGRID} for m in METHODS}
    TRUE = []
    GX = np.linspace(-3, 3, 61)
    wG = np.exp(-0.5 * GX ** 2)
    wG /= wG.sum()
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        f1 = calibrate(panel.Y, calR, cols, mode='1pl')
        f2 = calibrate(panel.Y, calR, cols, mode='2pl')
        typ = np.array([panel.sn[r] for r in calR])
        # calibration-planner response vectors per bank item (nan -> item mean)
        R = np.full((len(calR), len(cols)), np.nan)
        for a, rid in enumerate(calR):
            for b, pi in enumerate(cols):
                if (rid, pi) in panel.Y:
                    R[a, b] = panel.Y[(rid, pi)]
        pbar = np.nanmean(R, 1)
        Rf = np.where(np.isnan(R), pbar[:, None], R)
        for js in hp:
            bi, yy = panel.bank_rows(calR, js)
            n = len(bi)
            SR = yy.mean()
            TRUE.append(SR)
            b1, s1 = f1['b'][bi], f1['s'][bi]
            a2, b2 = f2['a'][bi], f2['b'][bi]
            ty = typ[bi]
            Rb, pb = Rf[bi], pbar[bi]
            Mf = marginal_curves(b1, s1)
            ones = np.ones(n)
            rng = np.random.RandomState(100 + seed * 20 + js)
            perm = list(rng.permutation(n))
            strat = stratified_order(ty, rng)
            # ---- static orders (planner-independent given the bank) -------
            disco = list(np.argsort(-(pb * (1 - pb))))
            th_cal = f2['th']
            info_tot = np.array([sum((a2[i] ** 2) * sig(a2[i] * (t - b2[i])) * (1 - sig(a2[i] * (t - b2[i])))
                                     for t in th_cal) for i in range(n)])
            tot_f = list(np.argsort(-info_tot))
            info_marg = np.array([((a2[i] ** 2) * sig(a2[i] * (GX - b2[i])) * (1 - sig(a2[i] * (GX - b2[i]))) * wG).sum()
                                  for i in range(n)])
            marg_f = list(np.argsort(-info_marg))
            # ---- adaptive trajectories to 120 (prefix property) ------------
            traj = {}
            S, t0 = [], 0.0
            for _ in range(max(BGRID)):
                rem = [i for i in range(n) if i not in S]
                p = sig(a2[rem] * (t0 - b2[rem]))
                S.append(rem[int(np.argmax((a2[rem] ** 2) * p * (1 - p)))])
                t0 = theta_map(b2[np.array(S)], yy[np.array(S)], a2[np.array(S)])
            traj['Fluid-style'] = S
            for name, pick in (('ours-EIG', lambda q, rem: eig_pick(q, Mf, rem)),
                               ('ours-SRVar', lambda q, rem: srvar_pick(Mf, q, rem))):
                S, q = [], PRIOR.copy()
                qs = {}
                for _ in range(max(BGRID)):
                    rem = [i for i in range(n) if i not in S]
                    S.append(pick(q, rem))
                    q = post_from(Mf, yy, S)
                    if len(S) in BGRID:
                        qs[len(S)] = q.copy()
                traj[name] = (S, qs)
            # ---- score every method at every budget ------------------------
            for B in BGRID:
                Sr = perm[:B]
                EST['Random (IRT-free mean)'][B].append(yy[Sr].mean())
                EST['Random + IRT'][B].append(pirt(b1, ones, yy, Sr))
                EST['Random-strat + IRT'][B].append(pirt(b1, ones, yy, strat[:B]))
                EST['DISCO-adapted'][B].append(pirt(b2, a2, yy, disco[:B]))
                km = KMeans(n_clusters=min(B, n), n_init=4, random_state=0).fit(Rb)
                est, tot = 0.0, 0
                for cl in range(km.n_clusters):
                    mem = np.where(km.labels_ == cl)[0]
                    if not len(mem):
                        continue
                    med = mem[np.argmin(((Rb[mem] - km.cluster_centers_[cl]) ** 2).sum(1))]
                    est += len(mem) * yy[med]
                    tot += len(mem)
                EST['AnchorPoints-adapted'][B].append(est / tot)
                EST['Total-Fisher static'][B].append(pirt(b2, a2, yy, tot_f[:B]))
                EST['Marginal-Fisher static'][B].append(pirt(b2, a2, yy, marg_f[:B]))
                EST['tinyBenchmarks'][B].append(pirt(b2, a2, yy, kmeans_anchors(a2, b2, B, n)))
                EST['metabench-lite'][B].append(pirt(b2, a2, yy, metabench_order(a2, b2, B, n)))
                EST['Fluid-style'][B].append(pirt(b2, a2, yy, traj['Fluid-style'][:B]))
                for name in ('ours-EIG', 'ours-SRVar'):
                    S, qs = traj[name]
                    EST[name][B].append(posterior_mean_sr(Mf, yy, S[:B], qs[B]))
                for m in METHODS:
                    ERR[m][B].append(abs(EST[m][B][-1] - SR))
        print(f'seed {seed} done', flush=True)

    TRUE = np.array(TRUE)
    print('\n===== efficiency frontier: SR-MAE by budget (48 samples) =====')
    print(f'{"method":24s} ' + ' '.join(f'{B:>6d}' for B in BGRID))
    for m in METHODS:
        print(f'{m:24s} ' + ' '.join(f'{np.mean(ERR[m][B]):6.4f}' for B in BGRID))
    print('\n===== pooled rank agreement (Spearman est vs true SR, 48 evals) =====')
    print(f'{"method":24s} ' + ' '.join(f'{B:>6d}' for B in BGRID))
    for m in METHODS:
        print(f'{m:24s} ' + ' '.join(f'{spearmanr(EST[m][B], TRUE).correlation:6.3f}' for B in BGRID))

    OUT.mkdir(exist_ok=True)
    json.dump({'err': {m: {str(B): list(map(float, ERR[m][B])) for B in BGRID} for m in METHODS},
               'est': {m: {str(B): list(map(float, EST[m][B])) for B in BGRID} for m in METHODS},
               'true': list(map(float, TRUE))},
              open(OUT / ('budget_frontier.json' if not custom else f'budget_frontier_grid{"-".join(map(str, BGRID))}.json'), 'w'))
    if custom:
        print('custom grid: anchors skipped')
        return
    assert abs(np.mean(ERR['ours-SRVar'][29]) - 0.0443) < 0.002
    assert abs(np.mean(ERR['Fluid-style'][29]) - 0.0375) < 0.002
    assert abs(np.mean(ERR['Random + IRT'][29]) - 0.0584) < 0.002
    assert abs(np.mean(ERR['metabench-lite'][69]) - 0.0285) < 0.002
    print('anchors OK')


if __name__ == '__main__':
    main()
