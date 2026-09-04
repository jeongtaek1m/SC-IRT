#!/usr/bin/env python3
"""Table 2 — what a CAT should target: metric uncertainty, not ability uncertainty.

The IRT model is held fixed (DriveAT's exact difficulty posterior, the
planner x type testlet, the posterior-median SR readout). Only the CAT
around it changes, so every difference below is attributable to the
objective and nothing else.

Selection (which route next), all greedy on the same posterior:
  Delta-R1     ours: the route that most reduces E|SR - SR_hat|   (metric)
  theta-EIG    expected information gain about the ability        (ability)
  Fisher       1PL maximum information at the ability estimate    (ability)
  Random       no objective

Stopping (when to halt), both calibrated leave-one-planner-out on the
calibration panel and never on evaluation planners:
  c * R1_t <= eps          ours: a bound on the reported metric's error
  c_th * SD(theta)_t <= .  ability precision, the classic CAT / Fluid rule;
                           its threshold is set on the calibration panel so
                           that its mean cost matches ours, so the two are
                           compared at equal budget.

Bank = all 220 routes (PROTOCOL section 2); K_cal in {4, 8, 12}.

    python experiments/run_cat_objective.py --seeds 0 2     # shard
    python experiments/run_cat_objective.py --merge         # tables + anchors
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveat.b2d import Panel
from driveat.splits import up_split, R_DRAWS
from driveat.calibration import calibrate
from driveat.bayes import bank_from_fit, track3
from driveat.acquisition import r1_pick, eig_pick, fisher_pick, traj
from driveat.metrics import paired_cluster_boot

OUT = Path(os.environ.get('DRIVEAT_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
KCALS = tuple(int(x) for x in os.environ.get('DRIVEAT_KCALS', '4,8,12').split(','))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PJ = 16
T0 = 10                      # no stop before 10 routes (PROTOCOL section 4)
EPS = (0.05, 0.03)
QUANT = 90                   # percentile of the LOO error / risk ratio
SEL = {'Delta-R1 (metric)': r1_pick, 'theta-EIG (ability)': eig_pick, 'Fisher (ability)': fisher_pick}
ORD = ('Delta-R1 (metric)', 'theta-EIG (ability)', 'Fisher (ability)', 'Random')
OURS = 'Delta-R1 (metric)'


def orders(bank, yy, seed, js):
    o = {k: traj(bank, yy, bank.n, p) for k, p in SEL.items()}
    o['Random'] = [int(i) for i in np.random.RandomState(100 + seed * PJ + js).permutation(bank.n)]
    return o


def tracks(bank, yy, seed, js):
    """Per order: the SR estimate, the metric risk R1 and the ability SD."""
    out = {}
    for k, S in orders(bank, yy, seed, js).items():
        Sh, R1, SE = track3(bank, yy, S)
        out[k] = {'Shat': [float(x) for x in Sh], 'R1': [float(x) for x in R1], 'SE': [float(x) for x in SE]}
    return out


def sub(cols, seed, Jc):
    if Jc >= len(cols):
        return list(cols)
    rs = np.random.RandomState(9000 + seed * 100 + Jc * 10)
    return sorted(np.array(cols)[rs.choice(len(cols), Jc, replace=False)].tolist())


def run(seeds):
    panel = Panel()
    recs = []
    for seed in seeds:
        hp, ht = up_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        typ = np.array([panel.sn[r] for r in calR])
        for K in KCALS:
            cs = sub(cols, seed, K)
            f0 = calibrate(panel.Y, calR, cs, mode='1pl', device=DEV, types=typ)
            for j in cs:                                    # LOO: calibrate the thresholds
                csl = [c for c in cs if c != j]
                f = calibrate(panel.Y, calR, csl, mode='1pl', device=DEV,
                              sigma_b=f0['sigma_b'], types=typ)
                bi, yy = panel.bank_rows(calR, j)
                bank = bank_from_fit(f, bi, typ, sigma_g=f0['sigma_g'])
                recs.append({'seed': seed, 'K': K, 'j': j, 'loo': True, 'SR': float(yy.mean()),
                             **{'tr': tracks(bank, yy, 700 + K, j)}})
            for js, j in enumerate(hp):                     # the evaluation planners
                bi, yy = panel.bank_rows(calR, j)
                bank = bank_from_fit(f0, bi, typ, sigma_g=f0['sigma_g'])
                recs.append({'seed': seed, 'K': K, 'j': j, 'js': js, 'loo': False,
                             'SR': float(yy.mean()), 'tr': tracks(bank, yy, seed, js)})
            print(f'seed {seed} K {K} done', flush=True)
    return recs


def stop_on(sig, thr, T):
    """First t >= T0 with sig[t] <= thr, else the end of the bank."""
    for t in range(T0 - 1, T):
        if sig[t] <= thr:
            return t + 1
    return T


def report(recs):
    loo = [r for r in recs if r['loo']]
    ev = [r for r in recs if not r['loo']]
    res = {}
    print('\n===== Table 2 — the CAT objective, with the IRT model held fixed =====')
    for K in KCALS:
        L = [r for r in loo if r['K'] == K]
        E = [r for r in ev if r['K'] == K]
        js = [r['js'] for r in E]
        # (a) the risk scale c of each selection rule, from its own LOO trajectories
        C = {}
        for o in ORD:
            rr = []
            for r in L:
                sh, r1 = np.array(r['tr'][o]['Shat']), np.array(r['tr'][o]['R1'])
                a, p = np.abs(sh[T0 - 1:] - r['SR']), r1[T0 - 1:]
                rr += list(a[p > 1e-6] / p[p > 1e-6])
            C[o] = float(np.percentile(rr, QUANT))
        print(f'\n-- K_cal = {K} --   risk scale c: ' + '  '.join(f'{o} {C[o]:.2f}' for o in ORD))
        print('   (a) selection objective, both stopping on the metric risk c*R1 <= eps')
        for eps in EPS:
            base = None
            for o in ORD:
                B, er = [], []
                for r in E:
                    t = stop_on(r['tr'][o]['R1'], eps / C[o], len(r['tr'][o]['R1']))
                    B.append(t)
                    er.append(abs(r['tr'][o]['Shat'][t - 1] - r['SR']))
                B, er = np.array(B, float), np.array(er)
                if o == OURS:
                    base = er
                d = (0, 0, 0) if o == OURS else paired_cluster_boot(er, base, js)
                res[f'K{K}|sel|{o}|{eps}'] = {'routes': float(B.mean()), 'mae': float(er.mean()),
                                              'cov': float(np.mean(er <= eps)), 'c': C[o]}
                print(f'      eps {eps:.2f}  {o:20s} routes {B.mean():5.1f} ({B.mean()/220:3.0%})  '
                      f'SR-MAE {er.mean():.4f}'
                      + ('' if o == OURS else f'   d {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]'))
        # (b) stopping objective at matched cost, our selection fixed
        print('   (b) stopping objective at matched cost, selection fixed to Delta-R1')
        for eps in EPS:
            tgt = np.mean([stop_on(r['tr'][OURS]['R1'], eps / C[OURS], len(r['tr'][OURS]['R1'])) for r in E])
            for sig, lab in (('R1', f'metric risk c*R1 <= {eps:.2f}'), ('SE', 'ability SD (matched cost)')):
                if sig == 'R1':
                    thr = eps / C[OURS]
                else:                                        # threshold set on the LOO panel to hit our cost
                    cand = np.unique(np.concatenate([np.array(r['tr'][OURS]['SE'])[T0 - 1:] for r in L]))
                    cost = [np.mean([stop_on(r['tr'][OURS]['SE'], c, len(r['tr'][OURS]['SE'])) for r in L])
                            for c in cand[::max(1, len(cand) // 240)]]
                    cs_ = cand[::max(1, len(cand) // 240)]
                    tl = np.mean([stop_on(r['tr'][OURS]['R1'], eps / C[OURS], len(r['tr'][OURS]['R1'])) for r in L])
                    thr = float(cs_[int(np.argmin(np.abs(np.array(cost) - tl)))])
                B, er = [], []
                for r in E:
                    t = stop_on(r['tr'][OURS][sig], thr, len(r['tr'][OURS][sig]))
                    B.append(t)
                    er.append(abs(r['tr'][OURS]['Shat'][t - 1] - r['SR']))
                B, er = np.array(B, float), np.array(er)
                res[f'K{K}|stop|{sig}|{eps}'] = {'routes': float(B.mean()), 'mae': float(er.mean()),
                                                 'cov': float(np.mean(er <= eps)), 'thr': thr}
                print(f'      eps {eps:.2f}  {lab:34s} routes {B.mean():5.1f} ({B.mean()/220:3.0%})  '
                      f'SR-MAE {er.mean():.4f}   (target {tgt:.1f})')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs=2, type=int, default=None)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.seeds:
        json.dump(run(range(*a.seeds)), open(OUT / f'cat_objective_{a.seeds[0]}_{a.seeds[1]}.json', 'w'))
        print('shard saved; run with --merge after all shards')
        return
    recs = ([r for f in sorted(glob.glob(str(OUT / 'cat_objective_*_*.json'))) for r in json.load(open(f))]
            if a.merge else run(range(R_DRAWS)))
    assert len(set(r['seed'] for r in recs)) == R_DRAWS, len(set(r['seed'] for r in recs))
    res = report(recs)
    json.dump(res, open(OUT / 'cat_objective.json', 'w'), indent=1)
    print(f'\nwritten: {OUT / "cat_objective.json"}')


if __name__ == '__main__':
    np.random.seed(0)
    main()
