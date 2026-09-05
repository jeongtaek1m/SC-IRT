#!/usr/bin/env python3
"""Route-level discrimination at fixed budget (RESULTS.md, after Table 1).

Table 1 scores one aggregate per evaluation (the full-bank SR). This is the
route-level diagnostic: after B routes have been rolled out, how well does the
posterior predict the outcome of the routes it did *not* buy?

For every (draw, K_cal, evaluation planner, B in {30, 55, 110, 165}) the state
`atdrive.bayes.State.predictive_all` gives P(y_s = 1 | D_B) for every bank route;
the B administered routes are removed and the remaining ones are scored against
that planner's true outcomes with

  AUROC   rank discrimination (Mann-Whitney, mid-ranks for ties)
  Brier   mean (p_s - y_s)^2
  calib.  mean predicted failure rate 1 - p_s vs mean realised failure rate

An evaluation whose unobserved outcomes are all one class has no AUROC; it is
dropped from the AUROC average only, and the drop count is reported per cell.
Brier and the two failure rates keep every evaluation.

Two controls decide how the scores can be read. (1) The ZERO-ROLLOUT predictor:
the posterior predictive before any route of the evaluation planner is
observed (calibration only), scored on each order's own residual set — it
cannot differ between orders through posterior quality, so whatever ordering
it reproduces is the ordering of the residual sets, not of the posteriors;
skill = the score minus this baseline. (2) The COMMON SET: at B in {30, 55},
the routes that none of the five orders administered, so every order is
scored on the same routes (at B >= 110 the union of five orders leaves too
few routes with both classes).

All five bank orders of Table 2 run under the COMMON ATDrive readout (the same
posterior, the same difficulty posteriors, the same testlet), so the only thing
that differs between rows is which routes the order bought — the same
native-vs-common convention as `run_adaptive.py`. The four baselines are the
Table 1 rows that have a probability readout under that common model
(Random, Random-strat, Fluid, metabench).

The ATDrive Delta-R1 order is not re-derived: it is the 165-route selection
already fixed by Table 1 and stored per record as `sel` in
results/up_frontier.json (route ids). Recomputing `r1_traj` for three records
spread over the (draw, K_cal) grid (seed 5 / K8, 11 / K12, 14 / K4, first
held-out planner) reproduced those selections exactly, 165 / 165 routes each,
so reuse costs nothing and keeps one selection of record.

    python experiments/run_route_discrimination.py                # all 16 draws
    python experiments/run_route_discrimination.py --seeds 0 4    # shard
    python experiments/run_route_discrimination.py --merge        # tables + anchors
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atdrive.b2d import Panel
from atdrive.splits import up_split, R_DRAWS
from atdrive.calibration import calibrate
from atdrive.bayes import bank_from_fit, State
from atdrive.baselines import fluid_order, metabench_order, stratified_order
from atdrive.metrics import paired_cluster_boot

OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
KCALS = tuple(int(x) for x in os.environ.get('ATDRIVE_KCALS', '4,8,12').split(','))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ORD = ('ATDrive', 'Fluid', 'metabench', 'Random', 'Random-strat')
BGRID = [30, 55, 110, 165]
COMMON_B = (30, 55)      # budgets at which the common (never-administered) set is scored
PJ = 16                  # planner count, set from the panel in run()
# Table 1 row names of the same orders, for the native-readout ranking check
NATIVE = {'ATDrive': 'ATDrive', 'Fluid': 'Fluid', 'metabench': 'metabench',
          'Random': 'Random + IRT', 'Random-strat': 'Random-strat + IRT'}


def subsample(cols, seed, Kc):
    if Kc >= len(cols):
        return list(cols)
    rs = np.random.RandomState(9000 + seed * 100 + Kc * 10 + 0)
    return sorted(np.array(cols)[rs.choice(len(cols), Kc, replace=False)].tolist())


def auroc(y, p):
    """Mann-Whitney AUROC with mid-ranks for ties; None when y is one class."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(p)
    return float((r[y > 0.5].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def orders_for(f2, bi, yy, typ, seed, js, sel, rid_of):
    """The five bank orders as bank indices. ATDrive's is Table 1's stored
    Delta-R1 selection (route ids -> bank indices); the others are the
    Table 2 constructions, whose prefixes at 165 are their own prefixes."""
    n = len(bi)
    T = max(BGRID)
    return {'ATDrive': [rid_of[r] for r in sel[:T]],
            'Fluid': fluid_order(f2['a'][bi], f2['b'][bi], yy, T),
            'metabench': [int(i) for i in metabench_order(f2['a'][bi], f2['b'][bi], T, n)],
            'Random': [int(i) for i in np.random.RandomState(100 + seed * PJ + js).permutation(n)[:T]],
            'Random-strat': [int(i) for i in stratified_order(typ[bi], np.random.RandomState(100 + seed * PJ + js))[:T]]}


def score(bank, yy, order, p0, U):
    """Route-level scores of one order at every budget in BGRID: on the order's
    own unobserved routes (auc, brier, ...), the zero-rollout predictor p0 on
    those same routes (auc0, brier0, psd0) and, for B in COMMON_B, on the common
    set U[B] of routes no order administered (auc_common, ...)."""
    st = State(bank, yy)
    out = {}
    for t, s in enumerate(order, 1):
        st.add(s)
        if t in BGRID:
            p = st.predictive_all()
            un = np.setdiff1d(np.arange(bank.n), np.array(st.S, int))
            pu, yu = p[un], yy[un]
            out[str(t)] = {'auc': auroc(yu, pu),
                           'brier': float(np.mean((pu - yu) ** 2)),
                           'pfail': float(np.mean(1 - pu)),
                           'rfail': float(np.mean(1 - yu)),
                           'psd': float(np.std(pu)),      # spread of what is left to rank
                           'n': int(len(un)),
                           'auc0': auroc(yu, p0[un]),     # the zero-rollout predictor, same routes
                           'brier0': float(np.mean((p0[un] - yu) ** 2)),
                           'psd0': float(np.std(p0[un]))}
            if t in U:
                uc = U[t]
                out[str(t)].update({'auc_common': auroc(yy[uc], p[uc]),
                                    'brier_common': float(np.mean((p[uc] - yy[uc]) ** 2)),
                                    'psd_common': float(np.std(p[uc])),
                                    'n_common': int(len(uc))})
    return out


def run(seeds):
    global PJ
    panel = Panel()
    PJ = panel.J
    fr = json.load(open(OUT / 'up_frontier.json'))
    SEL = {(r['seed'], r['K'], r['js']): r['sel'] for r in fr}
    recs = []
    for seed in seeds:
        hp, ht = up_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        typ = np.array([panel.sn[r] for r in calR])
        for Kc in KCALS:
            cs = subsample(cols, seed, Kc)
            f1 = calibrate(panel.Y, calR, cs, mode='1pl', types=typ, device=DEV)
            f2 = calibrate(panel.Y, calR, cs, mode='2pl', sigma_b=f1['sigma_b'], device=DEV)
            for js in hp:
                bi, yy = panel.bank_rows(calR, js)
                bank = bank_from_fit(f1, bi, typ)
                rid_of = {calR[bi[i]]: i for i in range(len(bi))}
                sel = SEL[(seed, Kc, int(js))]
                assert len(sel) >= max(BGRID) and all(r in rid_of for r in sel[:max(BGRID)])
                od = orders_for(f2, bi, yy, typ, seed, js, sel, rid_of)
                p0 = State(bank, yy).predictive_all()            # before any rollout of this planner
                U = {B: np.setdiff1d(np.arange(len(bi)), np.concatenate([od[o][:B] for o in ORD]))
                     for B in COMMON_B}                          # routes no order administered
                rec = {'seed': seed, 'K': Kc, 'js': int(js), 'SR': float(yy.mean())}
                for o in ORD:
                    rec[o] = score(bank, yy, od[o], p0, U)
                recs.append(rec)
            print(f'seed {seed} K{Kc} done', flush=True)
    return recs


def cell(recs, K, o, B, key):
    return [r[o][str(B)][key] for r in recs if r['K'] == K]


def report(recs):
    recs = sorted(recs, key=lambda r: (r['seed'], r['K'], r['js']))
    cells = [(K, B) for K in KCALS for B in BGRID]
    print(f'\n{len(recs)} planner evaluations ({len(recs) // len(KCALS)} per K_cal)')
    print('\n===== route-level AUROC on the UNOBSERVED routes; * = paired 95% CI vs ATDrive excludes 0 =====')
    print(f'{"order":14s} ' + ' '.join(f'K{K}B{B:<3d}' for K, B in cells) + '   macro')
    AUC = {}
    for o in ORD:
        row, mac = [], []
        for K, B in cells:
            a = cell(recs, K, o, B, 'auc')
            d = cell(recs, K, 'ATDrive', B, 'auc')
            j = [r['js'] for r in recs if r['K'] == K]
            ok = [i for i in range(len(a)) if a[i] is not None]
            v = float(np.mean([a[i] for i in ok]))
            AUC[(K, o, B)] = v
            mac.append(v)
            star = ' '
            if o != 'ATDrive':
                bo = [i for i in range(len(a)) if a[i] is not None and d[i] is not None]
                _, lo, hi = paired_cluster_boot([a[i] for i in bo], [d[i] for i in bo], [j[i] for i in bo])
                star = '*' if (lo > 0 or hi < 0) else ' '
            row.append(f'{v:.4f}{star}')
        print(f'{o:14s} ' + ' '.join(f'{c:>8s}' for c in row) + f'   {np.mean(mac):.4f}')
    print('\n===== evaluations with no AUROC (unobserved outcomes all one class), out of 64 per cell =====')
    print(f'{"order":14s} ' + ' '.join(f'K{K}B{B:<3d}' for K, B in cells))
    for o in ORD:
        print(f'{o:14s} ' + ' '.join(f'{sum(1 for a in cell(recs, K, o, B, "auc") if a is None):>8d}'
                                     for K, B in cells))
    print('\n===== Brier score on the unobserved routes (all 64 evaluations) =====')
    print(f'{"order":14s} ' + ' '.join(f'K{K}B{B:<3d}' for K, B in cells) + '   macro')
    BRI = {}
    for o in ORD:
        mac = []
        for K, B in cells:
            BRI[(K, o, B)] = float(np.mean(cell(recs, K, o, B, 'brier')))
            mac.append(BRI[(K, o, B)])
        print(f'{o:14s} ' + ' '.join(f'{BRI[(K, o, B)]:8.4f}' for K, B in cells) + f'   {np.mean(mac):.4f}')
    print('\n===== calibration of the fill: mean predicted failure rate - mean realised, unobserved routes =====')
    print(f'{"order":14s} ' + ' '.join(f'K{K}B{B:<3d}' for K, B in cells))
    for o in ORD:
        print(f'{o:14s} ' + ' '.join(
            f'{np.mean(cell(recs, K, o, B, "pfail")) - np.mean(cell(recs, K, o, B, "rfail")):+8.4f}'
            for K, B in cells))
    print('\n===== what is left to rank is not the same set for every order =====')
    print('mean realised failure rate on the unobserved routes')
    print(f'{"order":14s} ' + ' '.join(f'K{K}B{B:<3d}' for K, B in cells))
    for o in ORD:
        print(f'{o:14s} ' + ' '.join(f'{np.mean(cell(recs, K, o, B, "rfail")):8.4f}' for K, B in cells))
    print('SD of the predicted P(y=1) over the unobserved routes')
    print(f'{"order":14s} ' + ' '.join(f'K{K}B{B:<3d}' for K, B in cells) + '   macro')
    for o in ORD:
        sd = [np.mean(cell(recs, K, o, B, 'psd')) for K, B in cells]
        print(f'{o:14s} ' + ' '.join(f'{v:8.4f}' for v in sd) + f'   {np.mean(sd):.4f}')
    return AUC, BRI


def common_mae():
    """Common-readout SR-MAE per (K_cal, order, B) from results/adaptive.json."""
    ad = OUT / 'adaptive.json'
    if not ad.exists():
        return None
    A = json.load(open(ad))
    return {(K, o, B): float(np.mean([abs(r[o]['Shat'][B - 1] - r['SR']) for r in A if r['K'] == K]))
            for K in KCALS for o in ORD for B in BGRID}


def controls(recs, AUC, BRI):
    """The two controls and the pooled test the reading rests on. Returns the
    statistics the anchors pin."""
    recs = sorted(recs, key=lambda r: (r['seed'], r['K'], r['js']))
    cells = [(K, B) for K in KCALS for B in BGRID]
    MAE = common_mae()
    out = {}
    print('\n===== ATDrive vs Random-strat on AUROC: the pooled paired test (records where both are defined) =====')
    a, b, cl = [], [], []
    for r in recs:
        for B in BGRID:
            x, y = r['ATDrive'][str(B)]['auc'], r['Random-strat'][str(B)]['auc']
            if x is not None and y is not None:
                a.append(x)
                b.append(y)
                cl.append(r['js'])
    d, lo, hi = paired_cluster_boot(a, b, cl)
    ahead = sum(1 for K, B in cells if AUC[(K, 'ATDrive', B)] > AUC[(K, 'Random-strat', B)])
    mac = lambda D, o, skip=None: float(np.mean([D[(K, o, B)] for K, B in cells if (K, B) != skip]))
    rr = [r for r in recs if r['K'] == 4 and r['ATDrive']['165']['auc'] is not None]
    d4, lo4, hi4 = paired_cluster_boot([r['Random-strat']['165']['auc'] for r in rr],
                                       [r['ATDrive']['165']['auc'] for r in rr], [r['js'] for r in rr])
    CS = {o: [] for o in ORD}
    for K, B in cells:
        rs = [r for r in recs if r['K'] == K and all(r[o][str(B)]['auc'] is not None for o in ORD)]
        for o in ORD:
            CS[o].append(float(np.mean([r[o][str(B)]['auc'] for r in rs])))
    print(f'pooled ATDrive - Random-strat over {len(a)} paired evaluations: {d:+.4f} [{lo:+.4f},{hi:+.4f}]; '
          f'ATDrive ahead in {ahead} of 12 cells')
    print(f'macro without K4/B165: ATDrive {mac(AUC, "ATDrive", (4, 165)):.4f} vs Random-strat '
          f'{mac(AUC, "Random-strat", (4, 165)):.4f}; K4/B165 Random-strat - ATDrive {d4:+.4f} [{lo4:+.4f},{hi4:+.4f}] '
          f'on {len(rr)} records')
    print('common-support macro (records where all five orders have an AUROC): '
          + ', '.join(f'{o} {np.mean(CS[o]):.4f}' for o in ORD))
    out.update(pooled_d=(d, lo, hi), pooled_n=len(a), ahead=ahead)

    print('\n===== control 1: the zero-rollout predictor (calibration only, no evaluation-planner data), '
          'scored on each order\'s own residual sets =====')
    B0 = {(K, o, B): float(np.mean(cell(recs, K, o, B, 'brier0'))) for K, B in cells for o in ORD}
    A0 = {(K, o, B): float(np.mean([x for x in cell(recs, K, o, B, 'auc0') if x is not None]))
          for K, B in cells for o in ORD}
    print(f'{"order":14s} {"Brier0":>8s} {"Brier":>8s} {"AUROC0":>8s} {"AUROC":>8s}'
          + (f' {"SR-MAE":>8s}' if MAE else ''))
    for o in ORD:
        print(f'{o:14s} {mac(B0, o):8.4f} {mac(BRI, o):8.4f} {mac(A0, o):8.4f} {mac(AUC, o):8.4f}'
              + (f' {mac(MAE, o):8.4f}' if MAE else ''))
    sp0 = max(mac(B0, o) for o in ORD) - min(mac(B0, o) for o in ORD)
    sp1 = max(mac(BRI, o) for o in ORD) - min(mac(BRI, o) for o in ORD)
    print(f'between-order macro Brier spread: zero-rollout {sp0:.4f}, real posteriors {sp1:.4f}')
    if MAE:
        m = [mac(MAE, o) for o in ORD]
        print('macro Spearman with SR-MAE: Brier0 {:+.2f}, Brier {:+.2f}, AUROC0 {:+.2f}, AUROC {:+.2f}'.format(
            spearmanr([mac(B0, o) for o in ORD], m)[0], spearmanr([mac(BRI, o) for o in ORD], m)[0],
            spearmanr([-mac(A0, o) for o in ORD], m)[0], spearmanr([-mac(AUC, o) for o in ORD], m)[0]))
    print('skill over the zero-rollout predictor, paired vs ATDrive (pooled over records x B, planner clusters):')
    out['brier0'] = {o: mac(B0, o) for o in ORD}
    out['skill'] = {}
    for lab, f in (('Brier skill (brier0 - brier)', lambda r, o, B: r[o][str(B)]['brier0'] - r[o][str(B)]['brier']),
                   ('AUROC gain (auc - auc0)', lambda r, o, B: None if r[o][str(B)]['auc'] is None
                    else r[o][str(B)]['auc'] - r[o][str(B)]['auc0'])):
        for o in ('Random-strat', 'Random', 'Fluid'):
            a, b, cl = [], [], []
            for r in recs:
                for B in BGRID:
                    x, y = f(r, o, B), f(r, 'ATDrive', B)
                    if x is None or y is None:
                        continue
                    a.append(x)
                    b.append(y)
                    cl.append(r['js'])
            d, lo, hi = paired_cluster_boot(a, b, cl)
            print(f'  {lab:28s} {o:13s} - ATDrive: {np.mean(a):+.4f} vs {np.mean(b):+.4f}  '
                  f'd {d:+.4f} [{lo:+.4f},{hi:+.4f}]  n {len(a)}')
            out['skill'][(lab.split()[0], o)] = (d, lo, hi)

    print(f'\n===== control 2: the common set — routes no order administered, B in {COMMON_B} =====')
    cells2 = [(K, B) for K in KCALS for B in COMMON_B]
    nU = {B: np.mean([r['ATDrive'][str(B)]['n_common'] for r in recs]) for B in COMMON_B}
    print('common-set size: ' + ', '.join(f'{nU[B]:.1f} routes at B={B}' for B in COMMON_B) + '; both classes present in '
          + ', '.join('{:.1%} (B={})'.format(np.mean([all(r[o][str(B)]['auc_common'] is not None for o in ORD)
                                                       for r in recs]), B) for B in COMMON_B) + ' of evaluations')
    out['common'] = {}
    for key, sgn in (('auc', -1), ('brier', 1)):
        M = {(K, o, B): float(np.mean([x for x in cell(recs, K, o, B, key + '_common') if x is not None]))
             for K, B in cells2 for o in ORD}
        own = {o: float(np.mean([(AUC if key == 'auc' else BRI)[(K, o, B)] for K, B in cells2])) for o in ORD}
        macc = {o: float(np.mean([M[(K, o, B)] for K, B in cells2])) for o in ORD}
        print(f'-- {key} on the common set (macro over B in {COMMON_B}): '
              + ', '.join(f'{o} {macc[o]:.4f} (own residual set {own[o]:.4f})' for o in ORD))
        spc, spo = max(macc.values()) - min(macc.values()), max(own.values()) - min(own.values())
        print(f'   spread across orders: common {spc:.4f}, own residual set {spo:.4f}')
        out['common'][key] = {'macro': macc, 'spread': (spc, spo)}
        if MAE:
            pc = [spearmanr(sgn * np.array([M[(K, o, B)] for o in ORD]), [MAE[(K, o, B)] for o in ORD])[0]
                  for K, B in cells2]
            po = [spearmanr(sgn * np.array([(AUC if key == 'auc' else BRI)[(K, o, B)] for o in ORD]),
                            [MAE[(K, o, B)] for o in ORD])[0] for K, B in cells2]
            print(f'   per-cell mean Spearman with SR-MAE: common {np.mean(pc):+.2f}, own residual set {np.mean(po):+.2f}')
            out['common'][key]['rho'] = (float(np.mean(pc)), float(np.mean(po)))
        for o in ORD[1:]:
            a, b, cl = [], [], []
            for r in recs:
                for B in COMMON_B:
                    x, y = r[o][str(B)][key + '_common'], r['ATDrive'][str(B)][key + '_common']
                    if x is None or y is None:
                        continue
                    a.append(x)
                    b.append(y)
                    cl.append(r['js'])
            d, lo, hi = paired_cluster_boot(a, b, cl)
            print(f'   {o:13s} - ATDrive on the common set: {np.mean(a):.4f} vs {np.mean(b):.4f}  d {d:+.4f} [{lo:+.4f},{hi:+.4f}]')
            out['common'][key][o] = (d, lo, hi)
    return out


def ranking(AUC, BRI):
    """Does AUROC order the five methods the way SR-MAE does? Brier is carried
    alongside as the other route-level score; both are scored on each order's
    own residual set, which is why the zero-rollout control above is scored on
    the same sets.

    Two SR-MAE references: the common-readout errors of Table 2
    (results/adaptive.json — the like-for-like comparison, since the AUROC uses
    that same posterior) and the native-readout Table 1 errors
    (results/up_frontier.json)."""
    src = {}
    MAE = common_mae()
    if MAE:
        src['common readout (Table 2 machine)'] = MAE
    fr = OUT / 'up_frontier.json'
    if fr.exists():
        F = json.load(open(fr))
        src['native readout (Table 1)'] = {
            (K, o, B): float(np.mean([r['err'][NATIVE[o]][str(B)] for r in F if r['K'] == K]))
            for K in KCALS for o in ORD for B in BGRID}
    out = {}
    for name, MAE in src.items():
        print(f'\n===== ranking agreement with SR-MAE, {name} =====')
        print(f'{"cell":10s} {"rhoAUC":>7s} {"rhoBri":>7s}  {"best AUROC":14s} {"best Brier":14s} '
              f'{"best SR-MAE":14s}   ' + '  '.join(f'{o}' for o in ORD))
        ga, gb, hit = [], [], 0
        for K in KCALS:
            for B in BGRID:
                a = np.array([AUC[(K, o, B)] for o in ORD])
                s = np.array([BRI[(K, o, B)] for o in ORD])
                m = np.array([MAE[(K, o, B)] for o in ORD])
                ra, rb = float(spearmanr(-a, m)[0]), float(spearmanr(s, m)[0])
                ga.append(ra)
                gb.append(rb)
                ba, bs, bm = (ORD[int(np.argmax(a))], ORD[int(np.argmin(s))], ORD[int(np.argmin(m))])
                hit += (ba == bm)
                print(f'K{K}B{B:<6d} {ra:+7.2f} {rb:+7.2f}  {ba:14s} {bs:14s} {bm:14s}   ' +
                      '  '.join(f'{v:.4f}' for v in m))
        print(f'mean Spearman rho over the {len(ga)} cells: AUROC {np.mean(ga):+.2f}, Brier {np.mean(gb):+.2f} '
              f'(AUROC gives the identical order in {sum(1 for r in ga if r > 0.99)} cells, '
              f'names the SR-MAE winner in {hit})')
        mac = lambda D: np.array([np.mean([D[(K, o, B)] for K in KCALS for B in BGRID]) for o in ORD])
        print('macro order,  AUROC: ' + ' > '.join(np.array(ORD)[np.argsort(-mac(AUC))])
              + f'  (rho {spearmanr(-mac(AUC), mac(MAE))[0]:+.2f})')
        print('              Brier: ' + ' < '.join(np.array(ORD)[np.argsort(mac(BRI))])
              + f'  (rho {spearmanr(mac(BRI), mac(MAE))[0]:+.2f})')
        print('             SR-MAE: ' + ' < '.join(np.array(ORD)[np.argsort(mac(MAE))]))
        out[name] = float(np.mean(ga))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs=2, type=int, default=None)
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.merge:
        recs = sum([json.load(open(f)) for f in sorted(glob.glob(str(OUT / 'route_discrimination_*_*.json')))], [])
        if recs:
            json.dump(recs, open(OUT / 'route_discrimination.json', 'w'))
        else:                                   # no shards (a clone): score the results of record
            recs = json.load(open(OUT / 'route_discrimination.json'))
    elif args.seeds:
        lo, hi = args.seeds
        recs = run(range(lo, hi))
        json.dump(recs, open(OUT / f'route_discrimination_{lo}_{hi}.json', 'w'))
        print('shard saved; run with --merge after all shards')
        return
    else:
        recs = run(range(R_DRAWS))
        json.dump(recs, open(OUT / 'route_discrimination.json', 'w'))
    AUC, BRI = report(recs)
    ctl = controls(recs, AUC, BRI)
    rho = ranking(AUC, BRI)
    assert len(recs) == len(KCALS) * 64
    for (K, o, B, v) in ((4, 'ATDrive', 30, .7316), (4, 'Random-strat', 165, .7750), (8, 'metabench', 55, .7589),
                         (12, 'ATDrive', 165, .8281), (12, 'Fluid', 55, .7826), (12, 'Random', 110, .7957)):
        assert abs(AUC[(K, o, B)] - v) < .003, (K, o, B, AUC[(K, o, B)])
    mac = lambda D, o: float(np.mean([D[(K, o, B)] for K in KCALS for B in BGRID]))
    for o, a, b in (('ATDrive', .7768, .1341), ('Random-strat', .7767, .1687), ('metabench', .7500, .1804)):
        assert abs(mac(AUC, o) - a) < .002, (o, mac(AUC, o))       # macro AUROC: ATDrive and Random-strat .0001 apart ...
        assert abs(mac(BRI, o) - b) < .002, (o, mac(BRI, o))       # ... and .035 apart on Brier (own residual sets)
    assert sum(1 for r in recs for o in ORD for B in BGRID if r[o][str(B)]['auc'] is None) == 36
    assert abs(rho['common readout (Table 2 machine)'] - 0.533) < .02, rho    # not the SR-MAE ranking
    assert abs(rho['native readout (Table 1)'] - 0.633) < .02, rho
    # the pooled test: ATDrive and Random-strat are unresolved on AUROC, not tied
    d, lo, hi = ctl['pooled_d']
    assert abs(d - .0009) < .002 and lo < 0 < hi and ctl['pooled_n'] == 751 and ctl['ahead'] == 9, ctl
    # control 1: the zero-rollout predictor reproduces the Brier ordering; after subtracting it the
    # type-stratified order is ahead of ATDrive on both route-level scores
    for o, v in (('ATDrive', .1815), ('Fluid', .1864), ('metabench', .2297), ('Random', .2241), ('Random-strat', .2238)):
        assert abs(ctl['brier0'][o] - v) < .002, (o, ctl['brier0'][o])
    for key, v in (('Brier', .0076), ('AUROC', .0118)):
        d, lo, hi = ctl['skill'][(key, 'Random-strat')]
        assert abs(d - v) < .002 and lo > 0, (key, d, lo, hi)
    # control 2: with the evaluation set held fixed the orders barely separate
    c = ctl['common']
    assert abs(c['auc']['macro']['ATDrive'] - .7658) < .002 and abs(c['auc']['macro']['Random-strat'] - .7645) < .002, c['auc']['macro']
    d, lo, hi = c['auc']['Random-strat']
    assert abs(d + .0014) < .002 and lo < 0 < hi, (d, lo, hi)
    assert abs(c['brier']['spread'][0] - .0039) < .001 and abs(c['brier']['spread'][1] - .0175) < .002, c['brier']['spread']
    if 'rho' in c['auc']:
        assert abs(c['auc']['rho'][0] - 0.10) < .05 and abs(c['auc']['rho'][1] - 0.52) < .05, c['auc']['rho']
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
