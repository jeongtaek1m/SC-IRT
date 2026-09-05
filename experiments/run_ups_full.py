#!/usr/bin/env python3
"""UPS retargeted to the full 220-route success rate (RESULTS.md, after Table 3B).

`run_ups.py` (Table 3B) predicts only the block-D success rate: the 40
routes of the 8 held-out scenario types. Here the estimand is the whole
benchmark for the held-out planner,

    I = S_t  u  (C \\ S_t)  u  T

    S_t      the probed calibrated routes    -- observed responses, used as-is
    C \\ S_t  the rest of the 180 calibrated routes -- inferred from the
             response-calibrated difficulty posteriors p(b_s | A) (PROTOCOL 3)
    T        the 40 held-out-type routes     -- inferred from the
             scene-conditioned prior b_s ~ N(b_tilde_s, sigma^2) (PROTOCOL 3.1)

Everything else is `run_ups.py`: the 36:8 type split (`unified_split`), the
12:4 planner split, 16 draws, K_cal = 12, probe budgets B in {30, 55, 110}
placed on the calibrated routes only, and the ability posterior q_B built
from those probes exactly as before. Only the readout changes. The three
parts share one q_B, so the full-bank readout is the same L1 Bayes action
of PROTOCOL 4 with the unobserved total taken over (C \\ S_t) u T:

    SR_hat = ( sum_{s in S_t} y_s + median(T_unobs | D) ) / |I|

with mean/variance summed per scenario type -- calibration types carry the
probe-updated testlet posterior, target types keep the u prior, which is
what `atdrive.bayes.transfer` does for the T block alone.

Readout arms (all share one q_B, so they differ only in the difficulty
model used for the unobserved routes):

    naive                 the probed success rate, used for everything
    calC + priorT(marg)   C \\ S_t response-calibrated; T from the difficulty
                          prior b ~ N(0, sigma_b^2)  -- scene encoder OFF
    calC + priorT(const)  C \\ S_t response-calibrated; T from a single
                          constant b = mean_s b_tilde_s with the same
                          residual sigma -- the encoder's block level kept,
                          its per-route signal removed
    calC + sceneT         canonical: C \\ S_t response-calibrated, T from the
                          per-route RelGraph prior
    trueC + sceneT        C block revealed -- the error left is T's alone
    calC + trueT          T block revealed -- the error left is C's alone

Probe policies: Random, Delta-R1 on D (the canonical UPS rule of
`run_ups.py`, `acquisition.r1_pick_transfer`, the acquisition driven by the
scene prior), Delta-R1 on D against the scene-free target bank (the same
rule with the encoder removed from the acquisition as well, so the arm
'calC + priorT(marg)' on this order is encoder-free end to end) and Delta-R1
on the full bank (`acquisition.r1_pick` on the C u T bank, candidates
restricted to C) -- the acquisition retargeted to the new estimand.

Metrics: full-benchmark SR-MAE (primary), AUROC of the per-cell posterior
predictive on T, and the target-block SR-MAE as the diagnostic that makes
the number comparable with Table 3B. Paired planner-cluster bootstrap
intervals throughout.

    python experiments/run_ups_full.py --seeds 0 2   # shard (CPU, ~11 min)
    python experiments/run_ups_full.py --merge       # tables + anchors
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atdrive.b2d import Panel, DATA
from atdrive.splits import unified_split, R_DRAWS
from atdrive.calibration import calibrate
from atdrive.curves import marginal_curves, curves_from_posterior
from atdrive.bayes import Bank, State, bank_from_fit, mix_median, mix_l1, transfer
from atdrive.acquisition import r1_pick, r1_pick_transfer
from atdrive.metrics import paired_cluster_boot

OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
DEV = os.environ.get('ATDRIVE_DEVICE', 'cpu')
BP = (30, 55, 110)
TMAX = max(BP)
K_CAL = 12
ENC_RUN = 0                                     # RelGraph R2 run s0 (canonical)
POL = ('Random', 'Delta-R1 on D', 'Delta-R1 on D (scene-free)', 'Delta-R1 on full I')
CANP = 'Delta-R1 on D'                          # the probe rule of record (run_ups.py)
ARM = ('naive', 'calC + priorT(marg)', 'calC + priorT(const)', 'calC + sceneT',
       'trueC + sceneT', 'calC + trueT')
CANA = 'calC + sceneT'
TARM = ('naive', 'calC + priorT(marg)', 'calC + priorT(const)', 'calC + sceneT')   # arms with a T readout


def full_readout(q, yo, mu, var, n_unobs, N):
    """(SR_hat, R1) on the full bank from the pooled unobserved-total moments."""
    q2, mu2, sd2 = q[:, None], mu[:, None], np.sqrt(var + 1e-9)[:, None]
    c = mix_median(q2, mu2, sd2, float(n_unobs))
    return float((yo + c[0]) / N), float(mix_l1(q2, mu2, sd2, c)[0] / N)


def run(seeds):
    panel = Panel()
    pz = np.load(DATA / 'encoder' / f'relgraph_r2_s{ENC_RUN}.npz', allow_pickle=True)
    recs = []
    for seed in seeds:
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]      # K_cal = 12, no subsampling
        calR, newR = panel.split_routes(ht)
        typ = np.array([panel.sn[r] for r in calR])
        typD = np.array([panel.sn[r] for r in newR])
        f1 = calibrate(panel.Y, calR, cols, mode='1pl', types=typ, device=DEV)
        rt = [str(x) for x in pz[f'draw{seed}_rt']]
        lut = {rt[k]: float(pz[f'draw{seed}_bt'][k]) for k in range(len(rt))}
        bt = np.array([lut[r] for r in newR])
        sig_e = float(pz[f'draw{seed}_sigma'])
        for js in hp:
            dj = [i for i, r in enumerate(newR) if (r, js) in panel.Y]
            if len(dj) < 4:
                continue
            yD = np.array([panel.Y[(newR[i], js)] for i in dj], float)
            bi, yb = panel.bank_rows(calR, js)
            nC, nT = len(bi), len(dj)
            N = nC + nT
            btD, tD = bt[dj], typD[dj]
            bankC = bank_from_fit(f1, bi, typ)
            CD = {'calC + sceneT': marginal_curves(btD, np.full(nT, sig_e)),
                  'calC + priorT(marg)': marginal_curves(np.zeros(nT), np.full(nT, f1['sigma_b'])),
                  'calC + priorT(const)': marginal_curves(np.full(nT, btD.mean()), np.full(nT, sig_e))}
            bankD = {k: Bank(v, tD, f1['sigma_g']) for k, v in CD.items()}
            momT = {k: (lambda s: (s[0], s[1] ** 2))(State(v, np.zeros(nT)).stats())
                    for k, v in bankD.items()}                       # T is never observed
            bankF = Bank(np.hstack([curves_from_posterior(f1['W'][bi]), CD[CANA]]),
                         np.concatenate([typ[bi], tD]), f1['sigma_g'])
            # ---- probe orders (on the calibrated routes only) -------------
            S = {'Random': [int(i) for i in
                            np.random.RandomState(100 + seed * panel.J + js).permutation(nC)[:TMAX]]}
            for name, pick in ((CANP, lambda st, rem: r1_pick_transfer(st, bankD[CANA], rem)),
                               ('Delta-R1 on D (scene-free)',
                                lambda st, rem: r1_pick_transfer(st, bankD['calC + priorT(marg)'], rem)),
                               ('Delta-R1 on full I', r1_pick)):
                bk = bankF if name == 'Delta-R1 on full I' else bankC
                yy = np.concatenate([yb, np.zeros(len(yD))]) if name == 'Delta-R1 on full I' else yb   # D outcomes never enter the State (candidates are restricted to C)
                st, o = State(bk, yy), []
                for _ in range(TMAX):
                    o.append(pick(st, [i for i in range(nC) if i not in o]))
                    st.add(o[-1])
                S[name] = o
            # ---- readouts --------------------------------------------------
            rec = {'seed': int(seed), 'js': int(js), 'nC': nC, 'nT': nT,
                   'SR_full': float((yb.sum() + yD.sum()) / N), 'SR_C': float(yb.mean()),
                   'SR_T': float(yD.mean()), 'sigma_b': f1['sigma_b'], 'sigma_g': f1['sigma_g'],
                   'yT': [int(v) for v in yD], 'pol': {}}
            for pname, order in S.items():
                stC, out = State(bankC, yb), {}
                for b_i, s in enumerate(order):
                    stC.add(s)
                    if b_i + 1 not in BP:
                        continue
                    B = b_i + 1
                    q = stC.q
                    muC, sdC = stC.stats()
                    varC, yo = sdC ** 2, float(stC.yo)
                    srT = {k: transfer(q, v) for k, v in bankD.items()}   # (median SR on T, per-cell p)
                    pbar = float(yb[order[:B]].mean())
                    a = {'naive': {'sr': pbar, 'srT': pbar}}
                    for k in ('calC + priorT(marg)', 'calC + priorT(const)', CANA):
                        sr, r1 = full_readout(q, yo, muC + momT[k][0], varC + momT[k][1], N - B, N)
                        a[k] = {'sr': sr, 'r1': r1, 'srT': float(srT[k][0]),
                                'pT': [float(x) for x in srT[k][1]]}
                    mC = mix_median(q[:, None], muC[:, None], np.sqrt(varC + 1e-9)[:, None],
                                    float(nC - B))[0]
                    a['trueC + sceneT'] = {'sr': float((yb.sum() + srT[CANA][0] * nT) / N)}
                    a['calC + trueT'] = {'sr': float((yo + mC + yD.sum()) / N)}
                    out[str(B)] = {'arm': a, 'eC': float(q @ muC - (yb.sum() - yo)) / N,
                                   'eT': float(q @ momT[CANA][0] - yD.sum()) / N}
                rec['pol'][pname] = out
            recs.append(rec)
        print(f'seed {seed} done ({len(recs)} evaluations)', flush=True)
    return recs


def report(recs):
    recs = sorted(recs, key=lambda r: (r['seed'], r['js']))
    JS = [r['js'] for r in recs]
    E = lambda p, B, a: [abs(r['pol'][p][str(B)]['arm'][a]['sr'] - r['SR_full']) for r in recs]
    ET = lambda p, B, a: [abs(r['pol'][p][str(B)]['arm'][a]['srT'] - r['SR_T']) for r in recs]
    print(f'\n===== UPS on the FULL benchmark SR ({len(recs)} evaluations) =====')
    print('composition per evaluation: |C| = {:.1f} calibrated + |T| = {:.1f} target = {:.1f} routes; '
          'B of {:.1f} calibrated routes are probed'.format(
              np.mean([r['nC'] for r in recs]), np.mean([r['nT'] for r in recs]),
              np.mean([r['nC'] + r['nT'] for r in recs]), np.mean([r['nC'] for r in recs])))
    print('true SR: full {:.4f}, C block {:.4f}, T block {:.4f}; mean |SR_C - SR_T| = {:.4f}'.format(
        np.mean([r['SR_full'] for r in recs]), np.mean([r['SR_C'] for r in recs]),
        np.mean([r['SR_T'] for r in recs]), np.mean([abs(r['SR_C'] - r['SR_T']) for r in recs])))
    print('\n-- full-benchmark SR-MAE (primary) --')
    for p in POL:
        print(f'  probe order: {p}')
        for a in ARM:
            print(f'    {a:22s} ' + '  '.join(f'B{B}: {np.mean(E(p, B, a)):.4f}' for B in BP))
    print('\n-- target-block SR-MAE (diagnostic, comparable with Table 3B) --')
    for p in POL:
        print(f'  probe order: {p}')
        for a in TARM:
            print(f'    {a:22s} ' + '  '.join(f'B{B}: {np.mean(ET(p, B, a)):.4f}' for B in BP))
    print('\n-- AUROC on the T cells (pooled over evaluations / mean within evaluation) --')
    yT = np.concatenate([r['yT'] for r in recs])
    for p in POL:
        print(f'  probe order: {p}')
        for a in TARM[1:]:
            pool, per = [], []
            for B in BP:
                pv = np.concatenate([r['pol'][p][str(B)]['arm'][a]['pT'] for r in recs])
                w = [roc_auc_score(r['yT'], r['pol'][p][str(B)]['arm'][a]['pT'])
                     for r in recs if 0 < sum(r['yT']) < len(r['yT'])]
                pool.append(roc_auc_score(yT, pv))
                per.append(np.mean(w))
            print(f'    {a:22s} ' + '  '.join(f'B{B}: {pool[i]:.3f} / {per[i]:.3f}'
                                              for i, B in enumerate(BP)))
    print(f'    {"naive":22s} ' + '  '.join(f'B{B}: 0.500 / 0.500' for B in BP) + '   (constant p, degenerate)')
    print(f'\n-- paired delta full SR-MAE vs [{CANP} | {CANA}] (planner-cluster bootstrap) --')
    for p in POL:
        for a in ARM:
            if p == CANP and a == CANA:
                continue
            print(f'  {p:20s} {a:22s} ' + '  '.join('B{}: {:+.4f} [{:+.4f},{:+.4f}]'.format(
                B, *paired_cluster_boot(E(p, B, a), E(CANP, B, CANA), JS)) for B in BP))
    print('\n-- what the scene prior buys on the full SR (paired, within probe order; + = sceneT worse) --')
    for p in POL:
        for lab, a in (('vs priorT(marg)', 'calC + priorT(marg)'), ('vs priorT(const)', 'calC + priorT(const)'),
                       ('vs trueT (headroom)', 'calC + trueT')):
            print(f'  {p:20s} {lab:20s} ' + '  '.join('B{}: {:+.4f} [{:+.4f},{:+.4f}]'.format(
                B, *paired_cluster_boot(E(p, B, CANA), E(p, B, a), JS)) for B in BP))
    print('\n-- the encoder removed from the acquisition as well (+ = the scene-free version worse) --')
    SF, SFA = 'Delta-R1 on D (scene-free)', 'calC + priorT(marg)'
    print(f'  {"encoder-free end to end vs the canonical cell":52s} ' + '  '.join('B{}: {:+.4f} [{:+.4f},{:+.4f}]'.format(
        B, *paired_cluster_boot(E(SF, B, SFA), E(CANP, B, CANA), JS)) for B in BP))
    print(f'  {"acquisition channel alone (both read out scene-free)":52s} ' + '  '.join('B{}: {:+.4f} [{:+.4f},{:+.4f}]'.format(
        B, *paired_cluster_boot(E(SF, B, SFA), E(CANP, B, SFA), JS)) for B in BP))
    print('\n-- where the full-SR error comes from (posterior means, signed, in SR units) --')
    print('   part errors add to the full-SR error up to the median-vs-mean difference')
    for p in POL:
        for B in BP:
            eC = np.array([r['pol'][p][str(B)]['eC'] for r in recs])
            eT = np.array([r['pol'][p][str(B)]['eT'] for r in recs])
            print(f'  {p:20s} B{B:<4d} C-part |e| {np.abs(eC).mean():.4f} (signed {eC.mean():+.4f})   '
                  f'T-part |e| {np.abs(eT).mean():.4f} (signed {eT.mean():+.4f})   '
                  f'|sum| {np.abs(eC + eT).mean():.4f}')
    print('\n-- share of the T block: 40 of 220 routes = {:.1%} of the benchmark --'.format(
        np.mean([r['nT'] / (r['nC'] + r['nT']) for r in recs])))
    AU = lambda p, B, a: roc_auc_score(
        yT, np.concatenate([r['pol'][p][str(B)]['arm'][a]['pT'] for r in recs]))
    return E, ET, AU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs=2, type=int, default=None)
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.merge:
        recs = sum([json.load(open(f)) for f in sorted(glob.glob(str(OUT / 'ups_full_*_*.json')))], [])
        if recs:
            json.dump(recs, open(OUT / 'ups_full.json', 'w'))
        else:                                   # no shards (a clone): score the results of record
            recs = json.load(open(OUT / 'ups_full.json'))
    elif args.seeds:
        lo, hi = args.seeds
        json.dump(run(range(lo, hi)), open(OUT / f'ups_full_{lo}_{hi}.json', 'w'))
        print('shard saved; run with --merge after all shards')
        return
    else:
        recs = run(range(R_DRAWS))
        json.dump(recs, open(OUT / 'ups_full.json', 'w'))
    E, ET, AU = report(recs)
    assert len(recs) == 64, len(recs)
    JS = [r['js'] for r in sorted(recs, key=lambda r: (r['seed'], r['js']))]
    for (p, a, B, ref) in ANCHORS_FULL:
        assert abs(np.mean(E(p, B, a)) - ref) < .003, (p, a, B, np.mean(E(p, B, a)))
    for (p, a, B, ref) in ANCHORS_T:                # these reproduce Table 3B (results/ups.json)
        assert abs(np.mean(ET(p, B, a)) - ref) < .003, (p, a, B, np.mean(ET(p, B, a)))
    for (p, a, B, ref) in ANCHORS_AUC:
        assert abs(AU(p, B, a) - ref) < .006, (p, a, B, AU(p, B, a))
    # the null of record: the scene prior moves the full-SR error by less than .0025 with every paired
    # interval containing zero, in the readout and in the acquisition
    for B in BP:
        for x, y in (((CANP, CANA), (CANP, 'calC + priorT(marg)')),
                     (('Delta-R1 on D (scene-free)', 'calC + priorT(marg)'), (CANP, CANA))):
            d, lo, hi = paired_cluster_boot(E(x[0], B, x[1]), E(y[0], B, y[1]), JS)
            assert lo < 0 < hi and abs(d) < .0025, (x, y, B, d, lo, hi)
    print('anchors OK')


# (probe policy, readout arm, B, value) -- the numbers this script produced on the
# 16-draw x 4-planner protocol; ANCHORS_T are Table 3B's cells (results/ups.json).
ANCHORS_FULL = ((CANP, CANA, 30, .0467), (CANP, CANA, 55, .0319), (CANP, CANA, 110, .0211),
                (CANP, 'calC + priorT(marg)', 55, .0330), (CANP, 'trueC + sceneT', 110, .0173),
                (CANP, 'calC + trueT', 110, .0098), ('Delta-R1 on full I', CANA, 55, .0234),
                ('Random', 'naive', 30, .0698), ('Random', CANA, 110, .0276),
                ('Delta-R1 on D (scene-free)', 'calC + priorT(marg)', 30, .0454),
                ('Delta-R1 on D (scene-free)', 'calC + priorT(marg)', 55, .0326),
                ('Delta-R1 on D (scene-free)', 'calC + priorT(marg)', 110, .0205))
ANCHORS_T = (('Random', 'naive', 30, .1007), ('Random', CANA, 30, .1129),
             (CANP, CANA, 30, .0950), (CANP, CANA, 55, .0936), (CANP, CANA, 110, .0950))
ANCHORS_AUC = ((CANP, CANA, 30, .760), (CANP, 'calC + priorT(marg)', 30, .710))

if __name__ == '__main__':
    np.random.seed(0)
    main()
