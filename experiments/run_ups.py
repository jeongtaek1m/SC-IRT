#!/usr/bin/env python3
"""Table 3B — UPS: unseen planner x unseen scenes on the primary protocol.

The evaluation planner runs probes on the calibrated bank (budgets
B = {30, 55, 110}); its evaluation-scale ability posterior on the theta grid comes from
those probes (16 calibration planners); its success rate on the evaluation-type routes D
(zero rollouts there) is predicted by transporting the probe posterior as
is (scirt.bayes.transfer) through the scene-conditioned difficulty prior
N(b; b_tilde_s, sigma^2) with the testlet prior on the evaluation types:
the block-D success rate is its posterior median (MAE) and each D cell its
posterior predictive (NLL). b_tilde_s is the RelGraph R2 out-of-fold
prediction for that draw and sigma the per-draw shared residual SD the encoder
learned on the calibration block (data/encoder/relgraph_r2_s*.npz; run s0
is canonical, runs s1-s2 give the across-run SD). Probe policies: naive SR
transfer, Random,
Delta-R1 on the transported block-D success rate (canonical — acquire for
the quantity that must generalise), theta-EIG under the evaluation model and
the 2PL Fisher rule (ablations).

    python experiments/run_ups.py     # ~30 min, GPU
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirt.b2d import Panel, DATA
from scirt.splits import unified_split, R_DRAWS
from scirt.calibration import calibrate
from scirt.curves import marginal_curves
from scirt.bayes import Bank, State, bank_from_fit, state_from, transfer
from scirt.acquisition import eig_pick, r1_pick_transfer
from scirt.baselines import fluid_order
from scirt.metrics import paired_cluster_boot

OUT = Path(os.environ.get('SCIRT_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
BP = (30, 55, 110)
T = max(BP)
POL = ('Random', 'theta-EIG (abl.)', '2PL Fisher (abl.)', 'SC-IRT (Delta-R1 on D)')
CAN = 'SC-IRT (Delta-R1 on D)'
RUNS = (0, 1, 2)


def main():
    panel = Panel()
    RES = {p: {B: {'mae': [], 'nll': []} for B in BP} for p in ('naive',) + POL}
    RES3 = {run: {p: {B: [] for B in BP} for p in POL} for run in RUNS}
    JS = []  # planner id per evaluation, parallel to RES[*][*]['mae']
    RELG = {run: np.load(DATA / 'encoder' / f'relgraph_r2_s{run}.npz', allow_pickle=True) for run in RUNS}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, newR = panel.split_routes(ht)
        typ = np.array([panel.sn[r] for r in calR])
        typD = np.array([panel.sn[r] for r in newR])
        f1 = calibrate(panel.Y, calR, cols, mode='1pl', types=typ)
        f2 = calibrate(panel.Y, calR, cols, mode='2pl', sigma_b=f1['sigma_b'])
        PRI = {}
        for run in RUNS:
            pz = RELG[run]
            rt = [str(x) for x in pz[f'draw{seed}_rt']]
            lut = {rt[k]: float(pz[f'draw{seed}_bt'][k]) for k in range(len(rt))}
            PRI[run] = (np.array([lut[r] for r in newR]), float(pz[f'draw{seed}_sigma']))
        mu_D, tau = PRI[0]
        for js in hp:
            dj = [i for i, r in enumerate(newR) if (r, js) in panel.Y]
            if len(dj) < 4:
                continue
            yD = np.array([panel.Y[(newR[i], js)] for i in dj], float)
            SRD = float(yD.mean())
            JS.append(int(js))
            bankD = Bank(marginal_curves(mu_D[dj], np.full(len(dj), tau)), typD[dj], f1['sigma_g'])
            bankDR = {run: Bank(marginal_curves(PRI[run][0][dj], np.full(len(dj), PRI[run][1])), typD[dj], f1['sigma_g']) for run in RUNS}
            bi, yb = panel.bank_rows(calR, js)
            n = len(bi)
            bank = bank_from_fit(f1, bi, typ)
            S = {'Random': [int(i) for i in np.random.RandomState(100 + seed * panel.J + js).permutation(n)[:T]]}
            s_, st = [], State(bank, yb)
            for _ in range(T):
                rem = [i for i in range(n) if i not in s_]
                s_.append(eig_pick(st, rem))
                st.add(s_[-1])
            S['theta-EIG (abl.)'] = s_
            sd_, st = [], State(bank, yb)
            for _ in range(T):
                rem = [i for i in range(n) if i not in sd_]
                sd_.append(r1_pick_transfer(st, bankD, rem))
                st.add(sd_[-1])
            S[CAN] = sd_
            S['2PL Fisher (abl.)'] = fluid_order(f2['a'][bi], f2['b'][bi], yb, T)
            for B in BP:
                for p in POL:
                    q = state_from(bank, yb, S[p][:B]).q
                    srD, pD = transfer(q, bankD)
                    RES[p][B]['mae'].append(abs(srD - SRD))
                    RES[p][B]['nll'].append(float(-np.mean(yD * np.log(pD + 1e-9) + (1 - yD) * np.log(1 - pD + 1e-9))))
                    for run in RUNS:
                        RES3[run][p][B].append(abs(transfer(q, bankDR[run])[0] - SRD))
                nv = float(yb[S['Random'][:B]].mean())
                RES['naive'][B]['mae'].append(abs(nv - SRD))
                RES['naive'][B]['nll'].append(float(-np.mean(yD * np.log(nv + 1e-9) + (1 - yD) * np.log(1 - nv + 1e-9))))
        print(f'seed {seed} done', flush=True)
    nev = len(RES['naive'][30]['mae'])
    print(f'\n===== Table 3B: UPS, zero rollouts on D ({nev} evaluations) =====')
    print(f'{"probe policy":18s} ' + ' '.join(f'{"B=" + str(B) + " MAE":>10s} {"NLL":>7s}' for B in BP))
    for p in ('naive',) + POL:
        print(f'{p:18s} ' + ' '.join(f'{np.mean(RES[p][B]["mae"]):10.4f} {np.mean(RES[p][B]["nll"]):7.4f}' for B in BP))
    print('across-encoder-run SD of MAE (3 RelGraph runs):')
    for p in POL:
        print(f'  {p:18s} ' + '  '.join('B{}: {:.4f}'.format(B, np.std([np.mean(RES3[run][p][B]) for run in RUNS], ddof=1)) for B in BP))
    print(f'paired delta MAE vs {CAN}:')
    for p in ('Random', 'theta-EIG (abl.)', '2PL Fisher (abl.)'):
        print(f'  {p:18s} ' + '  '.join('B{}: {:+.4f} [{:+.4f},{:+.4f}]'.format(
            B, *paired_cluster_boot(RES[p][B]['mae'], RES[CAN][B]['mae'], JS)) for B in BP))
    OUT.mkdir(exist_ok=True)
    json.dump({p: {str(B): {m: [float(x) for x in v] for m, v in d.items()} for B, d in bb.items()}
               for p, bb in RES.items()}, open(OUT / 'ups.json', 'w'))
    for (p, B, ref) in (('naive', 30, .1233), ('Random', 30, .1047), (CAN, 30, .1030), (CAN, 55, .0971),
                        (CAN, 110, .0927), ('theta-EIG (abl.)', 30, .1041), ('2PL Fisher (abl.)', 110, .0873)):
        assert abs(np.mean(RES[p][B]['mae']) - ref) < .003, (p, B, np.mean(RES[p][B]['mae']))
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
