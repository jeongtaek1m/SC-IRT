#!/usr/bin/env python3
"""Table 3B — UPS: unseen planner x unseen scenes on the primary protocol.

The evaluation planner runs probes on the calibrated bank (budgets
B = {30, 55, 110}); its evaluation-scale ability is the MAP of the Rasch
posterior on those probes; its success rate on the evaluation-type routes D
(zero rollouts there) is predicted through the scene-conditioned
difficulty prior marginalised: P(y=1 | scene, B) = int sigmoid(theta_hat - b)
N(b; b_tilde_s, sigma^2) db, where b_tilde_s is the RelGraph R2 out-of-fold
prediction for that draw and sigma the per-draw shared residual SD the encoder
learned on the calibration block (data/encoder/relgraph_r2_s*.npz; run s0
is canonical, runs s1-s2 give the across-run SD). Probe policies: naive SR
transfer, Random,
theta-EIG under the evaluation model (canonical), and the 2PL Fisher rule
(ablation — no additional gain; the quantity that must generalise is the
evaluation-scale ability).

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
from scirt.curves import marginal_curves, PRIOR
from scirt.bayes import post_from
from scirt.acquisition import eig_pick
from scirt.baselines import fluid_order
from scirt.metrics import paired_seed_boot

OUT = Path(os.environ.get('SCIRT_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
BP = (30, 55, 110)
T = max(BP)
POL = ('Random', 'theta-EIG', '2PL Fisher (abl.)')
RUNS = (0, 1, 2)


def main():
    panel = Panel()
    RES = {p: {B: {'mae': [], 'nll': []} for B in BP} for p in ('naive',) + POL}
    RES3 = {run: {p: {B: [] for B in BP} for p in POL} for run in RUNS}
    RELG = {run: np.load(DATA / 'encoder' / f'relgraph_r2_s{run}.npz', allow_pickle=True) for run in RUNS}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, newR = panel.split_routes(ht)
        f1 = calibrate(panel.Y, calR, cols, mode='1pl')
        f2 = calibrate(panel.Y, calR, cols, mode='2pl')
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
            MD = marginal_curves(mu_D[dj], np.full(len(dj), tau))
            MDR = {run: marginal_curves(PRI[run][0][dj], np.full(len(dj), PRI[run][1])) for run in RUNS}
            bi, yb = panel.bank_rows(calR, js)
            n = len(bi)
            M1 = marginal_curves(f1['b'][bi], f1['s'][bi])
            S = {'Random': [int(i) for i in np.random.RandomState(100 + seed * 20 + js).permutation(n)[:T]]}
            s_ = []
            for _ in range(T):
                rem = [i for i in range(n) if i not in s_]
                s_.append(eig_pick(post_from(M1, yb, s_) if s_ else PRIOR.copy(), M1, rem))
            S['theta-EIG'] = s_
            S['2PL Fisher (abl.)'] = fluid_order(f2['a'][bi], f2['b'][bi], yb, T)
            for B in BP:
                for p in POL:
                    q = post_from(M1, yb, S[p][:B])
                    pD = MD[int(np.argmax(q))]
                    RES[p][B]['mae'].append(abs(float(pD.mean()) - SRD))
                    RES[p][B]['nll'].append(float(-np.mean(yD * np.log(pD + 1e-9) + (1 - yD) * np.log(1 - pD + 1e-9))))
                    for run in RUNS:
                        RES3[run][p][B].append(abs(float(MDR[run][int(np.argmax(q))].mean()) - SRD))
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
    print('paired delta MAE vs theta-EIG:')
    for p in ('Random', '2PL Fisher (abl.)'):
        print(f'  {p:18s} ' + '  '.join('B{}: {:+.4f} [{:+.4f},{:+.4f}]'.format(
            B, *paired_seed_boot(RES[p][B]['mae'], RES['theta-EIG'][B]['mae'], n_seeds=16, per_seed=6)) for B in BP))
    OUT.mkdir(exist_ok=True)
    json.dump({p: {str(B): {m: [float(x) for x in v] for m, v in d.items()} for B, d in bb.items()}
               for p, bb in RES.items()}, open(OUT / 'ups.json', 'w'))
    for (p, B, ref) in (('naive', 30, .1290), ('Random', 30, .1039), ('theta-EIG', 30, .1017),
                        ('theta-EIG', 55, .0946), ('theta-EIG', 110, .0884), ('2PL Fisher (abl.)', 30, .0983)):
        assert abs(np.mean(RES[p][B]['mae']) - ref) < .003, (p, B, np.mean(RES[p][B]['mae']))
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
