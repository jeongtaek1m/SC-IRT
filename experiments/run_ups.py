#!/usr/bin/env python3
"""Table 3B — UPS: unseen planner x unseen scenes on the primary protocol.

The evaluation planner runs probes on the calibrated bank (budgets
B = {30, 55, 110}); its evaluation-scale ability is the MAP of the Rasch
posterior on those probes; its success rate on the evaluation-type routes D
(zero rollouts there) is predicted through the feature path with the
difficulty prior marginalised: P(y=1 | x, B) = int sigmoid(theta_hat - b)
N(b; ridge(x), tau^2) db. Probe policies: naive SR transfer, Random,
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
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirt.b2d import Panel
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


def main():
    panel = Panel()
    RES = {p: {B: {'mae': [], 'nll': []} for B in BP} for p in ('naive',) + POL}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, newR = panel.split_routes(ht)
        f1 = calibrate(panel.Y, calR, cols, mode='1pl')
        f2 = calibrate(panel.Y, calR, cols, mode='2pl')
        Z = np.vstack([panel.feat[r] for r in calR])
        m0, s0 = Z.mean(0), Z.std(0) + 1e-9
        rg = Ridge(alpha=100.).fit((Z - m0) / s0, f1['b'])
        tau = float(np.sqrt((f1['b'] - rg.predict((Z - m0) / s0)).var()))
        mu_D = np.array([rg.predict(((panel.feat[r] - m0) / s0)[None])[0] for r in newR])
        for js in hp:
            dj = [i for i, r in enumerate(newR) if (r, js) in panel.Y]
            if len(dj) < 4:
                continue
            yD = np.array([panel.Y[(newR[i], js)] for i in dj], float)
            SRD = float(yD.mean())
            MD = marginal_curves(mu_D[dj], np.full(len(dj), tau))
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
                nv = float(yb[S['Random'][:B]].mean())
                RES['naive'][B]['mae'].append(abs(nv - SRD))
                RES['naive'][B]['nll'].append(float(-np.mean(yD * np.log(nv + 1e-9) + (1 - yD) * np.log(1 - nv + 1e-9))))
        print(f'seed {seed} done', flush=True)
    nev = len(RES['naive'][30]['mae'])
    print(f'\n===== Table 3B: UPS, zero rollouts on D ({nev} evaluations) =====')
    print(f'{"probe policy":18s} ' + ' '.join(f'{"B=" + str(B) + " MAE":>10s} {"NLL":>7s}' for B in BP))
    for p in ('naive',) + POL:
        print(f'{p:18s} ' + ' '.join(f'{np.mean(RES[p][B]["mae"]):10.4f} {np.mean(RES[p][B]["nll"]):7.4f}' for B in BP))
    print('paired delta MAE vs theta-EIG:')
    for p in ('Random', '2PL Fisher (abl.)'):
        print(f'  {p:18s} ' + '  '.join('B{}: {:+.4f} [{:+.4f},{:+.4f}]'.format(
            B, *paired_seed_boot(RES[p][B]['mae'], RES['theta-EIG'][B]['mae'], n_seeds=16, per_seed=6)) for B in BP))
    OUT.mkdir(exist_ok=True)
    json.dump({p: {str(B): {m: [float(x) for x in v] for m, v in d.items()} for B, d in bb.items()}
               for p, bb in RES.items()}, open(OUT / 'ups.json', 'w'))
    for (p, B, ref) in (('naive', 30, .1290), ('Random', 30, .1099), ('theta-EIG', 30, .1087),
                        ('theta-EIG', 55, .1003), ('theta-EIG', 110, .0979), ('2PL Fisher (abl.)', 30, .1042)):
        assert abs(np.mean(RES[p][B]['mae']) - ref) < .003, (p, B, np.mean(RES[p][B]['mae']))
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
