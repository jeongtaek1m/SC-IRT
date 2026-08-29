#!/usr/bin/env python3
"""Table 3B — UPS: unseen planner x unseen scenes.

The held-out planner runs 30 probes on the calibrated bank B; its
evaluation-scale ability is the MAP of the Rasch posterior on those probes;
its success rate on the unseen-type routes D (zero rollouts there) is
predicted through the feature path, marginalising the difficulty prior:

    P(y = 1 | x_D, B) = int sigmoid(theta_hat - b) N(b; b_tilde(x_D), tau^2) db

b_tilde = ridge(alpha = 100) on the SC-IRT descriptor stack fitted to the
calibration difficulties, tau = its residual SD. Probe policies compared
under the same readout: naive SR transfer (probe mean), Random, theta-EIG
under the evaluation model (canonical), and the UP localize rule (2PL
Fisher) — which brings no additional gain here because the quantity that
must generalise is the evaluation-scale ability (PROTOCOL section 4).

    python experiments/run_ups.py     # ~10 min, GPU
"""
import json
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
from scirt.acquisition import eig_pick, localize_cover
from scirt.metrics import paired_seed_boot

OUT = Path(__file__).resolve().parents[1] / 'results'
B_PROBE = 30
POL = ('Random', 'theta-EIG', 'Localize (2PL)')


def main():
    panel = Panel()
    RES = {p: {'mae': [], 'nll': []} for p in ('naive',) + POL}
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
            yD = np.array([panel.Y[(newR[i], js)] for i in dj], float)
            SRD = yD.mean()
            MD = marginal_curves(mu_D[dj], np.full(len(dj), tau))
            bi, yb = panel.bank_rows(calR, js)
            n = len(bi)
            M1 = marginal_curves(f1['b'][bi], f1['s'][bi])
            S = {'Random': list(np.random.RandomState(100 + seed * 20 + js).permutation(n)[:B_PROBE])}
            s_ = []
            for _ in range(B_PROBE):
                rem = [i for i in range(n) if i not in s_]
                s_.append(eig_pick(post_from(M1, yb, s_) if s_ else PRIOR.copy(), M1, rem))
            S['theta-EIG'] = s_
            S['Localize (2PL)'] = localize_cover(f2['a'][bi], f2['b'][bi], f2['th'], yb, K=B_PROBE, T=B_PROBE)
            for p in POL:
                q = post_from(M1, yb, S[p])
                pD = MD[int(np.argmax(q))]
                RES[p]['mae'].append(abs(pD.mean() - SRD))
                RES[p]['nll'].append(-np.mean(yD * np.log(pD + 1e-9) + (1 - yD) * np.log(1 - pD + 1e-9)))
            nv = yb[S['Random']].mean()
            RES['naive']['mae'].append(abs(nv - SRD))
            RES['naive']['nll'].append(-np.mean(yD * np.log(nv + 1e-9) + (1 - yD) * np.log(1 - nv + 1e-9)))
        print(f'seed {seed} done', flush=True)
    print(f'\n===== Table 3B: UPS, {B_PROBE} bank probes, zero rollouts on D ({len(RES["naive"]["mae"])} evaluations) =====')
    print(f'{"probe policy":16s} {"D-SR MAE":>9s} {"D NLL":>8s}')
    for p in ('naive',) + POL:
        print(f'{p:16s} {np.mean(RES[p]["mae"]):9.4f} {np.mean(RES[p]["nll"]):8.4f}')
    print('paired delta vs theta-EIG:')
    for p in ('Random', 'Localize (2PL)'):
        d, lo, hi = paired_seed_boot(RES[p]['mae'], RES['theta-EIG']['mae'])
        print(f'  {p:16s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]')
    OUT.mkdir(exist_ok=True)
    json.dump({k: {m: [float(x) for x in v] for m, v in d.items()} for k, d in RES.items()}, open(OUT / 'ups.json', 'w'))
    for p, ref in (('naive', .1282), ('Random', .1194), ('theta-EIG', .1034), ('Localize (2PL)', .1083)):
        assert abs(np.mean(RES[p]['mae']) - ref) < .003, (p, np.mean(RES[p]['mae']))
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
