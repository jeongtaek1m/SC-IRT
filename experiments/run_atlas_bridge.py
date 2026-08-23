#!/usr/bin/env python3
"""Table 4(e) — ATLAS tau <-> ours eps correspondence (native-vs-native).

ATLAS runs as published on our bank: 3PL calibration, top-5 randomesque
Fisher selection, SE(theta) <= tau stopping for tau in {0.3, 0.2, 0.1} with a
minimum of 30 items (its max-500 cap never binds on a 180-item bank — the
bank itself is the cap, recorded as pool exhaustion). For each tau we also
measure the SR-scale credible half-width its theta-precision actually
delivers. Ours records the posterior SE(theta) at its SR +-eps stops — the
reverse mapping.

Anchors: tau=0.3 -> 63.7 rollouts / MAE .0355 / cov 0.77; tau=0.1 -> 100%
pool exhaustion; ours +-10% -> SE(theta) 0.405, +-5% -> 0.278.
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
from scirt.curves import marginal_curves, point_curves_3pl, PRIOR
from scirt.bayes import post_from, sr_ci, theta_sd
from scirt.acquisition import srvar_pick, atlas_3pl_pick
from scirt.metrics import mean_se, coverage_str

np.random.seed(0)
torch.manual_seed(0)
TAUS = [0.3, 0.2, 0.1]
MINI = 30
EPSS = (0.10, 0.05)
OUT = Path(__file__).resolve().parents[1] / 'results'


def main():
    panel = Panel()
    ATL = {t: {'n': [], 'err': [], 'hw': [], 'cov': [], 'exh': []} for t in TAUS}
    OURS = {e: {'n': [], 'sd': [], 'err': [], 'cov': []} for e in EPSS}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        f1 = calibrate(panel.Y, calR, cols, mode='1pl')
        f3 = calibrate(panel.Y, calR, cols, mode='3pl')
        for js in hp:
            bi, yy = panel.bank_rows(calR, js)
            n = len(bi)
            SR = yy.mean()
            # ---- ATLAS native: 3PL curves, randomesque, SE(theta) stop ----
            a3, b3, c3 = f3['a'][bi], f3['b'][bi], f3['cc'][bi]
            M3 = point_curves_3pl(a3, b3, c3)
            rngA = np.random.RandomState(1)
            rngd = np.random.RandomState(11)
            S, q, done = [], PRIOR.copy(), set()
            for _ in range(n):
                rem = [i for i in range(n) if i not in S]
                S.append(atlas_3pl_pick(q, a3, b3, c3, rem, rngA))
                q = post_from(M3, yy, S)
                sd = theta_sd(q)
                exhausted = len(S) >= n
                for t in TAUS:
                    if t not in done and ((sd <= t and len(S) >= MINI) or exhausted):
                        lo, hi, m = sr_ci(M3, yy, S, q, rngd)
                        ATL[t]['n'].append(len(S))
                        ATL[t]['err'].append(abs(m - SR))
                        ATL[t]['hw'].append((hi - lo) / 2)
                        ATL[t]['cov'].append(1.0 if lo <= SR <= hi else 0.0)
                        ATL[t]['exh'].append(1.0 if (exhausted and sd > t) else 0.0)
                        done.add(t)
                if len(done) == len(TAUS):
                    break
            # ---- ours SRVar: record SE(theta) at the SR-eps stops ---------
            Mf = marginal_curves(f1['b'][bi], f1['s'][bi])
            rngo = np.random.RandomState(13)
            S, q, rec = [], PRIOR.copy(), {}
            for _ in range(min(120, n)):
                rem = [i for i in range(n) if i not in S]
                S.append(srvar_pick(Mf, q, rem))
                q = post_from(Mf, yy, S)
                lo, hi, m = sr_ci(Mf, yy, S, q, rngo)
                sd = theta_sd(q)
                for e in EPSS:
                    if e not in rec and (hi - lo <= 2 * e or len(S) >= min(120, n)):
                        rec[e] = 1
                        OURS[e]['n'].append(len(S))
                        OURS[e]['sd'].append(sd)
                        OURS[e]['err'].append(abs(m - SR))
                        OURS[e]['cov'].append(1.0 if lo <= SR <= hi else 0.0)
                if len(rec) == 2:
                    break
        print(f'seed {seed} done', flush=True)

    print('\n===== Table 4(e) — ATLAS-native SE(theta) <= tau (3PL, min30) =====')
    for t in TAUS:
        d = ATL[t]
        n_, ns = mean_se(d['n'])
        print(f'  tau={t}: {n_:6.1f}+-{ns:.1f} rollouts (exhausted {np.mean(d["exh"]):.0%})  '
              f'SR-MAE {np.mean(d["err"]):.4f}  SR-CI half-width {np.mean(d["hw"]):.3f}  '
              f'cov {coverage_str(d["cov"])}')
    print('\n===== ours SRVar — SE(theta) at the SR-eps stop =====')
    for e in EPSS:
        d = OURS[e]
        sd_, sds = mean_se(d['sd'])
        print(f'  +-{e:.0%}: {np.mean(d["n"]):5.1f} rollouts  SE(theta) {sd_:.3f}+-{sds:.3f}  '
              f'MAE {np.mean(d["err"]):.4f}  cov {coverage_str(d["cov"])}')

    OUT.mkdir(exist_ok=True)
    json.dump({'atl': {str(t): {k: list(map(float, ATL[t][k])) for k in ATL[t]} for t in TAUS},
               'ours': {str(e): {k: list(map(float, OURS[e][k])) for k in OURS[e]} for e in EPSS}},
              open(OUT / 'atlas_tau_bridge.json', 'w'))

    assert abs(np.mean(OURS[0.10]['n']) - 29.0) < 0.2
    assert abs(np.mean(OURS[0.10]['sd']) - 0.405) < 0.01
    assert abs(np.mean(ATL[0.3]['n']) - 63.7) < 1.0
    assert np.mean(ATL[0.1]['exh']) == 1.0
    print('anchors OK')


if __name__ == '__main__':
    main()
