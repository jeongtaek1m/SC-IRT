#!/usr/bin/env python3
"""Table 6 — UPS: the US x UP composition (unseen planner x unseen scenes).

Predict a held-out planner's success rate on routes whose scenario types have
zero calibration responses:  P(y | B, x) = integral sigmoid(theta - b)
dp(theta | B) dp(b | x).  theta comes from B = 30 bank probes (theta-EIG
selection — the probe's purpose is theta transport, so the SR-variance rule
does not apply there), the difficulty prior from the feature path
(ridge b_tilde with residual tau).

Sections
  (a) extend decomposition on a common Rasch scale: amortisation gap vs
      theta-transport gap (+ theta shift and its SE)
  (b) composition baselines: naive SR transfer and random-B composition
  (c) hybrid: warm-start posterior + D-probe recovery curve (D = 0..20)
  (d) standalone stress test (bank = the approx. 40 unseen-type routes,
      SRVar acquisition), and the pre-revision EIG arm as its ablation
  (e) the posterior-a UP variant (2PL bank + marginalisation, EIG) — the
      appendix row of Table 3

Anchors: (a) .1034/.1005/.0350, (b) .1259/.1206/.1035, (c) .1036/.0832/
.0654/.0470, (d) SRVar 21.0/.0495/46 and 30.5/.0254/45, (e) 24.7/67.6.
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
from scirt.calibration import calibrate, frozen_b
from scirt.curves import marginal_curves, THG, PRIOR
from scirt.bayes import post_from, sr_ci
from scirt.acquisition import srvar_pick, eig_pick
from scirt.metrics import mean_se, coverage_str

np.random.seed(0)
torch.manual_seed(0)
EPSS = [0.10, 0.05]
B_WARM = 30
DCK = [0, 5, 10, 20]
NDRAW = 4000
OUT = Path(__file__).resolve().parents[1] / 'results'


def cat_pool(Mf, yy, pick, rng, maxb):
    """Adaptive loop on a small bank; records pool-exhaustion fraction."""
    n = Mf.shape[1]
    SR = yy.mean()
    S, q, out = [], PRIOR.copy(), {}
    for _ in range(min(maxb, n)):
        rem = [i for i in range(n) if i not in S]
        S.append(pick(q, rem))
        q = post_from(Mf, yy, S)
        lo, hi, m = sr_ci(Mf, yy, S, q, rng)
        exhausted = len(S) >= min(maxb, n)
        for e in EPSS:
            if e not in out and (hi - lo <= 2 * e or exhausted):
                out[e] = (len(S), abs(m - SR), 1.0 if lo <= SR <= hi else 0.0, len(S) / n)
        if len(out) == len(EPSS):
            break
    for e in EPSS:
        if e not in out:
            out[e] = (len(S), abs(m - SR), 0.0, len(S) / n)
    return out


def main():
    panel = Panel()
    VAR = {e: {'n': [], 'err': [], 'cov': []} for e in EPSS}          # (e) posterior-a UP
    STD = {m: {e: {'n': [], 'err': [], 'cov': [], 'pool': []} for e in EPSS}
           for m in ('EIG', 'SRVar')}                                  # (d)
    DEC = {k: [] for k in ('e_ours', 'e_ob', 'e_ot', 'shift', 'seD')}  # (a)
    HYB = {d: {'err': [], 'cov': []} for d in DCK}                     # (c)
    UPSC = {m: [] for m in ('SR-transfer', 'random-B', 'ours(EIG-B)')} # (b)
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, newR = panel.split_routes(ht)
        f2 = calibrate(panel.Y, calR, cols, mode='2pl')
        f1 = calibrate(panel.Y, calR, cols, mode='1pl')
        Z = np.vstack([panel.feat[r] for r in calR])
        m0, s0 = Z.mean(0), Z.std(0) + 1e-9
        rg = Ridge(alpha=100.).fit((Z - m0) / s0, f1['b'])
        tau = float(np.sqrt((f1['b'] - rg.predict((Z - m0) / s0)).var()))
        mu_D = np.array([rg.predict(((panel.feat[r] - m0) / s0)[None])[0] for r in newR])
        bC, sC = frozen_b(panel.Y, newR, cols, f1['th'])
        for js in hp:
            # ---- (e) posterior-a UP variant (2PL bank + marginalisation) --
            bi, yb = panel.bank_rows(calR, js)
            Mb2 = marginal_curves(f2['b'][bi], f2['s'][bi], a=f2['a'][bi])

            def pk_eig2(q, rem, _M=Mb2):
                return eig_pick(q, _M, rem)

            o = cat_pool(Mb2, yb, pk_eig2, np.random.RandomState(7), 120)
            for e in EPSS:
                VAR[e]['n'].append(o[e][0])
                VAR[e]['err'].append(o[e][1])
                VAR[e]['cov'].append(o[e][2])
            # ---- (d) standalone: newR bank (1PL b_tilde + tau) ------------
            dj = [i for i, r in enumerate(newR) if (r, js) in panel.Y]
            if len(dj) < 4:
                continue
            yD = np.array([panel.Y[(newR[i], js)] for i in dj], float)
            nD = len(dj)
            SRD = yD.mean()
            M_ours = marginal_curves(mu_D[dj], np.full(nD, tau))

            def pk_eig_o(q, rem, _M=M_ours):
                return eig_pick(q, _M, rem)

            def pk_sv_o(q, rem, _M=M_ours):
                return srvar_pick(_M, q, rem)

            for m, pick, rs in (('EIG', pk_eig_o, 7), ('SRVar', pk_sv_o, 13)):
                o = cat_pool(M_ours, yD, pick, np.random.RandomState(rs), nD)
                for e in EPSS:
                    STD[m][e]['n'].append(o[e][0])
                    STD[m][e]['err'].append(o[e][1])
                    STD[m][e]['cov'].append(o[e][2])
                    STD[m][e]['pool'].append(o[e][3])
            # ---- (a) extend decomposition + (c) hybrid --------------------
            Mb1 = marginal_curves(f1['b'][bi], f1['s'][bi])
            S = []
            for _ in range(B_WARM):
                rem = [i for i in range(len(bi)) if i not in S]
                S.append(eig_pick(post_from(Mb1, yb, S), Mb1, rem))
            qB = post_from(Mb1, yb, S)
            M_oC = marginal_curves(bC[dj], sC[dj])
            qD = post_from(M_oC, yD, list(range(nD)))
            thB = float((qB * THG).sum())
            thD = float((qD * THG).sum())
            DEC['shift'].append(thB - thD)
            DEC['seD'].append(float(np.sqrt((qD * (THG - thD) ** 2).sum())))
            rng = np.random.RandomState(500 + seed * 20 + js)
            for key, q_, Md in (('e_ours', qB, M_ours), ('e_ob', qB, M_oC), ('e_ot', qD, M_oC)):
                ti = rng.choice(len(THG), size=NDRAW, p=q_)
                sr = ((rng.random((NDRAW, nD)) < Md[ti]).sum(1)) / nD
                DEC[key].append(abs(float(sr.mean()) - SRD))
            S2b = []
            for dq in range(max(DCK) + 1):
                qH = post_from(M_ours, yD, S2b, prior=qB)
                if dq in DCK:
                    un = [i for i in range(nD) if i not in S2b]
                    ti = rng.choice(len(THG), size=NDRAW, p=qH)
                    mm = M_ours[ti][:, un] if un else np.zeros((NDRAW, 0))
                    sr = (yD[S2b].sum() + (rng.random(mm.shape) < mm).sum(1)) / nD
                    m_, lo, hi = float(sr.mean()), np.percentile(sr, 2.5), np.percentile(sr, 97.5)
                    HYB[dq]['err'].append(abs(m_ - SRD))
                    HYB[dq]['cov'].append(1.0 if lo <= SRD <= hi else 0.0)
                rem = [i for i in range(nD) if i not in S2b]
                if rem and dq < max(DCK):
                    S2b.append(eig_pick(qH, M_ours, rem))
            # ---- (b) composition baselines (B = 30, zero D rollouts) ------
            yy = yb
            Mf1 = Mb1
            q, S = PRIOR.copy(), []
            for _ in range(30):
                rem = [i for i in range(len(bi)) if i not in S]
                S.append(eig_pick(q, Mf1, rem))
                q = post_from(Mf1, yy, S)
            UPSC['SR-transfer'].append(abs(yy[S].mean() - SRD))
            est = float(((M_ours * q[:, None]).sum(0)).mean())
            UPSC['ours(EIG-B)'].append(abs(est - SRD))
            rngb = np.random.RandomState(700 + seed * 20 + js)
            Sr = list(rngb.permutation(len(bi))[:30])
            qr = post_from(Mf1, yy, Sr)
            UPSC['random-B'].append(abs(float(((M_ours * qr[:, None]).sum(0)).mean()) - SRD))
        print(f'seed {seed} done', flush=True)

    print('\n===== Table 6 — UPS (unified split) =====')
    print('(e) posterior-a UP variant (2PL + marginalisation, EIG):')
    for e in EPSS:
        d = VAR[e]
        n_, ns = mean_se(d['n'])
        print(f'  +-{e:.0%}: {n_:5.1f}+-{ns:.1f}  MAE {np.mean(d["err"]):.4f}  cov {coverage_str(d["cov"])}')
    print('(d) standalone (newR bank, b_tilde + tau):')
    for m in ('SRVar', 'EIG'):
        for e in EPSS:
            d = STD[m][e]
            n_, ns = mean_se(d['n'])
            print(f'  {m:6s} +-{e:.0%}: {n_:5.1f}+-{ns:.1f} (pool {np.mean(d["pool"]):.0%})  '
                  f'MAE {np.mean(d["err"]):.4f}  cov {coverage_str(d["cov"])}')
    eo, _ = mean_se(DEC['e_ours'])
    eb, _ = mean_se(DEC['e_ob'])
    et, _ = mean_se(DEC['e_ot'])
    print(f'(a) decomposition: Err(thB,b~) {eo:.4f} / Err(thB,bC) {eb:.4f} / Err(thD,bC) {et:.4f}')
    print(f'    amortisation gap {eo - eb:+.4f} vs transport gap {eb - et:+.4f}')
    print(f'    |thB-thD| {np.abs(DEC["shift"]).mean():.2f}, SE(thD) {np.mean(DEC["seD"]):.2f}')
    print('(b) composition baselines (B=30, zero D rollouts):')
    for m in UPSC:
        e_, se = mean_se(UPSC[m])
        print(f'  {m:12s} |SR err| = {e_:.4f}+-{se:.4f}')
    print('(c) hybrid (B=30 warm + D probes):')
    for d in DCK:
        e_, _ = mean_se(HYB[d]['err'])
        print(f'  D={d:2d}: |err| {e_:.4f}  cov {coverage_str(HYB[d]["cov"])}')

    OUT.mkdir(exist_ok=True)
    json.dump({'var': {str(e): {k: list(map(float, VAR[e][k])) for k in VAR[e]} for e in EPSS},
               'std': {m: {str(e): {k: list(map(float, STD[m][e][k])) for k in STD[m][e]}
                           for e in EPSS} for m in STD},
               'dec': {k: list(map(float, DEC[k])) for k in DEC},
               'hyb': {str(d): {k: list(map(float, HYB[d][k])) for k in HYB[d]} for d in DCK},
               'upsc': {m: list(map(float, UPSC[m])) for m in UPSC}},
              open(OUT / 'ups.json', 'w'))

    assert abs(np.mean(VAR[0.10]['n']) - 24.7) < 0.2
    assert abs(np.mean(STD['SRVar'][0.10]['n']) - 21.0) < 0.2
    assert abs(np.mean(STD['EIG'][0.10]['n']) - 21.2) < 0.2
    assert abs(np.mean(DEC['e_ours']) - 0.1034) < 0.002
    assert abs(np.mean(DEC['e_ot']) - 0.0350) < 0.002
    assert abs(np.mean(HYB[20]['err']) - 0.0470) < 0.002
    assert abs(np.mean(UPSC['ours(EIG-B)']) - 0.1035) < 0.002
    print('anchors OK')


if __name__ == '__main__':
    main()
