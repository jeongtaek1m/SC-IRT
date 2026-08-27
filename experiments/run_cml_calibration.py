#!/usr/bin/env python3
"""Does a within-group (item-vs-item) estimator help under calibration scarcity?

The Rasch model contains a pure difficulty-vs-difficulty sub-model: for a
planner who passed exactly one of two routes, P(passed k, failed i) =
sigmoid(b_i - b_k) — theta cancels. Fitting b from those pairwise "votes"
is the conditional (pairwise) estimator, consistent in the number of items
regardless of how few planners there are, whereas the joint MAP we use
estimates theta and b together and is known to be biased with few raters
(our priors shrink that bias). This script swaps ONLY the bank estimator —
joint MAP vs pairwise CML (then theta by ML given b, same s_i formula, same
marginalised curves, same SRVar + SR-CI machine) — across J_cal in
{4, 7, 10, 13} on the Table 5 subsample streams.

Anchor: the joint-MAP arm must reproduce the Table 5 ours rows.
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
from scirt.curves import marginal_curves, sig, PRIOR
from scirt.bayes import post_from, sr_ci
from scirt.acquisition import srvar_pick
from scirt.metrics import mean_se, coverage_str, paired_seed_boot

np.random.seed(0)
torch.manual_seed(0)
EPS = 0.10
JCALS = (4, 7, 10, 13)
DEV = 'cuda'
OUT = Path(__file__).resolve().parents[1] / 'results'


def calibrate_cml(Y, routes, cols, it=800):
    """Pairwise conditional estimator for b, then theta by ML given b,
    theta-mean centring, and the same s_i formula as calibrate()."""
    n = len(routes)
    M = np.full((n, len(cols)), np.nan)
    for a, rid in enumerate(routes):
        for k, pi in enumerate(cols):
            if (rid, pi) in Y:
                M[a, k] = Y[(rid, pi)]
    mk = ~np.isnan(M)
    P = np.where(mk, np.nan_to_num(M), 0.0)          # pass indicator
    F = np.where(mk, 1.0 - np.nan_to_num(M), 0.0)    # fail indicator (0 where missing)
    N = torch.tensor(F @ P.T, dtype=torch.float32, device=DEV)   # N[i,k] = #planners: fail i, pass k
    b = torch.zeros(n, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([b], lr=0.05)
    for _ in range(it):
        diff = b[:, None] - b[None, :]
        loss = -(N * torch.nn.functional.logsigmoid(diff)).sum() / N.sum() + 1e-3 * b.pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    bb = b.detach()
    # theta by ML given b (same prior strength as the joint kernel)
    Mt = torch.tensor(np.nan_to_num(M), dtype=torch.float32, device=DEV)
    Wt = torch.tensor(mk, dtype=torch.float32, device=DEV)
    th = torch.zeros(len(cols), device=DEV, requires_grad=True)
    opt = torch.optim.Adam([th], lr=0.05)
    for _ in range(it):
        p = torch.sigmoid(th[None, :] - bb[:, None])
        nll = (-(Mt * torch.log(p + 1e-7) + (1 - Mt) * torch.log(1 - p + 1e-7)) * Wt).sum() / Wt.sum()
        (nll + 1e-2 * th.pow(2).mean()).backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        c = th.mean()
        out = dict(b=(bb - c).cpu().numpy(), th=(th - c).cpu().numpy())
    out['s'] = np.array([
        1 / np.sqrt(sum(sig(out['th'][k] - out['b'][i]) * (1 - sig(out['th'][k] - out['b'][i]))
                        for k in range(len(cols)) if mk[i, k]) + 1e-2) for i in range(n)])
    return out


def cat10(M, yy, rng):
    n = M.shape[1]
    SR = yy.mean()
    S, q = [], PRIOR.copy()
    for _ in range(min(120, n)):
        rem = [i for i in range(n) if i not in S]
        S.append(srvar_pick(M, q, rem))
        q = post_from(M, yy, S)
        lo, hi, m = sr_ci(M, yy, S, q, rng)
        if hi - lo <= 2 * EPS or len(S) >= min(120, n):
            return len(S), abs(m - SR), 1.0 if lo <= SR <= hi else 0.0


def main():
    panel = Panel()
    ARMS = ('joint-MAP', 'pairwise-CML')
    RES = {a: {k: {'n': [], 'err': [], 'cov': []} for k in JCALS} for a in ARMS}
    AGREE = {k: {'corr': [], 'rmse': [], 'sd_joint': [], 'sd_cml': [], 's_joint': [], 's_cml': []} for k in JCALS}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, _ = panel.split_routes(ht)
        for Jc in JCALS:
            if Jc == 13:
                cs = cols
            else:
                rs = np.random.RandomState(9000 + seed * 100 + Jc * 10 + 0)
                cs = sorted(np.array(cols)[rs.choice(len(cols), Jc, replace=False)].tolist())
            fj = calibrate(panel.Y, calR, cs, mode='1pl')
            fc = calibrate_cml(panel.Y, calR, cs)
            d = fc['b'] - fj['b']
            AGREE[Jc]['corr'].append(float(np.corrcoef(fj['b'], fc['b'])[0, 1]))
            AGREE[Jc]['rmse'].append(float(np.sqrt(((d - d.mean()) ** 2).mean())))
            AGREE[Jc]['sd_joint'].append(float(fj['b'].std()))
            AGREE[Jc]['sd_cml'].append(float(fc['b'].std()))
            AGREE[Jc]['s_joint'].append(float(fj['s'].mean()))
            AGREE[Jc]['s_cml'].append(float(fc['s'].mean()))
            for js in hp:
                bi, yy = panel.bank_rows(calR, js)
                for a, f in (('joint-MAP', fj), ('pairwise-CML', fc)):
                    M = marginal_curves(f['b'][bi], f['s'][bi])
                    n_, err, cov = cat10(M, yy, np.random.RandomState(13))
                    RES[a][Jc]['n'].append(n_)
                    RES[a][Jc]['err'].append(err)
                    RES[a][Jc]['cov'].append(cov)
        print(f'seed {seed} done', flush=True)

    print('\n===== bank estimator: joint MAP vs pairwise CML (SRVar + SR-CI, +-10%, 48) =====')
    for Jc in JCALS:
        for a in ARMS:
            dd = RES[a][Jc]
            n_, ns = mean_se(dd['n'])
            print(f'  J{Jc:2d} {a:13s} {n_:5.1f}+-{ns:.1f}  MAE {np.mean(dd["err"]):.4f}  cov {coverage_str(dd["cov"])}')
        dm, lo, hi = paired_seed_boot(RES['pairwise-CML'][Jc]['err'], RES['joint-MAP'][Jc]['err'])
        dc, lc, hc = paired_seed_boot(RES['pairwise-CML'][Jc]['cov'], RES['joint-MAP'][Jc]['cov'])
        dn, ln, hn = paired_seed_boot(RES['pairwise-CML'][Jc]['n'], RES['joint-MAP'][Jc]['n'])
        print(f'        CML-joint: dMAE {dm:+.4f} [{lo:+.4f},{hi:+.4f}]  dcov {dc:+.3f} [{lc:+.3f},{hc:+.3f}]  droll {dn:+.1f} [{ln:+.1f},{hn:+.1f}]')
    print('\n[b-hat agreement / dispersion per J_cal]')
    for Jc in JCALS:
        A = AGREE[Jc]
        print(f'  J{Jc:2d}: corr {np.mean(A["corr"]):.3f}  aligned RMSE {np.mean(A["rmse"]):.3f}  '
              f'sd(b) joint {np.mean(A["sd_joint"]):.2f} / CML {np.mean(A["sd_cml"]):.2f}  '
              f'mean s_i joint {np.mean(A["s_joint"]):.2f} / CML {np.mean(A["s_cml"]):.2f}')
    OUT.mkdir(exist_ok=True)
    json.dump({'res': {a: {str(k): {kk: list(map(float, RES[a][k][kk])) for kk in RES[a][k]} for k in JCALS} for a in RES},
               'agree': {str(k): {kk: list(map(float, AGREE[k][kk])) for kk in AGREE[k]} for k in JCALS}},
              open(OUT / 'cml_calibration.json', 'w'))
    j = RES['joint-MAP']
    assert abs(np.mean(j[13]['n']) - 29.0) < 0.2 and int(sum(j[13]['cov'])) == 48
    assert abs(np.mean(j[4]['err']) - 0.0630) < 0.004
    print('anchors OK')


if __name__ == '__main__':
    main()
