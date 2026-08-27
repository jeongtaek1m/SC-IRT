#!/usr/bin/env python3
"""Appendix: residual-interaction (multidimensional) IRT, dimension sweep.

    logit P(y_ij = 1) = theta_j - mu_i + u_j^T v_i,   u_j, v_i in R^d,  sum_j u_j = 0

For d in {1, 2, 4, 8} and prior strength lam in {0.05, 0.1, 0.2, 0.4} (weak
per-parameter L2 on U and V; SVD warm start from Fisher working residuals),
on the unified split, cell-level on the held-out-type block C:

  in-bank sd(u^T v)      how much interaction the fit deploys
  excess reliability     Procrustes-aligned split-half reliability of U
                         across two disjoint route halves, minus a row-shuffle
                         null (planner rows permuted) — is the structure real?
  oracle gain            cell AUROC / NLL when (mu, v) are fitted from the
                         C responses themselves (theta, U frozen) vs the
                         scalar oracle: the ceiling of what a vector could buy
  amortised gain         the deployable version: ridge x -> (mu~, v~) vs
                         ridge x -> b~
  G2                     Spearman(v~, v_oracle) per dimension (mean over dims)

The d = 2, lam = 0.1 row is the paper's canonical negative result (anchors).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirt.b2d import Panel
from scirt.splits import unified_split, R_DRAWS
from scirt.curves import sig

np.random.seed(0)
torch.manual_seed(0)
DEV = 'cuda'
DS = (1, 2, 4, 8)
LAMS = (0.05, 0.1, 0.2, 0.4)
N_SPLIT = 4
OUT = Path(__file__).resolve().parents[1] / 'results'


def calib_int(Y0, MK, rows, cols, d, lam, interaction=True):
    M = torch.tensor(Y0[np.ix_(rows, cols)], dtype=torch.float32).to(DEV)
    W = torch.tensor(MK[np.ix_(rows, cols)].astype(np.float32)).to(DEV)
    n, Jp = len(rows), len(cols)
    bb = torch.zeros(n, device=DEV, requires_grad=True)
    th = torch.zeros(Jp, device=DEV, requires_grad=True)
    opt0 = torch.optim.Adam([bb, th], lr=0.05)
    for _ in range(800):
        p = torch.sigmoid(th[None, :] - bb[:, None])
        nll = (-(M * torch.log(p + 1e-7) + (1 - M) * torch.log(1 - p + 1e-7)) * W).sum() / W.sum()
        (nll + 1e-2 * th.pow(2).mean() + 1e-3 * bb.pow(2).mean()).backward()
        opt0.step()
        opt0.zero_grad()
    if not interaction:
        with torch.no_grad():
            c = th.mean()
            return (bb - c).cpu().numpy(), (th - c).cpu().numpy(), None, None
    with torch.no_grad():
        p = torch.sigmoid(th[None, :] - bb[:, None])
        R = ((M - p) / torch.clamp(p * (1 - p), min=0.05)) * W
        Pm, Sv, Qt = torch.linalg.svd(R, full_matrices=False)
        sc = torch.sqrt(Sv[:d] / np.sqrt(n * Jp))
        V0 = Pm[:, :d] * sc[None, :]
        U0 = (Qt[:d, :].T) * sc[None, :]
    U = (0.5 * U0).clone().requires_grad_(True)
    V = (0.5 * V0).clone().requires_grad_(True)
    opt = torch.optim.Adam([bb, th, U, V], lr=0.05)
    for _ in range(1500):
        p = torch.sigmoid(th[None, :] - bb[:, None] + V @ U.T)
        nll = (-(M * torch.log(p + 1e-7) + (1 - M) * torch.log(1 - p + 1e-7)) * W).sum() / W.sum()
        (nll + 1e-2 * th.pow(2).mean() + 1e-3 * bb.pow(2).mean()
         + lam * U.pow(2).mean() + lam * V.pow(2).mean()).backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        Un, Vn = U.cpu().numpy(), V.cpu().numpy()
        thn, bbn = th.cpu().numpy(), bb.cpu().numpy()
    vm = Vn.mean(0)
    thn = thn + Un @ vm
    Vn = Vn - vm
    um = Un.mean(0)
    bbn = bbn - Vn @ um
    Un = Un - um
    c = thn.mean()
    return bbn - c, thn - c, Un, Vn


def fit_scene(y, mask_j, th, lam, U=None, it=60):
    k = 1 + (0 if U is None else U.shape[1])
    beta = np.zeros(k)
    X = np.hstack([-np.ones((mask_j.sum(), 1))] + ([U[mask_j]] if U is not None else []))
    off, yy = th[mask_j], y[mask_j]
    P0 = np.diag([1e-3] + [lam * 10] * (k - 1))
    for _ in range(it):
        p = sig(X @ beta + off)
        g = X.T @ (yy - p) - P0 @ beta
        H = (X * (p * (1 - p))[:, None]).T @ X + P0 + 1e-6 * np.eye(k)
        beta = np.clip(beta + np.linalg.solve(H, g), -6, 6)
    return beta


def proc_corr(A, B):
    W_, _, Vt_ = np.linalg.svd(A.T @ B)
    R = W_ @ Vt_
    return float(np.corrcoef((A @ R).ravel(), B.ravel())[0, 1])


def nll_of(y, p):
    p = np.clip(np.array(p), 1e-9, 1 - 1e-9)
    y = np.array(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    panel = Panel()
    Y0, MK = panel.dense()
    N, allr, sn = len(panel.allr), panel.allr, panel.sn
    feat = panel.feat
    # ---- scalar arms (shared by every (d, lam)) ------------------------
    SC = {'oracle': ([], []), 'amort': ([], [])}
    scal = {}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        tr = [i for i in range(N) if sn[allr[i]] not in ht]
        te = [i for i in range(N) if sn[allr[i]] in ht]
        b1, th1, _, _ = calib_int(Y0, MK, tr, cols, 1, 0.1, interaction=False)
        Z = np.vstack([feat[allr[i]] for i in tr])
        m0, s0 = Z.mean(0), Z.std(0) + 1e-9
        Ztr = (Z - m0) / s0
        Zte = (np.vstack([feat[allr[i]] for i in te]) - m0) / s0
        bt = Ridge(alpha=100.).fit(Ztr, b1).predict(Zte)
        for k, i in enumerate(te):
            mj, yy = MK[i, cols], Y0[i, cols]
            js = np.where(mj)[0]
            bo = fit_scene(yy, mj, th1, 0.1)[0]
            SC['oracle'][0].extend(yy[js]); SC['oracle'][1].extend(sig(th1[js] - bo))
            SC['amort'][0].extend(yy[js]); SC['amort'][1].extend(sig(th1[js] - bt[k]))
        scal[seed] = (cols, tr, te, Ztr, Zte)
    sc_or = (roc_auc_score(*SC['oracle']), nll_of(*SC['oracle']))
    sc_am = (roc_auc_score(*SC['amort']), nll_of(*SC['amort']))
    print(f'scalar: oracle AUROC {sc_or[0]:.4f} NLL {sc_or[1]:.4f} | amort AUROC {sc_am[0]:.4f} NLL {sc_am[1]:.4f}', flush=True)

    ROWS = []
    for d in DS:
        for lam in LAMS:
            CELL = {'oracle': ([], []), 'amort': ([], [])}
            SD, VO, VP = [], [], []
            for seed in range(R_DRAWS):
                cols, tr, te, Ztr, Zte = scal[seed]
                bI, thI, U_, V_ = calib_int(Y0, MK, tr, cols, d, lam)
                SD.append(float((V_ @ U_.T).std()))
                mt = Ridge(alpha=100.).fit(Ztr, bI).predict(Zte)
                Vt = np.stack([Ridge(alpha=100.).fit(Ztr, V_[:, k_]).predict(Zte) for k_ in range(d)], 1)
                for k, i in enumerate(te):
                    mj, yy = MK[i, cols], Y0[i, cols]
                    js = np.where(mj)[0]
                    mo = fit_scene(yy, mj, thI, lam, U_)
                    CELL['oracle'][0].extend(yy[js]); CELL['oracle'][1].extend(sig(thI[js] - mo[0] + U_[js] @ mo[1:]))
                    CELL['amort'][0].extend(yy[js]); CELL['amort'][1].extend(sig(thI[js] - mt[k] + U_[js] @ Vt[k]))
                    VO.append(mo[1:]); VP.append(Vt[k])
            VO, VP = np.array(VO), np.array(VP)
            g2 = float(np.mean([spearmanr(VP[:, k_], VO[:, k_]).correlation for k_ in range(d)]))
            # split-half reliability of U on the full panel, Procrustes-aligned, minus row-shuffle null
            rng = np.random.RandomState(0)
            rels, nulls = [], []
            for _ in range(N_SPLIT):
                perm = rng.permutation(N)
                half = N // 2
                Us = [calib_int(Y0, MK, list(part), list(range(panel.J)), d, lam)[2]
                      for part in (perm[:half], perm[half:])]
                rels.append(proc_corr(Us[0], Us[1]))
                nulls.append(np.mean([proc_corr(Us[0], Us[1][rng.permutation(panel.J)]) for _ in range(8)]))
            row = dict(d=d, lam=lam, sd_uv=float(np.mean(SD)), rel=float(np.mean(rels)), null=float(np.mean(nulls)),
                       auroc_or=roc_auc_score(*CELL['oracle']), nll_or=nll_of(*CELL['oracle']),
                       auroc_am=roc_auc_score(*CELL['amort']), nll_am=nll_of(*CELL['amort']),
                       g2=g2, sd_vp=float(VP.std()), sd_vo=float(VO.std()))
            ROWS.append(row)
            print(f'd={d} lam={lam:.2f}: sd(u.v) {row["sd_uv"]:.3f}  rel {row["rel"]:+.3f} (null {row["null"]:+.3f}, excess {row["rel"]-row["null"]:+.3f})  '
                  f'oracle AUROC {row["auroc_or"]:.4f} ({row["auroc_or"]-sc_or[0]:+.4f}) NLL {row["nll_or"]:.4f}  '
                  f'amort AUROC {row["auroc_am"]:.4f} ({row["auroc_am"]-sc_am[0]:+.4f})  G2 rho {g2:+.3f}', flush=True)

    OUT.mkdir(exist_ok=True)
    json.dump({'scalar': {'oracle': list(sc_or), 'amort': list(sc_am)}, 'rows': ROWS}, open(OUT / 'mirt_dsweep.json', 'w'))
    can = [r for r in ROWS if r['d'] == 2 and r['lam'] == 0.1][0]
    assert abs(sc_or[0] - 0.8761) < 0.003 and abs(sc_am[0] - 0.7604) < 0.003
    assert abs(can['auroc_or'] - 0.9094) < 0.004 and abs(can['auroc_am'] - 0.7610) < 0.004
    print('anchors OK')


if __name__ == '__main__':
    main()
