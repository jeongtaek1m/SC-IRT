#!/usr/bin/env python3
"""Three AV-testing baselines on the Table 1 protocol: FST (no public code), the GP adaptive sampling of Gong et al.
(through its official code, gpo_scene, and as our port, gp_*) and kernel test case sampling (ktcs_scene, from the
paper's Methods; see run_ktcs).

fst_*   Few-Shot Testing (Li, He, Yang, Hu, Zhang, Feng; IEEE T-ITS 2025, arXiv 2409.14369).  A FIXED test set of
        n = B routes and aggregation weights, chosen before the new planner is seen, from the K_cal calibration
        planners used as the surrogate vehicle set: cross-attention similarity network (MLP features, reciprocal
        L2 attention, softmax over the selected scenarios for every scenario of the space), weight of a selected
        route = the similarity mass of the whole bank it collects (Eq. 13-16), loss = the max over surrogates of
        |weighted estimate - true mean| (Eq. 17 / 20) plus the fluctuation term of Eq. 22-24 with w_M = 1 (the
        value used in the paper's experiments; under the query-wise normalisation of Eq. 14 that term equals the
        surrogate's own error, so it is the mean surrogate error here), training sets drawn from k-means clusters
        of the surrogate performance (the paper's critical distribution P_c), one network per cell shared by all
        budgets (as the paper reuses one network for n = 5, 10, 20), then the set optimised per budget.
        Deviation: the paper optimises continuous scenario coordinates by gradient descent; a route bank is
        discrete, so the set is optimised by best-improvement swap search under the same loss from the best of
        32 P_c draws.  fst_scene feeds the network the route descriptor (the paper's scenario state: the 25-d
        kinematics + 48-d risk descriptors of the US baselines), fst_resp the surrogates' response profile.
gp_*    Adaptive sampling with a Gaussian-process surrogate (Gong, Feng, Pan; IEEE T-ITS 2023, arXiv 2210.14114),
        single-fidelity method: GP regression on the new planner's outcomes over a route feature space, next
        route = argmax of the benefit B = U(D) - U(D, z~) (Eq. 8-12: the reduction of the variance bound of the
        accident rate when a hypothetical sample at the current mean is added), RBF kernel with hyperparameters
        by marginal likelihood at every step, n_init random routes first, then sequential selection.
        Deviations: the outcome is binary (regression on {0, 1} with delta = 0.5 and a constant prior mean),
        so an unexecuted route's success probability is that of its predictive outcome (latent + noise
        variance; the latent alone would call a constant-plus-noise fit certain); the candidates are the
        unexecuted routes (argmax instead of continuous optimisation); executed routes enter the estimate with
        their observed outcome and drop out of the variance bound (the paper's f is noise-free, where the two
        coincide).
        gp_scene uses the route descriptor, gp_resp the surrogates' response profile.

gpo_scene  the OFFICIAL code of Gong et al. (github.com/umbrellagong/MFGPreliability, commit 604bfce; see
        run_gp_official) on the route descriptor, unmodified: its discrete-grid acquisition, its GP and its readout.

Same draws, K_cal subsamples, banks and budgets as run_up_frontier.py (experiments/official/data.py), so every
cell is paired with the ATDrive cell; --merge prints SR-MAE, the paired delta against ATDrive and the pairwise
ranking accuracy of every cell and writes results/up_avbase.json.

    $P experiments/run_av_baselines.py --methods fst_scene fst_resp --seeds 0 4     # shard
    $P experiments/run_av_baselines.py --methods gp_scene gp_resp --seeds 0 4
    OMP_NUM_THREADS=2 $P experiments/run_av_baselines.py --methods gpo_scene --seeds 0 2   # official code, CPU
    $P experiments/run_av_baselines.py --methods ktcs_scene --seeds 0 2
    $P experiments/run_av_baselines.py --merge
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.special import ndtr
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from official.data import BGRID, KCALS, NDRAWS, protocol_cell, draw, panel     # noqa: E402
from atdrive.b2d import load_features                                          # noqa: E402
from atdrive.splits import up_split                                            # noqa: E402

OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
ALL_METHODS = ['fst_scene', 'fst_resp', 'gp_scene', 'gp_resp', 'gpo_scene', 'ktcs_scene']
MFGP = Path(os.environ.get('ATDRIVE_MFGP', '/data2/jeongtae/official_baselines/MFGPreliability/MFGPreliability'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GP_NINIT = 10


def cell_seed(seed, Kc, slot):
    return 100000 + 1000 * seed + 10 * Kc + slot                 # = run_up_official.cell_seed


def fill(R):
    """Surrogate response matrix (K_cal x bank), a missing cell filled with that surrogate's mean."""
    Rf = R.copy()
    for k in range(R.shape[0]):
        m = np.nanmean(R[k])
        Rf[k, np.isnan(Rf[k])] = m
    return Rf


_SCENE = None


def scene_feats(bi):
    """The route descriptors of the bank rows (25-d kinematics + 48-d risk, as the US baselines use them)."""
    global _SCENE
    if _SCENE is None:
        ck, gt = load_features('eval_cmdkin_stats'), load_features('eval_gtrisk')
        _, _, calR = draw(0)                                     # the 220-route bank (no held-out types in UP)
        _SCENE = np.stack([np.concatenate([ck[r], gt[r]]) for r in calR])
    return _SCENE[bi]


def standardise(F):
    return (F - F.mean(0)) / (F.std(0) + 1e-6)


# ------------------------------------------------------------------ FST ----------------------------------------
class SimNet(nn.Module):
    def __init__(self, d_in, d=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, d))

    def weights(self, Z, sets):
        """Z (N, d_in) every scenario; sets (C, n) selected indices -> W (C, n): Eq. 15-16 with d = 1 / L2 and the
        softmax over the n selected scenarios for every scenario of the space, W = S p with p = 1 / N."""
        E = self.enc(Z)
        Q = E[sets]                                              # (C, n, d)
        dist = torch.cdist(Q, E.unsqueeze(0).expand(sets.shape[0], -1, -1))   # (C, n, N)
        S = torch.softmax(1.0 / (dist + 1e-3), dim=1)
        return S.mean(-1)


def fst_loss(net, Z, P, mu, sets):
    """max over surrogates of |mu~ - mu| (Eq. 20) + the fluctuation term with w_M = 1 (Eq. 24), which under
    Eq. 14 equals the surrogate's own error and is taken as the mean over the surrogates."""
    W = net.weights(Z, sets)                                     # (C, n)
    Ps = P[:, sets]                                              # (s, C, n): the surrogates' outcomes on the sets
    est = torch.einsum('cn,scn->cs', W, Ps)                      # (C, s)
    err = (est - mu[None, :]).abs()
    return err.max(1).values + err.mean(1)


def pc_sample(rng, labels, n):
    """The paper's critical distribution: scenarios drawn from k-means clusters of the surrogate performance,
    round-robin over the clusters, uniformly inside a cluster."""
    byc = {}
    for i, c in enumerate(labels):
        byc.setdefault(int(c), []).append(i)
    pools = [list(rng.permutation(v)) for v in byc.values()]
    out, k = [], 0
    while len(out) < n:
        for pl in pools:
            if k < len(pl) and len(out) < n:
                out.append(int(pl[k]))
        k += 1
    return out


def fst_fit(R, feats, seed, steps=300, batch=16):
    """One similarity network per cell, trained on P_c sets of every budget (Eq. 19)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    P = torch.tensor(1.0 - fill(R), dtype=torch.float32, device=DEV)          # failure indicator per surrogate
    mu = P.mean(1)
    Z = torch.tensor(standardise(feats), dtype=torch.float32, device=DEV)
    N = Z.shape[0]
    prof = fill(R).T                                                           # bank x surrogates
    labels = {}
    for n in BGRID:
        k = min(n, 32, len(np.unique(prof, axis=0)))
        labels[n] = KMeans(k, n_init=4, random_state=seed).fit(prof).labels_
    net = SimNet(Z.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for it in range(steps):
        n = BGRID[it % len(BGRID)]
        sets = torch.tensor([pc_sample(rng, labels[n], n) for _ in range(batch)], device=DEV)
        loss = fst_loss(net, Z, P, mu, sets).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return {'net': net, 'Z': Z, 'P': P, 'mu': mu, 'labels': labels, 'rng': rng, 'N': N}


@torch.no_grad()
def fst_select(model, n, chunk=1024):
    """Best of 32 P_c draws, then best-improvement swap search under the trained loss (Eq. 20 / 24)."""
    net, Z, P, mu, N = model['net'], model['Z'], model['P'], model['mu'], model['N']
    rng = model['rng']

    def loss_of(sets):
        out = []
        for i in range(0, len(sets), chunk):
            out.append(fst_loss(net, Z, P, mu, torch.tensor(sets[i:i + chunk], device=DEV)))
        return torch.cat(out).cpu().numpy()

    cands = [pc_sample(rng, model['labels'][n], n) for _ in range(32)]
    L = loss_of(cands)
    cur, best = list(cands[int(np.argmin(L))]), float(L.min())
    for _ in range(60):
        others = [j for j in range(N) if j not in set(cur)]
        swaps = [(i, j) for i in range(n) for j in others]
        sets = []
        for i, j in swaps:
            s = list(cur)
            s[i] = j
            sets.append(s)
        L = loss_of(sets)
        k = int(np.argmin(L))
        if L[k] < best - 1e-7:
            best = float(L[k])
            cur = sets[k]
        else:
            break
    W = net.weights(Z, torch.tensor([cur], device=DEV))[0].cpu().numpy()
    return cur, W, best


def run_fst(R, y, feats, seed):
    model = fst_fit(R, feats, seed)
    out = {}
    for B in BGRID:
        sel, W, L = fst_select(model, B)
        out[B] = {'est': float((W * y[sel]).sum() / W.sum()), 'items': sel, 'note': f'train loss {L:.4f}'}
    return out


# ------------------------------------------------------------------ GP -----------------------------------------
def rbf(A, Bm, ell, tau):
    d2 = ((A[:, None, :] - Bm[None, :, :]) ** 2).sum(-1)
    return tau ** 2 * np.exp(-0.5 * d2 / ell ** 2)


def gp_fit(X, y, hp0):
    """Hyperparameters (log ell, log tau, log sigma_n) by marginal likelihood (L-BFGS-B), constant prior mean."""
    m = y.mean()
    yc = y - m

    def nll(h):
        ell, tau, sn = np.exp(h)
        K = rbf(X, X, ell, tau) + (sn ** 2 + 1e-6) * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e6
        a = np.linalg.solve(L.T, np.linalg.solve(L, yc))
        return 0.5 * yc @ a + np.log(np.diag(L)).sum()

    r = minimize(nll, hp0, method='L-BFGS-B', bounds=[(-3, 4), (-4, 2), (-4, 1)], options={'maxiter': 40})
    return r.x, m


def gp_posterior(X, y, Xall, hp, m):
    ell, tau, sn = np.exp(hp)
    K = rbf(X, X, ell, tau) + (sn ** 2 + 1e-6) * np.eye(len(X))
    Ks = rbf(Xall, X, ell, tau)
    Kss = rbf(Xall, Xall, ell, tau)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - m))
    V = np.linalg.solve(L, Ks.T)
    mean = m + Ks @ alpha
    cov = Kss - V.T @ V
    return mean, cov, sn ** 2


def run_gp(y, feats, seed, delta=0.5):
    """Algorithm 1 of Gong et al. on the bank: n_init random routes, then the benefit-maximising route."""
    rng = np.random.default_rng(seed)
    X = standardise(feats)
    N = len(y)
    S = list(rng.permutation(N)[:GP_NINIT])
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    hp = np.array([np.log(np.sqrt(np.median(d2[d2 > 0]))), np.log(0.5), np.log(0.3)])
    out = {}
    for t in range(GP_NINIT, max(BGRID) + 1):
        hp, m = gp_fit(X[S], y[S], hp)
        mean, cov, sn2 = gp_posterior(X[S], y[S], X, hp, m)
        var = np.clip(np.diag(cov), 1e-12, None)
        un = np.array([i for i in range(N) if i not in set(S)])
        # a binary outcome: the indicator probability of an unexecuted route is that of its predictive outcome
        # y* ~ N(mean, var + sigma_n^2) (the latent alone would call a constant-plus-noise fit certain)
        p_un = ndtr((mean[un] - delta) / np.sqrt(var[un] + sn2))                # P(success) of every unexecuted route
        if t in BGRID:
            out[t] = {'est': float((y[S].sum() + p_un.sum()) / N), 'items': [int(i) for i in S]}
        if t == max(BGRID):
            break
        # Eq. 8-13 on the bank: U = sum over the unexecuted routes of sqrt(p (1 - p)) / N; a hypothetical sample at
        # x~ removes x~'s own term (its outcome becomes known) and shrinks the others' variance by Eq. 11 with the
        # mean unchanged (Eq. 10); the next route maximises the reduction.
        var_new = var[un][None, :] - cov[np.ix_(un, un)] ** 2 / (var[un] + sn2)[:, None]
        a = (mean[un][None, :] - delta) / np.sqrt(np.clip(var_new, 1e-12, None) + sn2)
        pn = ndtr(a)
        bern = np.sqrt(np.clip(pn * (1 - pn), 0, None))
        np.fill_diagonal(bern, 0.0)                                             # the candidate's own term vanishes
        U1 = bern.sum(1) / N
        S.append(int(un[int(np.argmin(U1))]))                                   # argmax of U0 - U1
    return out


# ------------------------------------------------------------------ GP, official code -------------------------------
def run_gp_official(y, feats, seed, delta=0.5):
    """Gong, Feng, Pan's own single-fidelity code (github.com/umbrellagong/MFGPreliability, commit 604bfce), unmodified:
    DiscreteInputs over the bank (grid = the route descriptors, weights = 1/N), AcqIVR_FP, OptimalDesign.seq_sampling
    with discrete=True (the acquisition is evaluated on every grid route and the argmin taken) and its
    failure_probability readout (weighted share of routes whose posterior MEAN is below the limit 0).  f_h(x) = y - 1/2
    of the route whose descriptor is x, so f_h < 0 is a failure.  What is ours: the initial routes are drawn from the
    bank instead of a Latin hypercube (pyDOE stubbed), the kernel is an isotropic RBF (+ constant, + white noise)
    because the paper's per-dimension length scales are a 2-d choice, and the SR estimate is 1 - failure probability.
    The official discrete branch passes the grid INDEX to the acquisition; the wrapper resolves it to the grid row.
    The official loop may re-select an already executed route (nothing excludes it); such repeats are counted in the
    budget and reported."""
    import warnings
    for path in (str(Path(__file__).resolve().parent / 'official' / 'mfgp_shim'), str(MFGP)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from core import AcqIVR_FP, DiscreteInputs, OptimalDesign, failure_probability      # the official package
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    warnings.filterwarnings('ignore')
    X = standardise(feats)
    N, d = X.shape
    key = lambda x: tuple(np.round(np.asarray(x, float), 6))
    lut = {key(X[i]): i for i in range(N)}
    assert len(lut) == N, 'descriptor rows must be unique for the official lookup'

    def f_h(x):
        x = np.asarray(x, float)
        if x.ndim == 2:
            return np.array([f_h(r) for r in x])
        return float(y[lut[key(x)]] - delta)

    class BankInputs(DiscreteInputs):                      # LHS -> a random draw of bank routes
        def sampling(self, num, criterion=None):
            return X[np.random.default_rng(seed).permutation(N)[:num]].copy()

    class AcqBank(AcqIVR_FP):                              # the official discrete branch hands compute_value the grid
        def compute_value(self, x):                        # INDEX (optimaldesign.py, seq_sampling, discrete=True);
            if np.ndim(x) == 0:                            # resolve it to the grid row, then the official value
                x = self.grid[int(x)]
            return super().compute_value(x)

    inputs = BankInputs(np.array([[X[:, j].min(), X[:, j].max()] for j in range(d)]), d)
    inputs.set_pdf(X.copy(), np.ones(N) / N)
    kernel = C(1.0, (1e-2, 1e2)) * RBF(np.sqrt(d), (1e-1, 1e2)) + WhiteKernel(1e-1, (1e-3, 1e0))
    sgp = GaussianProcessRegressor(kernel, normalize_y=False, n_restarts_optimizer=2, random_state=seed)
    np.random.seed(seed)
    opt = OptimalDesign(f_h, inputs)
    opt.init_sampling(GP_NINIT)
    models = opt.seq_sampling(max(BGRID) - GP_NINIT, AcqBank(inputs), sgp, n_jobs=1, discrete=True, verbose=False)
    pf = failure_probability(models, inputs)               # model k = fit on the first GP_NINIT + k samples
    items = [lut[key(r)] for r in opt.DX]
    out = {}
    for B in BGRID:
        sel = items[:B]
        out[B] = {'est': float(1.0 - pf[B - GP_NINIT]), 'items': sel,
                  'note': f'{B - len(set(sel))} repeated route(s) in the first {B} samples'}
    return out


# ------------------------------------------------------------------ Kernel test case sampling -----------------------
def run_ktcs(y, feats, seed, steps=500, nrep=5):
    """Kernel Test Case Sampling (Qian, Xu, Xing, Guo, Nature Communications 17:3114, 2026), from the paper's Methods
    (its Code Ocean capsule was not reachable): a FIXED subset of M = B routes chosen from the route descriptors alone,
    no surrogate planner.  Step 1, coverage: importance weights w_theta(x) = softmax over the cases of a one-layer
    network, trained to minimise the information potential sum_{i != j} w_i w_j K(x_i, x_j); then Pareto-order
    sampling, U_i ~ U(0, 1), Q_i = U_i / (1 - U_i) * (1 - w_i) / w_i, the M smallest Q selected (nested over the
    budgets for one draw of U; nrep draws, the error averaged over draws like the random rows).  Step 2,
    representativeness: distribution-alignment weights lambda = argmin 1/2 lambda' K_zz lambda - lambda' Kbar,
    Kbar = (1/N) 1' K_xz, sum lambda = 1, lambda >= 0 (a convex QP, solved by SLSQP).  Readout = sum_j lambda_j y_j
    (the paper's accident-rate estimate with unit exposure and no correction factors).  RBF kernel; the paper does not
    state its bandwidth, the median pairwise distance is used."""
    from scipy.optimize import minimize
    X = standardise(feats)
    N = len(y)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    sig2 = np.median(d2[np.triu_indices(N, 1)])
    K = np.exp(-d2 / (2 * sig2))
    torch.manual_seed(seed)
    Kt = torch.tensor(K, dtype=torch.float32, device=DEV)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV)
    lin = nn.Linear(X.shape[1], 1).to(DEV)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    off = 1.0 - torch.eye(N, device=DEV)
    for _ in range(steps):                                   # Step 1: information potential of the weighted pool
        w = torch.softmax(lin(Xt).squeeze(-1), 0)
        loss = (w[:, None] * w[None, :] * Kt * off).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w = torch.softmax(lin(Xt).squeeze(-1), 0).detach().cpu().numpy().astype(float)
    Kbar_all = K.mean(0)                                     # (1/N) 1' K_xz for every candidate z
    rng = np.random.default_rng(seed)
    out = {B: {'ests': [], 'items': None} for B in BGRID}
    for rep in range(nrep):
        U = rng.uniform(1e-9, 1 - 1e-9, N)
        Q = U / (1 - U) * (1 - w) / np.clip(w, 1e-12, None)   # Pareto-order sampling
        order = np.argsort(Q)
        for B in BGRID:
            sel = order[:B]
            Kzz, Kbar = K[np.ix_(sel, sel)], Kbar_all[sel]
            lam0 = np.ones(B) / B                            # Step 2: distribution-alignment weights (convex QP)
            res = minimize(lambda l: 0.5 * l @ Kzz @ l - l @ Kbar, lam0, jac=lambda l: Kzz @ l - Kbar, method='SLSQP',
                           bounds=[(0, 1)] * B, constraints=[{'type': 'eq', 'fun': lambda l: l.sum() - 1, 'jac': lambda l: np.ones(B)}],
                           options={'maxiter': 300, 'ftol': 1e-10})
            lam = np.clip(res.x, 0, None)
            lam /= lam.sum()
            out[B]['ests'].append(float(lam @ y[sel]))
            if rep == 0:
                out[B]['items'] = [int(i) for i in sel]
    return {B: {'est': float(np.mean(v['ests'])), 'ests': v['ests'], 'items': v['items'],
                'note': f'{nrep} Pareto-order draws; est = mean of the per-draw estimates, err averaged per draw at merge'} for B, v in out.items()}


# ------------------------------------------------------------------ driver --------------------------------------
def run(methods, seeds, out_path):
    recs = json.load(open(out_path)) if out_path.exists() else []
    done = {(r['seed'], r['K'], r['slot'], r['method']) for r in recs}
    for seed in seeds:
        for Kc in KCALS:
            for slot in range(4):
                R, y, bi, SR = protocol_cell(seed, Kc, slot)
                for name in methods:
                    if (seed, Kc, slot, name) in done:
                        continue
                    t0 = time.time()
                    feats = scene_feats(bi) if name.endswith('scene') else fill(R).T
                    cs = cell_seed(seed, Kc, slot)
                    est = run_fst(R, y, feats, cs) if name.startswith('fst') else \
                        run_gp_official(y, feats, cs) if name.startswith('gpo') else \
                        run_ktcs(y, feats, cs) if name.startswith('ktcs') else run_gp(y, feats, cs)
                    rec = {'seed': seed, 'K': Kc, 'slot': slot, 'method': name, 'SR': SR, 'n_bank': len(bi),
                           'fit_s': time.time() - t0,
                           'budgets': {str(B): {'est': v['est'], 'err': float(np.mean([abs(e - SR) for e in v['ests']])) if 'ests' in v else abs(v['est'] - SR),
                                                'n_items': len(v['items']), 'items': [int(i) for i in v['items']], 'note': v.get('note', ''),
                                                **({'ests': v['ests']} if 'ests' in v else {})}
                                       for B, v in est.items()}}
                    recs.append(rec)
                    json.dump(recs, open(out_path, 'w'))
                    print(f'seed {seed} K{Kc} slot {slot} {name:10} {rec["fit_s"]:6.1f}s  '
                          + ' '.join(f'B{B}:{v["err"]:.4f}' for B, v in rec['budgets'].items()), flush=True)
    return recs


def merge():
    recs = []
    for f in sorted(glob.glob(str(OUT / 'up_avbase_*_[0-9]*_[0-9]*.json'))):
        recs += json.load(open(f))
    ref = {(r['seed'], r['K'], r['js']): r for r in json.load(open(OUT / 'up_frontier.json'))}
    p = panel()
    T = {k: float(p.bank_rows(p.allr, k)[1].mean()) for k in range(p.J)}
    js_of = {s: draw(s)[0] for s in range(NDRAWS)}

    def rank_acc(seed, js, est):
        held, _ = up_split(seed, p.utypes, p.J)
        out = []
        for c in range(p.J):
            if c in held:
                continue
            st, se = np.sign(T[js] - T[c]), np.sign(est - T[c])
            out.append(0.5 if (st == 0 or se == 0) else float(st == se))
        return float(np.mean(out))

    table = {}
    for r in recs:
        js = js_of[r['seed']][r['slot']]
        for B, v in r['budgets'].items():
            t = table.setdefault(r['method'], {}).setdefault((r['K'], int(B)), {'err': [], 'd': [], 'acc': []})
            t['err'].append(v['err'])
            t['d'].append(v['err'] - ref[(r['seed'], r['K'], js)]['err']['ATDrive'][B])
            t['acc'].append(rank_acc(r['seed'], js, v['est']))
    out = {'methods': {}}
    print(f"{'method':10}{'K':>3}{'B':>5}{'n_eval':>7}{'SR-MAE':>8}{'RankAcc':>8}{'d vs ATDrive':>14}")
    for mname, Tm in table.items():
        out['methods'][mname] = {}
        for (K, B), t in sorted(Tm.items()):
            row = {'n_eval': len(t['err']), 'sr_mae': float(np.mean(t['err'])), 'rank_acc': float(np.mean(t['acc'])),
                   'd_atdrive': float(np.mean(t['d']))}
            out['methods'][mname][f'K{K}_B{B}'] = row
            print(f"{mname:10}{K:3d}{B:5d}{row['n_eval']:7d}{row['sr_mae']:8.4f}{row['rank_acc']:8.3f}{row['d_atdrive']:+14.4f}")
        cells = [v for (K, B), v in Tm.items() if B in (30, 55, 110)]
        print(f"{mname:10} macro over the 9 Table 1 cells: SR-MAE {np.mean([np.mean(c['err']) for c in cells]):.4f}")
    json.dump(out, open(OUT / 'up_avbase.json', 'w'), indent=1)
    print(f"wrote {OUT / 'up_avbase.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument('--seeds', nargs=2, type=int, default=None)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.merge:
        merge()
        return
    lo, hi = a.seeds if a.seeds else (0, NDRAWS)
    tag = '-'.join(a.methods)
    run(a.methods, range(lo, hi), OUT / f'up_avbase_{tag}_{lo}_{hi}.json')


if __name__ == '__main__':
    main()
