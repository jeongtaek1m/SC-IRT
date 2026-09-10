#!/usr/bin/env python3
"""Kernel Test Case Sampling (Qian, Xu, Xing, Guo, "Test case sampling optimization for safety validation of
automated driving systems", Nature Communications 17:3114, 2026), from the paper's Methods — its Code Ocean
capsule (10.24433/CO.9203840.v1) was not reachable from this host.
  input      the route descriptors only, no surrogate planner [ours: the paper's 48 kinematic / environment /
             interaction features per naturalistic segment];
  model      Step 1, coverage: importance weights w = softmax over the cases of a one-layer network trained to
             minimise the information potential sum_{i != j} w_i w_j K(x_i, x_j) [paper; our impl.]; RBF kernel with
             the median pairwise distance as bandwidth [ours: the paper states none];
  selection  Pareto-order sampling: U_i ~ U(0, 1), Q_i = U_i / (1 - U_i) x (1 - w_i) / w_i, the M = B smallest Q
             [paper]; nested over the budgets for one draw of U, five draws, the error averaged per draw [ours];
  estimator  Step 2, representativeness: distribution-alignment weights lambda = argmin 1/2 lambda' K_zz lambda -
             lambda' Kbar, Kbar = (1/N) 1' K_xz, sum lambda = 1, lambda >= 0 (a convex QP; SLSQP) [paper]; the
             readout sum_j lambda_j y_j is the paper's accident-rate estimate with unit exposure and no correction
             factor [paper; ours: exposure];
  target     accident rate / scaling risk -> SR [paper role].
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

from official.av_common import DEV, features_for, standardise
from official.data import BGRID


def ktcs_fit(feats, seed, steps=500):
    X = standardise(feats)
    N = X.shape[0]
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
    return K, w


def ktcs_select(K, w, seed, budgets, nrep=5):
    """Pareto-order draws and the alignment weights of every budget; nothing of the new planner is read."""
    N = len(w)
    Kbar_all = K.mean(0)                                     # (1/N) 1' K_xz for every candidate z
    rng = np.random.default_rng(seed)
    draws = []
    for rep in range(nrep):
        U = rng.uniform(1e-9, 1 - 1e-9, N)
        Q = U / (1 - U) * (1 - w) / np.clip(w, 1e-12, None)   # Pareto-order sampling
        order = np.argsort(Q)
        per_b = {}
        for B in budgets:
            sel = order[:B]
            Kzz, Kbar = K[np.ix_(sel, sel)], Kbar_all[sel]
            lam0 = np.ones(B) / B                            # Step 2: distribution-alignment weights (convex QP)
            res = minimize(lambda l: 0.5 * l @ Kzz @ l - l @ Kbar, lam0, jac=lambda l: Kzz @ l - Kbar, method='SLSQP',
                           bounds=[(0, 1)] * B, constraints=[{'type': 'eq', 'fun': lambda l: l.sum() - 1, 'jac': lambda l: np.ones(B)}],
                           options={'maxiter': 300, 'ftol': 1e-10})
            lam = np.clip(res.x, 0, None)
            lam /= lam.sum()
            per_b[B] = (sel, lam)
        draws.append(per_b)
    return draws


class KTCS:
    def __init__(self, nrep=5):
        self.nrep = nrep

    def fit(self, R, seed, workdir, bi):
        K, w = ktcs_fit(features_for('desc', np.asarray(R, float), bi), seed)
        return {'draws': ktcs_select(K, w, seed, list(BGRID), self.nrep)}

    def estimate(self, model, y, budgets, seed):
        y = np.asarray(y, float)
        out = {}
        for B in budgets:
            ests = [float(lam @ y[sel]) for (sel, lam) in (d[B] for d in model['draws'])]
            out[B] = {'est': float(np.mean(ests)), 'ests': ests, 'items': [int(i) for i in model['draws'][0][B][0]],
                      'items_draws': [[int(i) for i in d[B][0]] for d in model['draws']],
                      'note': f'{self.nrep} Pareto-order draws; est = mean of the per-draw estimates, err averaged per draw at merge'}
        return out

    def stop(self, model, y, seed):
        return {}


METHODS = {'ktcs_scene': KTCS()}
