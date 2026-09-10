#!/usr/bin/env python3
"""Our port of the single-fidelity adaptive sampling of Gong, Feng, Pan ("An adaptive multi-fidelity sampling
framework for safety analysis of connected and automated vehicles", IEEE T-ITS 2023, arXiv 2210.14114). The
official code is run separately (mfgp.py); this port keeps the paper's algorithm and changes what a binary
outcome on a discrete bank forces:
  input      route features as the scenario coordinates (gp_scene: the route descriptor; gp_resp: the surrogates'
             response profile) [ours], the new planner's outcome y in {0, 1} as f [ours: the paper's f is continuous];
  model      GP regression, RBF kernel, hyperparameters by marginal likelihood at every step (Sec. III-A1) [paper;
             our impl.]; isotropic length scale where the paper uses per-dimension scales in 2-d [ours]; constant
             prior mean = the mean of the observed outcomes [ours];
  selection  the next sample maximises the reduction of the variance bound of the accident rate (Eq. 8-13) with the
             hypothetical sample at the current mean (Eq. 10-11) [paper], over the unexecuted routes (argmax in place
             of continuous optimisation) [ours], after n_init = 10 random routes [paper's structure; count ours];
             the indicator probability of an unexecuted route uses its predictive outcome (latent + noise variance)
             [ours: the latent alone calls a constant-plus-noise fit certain];
  estimator  observed outcomes on the executed routes plus the predictive success probability of the rest [ours:
             the paper's f is noise-free, where the two coincide with its surrogate-everywhere readout];
  target     accident rate = 1 - SR [paper role].
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from official.av_common import GP_NINIT, features_for, standardise
from official.data import BGRID


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


def run_gp(y, feats, seed, budgets, delta=0.5):
    """Algorithm 1 of Gong et al. on the bank: n_init random routes, then the benefit-maximising route."""
    rng = np.random.default_rng(seed)
    X = standardise(feats)
    N = len(y)
    S = list(rng.permutation(N)[:GP_NINIT])
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    hp = np.array([np.log(np.sqrt(np.median(d2[d2 > 0]))), np.log(0.5), np.log(0.3)])
    out = {}
    for t in range(GP_NINIT, max(budgets) + 1):
        hp, m = gp_fit(X[S], y[S], hp)
        mean, cov, sn2 = gp_posterior(X[S], y[S], X, hp, m)
        var = np.clip(np.diag(cov), 1e-12, None)
        un = np.array([i for i in range(N) if i not in set(S)])
        p_un = ndtr((mean[un] - delta) / np.sqrt(var[un] + sn2))                # P(success) of every unexecuted route
        if t in budgets:
            out[t] = {'est': float((y[S].sum() + p_un.sum()) / N), 'items': [int(i) for i in S]}
        if t == max(budgets):
            break
        var_new = var[un][None, :] - cov[np.ix_(un, un)] ** 2 / (var[un] + sn2)[:, None]   # Eq. 11 per candidate
        a = (mean[un][None, :] - delta) / np.sqrt(np.clip(var_new, 1e-12, None) + sn2)      # Eq. 10: mean unchanged
        pn = ndtr(a)
        bern = np.sqrt(np.clip(pn * (1 - pn), 0, None))
        np.fill_diagonal(bern, 0.0)                                             # the candidate's own term vanishes
        U1 = bern.sum(1) / N
        S.append(int(un[int(np.argmin(U1))]))                                   # argmax of U0 - U1 (Eq. 12-13)
    return out


class GPPort:
    def __init__(self, feature):
        self.feature = feature

    def fit(self, R, seed, workdir, bi):
        return {'feats': features_for(self.feature, np.asarray(R, float), bi)}

    def estimate(self, model, y, budgets, seed):
        return run_gp(np.asarray(y, float), model['feats'], seed, list(budgets))

    def stop(self, model, y, seed):
        return {}


METHODS = {'gp_scene': GPPort('desc'), 'gp_resp': GPPort('resp')}
