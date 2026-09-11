#!/usr/bin/env python3
"""Sim2Val (Luo, Yang, Watson, Sharma, Veer, Schmerling, Pavone, "Sim2Val: leveraging correlation across test
platforms for variance-reduced metric estimation", CoRL 2025, arXiv 2506.20553) through its OFFICIAL package
(github.com/NVlabs/sim2val, commit b1ea402; local checkout ATDRIVE_SIM2VAL), unmodified:
  method     control variates: n paired samples (F_i = the costly metric, G_i = a cheap surrogate of it) and k
             unpaired surrogate samples G'_j give
                 mu_hat = mean_paired(F - beta G) + beta mean_unpaired(G'),
                 beta = k / (k + n) Var(G)^-1 Cov(G, F)                    [official `control_variates_estimator`]
             with their variance estimate (`var_mu_hat_beta`) alongside;
  our setting F = the new planner's recorded outcome on an executed route (the costly platform); the cheap platform
             of the protocol is the historical response panel: G_i = the K_cal calibration planners' mean success
             on route i (`s2v_mean`, scalar) or their K_cal individual responses, a missing cell filled by that
             planner's mean (`s2v_vec`, the official matrix version) [ours: the surrogate definition; the paper's
             surrogates are nuPlan open-loop metrics or a trained metric correlator];
  selection  the paper's i.i.d. sampling assumption: a random route order, nested over the budgets, five draws,
             the error averaged per draw [ours: draws]; no adaptive selection, no stopping in the method;
  estimator  the official mu_hat, clipped to [0, 1] (an SR cannot leave it; the count of clipped estimates is
             reported); if the official code raises (zero surrogate variance among the paired routes, or a
             singular Var(G) in the matrix version) the plain mean of F is used for that draw and counted;
  target     mean of the costly metric over the population = the bank SR (P).
"""
import os
import sys
from pathlib import Path

import numpy as np

from official.av_common import fill
from official.data import BGRID

S2V = Path(os.environ.get('ATDRIVE_SIM2VAL', '/data2/jeongtae/official_baselines/sim2val/src'))
if str(S2V) not in sys.path:
    sys.path.insert(0, str(S2V))
from sim2val.control_variates import control_variates_estimator                       # the official package  # noqa: E402


def surrogate(R, kind):
    """G for every bank route from the calibration responses R (K_cal x bank, nan = missing)."""
    R = np.asarray(R, float)
    if kind == 'mean':
        g = np.nanmean(R, 0)
        g[np.isnan(g)] = np.nanmean(R)                              # a route no calibration planner recorded
        return g
    return fill(R).T                                                # (bank, K_cal)


class Sim2Val:
    def __init__(self, kind, nrep=5):
        self.kind, self.nrep = kind, nrep

    def fit(self, R, seed, workdir, bi):
        G = surrogate(R, self.kind)
        N = len(G)
        rng = np.random.default_rng(seed)
        return {'G': G, 'draws': [[int(i) for i in rng.permutation(N)[:max(BGRID)]] for _ in range(self.nrep)]}

    def estimate(self, model, y, budgets, seed):
        y = np.asarray(y, float)
        G, N = model['G'], len(model['G'])
        out = {}
        for B in budgets:
            ests, items, n_fb, n_clip, var = [], [], 0, 0, []
            for S in model['draws']:
                sel = np.array(S[:B])
                rest = np.setdiff1d(np.arange(N), sel)
                try:
                    r = control_variates_estimator(y[sel], G[sel], G[rest])
                    mu, v = float(r.mu_hat_beta), float(r.var_mu_hat_beta)
                except (ValueError, np.linalg.LinAlgError):
                    mu, v = float(y[sel].mean()), float(np.var(y[sel], ddof=1) / B)   # the plain Monte-Carlo estimate
                    n_fb += 1
                if mu < 0 or mu > 1:
                    n_clip += 1
                ests.append(float(np.clip(mu, 0.0, 1.0)))
                var.append(v)
                items.append([int(i) for i in sel])
            out[B] = {'est': float(np.mean(ests)), 'ests': ests, 'items': items[0], 'items_draws': items,
                      'var': var,
                      'note': f'{len(ests)} random draws; official control_variates_estimator, surrogate {self.kind}; '
                              f'{n_fb} fallback(s) to the plain mean, {n_clip} clipped to [0, 1]'}
        return out

    def stop(self, model, y, seed):
        return {}


METHODS = {'s2v_mean': Sim2Val('mean'), 's2v_vec': Sim2Val('vec')}
