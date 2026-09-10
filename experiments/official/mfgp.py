#!/usr/bin/env python3
"""Gong, Feng, Pan's OFFICIAL single-fidelity code (MFGPreliability, github.com/umbrellagong/MFGPreliability,
commit 604bfce — named in the paper; local checkout ATDRIVE_MFGP) on the route bank, unmodified:
  input      DiscreteInputs over the bank: grid = the route descriptors, weights = 1/N [official class; our data];
             f_h(x) = y - 1/2 of the route whose descriptor is x, so f_h < 0 is a failure [ours];
  model      their sklearn GaussianProcessRegressor; kernel = constant x isotropic RBF + white noise [ours: the
             paper's per-dimension length scales are a 2-d choice], n_restarts_optimizer = 2 [ours];
  selection  their AcqIVR_FP acquisition and OptimalDesign.seq_sampling(discrete=True): the acquisition is evaluated
             on every grid route and the argmin taken [official]; the official discrete branch hands the acquisition
             the grid INDEX (optimaldesign.py) — a subclass resolves it to the grid row [ours, no change to their
             files]; the initial n_init = 10 routes are drawn from the bank instead of a Latin hypercube (pyDOE
             stubbed) [ours]; nothing in the official loop excludes an executed route from re-selection, so repeats
             happen and are counted in the budget and reported [official behaviour];
  estimator  their failure_probability readout: the weighted share of grid routes whose posterior MEAN is below the
             limit 0 [official]; SR = 1 - that [ours];
  target     accident rate = 1 - SR [paper role].
The two unused dependencies of the official package (emukit for the bi-fidelity path, pyDOE for LHS) are stubbed
in experiments/official/mfgp_shim so that it imports unmodified.
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np

from official.av_common import GP_NINIT, features_for, standardise
from official.data import BGRID

MFGP = Path(os.environ.get('ATDRIVE_MFGP', '/data2/jeongtae/official_baselines/MFGPreliability/MFGPreliability'))


def run_gp_official(y, feats, seed, budgets, delta=0.5):
    for path in (str(Path(__file__).resolve().parent / 'mfgp_shim'), str(MFGP)):
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
    models = opt.seq_sampling(max(budgets) - GP_NINIT, AcqBank(inputs), sgp, n_jobs=1, discrete=True, verbose=False)
    pf = failure_probability(models, inputs)               # model k = fit on the first GP_NINIT + k samples
    items = [lut[key(r)] for r in opt.DX]
    out = {}
    for B in budgets:
        sel = items[:B]
        out[B] = {'est': float(1.0 - pf[B - GP_NINIT]), 'items': sel,
                  'note': f'{B - len(set(sel))} repeated route(s) in the first {B} samples'}
    return out


class MFGPOfficial:
    def fit(self, R, seed, workdir, bi):
        return {'feats': features_for('desc', np.asarray(R, float), bi)}

    def estimate(self, model, y, budgets, seed):
        return run_gp_official(np.asarray(y, float), model['feats'], seed, list(budgets))

    def stop(self, model, y, seed):
        return {}


METHODS = {'gpo_scene': MFGPOfficial()}
