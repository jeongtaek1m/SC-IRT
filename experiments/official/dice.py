#!/usr/bin/env python3
"""DICE (Farid, Schleede, Huang, Heckman, "Foundation models for rapid autonomy validation", ICRA 2025, arXiv
2411.03328), re-implemented end to end at the bank's scale — the paper's foundation model, difficulty head and
data are proprietary and unreleased (paper text, arXiv sources, GitHub, the authors' pages and citing works
checked; "patent pending").
  input      the bank's own scene tensors (experiments/dice_mae.py) [ours: the paper's 14M proprietary snippets];
  model      (i) a masked autoencoder with the paper's recipe (ego / agent-track / lane-polyline tokens, mask ratio
             .5 to a zero vector, a 4-layer transformer, L1 reconstruction with equal type weights, the ego-token
             mean as the 64-d embedding; Sec. III, VI-A) trained on the bank's 2,656 windows, 154k parameters
             [paper recipe; our scale and impl.]; (ii) the difficulty head (Sec. IV-B): backbone frozen, the pooled
             ego / track / road embeddings concatenated, an MLP to a scalar, binary cross-entropy on the simulation
             outcomes of previous software versions — here every (route, calibration planner) cell with a response,
             trained per cell on the K_cal calibration planners [paper; our impl.];
  selection  Algorithm 2: concat(z, d) clustered by k-means into M groups, a cluster drawn with probability
             proportional to K_0 + mean(d), a route uniformly inside it without replacement [paper]; M = 10, K_0 = 1
             (the value the paper discusses), the difficulty block scaled to the variance of the embedding block
             [ours: the paper leaves them open]; nested over the budgets, five draws, the error averaged per draw
             [ours];
  estimator  the stratified count the paper mentions (Sec. VI-B): the failures found in a cluster scaled by the
             inverse of its sampled share, an unsampled cluster contributing none; SR = 1 - failures / N [paper
             mentions, does not evaluate];
  target     collision rate -> SR [paper role; the paper's own metric is the share of collisions found].
Variants: dice_full = (i) + (ii) + Algorithm 2; dice_mae replaces the head by the calibration planners' failure
rate; dice_desc additionally replaces the embedding by the route descriptor.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from official.av_common import features_for, fill, route_feats, standardise
from official.data import BGRID


def dice_head(R, pooled, seed, epochs=300):
    """The paper's difficulty head: frozen backbone features -> MLP -> collision probability, BCE on every (route,
    calibration planner) outcome. Returns the head's probability for every bank route (Sec. IV-C)."""
    torch.manual_seed(seed)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)                                      # the head's outcome must not depend on the BLAS thread count
    X = torch.tensor(standardise(pooled), dtype=torch.float32)
    rows, labels = np.where(~np.isnan(R.T))                       # (route, planner) examples
    yb = torch.tensor(1.0 - R.T[rows, labels], dtype=torch.float32)  # 1 = failure
    Xb = X[torch.tensor(rows)]
    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        loss = nn.functional.binary_cross_entropy_with_logits(net(Xb).squeeze(-1), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        d = torch.sigmoid(net(X).squeeze(-1)).numpy().astype(float)
    torch.set_num_threads(n_threads)
    return d


def dice_plan(Z, d, seed, budgets, M=10, K0=1.0, nrep=5):
    """Algorithm 2 sampling for every draw and budget; nothing of the new planner is read."""
    Z = standardise(Z)
    N = len(d)
    dz = (d - d.mean()) / (d.std() + 1e-9) * np.sqrt(Z.shape[1])
    lab = KMeans(M, n_init=10, random_state=seed).fit(np.hstack([Z, dz[:, None]])).labels_
    sizes = np.bincount(lab, minlength=M)
    w = np.array([K0 + d[lab == c].mean() if sizes[c] else 0.0 for c in range(M)])
    rng = np.random.default_rng(seed)
    draws = []
    for rep in range(nrep):
        pools = {c: list(rng.permutation(np.where(lab == c)[0])) for c in range(M)}
        S = []
        while len(S) < max(budgets):
            c = rng.choice(M, p=w / w.sum())
            if pools[c]:
                S.append(int(pools[c].pop()))
        draws.append(S)
    return {'lab': lab, 'sizes': sizes, 'draws': draws, 'M': M, 'K0': K0, 'N': N}


def dice_estimate(plan, y, budgets):
    y = np.asarray(y, float)
    lab, sizes, M, N = plan['lab'], plan['sizes'], plan['M'], plan['N']
    out = {}
    for B in budgets:
        ests, items = [], []
        for S in plan['draws']:
            sel = np.array(S[:B])
            fails = 0.0
            for c in range(M):
                sc = sel[lab[sel] == c]
                if len(sc):
                    fails += (1.0 - y[sc]).sum() * sizes[c] / len(sc)
            ests.append(float(1.0 - fails / N))
            items.append([int(i) for i in sel])
        out[B] = {'est': float(np.mean(ests)), 'ests': ests, 'items': items[0], 'items_draws': items,
                  'note': f'{len(ests)} draws, M={M}, K0={plan["K0"]}, cluster sizes {sizes.tolist()}'}
    return out


class DICE:
    def __init__(self, embedding, head):
        self.embedding, self.head = embedding, head              # embedding 'desc' | 'mae'; head 'surrogate' | 'mlp'

    def fit(self, R, seed, workdir, bi):
        R = np.asarray(R, float)
        Z = features_for(self.embedding, R, bi)
        d = dice_head(R, route_feats(bi, 'pooled'), seed) if self.head == 'mlp' else 1.0 - fill(R).mean(0)
        return dice_plan(Z, d, seed, list(BGRID))

    def estimate(self, model, y, budgets, seed):
        return dice_estimate(model, y, budgets)

    def stop(self, model, y, seed):
        return {}


METHODS = {'dice_desc': DICE('desc', 'surrogate'), 'dice_mae': DICE('mae', 'surrogate'), 'dice_full': DICE('mae', 'mlp')}
