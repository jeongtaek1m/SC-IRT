#!/usr/bin/env python3
"""Few-Shot Testing (Li, He, Yang, Hu, Zhang, Feng, "Few-shot testing of autonomous vehicles with scenario
similarity learning", IEEE T-ITS 2025, arXiv 2409.14369), re-implemented from the paper — no code was released
(paper text, arXiv sources, the authors' GitHub accounts and the Feng group's release organisation checked).

What the paper proposes (Sec. III-IV) and what runs here, stage by stage:
  input      the surrogate vehicle set M = the K_cal calibration planners' outcomes on the bank (Eq. 8, the
             role the paper gives its surrogate IDMs) [paper]; the scenario content fed to the similarity network
             = the route descriptor (fst_scene: the paper's scenario state) or the surrogates' response profile
             (fst_resp) [ours];
  model      cross-attention similarity network: MLP features, reciprocal-L2 attention between the selected
             scenarios (queries) and every scenario (keys), softmax over the queries (Eq. 15), weight of a selected
             scenario = the similarity mass it collects, W = S p with p = 1/N (Eq. 13-16) [paper; our impl.];
             loss = max over surrogates of |weighted estimate - true mean| (Eq. 17/20) plus the fluctuation term
             with w_M = 1 (Eq. 24; under Eq. 14 it equals a surrogate's own error, taken as the mean over the
             surrogates) [paper]; training sets drawn from k-means clusters of the surrogate performance (the
             critical distribution P_c, Sec. IV-C) [paper]; one network per cell serves every budget, as the paper
             reuses one network for n = 5, 10, 20 [paper];
  selection  a FIXED set of n = B routes chosen before the new planner is seen (Eq. 4) [paper]; the paper
             optimises continuous scenario coordinates by gradient descent (Sec. IV-D) — the bank is discrete, so
             the set is optimised by best-improvement swap search from the best of 32 P_c draws [ours];
  estimator  the weighted sum of the observed outcomes on the selected routes (Eq. 10) [paper];
  target     the performance index mu = sum_x P(A|x) p(x) (Eq. 1-2), which with p = 1/N and A = success is the
             benchmark SR [paper role; the paper's own experiments estimate a crash rate].
Hyperparameters the paper does not state, chosen once: MLP 64 -> 32, Adam 1e-3, 300 steps x 16 sets, k-means
k = min(n, 32, #distinct profiles), 32 initial draws, <= 60 swaps, L2-reciprocal epsilon 1e-3.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from official.av_common import DEV, features_for, fill, standardise
from official.data import BGRID


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


class FST:
    """Wrapper contract: fit(R, seed, workdir, bi) trains the network AND fixes the set for every budget (nothing
    of the new planner is read); estimate(model, y, budgets, seed) reads the selected routes' outcomes only."""

    def __init__(self, feature):
        self.feature = feature                                   # 'desc' (the paper's scenario state) or 'resp'

    def fit(self, R, seed, workdir, bi):
        model = fst_fit(np.asarray(R, float), features_for(self.feature, np.asarray(R, float), bi), seed)
        sets = {B: fst_select(model, B) for B in BGRID}
        return {'sets': sets}

    def estimate(self, model, y, budgets, seed):
        y = np.asarray(y, float)
        out = {}
        for B in budgets:
            sel, W, L = model['sets'][B]
            out[B] = {'est': float((W * y[sel]).sum() / W.sum()), 'items': [int(i) for i in sel], 'note': f'train loss {L:.4f}'}
        return out

    def stop(self, model, y, seed):
        return {}


METHODS = {'fst_scene': FST('desc'), 'fst_resp': FST('resp')}
