#!/usr/bin/env python3
"""REEval's amortized calibration (Truong, Tu, Liang, Li, Koyejo, "Reliable and efficient amortized model-based
evaluation", ICLR 2025, arXiv 2503.13335; github.com/sangttruong/reeval) as a Table 3A row: difficulty of a
new route predicted from its content by the paper's amortized Rasch calibration, scored on the unified split.

What the paper does (Sec. 4.1, calibration/calibration.ipynb cell 8): the Rasch difficulty of an item with a
content embedding e_j is z_j = w . e_j + b, fitted JOINTLY with the response likelihood
    P(y_ij = 1) = sigmoid(theta_i + z_j)                (the mirt sign convention: large z = easy)
by L-BFGS over (z_free, w, b) with 150 nuisance abilities theta ~ N(0, 1) (the notebook's marginalisation
device), then the test takers' thetas by L-BFGS with z fixed; items with a feature take the amortized z, the
others a free z. The paper evaluates the amortized model by the AUC of held-out responses and by its
agreement with the traditional calibration.

What is copied verbatim from the notebook: `trainer` (L-BFGS loop with the three-way convergence test,
n_iter = 100), the two closures of cell 8 (nuisance-theta z fit, then theta fit), the sign convention and
the AUC metric (torchmetrics AUROC on held-out cells). What is ours: the "content embedding" of a route is
the 73-d rollout descriptor of Table 3A (25-d kinematics + 48-d risk), standardised on the calibration
routes — the paper embeds question text with an LLM; the split is the unified split of the protocol (per
draw: the 36 calibration types x 12 calibration planners are the training responses with features, the 8
evaluation types are the held-out items whose z comes from w . e + b alone); and the three Table 3A metrics
are computed from the amortized model's own quantities (AUROC of P over the held-out cells — the paper's own
metric; route-level MAE between the mean predicted failure 1 - P over the calibration planners and the
observed failure rate; Spearman of the difficulty -z with the observed failure rate). No Ridge, no
ATDrive calibration, no encoder enters this row.

    CUDA_VISIBLE_DEVICES=3 python experiments/us_official/reeval_amortized.py   -> results/us_reeval_amortized.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.distributions import Bernoulli
from torch.optim import LBFGS

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from atdrive.b2d import Panel, load_features                                   # noqa: E402
from atdrive.splits import unified_split, R_DRAWS                              # noqa: E402

OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', REPO / 'results'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)


def trainer(parameters, optim, closure, n_iter=100, verbose=False):
    """calibration.ipynb cell 0, verbatim (tqdm bar dropped)."""
    for iteration in range(n_iter):
        if iteration > 0:
            previous_parameters = [p.clone() for p in parameters]
            previous_loss = loss.clone()
        loss = optim.step(closure)
        if iteration > 0:
            d_loss = (previous_loss - loss).item()
            d_parameters = sum(torch.norm(prev - curr, p=2).item() for prev, curr in zip(previous_parameters, parameters))
            grad_norm = sum(torch.norm(p.grad, p=2).item() for p in parameters if p.grad is not None)
            if d_loss < 1e-5 and d_parameters < 1e-5 and grad_norm < 1e-5:
                break
    return parameters


def amortized_fit(data_with0, data_idtor, features, has_feature, train_cols, seed):
    """calibration.ipynb cell 8 for one 'scenario', verbatim up to the data plumbing: data (test takers x items,
    missing = 0 with data_idtor = 0), features (items x d), has_feature (items), train_idtor = the training
    columns' observed cells. Returns z (items), thetas (test takers)."""
    torch.manual_seed(seed)
    n_test_takers, n_items = data_with0.shape
    train_idtor = (train_cols[None, :].float().repeat(n_test_takers, 1) * data_idtor)
    has_feature_train = (has_feature * train_cols).int()
    B = 50000
    thetas_nuisance = torch.randn(150, n_test_takers, device=device)
    w = torch.randn(features.shape[1], requires_grad=True, device=device)
    b = torch.randn(1, requires_grad=True, device=device)
    z_free = torch.zeros(n_items, requires_grad=True, device=device)
    optim_z = LBFGS([z_free, w, b], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

    def closure_z():
        idx = torch.randperm(n_items)[:B]
        data_batch = data_with0[:, idx]
        train_idtor_batch = train_idtor[:, idx]
        features_batch = features[idx]
        has_feature_train_batch = has_feature_train[idx]
        z_free_batch = z_free[idx]
        optim_z.zero_grad()
        z = z_free_batch * (1 - has_feature_train_batch) + (features_batch @ w + b) * has_feature_train_batch
        probs = torch.sigmoid(thetas_nuisance[:, :, None] + z[None, None, :])
        loss = -(Bernoulli(probs=probs).log_prob(data_batch) * train_idtor_batch).mean()
        loss.backward()
        return loss

    z_free, w, b = trainer([z_free, w, b], optim_z, closure_z)
    z = z_free * (1 - has_feature) + (features @ w + b) * has_feature          # every route has a feature here
    z = z.detach()
    thetas = torch.randn(n_test_takers, requires_grad=True, device=device)
    optim_theta = LBFGS([thetas], lr=0.1, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

    def closure_theta():
        optim_theta.zero_grad()
        probs = torch.sigmoid(thetas[:, None] + z[None, :])
        loss = -(Bernoulli(probs=probs).log_prob(data_with0) * train_idtor).mean()
        loss.backward()
        return loss

    thetas = trainer([thetas], optim_theta, closure_theta)[0].detach()
    return z, thetas, w.detach(), b.detach()


def main():
    panel = Panel()
    Y0, MK = panel.dense()
    N, J = Y0.shape
    allr, sn = panel.allr, panel.sn
    ck, gt = load_features('eval_cmdkin_stats'), load_features('eval_gtrisk')
    F = np.stack([np.concatenate([ck[r], gt[r]]) for r in allr]).astype(np.float32)
    P, Yc, RP, RO, BT, FL = [], [], [], [], [], []
    per_draw = []
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, J)
        cols = [c for c in range(J) if c not in hp]
        tr = np.array([sn[allr[i]] not in ht for i in range(N)])
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
        feats = torch.tensor((F - mu) / sd, dtype=torch.float, device=device)
        data = torch.tensor(Y0[:, cols].T, dtype=torch.float, device=device)          # test takers x items
        idtor = torch.tensor(MK[:, cols].T.astype(np.float32), device=device)
        z, th, w, b = amortized_fit(data * idtor, idtor, feats, torch.ones(N, device=device),
                                    torch.tensor(tr, device=device), seed)
        probs = torch.sigmoid(th[:, None] + z[None, :]).cpu().numpy()                   # planners x routes
        te = np.where(~tr)[0]
        for i in te:
            js = [k for k, c in enumerate(cols) if MK[i, c]]
            if not js:
                continue
            p = probs[js, i]
            y = Y0[i, [cols[k] for k in js]]
            P += p.tolist()
            Yc += y.tolist()
            RP.append(float((1 - p).mean()))
            RO.append(float(1 - y.mean()))
            BT.append(float(-z[i].item()))
            FL.append(float(1 - y.mean()))
        per_draw.append({'seed': seed, 'n_te': int(len(te)), 'w_norm': float(w.norm()), 'b': float(b.item())})
        print(f'draw {seed:2d}: {len(te)} held-out routes, |w| {float(w.norm()):.3f}, b {float(b.item()):+.3f}', flush=True)
    auc = roc_auc_score(Yc, P)
    mae = float(np.mean(np.abs(np.array(RP) - np.array(RO))))
    rho = float(spearmanr(BT, FL).correlation)
    print(f'\nREEval amortized calibration (official notebook fit, 73-d rollout descriptor as the content embedding):')
    print(f'  AUROC {auc:.3f}   scene-MAE {mae:.3f}   rho(-z, fail rate) {rho:+.3f}   over {len(RP)} route evaluations, {len(P)} cells')
    json.dump({'row': 'REEval amortized calibration (official notebook fit; descriptor as content embedding)',
               'auroc': auc, 'scene_mae': mae, 'rho': rho, 'n_routes': len(RP), 'n_cells': len(P), 'per_draw': per_draw,
               'source': 'github.com/sangttruong/reeval calibration/calibration.ipynb cell 0 (trainer) + cell 8 (amortized fit), commit 726cfe1'},
              open(OUT / 'us_reeval_amortized.json', 'w'), indent=1)
    print(f"wrote {OUT / 'us_reeval_amortized.json'}")


if __name__ == '__main__':
    main()
