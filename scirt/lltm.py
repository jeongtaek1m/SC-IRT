"""One-stage explanatory random-item IRT — LLTM+e.

    z_ij = theta_j - (w^T x_i + eps_i),   eps_i ~ N(0, sigma^2)

(theta, w, log sigma) fitted jointly by MAP on the calibration block, with
eps marginalised by 21-node Gauss-Hermite. This is the canonical descriptor
estimator for Table 3A (Fischer 1973; Janssen et al. 2004; De Boeck 2008) —
sigma-hat is the model's own estimate of the difficulty share features
cannot explain, and it instantiates the b|x ~ N(b_tilde(x), sigma^2) prior
of the unified generative model.
"""
import numpy as np

from .curves import GX, GW


def lltm_e(Y0, MK, rows, cols, X, it=1500, lam_w=0.5, device='cuda'):
    """Joint MAP of (w, sigma, theta) on the calibration block.

    X must already be standardised on the training rows.
    Returns (w, sigma, theta_centred, theta_centre).
    """
    import torch
    M = torch.tensor(Y0[np.ix_(rows, cols)], dtype=torch.float32).to(device)
    W = torch.tensor(MK[np.ix_(rows, cols)].astype(np.float32)).to(device)
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    gx = torch.tensor(GX, dtype=torch.float32).to(device)
    gw = torch.tensor(GW, dtype=torch.float32).to(device)
    w = torch.zeros(X.shape[1], device=device, requires_grad=True)
    th = torch.zeros(len(cols), device=device, requires_grad=True)
    ls = torch.tensor(0.0, device=device, requires_grad=True)  # log sigma
    opt = torch.optim.Adam([w, th, ls], lr=0.05)
    for _ in range(it):
        mu = Xt @ w
        sg = torch.exp(ls)
        z = th[None, :, None] - (mu[:, None, None] + sg * gx[None, None, :])
        p = torch.sigmoid(z)
        ll_cell = M[:, :, None] * torch.log(p + 1e-7) + (1 - M[:, :, None]) * torch.log(1 - p + 1e-7)
        ll_item = (ll_cell * W[:, :, None]).sum(1)          # per-item conditional LL, (n, G)
        Li = torch.logsumexp(ll_item + torch.log(gw)[None, :], dim=1)  # eps marginalised
        nll = -(Li.sum()) / W.sum()
        loss = nll + 1e-2 * th.pow(2).mean() + lam_w * w.pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        c = th.mean()
        return (w.cpu().numpy(), float(torch.exp(ls)),
                (th - c).cpu().numpy(), float(c))
