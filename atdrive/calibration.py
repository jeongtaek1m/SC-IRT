"""Calibration of the item bank on the calibration block (PROTOCOL section 3).

MAP fit with explicit priors — theta_k ~ N(0, 1) (the same prior the
evaluation posterior uses for a new planner), b_s ~ N(0, sigma_b^2) with
sigma_b chosen by empirical Bayes on a grid, log a_s ~ N(0, .5^2) and
logit c_s ~ N(-2.2, 1) for the 2PL/3PL baselines — followed, for the
Rasch evaluation model, by the exact conditional posterior of every
difficulty given the fitted abilities (`curves.item_posteriors`) and the
testlet SD sigma_g of the planner x scenario-type effect (profile
marginal likelihood on a grid).
"""
import numpy as np
import torch
from scipy.special import logsumexp

from .curves import sig, item_loglik, item_posteriors, item_marginal_loglik, UG

SIGMA_THETA = 1.0
SIGMA_LOGA = 0.5
SIGMA_B_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
SIGMA_G_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'   # the paper's numbers were fitted on a GPU; CPU float32 moves third decimals


def _fit(M, mk, mode, sigma_b, it=800, device=DEVICE, freeze_b0=False):
    """MAP fit on a (n items x K planners) response matrix with nan mask mk.
    Returns dict(b, th, a, cc). The zero-centred priors already identify the
    location, so the theta-mean centring applied before returning (skipped when
    freeze_b0) is a change of origin, not identification: the returned pair is a
    translation of the MAP by c = mean(theta_hat), and every quantity downstream
    is read in that frame with its prior left at zero. PROTOCOL.md section 3."""
    import torch
    Mt = torch.tensor(np.nan_to_num(M), dtype=torch.float32).to(device)
    Wt = torch.tensor(mk, dtype=torch.float32).to(device)
    n, K = M.shape
    bb = torch.zeros(n, device=device, requires_grad=not freeze_b0)
    th = torch.zeros(K, device=device, requires_grad=True)
    la = torch.zeros(n, device=device, requires_grad=(mode != '1pl'))
    gc = torch.full((n,), -2.2, device=device, requires_grad=(mode == '3pl'))
    params = ([bb] if not freeze_b0 else []) + [th] + ([la] if mode != '1pl' else []) \
        + ([gc] if mode == '3pl' else [])
    opt = torch.optim.Adam(params, lr=0.05)
    for _ in range(it):
        base = torch.sigmoid((torch.exp(la)[:, None] if mode != '1pl' else 1.0)
                             * (th[None, :] - bb[:, None]))
        p = (torch.sigmoid(gc)[:, None] + (1 - torch.sigmoid(gc)[:, None]) * base) \
            if mode == '3pl' else base
        nll = -((Mt * torch.log(p + 1e-7) + (1 - Mt) * torch.log(1 - p + 1e-7)) * Wt).sum()
        loss = nll + 0.5 * th.pow(2).sum() / SIGMA_THETA ** 2
        if not freeze_b0:
            loss = loss + 0.5 * bb.pow(2).sum() / sigma_b ** 2
        if mode != '1pl':
            loss = loss + 0.5 * la.pow(2).sum() / SIGMA_LOGA ** 2
        if mode == '3pl':
            loss = loss + 0.5 * ((gc + 2.2) ** 2).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        c = 0.0 if freeze_b0 else th.mean()
        return dict(b=(bb - c).cpu().numpy(), th=(th - c).cpu().numpy(),
                    a=torch.exp(la).cpu().numpy(), cc=torch.sigmoid(gc).cpu().numpy())


def _eb_fit(M, mk, mode, sigma_b, it, device):
    """Fit; if sigma_b is None choose it by empirical Bayes over SIGMA_B_GRID."""
    if sigma_b is not None:
        f = _fit(M, mk, mode, sigma_b, it, device)
        f['sigma_b'] = float(sigma_b)
        return f
    best = None
    for sb in SIGMA_B_GRID:
        f = _fit(M, mk, mode, sb, it, device)
        mll = item_marginal_loglik(item_loglik(M, f['th'], None if mode == '1pl' else f['a']), sb)
        if best is None or mll > best[0]:
            best = (mll, sb, f)
    f = best[2]
    f['sigma_b'] = float(best[1])
    return f


def testlet_sd(M, th, b, types):
    """Profile marginal likelihood of the planner x type effect
    u_kg ~ N(0, sigma_g^2) on SIGMA_G_GRID, at the fitted (theta, b)."""
    mk = ~np.isnan(M)
    Y = np.nan_to_num(M)
    types = np.asarray(types)
    best, bestv = 0.0, -np.inf
    for sg in SIGMA_G_GRID:
        if sg == 0.0:
            ug, lpu = np.zeros(1), np.zeros(1)
        else:
            ug = UG
            lpu = -0.5 * (UG / sg) ** 2
            lpu = lpu - logsumexp(lpu)
        P = sig(th[None, :, None] - b[:, None, None] + ug[None, None, :])       # (n, K, J)
        ll = mk[:, :, None] * (Y[:, :, None] * np.log(P + 1e-12)
                               + (1 - Y[:, :, None]) * np.log(1 - P + 1e-12))
        tot = 0.0
        for t in np.unique(types):
            lt = ll[types == t].sum(0)                                          # (K, J)
            tot += logsumexp(lt + lpu[None, :], axis=1).sum()
        if tot > bestv:
            bestv, best = tot, sg
    return float(best)


def calibrate(Y, routes, cols, it=800, mode='1pl', device=DEVICE, sigma_b=None, types=None):
    """Fit item parameters from the sparse response dict.

    Y      : {(route_id, planner_idx): 0/1}
    routes : bank route ids (rows)
    cols   : calibration planner indices (columns)
    Returns dict with b, th (theta-mean centred), a, cc, sigma_b and — for
    the Rasch evaluation model — W (n, 801) exact difficulty posteriors on
    BG and sigma_g (0 when `types` is None).
    """
    M = np.full((len(routes), len(cols)), np.nan)
    for a_, rid in enumerate(routes):
        for b_, pi in enumerate(cols):
            if (rid, pi) in Y:
                M[a_, b_] = Y[(rid, pi)]
    mk = ~np.isnan(M)
    f = _eb_fit(M, mk, mode, sigma_b, it, device)
    if mode == '1pl':
        f['W'] = item_posteriors(item_loglik(M, f['th']), f['sigma_b'])
        f['sigma_g'] = testlet_sd(M, f['th'], f['b'], types) if types is not None else 0.0
    return f


def calibrate_dense(Y0, MK, rows, cols, it=800, device=DEVICE, freeze_b0=False, sigma_b=None):
    """1PL calibration on the dense (N x J) panel view (US-side scripts).

    freeze_b0=True fits the planner-only null (b == 0, theta free) and
    returns (None, theta) — theta is NOT centred in that case, matching the
    null convention of Table 3A. Otherwise returns (b, theta, sigma_b)."""
    M = np.where(MK[np.ix_(rows, cols)], Y0[np.ix_(rows, cols)], np.nan)
    mk = ~np.isnan(M)
    if freeze_b0:
        return None, _fit(M, mk, '1pl', 1.0, it, device, freeze_b0=True)['th']
    f = _eb_fit(M, mk, '1pl', sigma_b, it, device)
    return f['b'], f['th'], f['sigma_b']


def frozen_b_dense(Y0, MK, rows, cols, thA, sigma_b, it=60):
    """Refit item difficulties with theta frozen (Newton, descent) under the
    same b prior — the response-calibrated oracle for held-out routes,
    anchored to the training scale in both location and unit."""
    out = np.zeros(len(rows))
    for k, i in enumerate(rows):
        js = [c for c in cols if MK[i, c]]
        ys = Y0[i, js]
        tj = thA[[cols.index(c) for c in js]]
        b = 0.0
        for _ in range(it):
            p = sig(tj - b)
            g = (ys - p).sum() + b / sigma_b ** 2
            hh = (p * (1 - p)).sum() + 1 / sigma_b ** 2
            b = float(np.clip(b - g / hh, -8, 8))
        out[k] = b
    return out
