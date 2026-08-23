"""Item-bank calibration kernels (MAP, Adam) for the unified protocol.

These are verbatim ports of the kernels every paper number was produced with:
zeros initialisation, Adam lr 0.05, 800 iterations, the exact regularisation
terms, and theta-mean centering. Do not "improve" them — bit-compatibility
with the published tables is the point.

Models
------
1pl : P = sigmoid(theta - b)                      (the paper's main model)
2pl : P = sigmoid(a (theta - b)),  log a ~ N(0, .)      (posterior-a variant,
      Fluid-style baselines)
3pl : P = c + (1-c) sigmoid(a (theta - b)), logit c ~ N(-2.2, .)  (ATLAS-style)

`s` is the per-item difficulty posterior SD from summed Fisher information —
the quantity the marginalised item curves carry forward.
"""
import numpy as np

from .curves import sig


def calibrate(Y, routes, cols, it=800, mode='1pl', device='cuda'):
    import torch
    """Fit item difficulties from the sparse response dict.

    Y      : {(route_id, planner_idx): 0/1}
    routes : bank route ids (rows)
    cols   : calibration planner indices (columns)
    Returns dict with b, th (both theta-mean centred), a, cc (3pl guessing),
    and s (difficulty posterior SD).
    """
    M = np.full((len(routes), len(cols)), np.nan)
    for a, rid in enumerate(routes):
        for b, pi in enumerate(cols):
            if (rid, pi) in Y:
                M[a, b] = Y[(rid, pi)]
    mk = ~np.isnan(M)
    Mt = torch.tensor(np.nan_to_num(M), dtype=torch.float32).to(device)
    Wt = torch.tensor(mk, dtype=torch.float32).to(device)
    n = len(routes)
    bb = torch.zeros(n, device=device, requires_grad=True)
    th = torch.zeros(len(cols), device=device, requires_grad=True)
    la = torch.zeros(n, device=device, requires_grad=(mode != '1pl'))
    gc = torch.full((n,), -2.2, device=device, requires_grad=(mode == '3pl'))
    params = [bb, th] + ([la] if mode != '1pl' else []) + ([gc] if mode == '3pl' else [])
    opt = torch.optim.Adam(params, lr=0.05)
    for _ in range(it):
        base = torch.sigmoid((torch.exp(la)[:, None] if mode != '1pl' else 1.0)
                             * (th[None, :] - bb[:, None]))
        p = (torch.sigmoid(gc)[:, None] + (1 - torch.sigmoid(gc)[:, None]) * base) \
            if mode == '3pl' else base
        nll = (-(Mt * torch.log(p + 1e-7) + (1 - Mt) * torch.log(1 - p + 1e-7)) * Wt).sum() / Wt.sum()
        loss = nll + 1e-2 * th.pow(2).mean() + 1e-3 * bb.pow(2).mean()
        if mode != '1pl':
            loss = loss + 0.5 * la.pow(2).mean()
        if mode == '3pl':
            loss = loss + 0.5 * ((gc + 2.2) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        c = th.mean()
        out = dict(b=(bb - c).cpu().numpy(), th=(th - c).cpu().numpy(),
                   a=torch.exp(la).cpu().numpy(), cc=torch.sigmoid(gc).cpu().numpy())
    out['s'] = np.array([
        1 / np.sqrt(sum((out['a'][i] ** 2)
                        * sig(out['a'][i] * (out['th'][k] - out['b'][i]))
                        * (1 - sig(out['a'][i] * (out['th'][k] - out['b'][i])))
                        for k in range(len(cols)) if mk[i, k]) + 1e-2)
        for i in range(n)])
    return out


def calibrate_dense(Y0, MK, rows, cols, it=800, device='cuda', freeze_b0=False):
    """1PL calibration on the dense (N x J) panel view (US-side scripts).

    freeze_b0=True fits the planner-only null (b == 0, theta free) and
    returns (None, theta) — theta is NOT centred in that case, matching the
    null convention of the published Table 1.
    """
    import torch
    M = torch.tensor(Y0[np.ix_(rows, cols)], dtype=torch.float32).to(device)
    W = torch.tensor(MK[np.ix_(rows, cols)].astype(np.float32)).to(device)
    bb = torch.zeros(len(rows), device=device, requires_grad=not freeze_b0)
    th = torch.zeros(len(cols), device=device, requires_grad=True)
    opt = torch.optim.Adam(([bb] if not freeze_b0 else []) + [th], lr=0.05)
    for _ in range(it):
        p = torch.sigmoid(th[None, :] - bb[:, None])
        nll = (-(M * torch.log(p + 1e-7) + (1 - M) * torch.log(1 - p + 1e-7)) * W).sum() / W.sum()
        (nll + 1e-2 * th.pow(2).mean()
         + (1e-3 * bb.pow(2).mean() if not freeze_b0 else 0.)).backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        if freeze_b0:
            return None, th.cpu().numpy()
        c = th.mean()
        return (bb - c).cpu().numpy(), (th - c).cpu().numpy()


def calibrate_dense_se(Y0, MK, rows, cols, it=800, device='cuda'):
    """As calibrate_dense but also return the 1PL difficulty posterior SD
    (used by the plausible-values decomposition)."""
    bh, tv = calibrate_dense(Y0, MK, rows, cols, it=it, device=device)
    se = np.array([
        1 / np.sqrt(sum(sig(tv[k] - bh[i]) * (1 - sig(tv[k] - bh[i]))
                        for k in range(len(cols)) if MK[rows[i], cols[k]]) + 1e-2)
        for i in range(len(rows))])
    return bh, tv, se


def frozen_b_dense(Y0, MK, rows, cols, thA, it=60):
    """Refit item difficulties with theta frozen (Newton, descent) — the
    response-calibrated oracle for held-out routes, anchored to the training
    scale in both location and unit."""
    out = np.zeros(len(rows))
    for k, i in enumerate(rows):
        js = [c for c in cols if MK[i, c]]
        ys = Y0[i, js]
        tj = thA[[cols.index(c) for c in js]]
        b = 0.0
        for _ in range(it):
            p = sig(tj - b)
            g = (ys - p).sum() + 1e-3 * b
            hh = (p * (1 - p)).sum() + 1e-3
            b = float(np.clip(b - g / hh, -6, 6))
        out[k] = b
    return out


def frozen_b(Y, routes, cols, thA, it=60):
    """Sparse-dict variant of frozen_b_dense; also returns the SE.
    (UPS-extend decomposition uses both.)"""
    out = np.zeros(len(routes))
    se = np.zeros(len(routes))
    for i, rid in enumerate(routes):
        js = [k for k, pi in enumerate(cols) if (rid, pi) in Y]
        ys = np.array([Y[(rid, cols[k])] for k in js], float)
        tj = thA[js]
        b = 0.0
        for _ in range(it):
            p = sig(tj - b)
            g = (ys - p).sum() + 1e-3 * b
            hh = (p * (1 - p)).sum() + 1e-3
            b = float(np.clip(b - g / hh, -6, 6))
        p = sig(tj - b)
        out[i] = b
        se[i] = 1 / np.sqrt((p * (1 - p)).sum() + 1e-2)
    return out, se
