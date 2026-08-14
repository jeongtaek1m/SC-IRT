"""The IRT calibration kernel and its three identification policies.

Every calibration in this package is the same MAP fit of a 1PL or 2PL model::

    P(y_ij = 1) = sigmoid( a_i * (theta_j - b_i) ),   a_i = exp(loga_i)

    loss = masked-mean NLL
         + reg_theta * mean(theta^2)
         + reg_b     * mean(b^2)
         + reg_loga  * mean(loga^2)

optimised by Adam from a zero initialisation for a fixed number of steps.

The original code inlined this loop in ten places with three different step
counts, two model variants, two regulariser sets and three post-hoc
identification rules. Those differences are load-bearing, so this module
deliberately does **not** hide them behind defaults: `model` and `it` are
required keyword arguments, and identification is applied by a separate named
helper rather than inside the fit.

**Why one kernel is safe.** The originals also differ in how the loss terms are
parenthesised, which is a different float32 summation order in the forward pass.
It cannot change the result: a sum node's backward pass hands `grad_output` to
each operand unchanged, and every leaf accumulates from exactly two paths, so
gradient accumulation is a two-term addition and commutative under IEEE 754.
`tests/test_kernel_equiv.py` asserts this empirically at max|delta| == 0.0
rather than relying on the argument.
"""

from dataclasses import dataclass

import numpy as np
import torch

from . import runtime


@dataclass
class IrtFit:
    """Raw, **uncentred** MAP estimates. Apply an identification policy before use."""

    b: np.ndarray  # item difficulty
    theta: np.ndarray  # rater ability
    loga: np.ndarray  # log discrimination (zeros for a 1PL fit)
    theta_mean: np.floating  # centering constant, computed in the fit's dtype

    @property
    def a(self):
        """Item discrimination, exp(loga). Identically 1.0 for a 1PL fit."""
        return np.exp(self.loga)


def fit_irt_map(
    M,
    W,
    *,
    model,
    it,
    freeze_b=None,
    lr=0.05,
    reg_theta=1e-2,
    reg_b=1e-3,
    reg_loga=0.5,
    eps=1e-7,
    device=None,
):
    """Fit a 1PL/2PL IRT model by MAP and return raw parameters.

    Args:
        M: (n_items, n_raters) responses; missing entries may be NaN.
        W: (n_items, n_raters) observation mask, truthy where M is observed.
        model: '1pl' (discrimination fixed at 1) or '2pl' (discrimination fitted).
        it: number of Adam steps. Required — the live values are 400, 600 and 800
            and a shared default would silently rewrite whichever site omits it.
        freeze_b: optional difficulty vector to hold fixed, making b data rather
            than a parameter. Used by the encoder evaluation, which fits theta
            against a frozen predicted difficulty.
        reg_theta, reg_b, reg_loga: MAP prior weights. The frozen-b site passes
            reg_b=0 and reg_loga=0.

    Returns:
        IrtFit with uncentred estimates.
    """
    if model not in ("1pl", "2pl"):
        raise ValueError(f"model must be '1pl' or '2pl', got {model!r}")
    device = runtime.DEVICE if device is None else device

    n_items, n_raters = M.shape
    Mt = torch.tensor(np.nan_to_num(M), dtype=runtime.DTYPE).to(device)
    Wt = torch.tensor(np.asarray(W), dtype=runtime.DTYPE).to(device)

    if freeze_b is None:
        bb = torch.zeros(n_items, device=device, dtype=runtime.DTYPE, requires_grad=True)
        params = [bb]
    else:
        bb = torch.tensor(np.asarray(freeze_b), dtype=runtime.DTYPE, device=device)
        params = []
    th = torch.zeros(n_raters, device=device, dtype=runtime.DTYPE, requires_grad=True)
    params.append(th)

    two_pl = model == "2pl"
    la = torch.zeros(n_items, device=device, dtype=runtime.DTYPE, requires_grad=two_pl)
    if two_pl:
        params.append(la)

    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(it):
        if two_pl:
            p = torch.sigmoid(torch.exp(la)[:, None] * (th[None, :] - bb[:, None]))
        else:
            p = torch.sigmoid(th[None, :] - bb[:, None])
        nll = (
            -(Mt * torch.log(p + eps) + (1 - Mt) * torch.log(1 - p + eps)) * Wt
        ).sum() / Wt.sum()
        loss = nll + reg_theta * th.pow(2).mean()
        if freeze_b is None and reg_b:
            loss = loss + reg_b * bb.pow(2).mean()
        if two_pl and reg_loga:
            loss = loss + reg_loga * la.pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        # The centering constant is computed here, in torch, in the fit's own
        # dtype. Recomputing it later as a Python float would promote the
        # identification subtraction to float64 and move every difficulty by
        # ~6e-8 — enough to change a printed CI edge and to reorder CAT items.
        return IrtFit(
            b=bb.detach().cpu().numpy().copy(),
            theta=th.detach().cpu().numpy().copy(),
            loga=la.detach().cpu().numpy().copy(),
            theta_mean=th.mean().detach().cpu().numpy().copy(),
        )


# --- identification policies -------------------------------------------------
# The model is invariant to a common shift of theta and b, so one of these must
# be applied before difficulties are compared across fits. They are separate
# named functions because the choice differs per experiment and getting it wrong
# is silent: routing the encoder evaluation through center_both moves Table I's
# headline row from 0.762/0.173 to 0.765/0.171.


def center_b(fit):
    """Shift difficulty by mean(theta). The convention of PROTOCOL section 2.1."""
    return fit.b - fit.theta_mean


def center_both(fit):
    """Shift theta and difficulty by the same constant, preserving their difference."""
    c = fit.theta_mean
    return fit.theta - c, fit.b - c


def uncentred_theta(fit):
    """Return theta as fitted.

    Used only where difficulty is frozen input rather than a free parameter: there
    the supplied difficulty vector *is* the scale anchor, so re-centering would
    move the estimate off the scale the prediction was made on.
    """
    return fit.theta


# --- panel assembly ----------------------------------------------------------


def calibrate_panel(panel, routes, *, planner_mask, model, it, **kw):
    """Calibrate on a sparse response panel, selecting items by route id.

    Args:
        panel: `data.ResponsePanel`.
        routes: item ids, in the order they should occupy the item axis.
        planner_mask: per-planner booleans selecting the rater axis.

    Returns:
        (IrtFit, kept_planner_indices)
    """
    keep = [j for j in range(panel.n_planners) if planner_mask[j]]
    M = panel.dense(routes, keep)
    fit = fit_irt_map(M, ~np.isnan(M), model=model, it=it, **kw)
    return fit, keep
