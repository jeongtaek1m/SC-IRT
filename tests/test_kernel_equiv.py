"""Prove the extracted kernel is the original inlined body, bit for bit.

Each test reimplements one of the original `calib` variants verbatim and asserts
`max|delta| == 0.0` against `scirt.irt.fit_irt_map` on the same inputs. Exact
equality, not `allclose`: the whole point of the refactor contract is that no
published digit moves, and a 1e-9 drift in a fitted difficulty is amplified into
a different item trajectory by the CAT selectors.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scirt import data, irt, runtime  # noqa: E402

runtime.configure()

DEV = runtime.DEVICE


@pytest.fixture(scope="module")
def panel_matrix():
    """A real (routes x planners) response block, not a synthetic one."""
    panel = data.read_response_panel()
    routes = panel.route_ids[:60]
    M = panel.dense(routes, list(range(panel.n_planners)))
    return M, ~np.isnan(M)


def _original_2pl(M, mk, it):
    """Verbatim body shared by gold_anchor, descriptor_table, hybrid_prereg."""
    Mt = torch.tensor(np.nan_to_num(M), dtype=torch.float32).to(DEV)
    Wt = torch.tensor(mk, dtype=torch.float32).to(DEV)
    n, J = M.shape
    bb = torch.zeros(n, device=DEV, requires_grad=True)
    th = torch.zeros(J, device=DEV, requires_grad=True)
    la = torch.zeros(n, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([bb, th, la], lr=0.05)
    for _ in range(it):
        p = torch.sigmoid(torch.exp(la)[:, None] * (th[None, :] - bb[:, None]))
        nll = (-(Mt * torch.log(p + 1e-7) + (1 - Mt) * torch.log(1 - p + 1e-7)) * Wt).sum() / Wt.sum()
        (nll + 1e-2 * th.pow(2).mean() + 1e-3 * bb.pow(2).mean()
         + 0.5 * la.pow(2).mean()).backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        return (bb.cpu().numpy().copy(), th.cpu().numpy().copy(),
                la.cpu().numpy().copy())


def _original_1pl(M, mk, it):
    """Verbatim body of the cat_ups / cat_up 1PL variant (no discrimination)."""
    Mt = torch.tensor(np.nan_to_num(M), dtype=torch.float32).to(DEV)
    Wt = torch.tensor(mk, dtype=torch.float32).to(DEV)
    n, J = M.shape
    bb = torch.zeros(n, device=DEV, requires_grad=True)
    th = torch.zeros(J, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([bb, th], lr=0.05)
    for _ in range(it):
        p = torch.sigmoid(th[None, :] - bb[:, None])
        nll = (-(Mt * torch.log(p + 1e-7) + (1 - Mt) * torch.log(1 - p + 1e-7)) * Wt).sum() / Wt.sum()
        (nll + 1e-2 * th.pow(2).mean() + 1e-3 * bb.pow(2).mean()).backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        return bb.cpu().numpy().copy(), th.cpu().numpy().copy()


def _original_frozen_b(M, mk, bvals, it):
    """Verbatim body of encoder_us: difficulty frozen, theta only, theta prior only."""
    Mt = torch.tensor(np.nan_to_num(M), dtype=torch.float32).to(DEV)
    Wt = torch.tensor(mk, dtype=torch.float32).to(DEV)
    J = M.shape[1]
    bb = torch.tensor(bvals, dtype=torch.float32, device=DEV)
    th = torch.zeros(J, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([th], lr=0.05)
    for _ in range(it):
        p = torch.sigmoid(th[None, :] - bb[:, None])
        nll = (-(Mt * torch.log(p + 1e-7) + (1 - Mt) * torch.log(1 - p + 1e-7)) * Wt).sum() / Wt.sum()
        (nll + 1e-2 * th.pow(2).mean()).backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        return th.cpu().numpy().copy()


@pytest.mark.parametrize("it", [400, 800])
def test_2pl_kernel_matches_original(panel_matrix, it):
    M, mk = panel_matrix
    b0, t0, l0 = _original_2pl(M, mk, it)
    fit = irt.fit_irt_map(M, mk, model="2pl", it=it)
    assert np.abs(fit.b - b0).max() == 0.0
    assert np.abs(fit.theta - t0).max() == 0.0
    assert np.abs(fit.loga - l0).max() == 0.0


@pytest.mark.parametrize("it", [600, 800])
def test_1pl_kernel_matches_original(panel_matrix, it):
    M, mk = panel_matrix
    b0, t0 = _original_1pl(M, mk, it)
    fit = irt.fit_irt_map(M, mk, model="1pl", it=it)
    assert np.abs(fit.b - b0).max() == 0.0
    assert np.abs(fit.theta - t0).max() == 0.0
    # A 1PL fit must report unit discrimination, not a fitted one.
    assert np.array_equal(fit.a, np.ones_like(fit.a))


def test_frozen_b_kernel_matches_original(panel_matrix):
    M, mk = panel_matrix
    rng = np.random.RandomState(0)
    bvals = rng.normal(size=M.shape[0])
    t0 = _original_frozen_b(M, mk, bvals, 400)
    fit = irt.fit_irt_map(M, mk, model="1pl", it=400,
                          freeze_b=bvals, reg_b=0, reg_loga=0)
    assert np.abs(fit.theta - t0).max() == 0.0
    # Frozen difficulty must come back untouched.
    assert np.abs(fit.b - bvals.astype(np.float32)).max() == 0.0


def test_frozen_b_is_not_recentred(panel_matrix):
    """Guard R3: the encoder site must not have theta centred.

    Centering here would move Table I's headline row from 0.762/0.173 to
    0.765/0.171, and the edit that does it looks like a protocol-compliance fix.
    """
    M, mk = panel_matrix
    rng = np.random.RandomState(0)
    fit = irt.fit_irt_map(M, mk, model="1pl", it=400,
                          freeze_b=rng.normal(size=M.shape[0]),
                          reg_b=0, reg_loga=0)
    assert abs(irt.uncentred_theta(fit).mean()) > 1e-3


def test_center_both_preserves_difference(panel_matrix):
    """center_both shifts theta and b by the same constant.

    Tolerance is one float32 ulp at this magnitude, not zero: subtracting the
    same constant from two float32 vectors rounds independently per element.
    What must hold exactly is that both use the *same* constant, which is what
    the second assertion pins down.
    """
    M, mk = panel_matrix
    fit = irt.fit_irt_map(M, mk, model="2pl", it=400)
    theta_c, b_c = irt.center_both(fit)
    shift_theta = fit.theta - theta_c
    shift_b = fit.b - b_c
    assert np.abs(shift_theta - shift_b[0]).max() < 1e-6
    assert np.abs(b_c - irt.center_b(fit)).max() == 0.0
