"""Fast invariants of the protocol + pointers to the slow anchors.

The full anchor suite lives in the experiment entry points themselves — each
`experiments/run_*.py` ends by asserting the published table numbers and
prints `anchors OK`. Those runs need a GPU and minutes-to-hours; here we pin
everything that is cheap and deterministic on CPU.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atdrive.b2d import Panel
from atdrive.splits import unified_split, up_split, R_DRAWS, H_P, H_S
from atdrive.curves import (THG, XG, I0, PRIOR, BG, UG, USHIFT, GX, GW, marginal_curves, sig,
                          item_loglik, item_posteriors, curves_from_posterior, posterior_sd)
from atdrive.bayes import Bank, State, state_from, readout, track, transfer, stop_at
from atdrive.acquisition import r1_pick, r1_scores, eig_pick, r1_traj
from atdrive.metrics import ies, paired_cluster_boot
from atdrive.baselines import kmeans_anchors, anchorpoints_select, phi_distance, pam_medoids


@pytest.fixture(scope='module')
def panel():
    return Panel()


def test_panel_invariants(panel):
    assert len(panel.allr) == 220
    assert len(panel.utypes) == 44
    assert panel.J == 16
    assert len(panel.Y) == 3482              # 3520 cells minus the missing ones
    assert 'PDM-Lite' not in panel.names
    assert panel.sn['11755'] == 'EnterActorFlow'


def test_split_protocol_constants():
    assert (R_DRAWS, H_P, H_S) == (16, 4, 8)


def test_split_draw0_pinned(panel):
    hp, ht = unified_split(0, panel.utypes, panel.J)
    assert hp == [2, 5, 9, 13]
    assert sorted(ht) == ['ConstructionObstacle', 'CrossingBicycleFlow', 'EnterActorFlow',
                          'HardBreakRoute', 'NonSignalizedJunctionLeftTurn',
                          'ParkingCrossingPedestrian', 'ParkingCutIn', 'StaticCutIn']


def test_splits_are_paired_across_regimes(panel):
    """Every draw partitions both axes at once, so US/UPS share it with UP's planners."""
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        assert len(hp) == 4 and len(ht) == 8
        cal, new = panel.split_routes(ht)
        assert len(cal) + len(new) == 220
        assert len(new) == 40                       # the US / UPS target block: 8 types x 5 routes
        assert all(panel.sn[r] in ht for r in new)


def test_up_split_is_the_whole_benchmark(panel):
    """UP holds out planners only: its bank is all 220 routes and 44 types,
    while the US / UPS target block keeps the 36:8 type hold-out."""
    for seed in range(R_DRAWS):
        hp_up, ht_up = up_split(seed, panel.utypes, panel.J)
        assert hp_up == unified_split(seed, panel.utypes, panel.J)[0]
        assert ht_up == set()
        cal, new = panel.split_routes(ht_up)
        assert len(cal) == 220 and new == []
        assert len(set(panel.sn[r] for r in cal)) == 44
        assert min(len(panel.bank_rows(cal, j)[0]) for j in hp_up) >= 165   # every fixed budget is reachable


def test_grid_constants():
    assert THG.shape == (241,) and THG[0] == -6 and THG[-1] == 6
    assert np.allclose(XG[I0:I0 + 241], THG) and XG.shape == (361,)
    assert abs(PRIOR.sum() - 1) < 1e-12
    assert BG.shape == (801,) and UG.shape == (61,)
    assert np.allclose(XG[I0 + 100 + USHIFT], THG[100] + UG)       # theta + u is an index shift
    assert GX.shape == (21,) and abs(GW.sum() - 1) < 1e-12


def test_marginal_curves_reduce_to_point():
    mu = np.array([-1.0, 0.5])
    m = marginal_curves(mu, np.array([1e-9, 1e-9]))
    assert m.shape == (361, 2)
    assert np.allclose(m, sig(XG[:, None] - mu[None, :]), atol=1e-6)


def _toy_bank(sigma_g=0.6, seed=0):
    rng = np.random.RandomState(seed)
    n = 40
    types = np.repeat(np.arange(8), 5)
    b, th = rng.randn(n), rng.randn(6)
    R = (rng.rand(n, 6) < sig(th[None, :] - b[:, None])).astype(float)
    R[rng.rand(n, 6) < 0.1] = np.nan
    W = item_posteriors(item_loglik(R, th), 1.5)
    assert np.allclose(W.sum(1), 1) and (posterior_sd(W) > 0).all()
    y = (rng.rand(n) < sig(0.3 - b + 0.6 * rng.randn(8)[types])).astype(float)
    return Bank(curves_from_posterior(W), types, sigma_g), y


def test_exact_item_posterior_sharpens_with_more_planners():
    rng = np.random.RandomState(1)
    b = rng.randn(30)
    sds = []
    for K in (4, 16, 64):
        th = rng.randn(K)
        R = (rng.rand(30, K) < sig(th[None, :] - b[:, None])).astype(float)
        sds.append(posterior_sd(item_posteriors(item_loglik(R, th), 1.5)).mean())
    assert sds[0] > sds[1] > sds[2]


def test_readout_and_risk_are_deterministic_and_consistent():
    bank, y = _toy_bank()
    S = [3, 17, 25, 8, 31]
    st = state_from(bank, y, S)
    sh, r1 = st.readout()
    assert 0 <= sh <= 1 and r1 > 0
    assert readout(bank, y, S) == sh
    assert abs(st.q.sum() - 1) < 1e-12
    Sh, R1 = track(bank, y, S)
    assert Sh[-1] == sh and R1[-1] == r1
    Shf, R1f = track(bank, y, list(range(bank.n)))
    assert abs(Shf[-1] - y.mean()) < 1e-12 and R1f[-1] == 0.0      # full observation is exact
    assert stop_at(R1, 1.0) == 1 and stop_at(R1, 0.0) == len(R1)
    p = st.predictive_all()
    assert p.shape == (bank.n,) and ((p >= 0) & (p <= 1)).all()


def test_no_testlet_is_the_independent_model():
    bank, y = _toy_bank(sigma_g=0.0)
    C = bank.M3[:, 0, :]
    XC = np.zeros((361, bank.n))
    XC[I0:I0 + 241] = C
    flat = Bank(XC, np.arange(bank.n), 0.0)          # every item its own type, no u
    assert bank.M3.shape[1] == 1 and flat.M3.shape[1] == 1
    S = [1, 2, 3, 20, 21]
    assert abs(readout(bank, y, S) - readout(flat, y, S)) < 1e-12
    assert r1_pick(State(bank, y), list(range(bank.n))) == r1_pick(State(flat, y), list(range(bank.n)))


def test_acquisition_rules_are_deterministic():
    bank, y = _toy_bank()
    st = State(bank, y)
    rem = list(range(bank.n))
    ev = r1_scores(st, rem)
    assert ev.shape == (bank.n,) and r1_pick(st, rem) == rem[int(np.argmin(ev))]
    S1 = r1_traj(bank, y, 8)
    S2 = r1_traj(bank, y, 8)
    assert S1 == S2 and len(set(S1)) == 8
    e = eig_pick(State(bank, y), rem)
    assert e in rem
    st2 = state_from(bank, y, S1)
    sr_d, pD = transfer(st2.q, bank)
    assert 0 <= sr_d <= 1 and pD.shape == (bank.n,)


def test_ies_definition():
    assert ies(0.04, 55, 0.04) == pytest.approx(1.0)
    assert ies(0.0342, 54.2, 0.0408) == pytest.approx(0.826, abs=0.001)


def test_slow_anchor_entry_points_exist():
    exp = Path(__file__).resolve().parents[1] / 'experiments'
    for name in ('run_up_frontier', 'run_tau_calibration', 'run_adaptive', 'run_ablation',
                 'run_system_ablation', 'run_system_comparison', 'run_cat_objective',
                 'run_policy_matrix', 'run_ranking_quality', 'run_route_discrimination',
                 'run_readout_dropin', 'run_us', 'run_ups', 'run_ups_full', 'run_nuplan_zeroshot',
                 'run_model_adequacy', 'make_figures', 'make_icc_figure', 'make_uncertainty_figure',
                 'eval_us_predictions', 'build_data'):
        assert (exp / f'{name}.py').exists(), name


def test_acquisition_and_readout_never_peek():
    bank, y = _toy_bank()
    S = r1_traj(bank, y, 6)
    y2 = y.copy()
    for i in range(bank.n):
        if i not in S:
            y2[i] = 1 - y2[i]                           # flip every unobserved outcome
    assert r1_traj(bank, y2, 6) == S
    assert readout(bank, y, S) == readout(bank, y2, S)
    st, st2 = state_from(bank, y, S), state_from(bank, y2, S)
    assert np.array_equal(r1_scores(st, [i for i in range(bank.n) if i not in S]),
                          r1_scores(st2, [i for i in range(bank.n) if i not in S]))
    assert eig_pick(st, [i for i in range(bank.n) if i not in S]) == eig_pick(st2, [i for i in range(bank.n) if i not in S])


def test_mixture_median_and_l1_against_brute_force():
    from atdrive.bayes import mix_median, mix_l1
    from scipy.stats import norm
    q = np.array([[0.3], [0.7]]); mu = np.array([[10.0], [20.0]]); sd = np.array([[2.0], [3.0]])
    c = mix_median(q, mu, sd, 40.0)[0]
    assert abs((q[:, 0] * norm.cdf((c - mu[:, 0]) / sd[:, 0])).sum() - 0.5) < 1e-6
    x = np.linspace(-20, 60, 200001)
    dens = (q[:, 0][:, None] * norm.pdf((x[None, :] - mu[:, 0][:, None]) / sd[:, 0][:, None]) / sd[:, 0][:, None]).sum(0)
    brute = np.trapz(np.abs(x - c) * dens, x)
    assert abs(mix_l1(q, mu, sd, np.array([c]))[0] - brute) < 1e-3


def test_paired_cluster_boot_weights_evaluations_equally():
    a = np.array([1.0, 1.0, 1.0, 5.0]); b = np.zeros(4); cl = np.array([0, 0, 0, 1])
    m, lo, hi = paired_cluster_boot(a, b, cl, B=200)
    assert m == 2.0 and lo <= m <= hi
    assert stop_at([0.5, 0.2, 0.1, 0.05], 0.15) == 3


def test_anchor_selectors_spend_the_budget():
    rng = np.random.RandomState(0)
    a2 = np.repeat(np.array([1.0, 1.2, 0.8]), 20); b2 = np.repeat(rng.randn(3), 20)   # 3 distinct (a, b) points
    sel = kmeans_anchors(a2, b2, 25, 60)
    assert len(sel) == 25 and len(set(sel)) == 25
    R = (rng.rand(60, 7) < 0.5).astype(float); R[10:20] = R[0]                          # duplicate rows
    med, w = anchorpoints_select(R, 15)
    assert len(med) == 15 and len(set(med)) == 15 and w.sum() == 60
    D = phi_distance(R)
    assert D.shape == (60, 60) and np.allclose(np.diag(D), 0) and (D >= -1e-12).all() and D[0, 10] == 0
    m2, _ = pam_medoids(D, 3)
    assert len(m2) == 3
