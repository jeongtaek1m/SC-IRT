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
from scirt.b2d import Panel
from scirt.splits import unified_split, R_DRAWS, H_P, H_S
from scirt.curves import THG, PRIOR, GX, GW, marginal_curves, sig
from scirt.bayes import post_from, map_fill, r1_risk, track, stop_at
from scirt.acquisition import localize_cover, eig_pick, population_fisher, K_LOCALIZE
from scirt.baselines import fluid_order, metabench_order
from scirt.metrics import ies


@pytest.fixture(scope='module')
def panel():
    return Panel()


def test_panel_invariants(panel):
    assert len(panel.allr) == 220
    assert len(panel.utypes) == 44
    assert panel.J == 16
    assert len(panel.Y) == 3476              # 3520 cells - 44 missing
    assert 'PDM-Lite' not in panel.names
    assert panel.sn['11755'] == 'EnterActorFlow'


def test_split_protocol_constants():
    assert (R_DRAWS, H_P, H_S) == (16, 3, 8)
    assert K_LOCALIZE == 20


def test_split_draw0_pinned(panel):
    hp, ht = unified_split(0, panel.utypes, panel.J)
    assert hp == [2, 5, 13]
    assert sorted(ht) == ['ConstructionObstacle', 'CrossingBicycleFlow',
                          'EnterActorFlow', 'HardBreakRoute',
                          'NonSignalizedJunctionLeftTurn',
                          'ParkingCrossingPedestrian', 'ParkingCutIn',
                          'StaticCutIn']


def test_splits_are_paired_across_regimes(panel):
    """Every draw partitions both axes at once, so US/UP/UPS share it."""
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        assert len(hp) == 3 and len(ht) == 8
        cal, new = panel.split_routes(ht)
        assert len(cal) + len(new) == 220
        assert all(panel.sn[r] in ht for r in new)


def test_grid_constants():
    assert THG.shape == (241,) and THG[0] == -6 and THG[-1] == 6
    assert abs(PRIOR.sum() - 1) < 1e-12
    assert GX.shape == (21,) and abs(GW.sum() - 1) < 1e-12


def test_marginal_curves_reduce_to_point():
    mu = np.array([0.0, 1.0])
    m = marginal_curves(mu, np.array([1e-9, 1e-9]))
    assert np.allclose(m, sig(THG[:, None] - mu[None, :]), atol=1e-6)


def _toy():
    mu = np.linspace(-2, 2, 40)
    M = marginal_curves(mu, np.full(40, 0.3))
    y = (mu < 0.3).astype(float)
    return mu, M, y


def test_readout_and_risk_are_deterministic_and_consistent():
    mu, M, y = _toy()
    S = [0, 3, 7]
    q = post_from(M, y, S)
    assert abs(q.sum() - 1) < 1e-12
    assert map_fill(M, y, S, q) == map_fill(M, y, S)
    r = r1_risk(M, y, S, q)
    assert r > 0
    assert r1_risk(M, y, list(range(40))) == 0.0            # nothing left to fill
    Sh, R1 = track(M, y, list(range(40)))
    assert len(Sh) == 40 and abs(Sh[-1] - y.mean()) < 1e-12 and R1[-1] == 0.0
    assert stop_at(R1, 1e-9) == 40 and stop_at(R1, 10.0) == 1


def test_acquisition_rules_are_deterministic():
    mu, M, y = _toy()
    a = np.ones(40)
    th = np.linspace(-1, 1, 7)
    o1 = localize_cover(a, mu, th, y, K=5, T=20)
    o2 = localize_cover(a, mu, th, y, K=5, T=20)
    assert o1 == o2 and len(set(o1)) == 20
    assert o1[5:] == [i for i in np.argsort(-population_fisher(a, mu, th)) if i not in set(o1[:5])][:15]
    assert localize_cover(a, mu, th, y, K=20, T=20) == fluid_order(a, mu, y, 20)   # K = T: pure localize
    q = post_from(M, y, [0, 3, 7])
    rem = [i for i in range(40) if i not in (0, 3, 7)]
    assert eig_pick(q, M, rem) == eig_pick(q, M, rem)
    mb = metabench_order(a, mu, 20, 40)
    assert mb[:10] == metabench_order(a, mu, 10, 40)        # prefix property


def test_ies_definition():
    assert ies(0.0347, 60, 0.0347) == pytest.approx(1.0)
    assert ies(0.0360, 31.0, 0.0449) == pytest.approx(0.414, abs=0.001)


def test_slow_anchor_entry_points_exist():
    exp = Path(__file__).resolve().parents[1] / 'experiments'
    for name in ('run_up_frontier', 'run_adaptive', 'run_tau_calibration', 'run_k_calibration',
                 'run_readout_dropin', 'run_us', 'run_ups', 'run_model_adequacy',
                 'run_calibration_stability', 'eval_us_predictions', 'build_data', 'make_figures'):
        assert (exp / f'{name}.py').exists(), name
