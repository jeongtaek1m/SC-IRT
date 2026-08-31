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
from scirt.acquisition import r1_pick, eig_pick
from scirt.baselines import fluid_order, metabench_order, population_fisher
from scirt.metrics import ies


@pytest.fixture(scope='module')
def panel():
    return Panel()


def test_panel_invariants(panel):
    assert len(panel.allr) == 220
    assert len(panel.utypes) == 44
    assert panel.J == 22
    assert len(panel.Y) == 4796              # 4840 cells - 44 missing
    assert 'PDM-Lite' not in panel.names
    assert panel.sn['11755'] == 'EnterActorFlow'


def test_split_protocol_constants():
    assert (R_DRAWS, H_P, H_S) == (16, 6, 8)


def test_split_draw0_pinned(panel):
    hp, ht = unified_split(0, panel.utypes, panel.J)
    assert hp == [3, 5, 6, 17, 18, 20]
    assert sorted(ht) == ['BlockedIntersection', 'ConstructionObstacle',
                          'EnterActorFlow', 'HardBreakRoute',
                          'NonSignalizedJunctionLeftTurn',
                          'ParkedObstacleTwoWays',
                          'SignalizedJunctionRightTurn',
                          'VanillaNonSignalizedTurnEncounterStopsign']


def test_splits_are_paired_across_regimes(panel):
    """Every draw partitions both axes at once, so US/UP/UPS share it."""
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        assert len(hp) == 6 and len(ht) == 8
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
    S, q = [], post_from(M, y, [])
    for _ in range(8):
        rem = [i for i in range(40) if i not in S]
        S.append(r1_pick(M, y, S, q, rem))
        q = post_from(M, y, S)
    assert len(set(S)) == 8
    S2, q = [], post_from(M, y, [])
    for _ in range(8):
        rem = [i for i in range(40) if i not in S2]
        S2.append(r1_pick(M, y, S2, q, rem))
        q = post_from(M, y, S2)
    assert S == S2
    q = post_from(M, y, [0, 3, 7])
    rem = [i for i in range(40) if i not in (0, 3, 7)]
    assert eig_pick(q, M, rem) == eig_pick(q, M, rem)
    assert len(population_fisher(a, mu, th)) == 40
    mb = metabench_order(a, mu, 20, 40)
    assert mb[:10] == metabench_order(a, mu, 10, 40)        # prefix property


def test_ies_definition():
    assert ies(0.04, 55, 0.04) == pytest.approx(1.0)
    assert ies(0.0342, 54.2, 0.0408) == pytest.approx(0.826, abs=0.001)


def test_slow_anchor_entry_points_exist():
    exp = Path(__file__).resolve().parents[1] / 'experiments'
    for name in ('run_up_frontier', 'run_adaptive', 'run_tau_calibration',
                 'run_readout_dropin', 'run_us', 'run_ups', 'run_model_adequacy',
                 'run_ablation', 'run_navhard',
                 'run_calibration_stability', 'eval_us_predictions', 'build_data', 'make_figures'):
        assert (exp / f'{name}.py').exists(), name
