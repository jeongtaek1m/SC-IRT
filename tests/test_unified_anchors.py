"""Fast invariants of the unified protocol + pointers to the slow anchors.

The full anchor suite lives in the experiment entry points themselves — each
`experiments/run_*.py` ends by asserting the published table numbers and
prints `anchors OK`. Those runs need a GPU and minutes-to-an-hour; here we
pin everything that is cheap and deterministic on CPU.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirt.b2d import Panel
from scirt.splits import unified_split, R_DRAWS, H_P, H_S
from scirt.curves import THG, PRIOR, GX, GW, marginal_curves, sig
from scirt.bayes import post_from, sr_ci, theta_sd
from scirt.acquisition import srvar_pick, eig_pick
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


def test_posterior_and_srvar_are_deterministic():
    mu = np.linspace(-2, 2, 12)
    M = marginal_curves(mu, np.full(12, 0.3))
    y = (mu < 0).astype(float)
    q = post_from(M, y, [0, 3, 7])
    assert abs(q.sum() - 1) < 1e-12
    rem = [i for i in range(12) if i not in (0, 3, 7)]
    assert srvar_pick(M, q, rem) == srvar_pick(M, q, rem)
    assert eig_pick(q, M, rem) == eig_pick(q, M, rem)
    assert theta_sd(PRIOR.copy()) > 0.9          # standard-normal prior


def test_sr_ci_consumes_fixed_rng_stream():
    mu = np.linspace(-2, 2, 12)
    M = marginal_curves(mu, np.full(12, 0.3))
    y = (mu < 0).astype(float)
    q = post_from(M, y, [0, 3, 7])
    lo1, hi1, m1 = sr_ci(M, y, [0, 3, 7], q, np.random.RandomState(7))
    lo2, hi2, m2 = sr_ci(M, y, [0, 3, 7], q, np.random.RandomState(7))
    assert (lo1, hi1, m1) == (lo2, hi2, m2)
    assert lo1 <= m1 <= hi1


def test_ies_definition():
    assert ies(0.0217, 100, 0.0217) == pytest.approx(1.0)
    assert ies(0.0463, 29.0, 0.0217) == pytest.approx(0.618, abs=0.001)


def test_slow_anchor_entry_points_exist():
    exp = Path(__file__).resolve().parents[1] / 'experiments'
    for name in ('run_up_main', 'run_up_baselines', 'run_atlas_bridge',
                 'run_scarcity', 'run_us', 'run_ups', 'run_sel_diversity'):
        assert (exp / f'{name}.py').exists()
