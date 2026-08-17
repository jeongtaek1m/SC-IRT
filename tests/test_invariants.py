"""Structural facts the reported numbers depend on.

Cheap assertions that fail loudly if a future edit quietly changes the evaluation
bank, the dtype, the device policy, or an iteration count. Each corresponds to a
way the numbers moved, or nearly moved, during the refactor.
"""

import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scirt import data, features, irt, runtime  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runtime.configure()


def _bank():
    panel = data.read_response_panel()
    types = data.read_route_types()
    CK = features.load_st("eval_cmdkin_stats")
    return panel, types, data.route_universe(panel.route_ids, types, CK)


def test_evaluation_bank_is_220_routes():
    """Route 11755 was re-collected; the bank covers all 220 routes."""
    panel, types, allr = _bank()
    data.assert_canonical_universe(allr)
    assert len(panel.route_ids) == 220
    assert "11755" in types      # recovered: type EnterActorFlow, confirmed across 79 artifacts


def test_panel_is_16_planners():
    panel = data.read_response_panel()
    assert panel.n_planners == 16
    assert data.EXCLUDED_PLANNER not in panel.planners


def test_cluster_structure_is_44_types():
    """44 types of exactly five routes - the bootstrap's resampling unit.

    Recovering route 11755 completes EnterActorFlow, the one type that had four:
    the design is balanced, and the earlier 4+5*43 shape was the missing route.
    """
    _, types, allr = _bank()
    sizes = sorted(len(c) for c in data.type_clusters(allr, types))
    assert len(sizes) == 44
    assert sizes == [5] * 44
    assert sum(sizes) == 220


def test_noise_ceiling_bank_is_220():
    """The reliability estimate runs on the full collection, not the filtered bank.

    Its filter is 'at least eight observed responses', which excludes nothing.
    Every route clears it, so the ceiling (0.899, 1PL) uses the same 220 bank.
    """
    Y = data.read_response_panel().dense_all()
    assert sum((~np.isnan(Y[i])).sum() >= 8 for i in range(Y.shape[0])) == 220


def test_features_and_fits_are_float32():
    """Promotion to float64 moves every reported value in the third decimal."""
    CK = features.load_st("eval_cmdkin_stats")
    assert next(iter(CK.values())).dtype == np.float32
    assert torch.get_default_dtype() == torch.float32

    panel = data.read_response_panel()
    M = panel.dense(panel.route_ids[:40], list(range(panel.n_planners)))
    fit = irt.fit_irt_map(M, ~np.isnan(M), model="2pl", it=50)
    assert fit.b.dtype == np.float32
    # The centering constant must stay in the fit's dtype: computing it as a
    # Python float promotes the subtraction to float64 and shifts every
    # difficulty by ~6e-8.
    assert np.asarray(fit.theta_mean).dtype == np.float32
    assert irt.center_b(fit).dtype == np.float32


def test_no_module_auto_selects_a_device():
    """Device is pinned in runtime.py; nothing else may probe for a GPU."""
    offenders = []
    for sub in ("scirt", "train", "tutorials"):
        for name in sorted(os.listdir(os.path.join(ROOT, sub))):
            if name.endswith(".py") and name != "runtime.py":
                src = open(os.path.join(ROOT, sub, name), encoding="utf-8").read()
                if "cuda.is_available" in src:
                    offenders.append(f"{sub}/{name}")
    assert offenders == []


def test_every_fit_call_names_its_iteration_count():
    """400, 600 and 800 are all live; a defaulted it= would silently rewrite one."""
    missing = []
    call = re.compile(r"(fit_irt_map|calibrate_panel|map_theta)\s*\(")
    for name in sorted(os.listdir(os.path.join(ROOT, "scirt"))):
        if not name.endswith(".py"):
            continue
        src = open(os.path.join(ROOT, "scirt", name), encoding="utf-8").read()
        for m in call.finditer(src):
            if src[max(0, m.start() - 4):m.start()].endswith("def "):
                continue                     # definitions declare it=, calls must name it
            depth, i = 0, m.end() - 1
            while i < len(src):
                depth += src[i] == "("
                depth -= src[i] == ")"
                if depth == 0:
                    break
                i += 1
            if "it=" not in src[m.end():i]:
                missing.append(f"{name}:{src[:m.start()].count(chr(10)) + 1}")
    assert missing == [], f"calls without an explicit it=: {missing}"


def test_api_matches_reference_outputs():
    """scirt.evaluate on the shipped artifact reproduces the published row."""
    import scirt

    r = scirt.evaluate(scirt.encoder_predictions())
    assert abs(r["rho_scene"] - 0.560) < 2e-3
    assert abs(r["auroc"] - 0.771) < 2e-3
    assert abs(r["scene_mae"] - 0.167) < 2e-3
    assert r["n_routes"] == 220


def test_scene_metrics_beat_the_planner_only_null():
    """The comparison point is the planner-only null, not chance (not a strict floor)."""
    import scirt

    r = scirt.evaluate(scirt.encoder_predictions())
    assert r["d_auroc"] > 0            # 0.771 vs 0.706
    assert r["d_nll"] > 0              # scene information improves calibrated likelihood
    assert 0 < r["oracle_gap_recovery"] < 1
    assert r["scene_mae_oracle"] < r["scene_mae"] < r["scene_mae_theta_only"]


def test_estimate_planner_recovers_a_panel_row():
    """Replaying a full panel row must recover its success rate almost exactly."""
    import scirt
    from scirt import data

    panel = data.read_response_panel()
    j = 3
    truth = {r: panel.y[(r, j)] for r in panel.route_ids if panel.observed(r, j)}
    est = scirt.estimate_planner(truth)          # administer everything
    true_sr = sum(truth.values()) / len(truth)
    assert abs(est["sr_hat"] - true_sr) < 0.02   # p-IRT keeps observed outcomes
    assert est["se"] < 0.2
