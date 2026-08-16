"""High-level API: estimate scene difficulty and score difficulty predictors.

    import scirt

    gold = scirt.gold()                       # frozen 2PL difficulty anchor
    bt   = scirt.encoder_predictions()       # or any {route_id: difficulty}
    print(scirt.evaluate(bt))                # {'auroc': ..., 'mae': ..., 'rho': ...}

`evaluate` follows the paper's protocol exactly: ability is refit per held-out
scenario type with difficulty frozen at the prediction, and all metrics pool
across the 44 leave-one-type-out folds. See PROTOCOL.md.
"""

import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from . import data, irt, paths, runtime
from .theta import sig


def _setup():
    runtime.configure()
    runtime.set_global_seeds(0)


def gold(recompute=False):
    """The frozen 2PL difficulty anchor as {route_id: difficulty}.

    Loads results/gold_anchor.json when present; otherwise (or with
    recompute=True) fits the full-panel calibration and writes it.
    """
    _setup()
    cached = paths.result("gold_anchor.json")
    if not recompute and os.path.exists(cached):
        return json.load(open(cached))["gold"]
    from .features import load_st
    panel = data.read_response_panel()
    types = data.read_route_types()
    allr = data.route_universe(panel.route_ids, types, load_st("eval_cmdkin_stats"))
    data.assert_canonical_universe(allr)
    fit, _ = irt.calibrate_panel(panel, allr, planner_mask=[True] * panel.n_planners,
                                 model="2pl", it=800)
    g = {r: float(b) for r, b in zip(allr, irt.center_b(fit))}
    a = {r: float(v) for r, v in zip(allr, fit.a)}
    paths.ensure_results()
    json.dump({"gold": g, "a": a}, open(cached, "w"))
    return g


def encoder_predictions(ensemble="logit6"):
    """The released encoder's out-of-fold difficulty, as {route_id: b_tilde}.

    ensemble: 'logit6' (the Table I arm — mean of the six runs' logits) or a
    run key such as 'd64_s0'.
    """
    d = np.load(f"{paths.INTERACT}/interact_b2d_w2a_final.npz", allow_pickle=True)
    routes = [str(r).replace("route_", "") for r in d["routes"]]
    if ensemble == "logit6":
        keys = ["d64_s0", "d64_s1", "d64_s2", "d96_s0", "d96_s1", "d96_s2"]
        v = np.mean([np.array(d[k], dtype=np.float64) for k in keys], 0)
    else:
        v = np.array(d[ensemble], dtype=np.float64)
    return dict(zip(routes, v.tolist()))


def evaluate(b_tilde, anchor=None):
    """Score a difficulty prediction against the frozen anchor.

    Args:
        b_tilde: {route_id: predicted difficulty}, higher = harder. Predictions
            must be out-of-fold if you intend to compare with the paper.
        anchor: optional {route_id: gold difficulty}; defaults to `gold()`.

    Returns:
        {'auroc': cell-ranking AUROC, 'mae': per-scene pass-rate MAE,
         'rho': pooled Spearman against the anchor, 'n_routes': ...}
    """
    _setup()
    g = anchor if anchor is not None else gold()
    panel = data.read_response_panel()
    types = data.read_route_types()
    J = panel.n_planners

    keep = [r for r in panel.route_ids if r in b_tilde and r in g and r in types]
    gv = np.array([g[r] for r in keep])
    bt = {r: float(b_tilde[r]) for r in keep}
    held_out = sorted(set(types[r] for r in keep))

    observed, predicted, route_err = [], [], []
    for t in held_out:
        train = [r for r in keep if types[r] != t]
        test = [r for r in keep if types[r] == t]
        M = panel.dense(train, list(range(J)))
        fit = irt.fit_irt_map(M, ~np.isnan(M), model="1pl", it=400,
                              freeze_b=np.array([bt[r] for r in train], dtype=np.float64),
                              reg_b=0, reg_loga=0)
        th = irt.uncentred_theta(fit)
        for r in test:
            ps = [sig(th[j] - bt[r]) for j in range(J) if panel.observed(r, j)]
            ys = [panel.y[(r, j)] for j in range(J) if panel.observed(r, j)]
            if ys:
                route_err.append(abs(np.mean(ps) - np.mean(ys)))
                observed += ys
                predicted += ps
    return {
        "auroc": float(roc_auc_score(np.array(observed, float), predicted)),
        "mae": float(np.mean(route_err)),
        "rho": float(spearmanr(gv, [bt[r] for r in keep]).correlation),
        "n_routes": len(keep),
    }


def noise_ceiling(n_splits=20, seed=0):
    """The panel's reliability ceiling: split-half -> Spearman-Brown -> sqrt."""
    _setup()
    panel = data.read_response_panel()
    M, W = panel.dense_all(), None
    J = panel.n_planners
    rng = np.random.RandomState(seed)
    halves = []
    for _ in range(n_splits):
        pm = rng.permutation(J)
        bs = []
        for cols in (pm[: J // 2], pm[J // 2:]):
            Mc = M[:, cols]
            fit = irt.fit_irt_map(Mc, ~np.isnan(Mc), model="2pl", it=800)
            bs.append(irt.center_b(fit))
        halves.append(spearmanr(bs[0], bs[1]).correlation)
    r = float(np.mean(halves))
    rel = 2 * r / (1 + r)
    return {"split_half": r, "reliability": rel, "ceiling": float(np.sqrt(rel))}


def _anchor_params():
    _setup()
    cached = paths.result("gold_anchor.json")
    if not os.path.exists(cached):
        gold(recompute=True)
    d = json.load(open(cached))
    if "a" not in d:                      # anchor written by an older version
        gold(recompute=True)
        d = json.load(open(cached))
    return d["gold"], d["a"]


def estimate_planner(responses, it=50):
    """Estimate a new planner from a handful of closed-loop rollouts.

    The tinyBenchmarks use case, for driving: run a planner on a few routes,
    pass the observed {route_id: 0|1} outcomes, and get back its ability, the
    measurement's standard error, and a p-IRT estimate of its success rate over
    the full 219-route bank (observed outcomes kept as-is, IRT probabilities
    fill in the rest).
    """
    from .theta import map_theta, sig as _sig
    b, a = _anchor_params()
    admin = [r for r in responses if r in b]
    bs = np.array([b[r] for r in admin])
    aa = np.array([a[r] for r in admin])
    ys = np.array([float(responses[r]) for r in admin])
    th = map_theta(bs, ys, aa, it=it)
    p_admin = _sig(aa * (th - bs))
    se = 1.0 / np.sqrt((aa**2 * p_admin * (1 - p_admin)).sum() + 1.0)
    rest = [r for r in b if r not in responses]
    p_rest = [_sig(a[r] * (th - b[r])) for r in rest]
    sr = (ys.sum() + float(np.sum(p_rest))) / (len(admin) + len(rest))
    return {"theta": float(th), "se": float(se), "sr_hat": float(sr),
            "n_administered": len(admin), "n_bank": len(admin) + len(rest)}


def next_route(responses):
    """The most informative unadministered route (2PL Fisher information)."""
    from .theta import map_theta, sig as _sig
    b, a = _anchor_params()
    admin = [r for r in responses if r in b]
    th = map_theta(np.array([b[r] for r in admin]),
                   np.array([float(responses[r]) for r in admin]),
                   np.array([a[r] for r in admin]), it=50) if admin else 0.0
    rest = [r for r in b if r not in responses]
    info = [a[r] ** 2 * _sig(a[r] * (th - b[r])) * (1 - _sig(a[r] * (th - b[r]))) for r in rest]
    return rest[int(np.argmax(info))]
