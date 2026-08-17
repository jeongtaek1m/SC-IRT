"""High-level API: predict scene difficulty, and evaluate a planner cheaply.

    import scirt

    bt = scirt.encoder_predictions()          # or any {route_id: difficulty}
    print(scirt.evaluate(bt))                 # unseen-scene metrics

    resp = {"3021": 1, "11755": 0}            # closed-loop outcomes so far
    print(scirt.estimate_planner(resp))       # ability, success rate, 95% CI
    print(scirt.next_route(resp))             # most informative route to run next

Two regimes, two parameterisations, one reason:

    unseen scene (evaluate)      P = sigmoid(theta - b_tilde(x))        Rasch
    calibrated bank (estimate)   P = sigmoid(a_i (theta - b_i))         2PL

Discrimination is used **only where it is response-calibrated**. A held-out
scene has no responses, so a_i cannot be estimated there at all and the model
reverts to the identifiable Rasch form. On the calibrated bank a_i is worth
having: it cuts rollouts by 3.1-5.1 with a 95% CI excluding zero, and does not
move the error. See PROTOCOL.md section 2.1b.
"""

import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from . import data, irt, paths, posterior, runtime
from .theta import sig


def _setup():
    runtime.configure()
    runtime.set_global_seeds(0)


def _universe():
    from .features import load_st
    panel = data.read_response_panel()
    types = data.read_route_types()
    allr = data.route_universe(panel.route_ids, types, load_st("eval_cmdkin_stats"))
    data.assert_canonical_universe(allr)
    return panel, types, allr


def reference(recompute=False):
    """Full-panel reference difficulty as {route_id: b}, Rasch.

    A 16-rater estimate used as a *diagnostic* anchor, not as truth: the primary
    unseen-scene metric ranks predictions against observed failure rates
    instead. Cached in results/reference_difficulty.json.
    """
    _setup()
    cached = paths.result("reference_difficulty.json")
    if not recompute and os.path.exists(cached):
        return json.load(open(cached))["b"]
    panel, _, allr = _universe()
    fit, _ = irt.calibrate_panel(panel, allr, planner_mask=[True] * panel.n_planners,
                                 model="1pl", it=800)
    b = {r: float(v) for r, v in zip(allr, irt.center_b(fit))}
    paths.ensure_results()
    json.dump({"b": b}, open(cached, "w"))
    return b


def calibrated_bank(recompute=False):
    """The 2PL item bank for adaptive planner evaluation.

    Returns {route_id: (b, a, s)} where s is the posterior SD of the difficulty
    — the bank is an estimate from 16 raters, and `estimate_planner` integrates
    over that uncertainty rather than pretending b is known.
    Cached in results/calibrated_bank.json.
    """
    _setup()
    cached = paths.result("calibrated_bank.json")
    if not recompute and os.path.exists(cached):
        d = json.load(open(cached))
        return {r: tuple(v) for r, v in d.items()}
    panel, _, allr = _universe()
    fit, keep = irt.calibrate_panel(panel, allr, planner_mask=[True] * panel.n_planners,
                                    model="2pl", it=800)
    th, b = irt.center_both(fit)
    a = fit.a
    s = np.empty(len(allr))
    for i, r in enumerate(allr):
        info = 0.0
        for jj, j in enumerate(keep):
            if panel.observed(r, j):
                p = sig(a[i] * (th[jj] - b[i]))
                info += a[i] ** 2 * p * (1 - p)
        s[i] = 1.0 / np.sqrt(info + 1e-2)
    bank = {r: (float(b[i]), float(a[i]), float(s[i])) for i, r in enumerate(allr)}
    paths.ensure_results()
    json.dump(bank, open(cached, "w"))
    return bank


def encoder_predictions():
    """The released encoder's out-of-fold difficulty, as {route_id: b_tilde}.

    The mean of the six released runs' logits; every value is a prediction made
    by a model that never saw that route's scenario type.
    """
    d = np.load(f"{paths.INTERACT}/interact_b2d_w2a_final.npz", allow_pickle=True)
    routes = [str(r).replace("route_", "") for r in d["routes"]]
    keys = ["d64_s0", "d64_s1", "d64_s2", "d96_s0", "d96_s1", "d96_s2"]
    v = np.mean([np.array(d[k], dtype=np.float64) for k in keys], 0)
    return dict(zip(routes, v.tolist()))


def _fold_predictions(bt, panel, types, keep, J):
    """Leave-one-type-out pass: per held-out cell, the predicted probability."""
    cells, per_route = [], {}
    for t in sorted(set(types[r] for r in keep)):
        train = [r for r in keep if types[r] != t]
        test = [r for r in keep if types[r] == t]
        M = panel.dense(train, list(range(J)))
        fit = irt.fit_irt_map(M, ~np.isnan(M), model="1pl", it=400,
                              freeze_b=np.array([bt[r] for r in train], dtype=np.float64),
                              reg_b=0, reg_loga=0)
        th = irt.uncentred_theta(fit)
        for r in test:
            js = [j for j in range(J) if panel.observed(r, j)]
            if not js:
                continue
            ps = [sig(th[j] - bt[r]) for j in js]
            ys = [panel.y[(r, j)] for j in js]
            per_route[r] = (ps, ys)
            cells += list(zip(ps, ys))
    return cells, per_route


def _theta_only(panel, types, keep, J):
    """Planner-only null: b_i = 0 for every scene, same MAP convention."""
    out = {}
    for t in sorted(set(types[r] for r in keep)):
        train = [r for r in keep if types[r] != t]
        M = panel.dense(train, list(range(J)))
        fit = irt.fit_irt_map(M, ~np.isnan(M), model="1pl", it=400,
                              freeze_b=np.zeros(len(train)), reg_b=0, reg_loga=0)
        th = irt.uncentred_theta(fit)
        for r in [r for r in keep if types[r] == t]:
            js = [j for j in range(J) if panel.observed(r, j)]
            if js:
                out[r] = ([sig(th[j]) for j in js], [panel.y[(r, j)] for j in js])
    return out


def _nll(pairs):
    ps = np.clip(np.array([p for p, _ in pairs], float), 1e-9, 1 - 1e-9)
    ys = np.array([y for _, y in pairs], float)
    return float(-np.mean(ys * np.log(ps) + (1 - ys) * np.log(1 - ps)))


def evaluate(b_tilde, anchor=None):
    """Score an unseen-scene difficulty prediction (the US regime).

    Ability is refit per held-out scenario type with difficulty frozen at the
    prediction; all metrics pool across the 44 leave-one-type-out folds.

    Every headline number is reported against the planner-only null
    `P = sigmoid(theta_j)` — the model that says all scenes are equally hard.
    Random (0.5 AUROC) is not the right floor and is not reported.

    Returns:
        rho_scene   Spearman(prediction, observed failure rate)   -- primary
        auroc, auroc_theta_only, d_auroc
        scene_mae, scene_mae_theta_only, scene_mae_oracle, oracle_gap_recovery
        d_nll       NLL(planner-only) - NLL(planner+scene); >0 means scene helps
        rho_ref     Spearman against the full-panel reference    -- diagnostic
    """
    _setup()
    panel = data.read_response_panel()
    types = data.read_route_types()
    ref = anchor if anchor is not None else reference()
    J = panel.n_planners

    keep = [r for r in panel.route_ids if r in b_tilde and r in ref and r in types]
    bt = {r: float(b_tilde[r]) for r in keep}

    cells, per_route = _fold_predictions(bt, panel, types, keep, J)
    null = _theta_only(panel, types, keep, J)
    orc = _fold_predictions({r: ref[r] for r in keep}, panel, types, keep, J)[1]

    routes = sorted(per_route)
    fail = np.array([1 - np.mean(per_route[r][1]) for r in routes])
    mae = float(np.mean([abs(np.mean(per_route[r][0]) - np.mean(per_route[r][1])) for r in routes]))
    mae0 = float(np.mean([abs(np.mean(null[r][0]) - np.mean(null[r][1])) for r in routes]))
    maeo = float(np.mean([abs(np.mean(orc[r][0]) - np.mean(orc[r][1])) for r in routes]))

    null_cells = [(p, y) for r in routes for p, y in zip(*null[r])]
    auc = float(roc_auc_score([y for _, y in cells], [p for p, _ in cells]))
    auc0 = float(roc_auc_score([y for _, y in null_cells], [p for p, _ in null_cells]))
    return {
        "rho_scene": float(spearmanr([bt[r] for r in routes], fail).correlation),
        "auroc": auc, "auroc_theta_only": auc0, "d_auroc": auc - auc0,
        "scene_mae": mae, "scene_mae_theta_only": mae0, "scene_mae_oracle": maeo,
        "oracle_gap_recovery": (mae0 - mae) / (mae0 - maeo) if mae0 > maeo else float("nan"),
        "d_nll": _nll(null_cells) - _nll(cells),
        "rho_ref": float(spearmanr([ref[r] for r in routes], [bt[r] for r in routes]).correlation),
        "n_routes": len(routes),
    }


def noise_ceiling(n_splits=20, seed=0):
    """The panel's reliability ceiling: split-half -> Spearman-Brown -> sqrt."""
    _setup()
    panel = data.read_response_panel()
    M = panel.dense_all()
    J = panel.n_planners
    rng = np.random.RandomState(seed)
    halves = []
    for _ in range(n_splits):
        pm = rng.permutation(J)
        bs = []
        for cols in (pm[: J // 2], pm[J // 2:]):
            Mc = M[:, cols]
            fit = irt.fit_irt_map(Mc, ~np.isnan(Mc), model="1pl", it=800)
            bs.append(irt.center_b(fit))
        halves.append(spearmanr(bs[0], bs[1]).correlation)
    r = float(np.mean(halves))
    rel = 2 * r / (1 + r)
    return {"split_half": r, "reliability": rel, "ceiling": float(np.sqrt(rel))}


def _bank_arrays(bank, order):
    arr = np.array([bank[r] for r in order], float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def estimate_planner(responses, marginalise=True, n_draws=4000, seed=0):
    """Estimate a new planner from a handful of closed-loop rollouts.

    Run a planner on a few routes, pass the observed {route_id: 0|1} outcomes,
    and get back its ability, a success-rate estimate over the whole bank
    (observed outcomes kept as-is, model probabilities filling in the rest), and
    a 95% posterior-predictive interval for that rate.

    Args:
        marginalise: integrate over difficulty uncertainty. Costs a few extra
            rollouts to reach a given precision but is what makes the interval
            honest (coverage 0.94 vs 0.88 at +-10%). Set False for the plug-in
            variant reported as an ablation.
    """
    bank = calibrated_bank()
    order = sorted(bank)
    b, a, s = _bank_arrays(bank, order)
    if not marginalise:
        s = np.zeros_like(s)
    idx = {r: i for i, r in enumerate(order)}
    admin = [idx[r] for r in responses if r in idx]
    y = np.zeros(len(order))
    for r in responses:
        if r in idx:
            y[idx[r]] = float(responses[r])

    curves = posterior.item_curves(b, a, s)
    post = posterior.theta_posterior(curves, y, admin)
    sr = posterior.success_rate(post, curves, y, admin, len(order),
                                n_draws=n_draws, seed=seed)
    return {"theta": float(posterior.THETA_GRID[int(np.argmax(post))]),
            "sr_hat": sr["mean"], "ci95": (sr["lo"], sr["hi"]), "se": sr["se"],
            "n_administered": len(admin), "n_bank": len(order)}


def next_route(responses, marginalise=True):
    """The most informative unadministered route, by target-EIG.

        A_i = h(E_theta[m_i]) - E_theta[h(m_i)]

    Difficulty is marginalised *inside* m_i, so this scores what the response
    tells us about ability. Integrating over (theta, b) jointly instead scores
    I((theta,b); Y) and picks items that are informative about the bank rather
    than about the planner.

    Returns None once every route in the bank has been administered.
    """
    bank = calibrated_bank()
    order = sorted(bank)
    b, a, s = _bank_arrays(bank, order)
    if not marginalise:
        s = np.zeros_like(s)
    idx = {r: i for i, r in enumerate(order)}
    admin = [idx[r] for r in responses if r in idx]
    y = np.zeros(len(order))
    for r in responses:
        if r in idx:
            y[idx[r]] = float(responses[r])

    curves = posterior.item_curves(b, a, s)
    post = posterior.theta_posterior(curves, y, admin)
    rest = [i for i in range(len(order)) if i not in set(admin)]
    if not rest:                      # bank exhausted: nothing left to administer
        return None
    gain = posterior.expected_information_gain(post, curves, rest)
    return order[rest[int(np.argmax(gain))]]
