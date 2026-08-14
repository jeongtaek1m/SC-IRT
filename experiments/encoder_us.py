#!/usr/bin/env python3
"""Table I, "Ours" row: response-ranking and reconstruction quality of the encoder.

Scores the difficulty-supervised interaction encoder in exactly the geometry it
was trained in: P(success) = sigmoid(theta_j - b_tilde_i), with the encoder's
out-of-fold predicted difficulty held fixed and only planner ability fitted.

The predicted difficulties are read from a frozen artifact rather than retrained
here; `data/interact/interact_b2d_w2a_final.npz` holds the six training runs
(two widths x three seeds) and their ensembles, all out-of-fold under the same
44-type leave-one-type-out split used everywhere else.

Reported arm is ens6L, the logit-mean of the six runs.
"""

import csv
import json
import sys
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scirt import data, irt, paths, runtime  # noqa: E402
from scirt.theta import sig  # noqa: E402

runtime.configure()
runtime.set_global_seeds(0)

d = np.load(f"{paths.INTERACT}/interact_b2d_w2a_final.npz", allow_pickle=True)
routes = [str(r).replace("route_", "") for r in d["routes"]]
types = [str(t) for t in d["types"]]

runs = [np.array(d[k], dtype=np.float64)
        for k in ["d64_s0", "d64_s1", "d64_s2", "d96_s0", "d96_s1", "d96_s2"]]
ARMS = {
    "ens6L (6-seed logit mean)": np.mean(runs, 0),
    "ens_d64": np.array(d["ens_d64"], dtype=np.float64),
}

panel = data.read_response_panel()
J = panel.n_planners

gold = json.load(open(paths.result("gold_anchor.json")))["gold"]
keep = [i for i, r in enumerate(routes) if r in gold]
held_out_types = sorted(set(types[i] for i in keep))
gv = np.array([gold[routes[i]] for i in keep])


def fit_theta_against_frozen_difficulty(train_routes, b_frozen):
    """Fit planner ability with difficulty held fixed at the encoder's prediction.

    Identification note: the result is deliberately **not** re-centred. Elsewhere
    in the package theta is centred to fix the location of a jointly-estimated
    scale, but here the supplied difficulty vector is itself the anchor —
    subtracting mean(theta) would shift the estimate off the scale the prediction
    was made on. Doing so moves this row from 0.762/0.173 to 0.765/0.171.
    """
    M = panel.dense(train_routes, list(range(J)))
    fit = irt.fit_irt_map(
        M, ~np.isnan(M),
        model="1pl", it=400,
        freeze_b=np.asarray(b_frozen, dtype=np.float64),
        reg_b=0, reg_loga=0,
    )
    return irt.uncentred_theta(fit)


for name, b_all in ARMS.items():
    observed, predicted, route_err = [], [], []
    bt = {routes[i]: b_all[i] for i in keep}

    for t in held_out_types:
        train = [routes[i] for i in keep if types[i] != t]
        test = [routes[i] for i in keep if types[i] == t]
        th = fit_theta_against_frozen_difficulty(train, [bt[r] for r in train])

        for r in test:
            resp = panel.responses_for(r)
            if not resp:
                continue
            ps = [sig(th[pi] - bt[r]) for pi, _ in resp]
            ys = [y for _, y in resp]
            # Aggregate-level MAE: how well the predicted pass rate of a scene
            # matches the observed one. Per-cell |p - y| is improper and is not
            # reported (see PROTOCOL section 4.3).
            route_err.append(abs(np.mean(ps) - np.mean(ys)))
            observed += ys
            predicted += ps

    auc = roc_auc_score(np.array(observed, float), predicted)
    rho = spearmanr(gv, np.array([bt[routes[i]] for i in keep])).correlation
    print(f"{name:24s} US AUROC={auc:.3f}  US MAE={np.mean(route_err):.3f}  ρ={rho:+.3f}")
