"""Scene descriptors: the feature side of the explanatory model.

A descriptor maps a route id to a fixed-length vector summarising the scene,
computed without ever driving it. Table I asks how far each descriptor family
gets toward the difficulty a full response panel would reveal.

The registry is an explicit ordered list. The original selected rows by
substring-matching a command-line argument against a 119-key dictionary, which
had three documented failure modes: a Table IV row was registered inside an
unrelated `if` block and vanished when an apparently-unused feature file was
pruned; a token could match two rows depending on load order; and three rows
were printed purely as substring collateral. An explicit list also means only
the files actually cited need to ship.
"""

import csv
from collections import OrderedDict

import numpy as np

from . import paths


def load_st(name):
    """Load a `stats`-format descriptor npz into {route_id: float32 vector}."""
    d = np.load(f"{paths.FEATURES}/{name}.npz", allow_pickle=True)
    names = [str(x).replace("route_", "") for x in d["names"]]
    return {names[i]: d["stats"][i].astype(np.float32) for i in range(len(names))}


def concat(*dicts):
    """Concatenate descriptors route-wise, keeping only routes present in all.

    Key order follows the first argument; values are concatenated in argument
    order. Both matter downstream — the first determines the dimensionality that
    picks the ridge penalty, and the second fixes the column layout the
    standardisation statistics are computed over.
    """
    return {
        k: np.concatenate([d[k] for d in dicts])
        for k in dicts[0]
        if all(k in d for d in dicts)
    }


def load_kin_den():
    """Agent density + ego kinematics (18d), the strongest hand-crafted Table I row."""
    d = np.load(f"{paths.FEATURES}/baseline_kin_den.npz", allow_pickle=True)
    kin = {str(n): d["kin"][i] for i, n in enumerate(d["kin_names"])}
    den = {str(n): d["den"][i] for i, n in enumerate(d["den_names"])}
    return {
        r: np.concatenate([kin[r], den[r]]).astype(np.float32) for r in kin if r in den
    }


def load_traffic_csv():
    """Risk-field (17d) and minimum time-to-collision (1d) from the traffic CSV.

    Min-TTC is the classical criticality scalar and the natural strawman: it is
    the single number the surrogate-safety literature would reach for first.
    """
    rows = list(csv.reader(open(f"{paths.FEATURES}/b2d_traffic_features_220.csv")))
    hdr = rows[0]
    col = {c: i for i, c in enumerate(hdr)}

    def as_float(v):
        try:
            return float(v)
        except ValueError:
            return 0.0

    risk_field = {
        r[0]: np.array([as_float(r[col[c]]) for c in hdr[2:]], np.float32) for r in rows[1:]
    }
    min_ttc = {
        r[0]: np.array([as_float(r[col["ssm_min_ttc"]])], np.float32) for r in rows[1:]
    }
    return risk_field, min_ttc


def build_descriptors():
    """The eight descriptor rows reported in the paper.

    Six are Table I baselines; the last two are the Table IV reference rows
    (kin-only ridge, and the hand-crafted ground-truth stack the encoder is
    compared against). Build order is print order.

    Table I's seventh row, Random, is definitional (AUROC 0.500, rho 0.000) and
    is not computed.
    """
    risk_field, min_ttc = load_traffic_csv()
    cmdkin = load_st("eval_cmdkin_stats")
    gtrisk = load_st("eval_gtrisk")

    return OrderedDict(
        [
            ("routegeom(16)", load_st("eval_routegeom")),  # Table I  Route geometry
            ("agentjepa(12)", load_st("eval_agentjepa")),  # Table I  Agent-JEPA
            ("bl-cmdkin(25)", cmdkin),                     # Table IV kin-only ridge
            ("GT:ck+gtrisk", concat(cmdkin, gtrisk)),      # Table IV hand-crafted GT stack
            ("bl-kin+den(18)", load_kin_den()),            # Table I  Agent density + kinematics
            ("risk-field(17)", risk_field),                # Table I  Risk field
            ("minTTC(1)", min_ttc),                        # Table I  Min-TTC
            ("smart-ent(1)", load_st("eval_smart_ent")),   # Table I  Traffic entropy
        ]
    )
