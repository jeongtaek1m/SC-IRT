"""B2D data loading for the unified protocol.

Everything is read from the repo's own `data/` tree — no external paths.
Route order is the response-matrix CSV column order throughout; the scripts
that produced the paper numbers iterate routes in exactly this order, so it
is part of the reproduction contract.
"""
import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / 'data'

EXCLUDED_PLANNER = 'PDM-Lite'


def load_features(name):
    """One feature npz -> {route_id: float64 vector} (route_ prefix stripped)."""
    d = np.load(DATA / 'features' / f'{name}.npz', allow_pickle=True)
    names = [str(x).replace('route_', '') for x in d['names']]
    return {names[i]: d['stats'][i].astype(np.float64) for i in range(len(names))}


def load_scene_features():
    """The SC-IRT descriptor stack: cmdkin (25d) + scenparamz (31d) -> 56d."""
    ck = load_features('eval_cmdkin_stats')
    spz = load_features('eval_scenparamz')
    return {k: np.concatenate([ck[k], spz[k]]) for k in ck if k in spz}


def load_route_types():
    """{route_id: scenario_type}. Derived from the CARLA checkpoint JSONs
    (verified identical by experiments/build_data.py; includes the one route,
    11755, missing from the JSONs)."""
    return dict(csv.reader(open(DATA / 'matrices' / 'b2d_route_types.csv')))


def load_response_rows():
    """Raw response matrix: (route_ids in CSV order, planner rows)."""
    rows = list(csv.reader(open(DATA / 'matrices' / 'b2d_e2e16_response_matrix.csv')))
    rids = rows[0][1:]
    planners = [r for r in rows[1:] if r[0] != EXCLUDED_PLANNER]
    return rids, planners


class Panel:
    """The 16-planner x 220-route response panel plus everything the
    experiments need, in the canonical ordering.

    Attributes
    ----------
    rids      : route ids, CSV column order (the canonical route order)
    names     : planner names, CSV row order
    J         : number of planners (16)
    Y         : {(route_id, planner_idx): 0/1} sparse dict (44 cells missing)
    sn        : {route_id: scenario_type}
    feat      : {route_id: 56d descriptor} (cmdkin + scenparamz)
    allr      : routes present in both feat and sn, CSV order (220)
    utypes    : sorted unique scenario types (44)
    """

    def __init__(self, extra_feature_dicts=()):
        rids, planners = load_response_rows()
        self.rids = rids
        self.names = [r[0] for r in planners]
        self.J = len(planners)
        self.Y = {}
        for pi, row in enumerate(planners):
            for j, rid in enumerate(rids):
                if row[1 + j] != '':
                    self.Y[(rid, pi)] = int(float(row[1 + j]))
        self.sn = load_route_types()
        self.feat = load_scene_features()
        keys = set(self.feat)
        for d in extra_feature_dicts:
            keys &= set(d)
        self.allr = [r for r in rids if r in keys and r in self.sn]
        self.utypes = sorted(set(self.sn[r] for r in self.allr))

    # dense (N x J) view used by the US-side scripts -------------------------
    def dense(self):
        """Return (Y0, MK): nan-filled responses as 0-filled matrix + mask,
        rows in `allr` order, columns in planner order."""
        idx = {r: i for i, r in enumerate(self.rids)}
        n = len(self.allr)
        Y = np.full((n, self.J), np.nan)
        rows = load_response_rows()[1]
        for a, r in enumerate(self.allr):
            for pi, row in enumerate(rows):
                v = row[1 + idx[r]]
                if v != '':
                    Y[a, pi] = float(v)
        return np.nan_to_num(Y), ~np.isnan(Y)

    # per-planner bank view used by the CAT-side scripts ---------------------
    def bank_rows(self, cal_routes, planner):
        """Indices into `cal_routes` observed for `planner`, plus responses."""
        bi = [i for i, r in enumerate(cal_routes) if (r, planner) in self.Y]
        yy = np.array([self.Y[(cal_routes[i], planner)] for i in bi], float)
        return bi, yy

    def split_routes(self, held_out_types):
        """(calibration routes, held-out-type routes), both in CSV order."""
        cal = [r for r in self.allr if self.sn[r] not in held_out_types]
        new = [r for r in self.allr if self.sn[r] in held_out_types]
        return cal, new
