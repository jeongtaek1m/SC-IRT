"""B2D response panel (16 planners x 220 routes) and the route->type map.

Route 11755 failed collection (no type, NaN descriptors) which leaves the
219-route evaluation bank. Sparse dict view for item selection by route id;
dense array view for the noise-ceiling's planner-column splits.
"""

import csv
import hashlib

import numpy as np

from . import paths

#: SHA1 of the canonical 219-route evaluation bank, in response-matrix column
#: order. Guards against a reordered or silently re-filtered universe, which
#: would move the frozen gold difficulty and therefore every downstream number.
CANONICAL_UNIVERSE_SHA1 = None  # populated on first call; see assert_canonical_universe

#: Planner excluded from the panel. Present as a defensive filter in every
#: original script; it matches no row in the shipped matrix but defines J=16.
EXCLUDED_PLANNER = "PDM-Lite"


class ResponsePanel:
    """Binary success responses for a planner panel over a route bank."""

    def __init__(self, route_ids, planners, sparse):
        self.route_ids = route_ids  #: routes in response-matrix column order
        self.planners = planners  #: planner names in row order
        self.y = sparse  #: {(route_id, planner_index): 0|1}

    @property
    def n_planners(self):
        return len(self.planners)

    def observed(self, route_id, planner_index):
        return (route_id, planner_index) in self.y

    def dense(self, routes, planner_indices):
        """(len(routes), len(planner_indices)) array, NaN where unobserved."""
        M = np.full((len(routes), len(planner_indices)), np.nan)
        for a, rid in enumerate(routes):
            for b, pi in enumerate(planner_indices):
                if (rid, pi) in self.y:
                    M[a, b] = self.y[(rid, pi)]
        return M

    def dense_all(self):
        """(n_routes, n_planners) array over every collected route, NaN where unobserved."""
        M = np.full((len(self.route_ids), self.n_planners), np.nan)
        for pi in range(self.n_planners):
            for j, rid in enumerate(self.route_ids):
                if (rid, pi) in self.y:
                    M[j, pi] = self.y[(rid, pi)]
        return M

    def responses_for(self, route_id):
        """Observed (planner_index, response) pairs for one route, in planner order."""
        return [(pi, self.y[(route_id, pi)]) for pi in range(self.n_planners)
                if (route_id, pi) in self.y]


def read_response_panel(path=None):
    """Load the B2D response matrix."""
    path = path or f"{paths.MATRICES}/b2d_e2e16_response_matrix.csv"
    rows = list(csv.reader(open(path)))
    route_ids = rows[0][1:]
    body = [r for r in rows[1:] if r[0] != EXCLUDED_PLANNER]
    sparse = {}
    for pi, row in enumerate(body):
        for j, rid in enumerate(route_ids):
            if row[1 + j] != "":
                sparse[(rid, pi)] = int(float(row[1 + j]))
    return ResponsePanel(route_ids, [r[0] for r in body], sparse)


def read_route_types(path=None):
    """Load the route -> scenario-type map (44 types over 219 routes, no header)."""
    path = path or f"{paths.MATRICES}/b2d_route_types.csv"
    return {
        line.split(",")[0]: line.split(",")[1].strip()
        for line in open(path)
        if "," in line
    }


def route_universe(route_ids, types, *feature_dicts):
    """Routes present in the type map and in every supplied feature dict.

    Iteration follows `route_ids` — response-matrix column order — and never
    sorts. The order sets the item axis of every calibration, and reduction
    order is visible in the last bits of a float32 fit.
    """
    return [
        r
        for r in route_ids
        if r in types and all(r in f for f in feature_dicts)
    ]


def assert_canonical_universe(routes, expect_n=219):
    """Assert the evaluation bank is the canonical one.

    Three separate intersection paths in the original code converge on the same
    219 ids in the same order. Any change to the feature set or the filter order
    would move the frozen gold difficulty silently, so it is checked explicitly.
    """
    global CANONICAL_UNIVERSE_SHA1
    digest = hashlib.sha1(",".join(routes).encode()).hexdigest()
    if CANONICAL_UNIVERSE_SHA1 is None:
        CANONICAL_UNIVERSE_SHA1 = digest
    if len(routes) != expect_n:
        raise AssertionError(
            f"route universe has {len(routes)} entries, expected {expect_n}"
        )
    if digest != CANONICAL_UNIVERSE_SHA1:
        raise AssertionError(
            "route universe differs from the one established earlier in this run"
        )
    return routes


def type_clusters(routes, types):
    """Index arrays grouping routes by scenario type, in sorted type order.

    This is the resampling unit of every cluster bootstrap in the package.
    """
    labels = sorted(set(types[r] for r in routes))
    return [
        np.array([i for i, r in enumerate(routes) if types[r] == t])
        for t in labels
    ]
