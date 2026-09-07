#!/usr/bin/env python3
"""NavSim / nuPlan relational scene graph extractor (relgraph v2).

Twin of b2d_extract.py. KEYS.md is the single source of truth for key names, channel indices,
dtypes and sizes; nothing here may drift from it. Only PRIMITIVE relations are stored -- no
conflict / TTC / risk / hardness channel, no merges_into_route / crosses_route, no raw nuPlan
lane ids in the tensor.

    python navsim_extract.py --debug            # build debug_ids_navsim.json + navsim_relgraph_debug.npz
    python navsim_extract.py --ids FILE         # extract exactly the ids in FILE
    python navsim_extract.py --full             # all 12,146 windows -> navsim_relgraph_v2.npz

Run with /home/jeongtae/miniconda3/envs/smart/bin/python (nuplan-devkit lives only there).

--------------------------------------------------------------------------------------------
SHARED GEOMETRY CONTRACT -- these definitions must read identically in b2d_extract.py.
Anything ambiguous in KEYS.md is pinned down here, and every pinned choice is listed in the
report so the twin can be checked against it.

  GEOMETRY SOURCE   every relation (A2L, route_rel, OPPOSITE_DIRECTION, SUCCESSOR continuity) is
                    computed from the stored P=10 resampled polyline, never from a finer internal
                    one, so a validator can recompute all of it from the tensor alone.

  lane polyline     the centreline is sampled at LANE_SAMPLE = 2 m; of the runs of consecutive
                    samples inside the R_QUERY disc the one passing closest to the ego is kept
                    (a lane that leaves and re-enters is never bridged), and that run is
                    resampled to exactly P points.  A lane with fewer than two samples inside
                    the disc is dropped.  lane_feat[1] = that run's arclength / 95.

  lanes[..., 2:4]   segment vector from this point to the next, in travel direction.  Row P-1
                    has no following segment and is ZERO, so pts[i] + seg[i] == pts[i+1] holds.

  LATERAL AXIS      nuPlan is right-handed, so x forward / y LEFT here, and d_lat > 0 means left
                    of the lane's travel direction as KEYS.md specifies.  b2d_extract.py measured
                    that the frozen B2D window tensor lives in CARLA's LEFT-handed world and is
                    therefore mirrored in y; that clash between the frozen source tensor and the
                    frozen cross-domain rule is recorded on both sides, not patched here.

  a2l_rel[..., 1]   d_lon is an arclength coordinate along the lane centreline whose ORIGIN is
                    p_L*, the point of that lane closest to the ego anchor -- the same origin
                    KEYS.md mandates for route_order_norm and for the same reason: any origin
                    taken from the polyline start moves when the window clipping moves.
                    d_lon > 0 means the agent is further along that lane than the ego is.

  A2L containment   KEYS.md writes tier 1 as |d_lat| <= width/2.  nuPlan does not expose lane
                    width (get_width_left_right raises NotImplementedError), so the identical
                    predicate is read off the lane polygon: lane.polygon.contains(agent).

  OPPOSITE_DIRECTION  evaluated per ORDERED pair exactly as KEYS.md writes it (mid of i against
                    the polyline of j), so a mutually opposite pair yields both i->j and j->i.

  route_reachable_3hop  1..3 SUCCESSOR hops over the lanes selected in this window; a lane never
                    counts as reaching itself in 0 hops.

  shares_downstream_with_route  lane L and some OTHER in-window corridor lane have a common
                    immediate successor lane in the map lane graph.

  E overflow        edges touching a corridor lane first, then by distance to the ego, then by
                    type / src / dst -- the ordering frozen in b2d_extract.py.

  route polyline (NavSim side of the B2D XML reconstruction)
      frame['roadblock_ids'] is an ordered bundle list, not a path.  One lane per roadblock is
      chosen by a DP that forces the roadblock nearest the ego to be entered on the lane the ego
      is actually on and otherwise minimises lateral movement; missing links are bridged by a
      successor BFS.  A lane change is emitted one lane at a time so every join is one lane
      width, which is what the 5 m failure threshold is sized for; a stretch that cannot be
      connected at all leaves a real gap and fails the window instead of being drawn through.
      Only +-ROUTE_SPAN of chain around the ego is reconstructed -- route_order_norm is clipped
      at H_ROUTE and the lanes never leave the R_QUERY window, so nothing beyond it is read.
--------------------------------------------------------------------------------------------
"""
import argparse, json, os, pickle, sys, time
from collections import deque

import numpy as np

os.environ.setdefault('NUPLAN_MAPS_ROOT', '/data2/jeongtae/navsim_meta/maps')

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer as S
from nuplan.common.maps.nuplan_map.lane_connector import NuPlanLaneConnector
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from shapely.geometry import LineString, Point

# ---------------------------------------------------------------- sizes (KEYS.md)
A, T, M, P, K, E = 48, 12, 128, 10, 8, 384   # E: 2026-08-26 256 -> 384. hard case 절단 전 최대 수요 265, 위상 edge 까지 잘렸다
R_QUERY = 60.0            # lane query radius, m
H_ROUTE = 100.0           # route_order_norm normaliser, m
ANCHOR_T = 3              # navtest_tensors_v2 anchor timestep: ego[:, 3, :2] is exactly 0
A2L_RADIUS = 5.0
OPP_MIN, OPP_MAX = 1.0, 6.0
OPP_ANGLE = np.radians(150.0)
SUCC_GAP = 5.0            # geometric guard on SUCCESSOR (2 m sampling on both sides)
LANE_SAMPLE = 2.0         # centreline sampling step, m
ARCLEN_NORM = 95.0        # lane_feat[1] normaliser
REACH_HOPS = 3
ROUTE_BAND = 60           # +-60 route samples (= +-120 m) searched for a projection
ROUTE_STEP = 2.0          # densify / resample step for the route polyline
ROUTE_FAIL_GAP = 5.0      # KEYS.md extractor failure condition
ROUTE_SPAN = 300.0        # how far along the roadblock chain is reconstructed at all

SUCCESSOR, LEFT_NEIGHBOR, RIGHT_NEIGHBOR, OPPOSITE_DIRECTION = 0, 1, 2, 3

# ---- 채널 심볼 (KEYS.md, 2026-08-26 freeze). 숫자 인덱스를 코드에 쓰지 않는다.
# pick_debug 에서 실제로 rr[:, 4] 를 intersect 로 읽는 버그가 나왔다. 재배열은 조용히 틀린다.
SIDECAR_MAP = {'n_lane_before_M': 'n_lane_in_window', 'n_lane_after_M': 'm_used', 'n_route_lane_before_M': 'n_corridor', 'n_route_lane_after_M': 'n_corridor_kept', 'edge_count_before_E': 'n_edges', 'edge_count_after_E': 'n_edge_kept', 'n_agent_containing_before_K': 'a2l_contained_tot', 'n_agent_containing_after_K': 'a2l_contained_kept', 'route_fail': 'route_fail'}

R_ON_ROUTE, R_REACH, R_SHARES, R_XSECT, R_ORDER = 0, 1, 2, 3, 4
LF_JUNCTION, LF_ARCLEN = 0, 1
A2L_DLAT, A2L_DLON, A2L_COS, A2L_SIN = 0, 1, 2, 3


MAPS_ROOT = os.environ['NUPLAN_MAPS_ROOT']
ROOT = '/data2/jeongtae/relgraph_e16sel'
TENSORS = '/data2/jeongtae/navsim_interact/navtest_tensors_v2.npz'
ANCHORS = '/data2/jeongtae/relgraph_e16sel/navtest_anchor_pose.npz'
META = '/data2/jeongtae/navsim_meta/openscene-v1.1/meta_datas/test'


# ================================================================ pure geometry
# This block is the cross-domain contract surface.  Every definition below must read the same in
# b2d_extract.py; where KEYS.md left something open, the choice frozen there is repeated here.
def densify(c, step):
    """resample a polyline at <= step spacing, endpoints preserved."""
    c = np.asarray(c, float)
    if len(c) < 2:
        return c
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 1e-9:
        return c[:1]
    n = int(np.ceil(s[-1] / step)) + 1
    si = np.linspace(0.0, s[-1], n)
    return np.c_[np.interp(si, s, c[:, 0]), np.interp(si, s, c[:, 1])]


def resample(poly, n):
    """resample to exactly n points, uniform in arclength; returns (poly, arclength)."""
    poly = np.asarray(poly, float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 1e-3:
        return None, 0.0
    si = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(si, s, poly[:, 0]), np.interp(si, s, poly[:, 1])], 1), float(s[-1])


def proj_to_segs(pts, segA, segB):
    """pts (n,2) onto segments segA/segB (m,2) -> dist (n,m), t (n,m), proj (n,m,2)."""
    d = segB - segA
    L2 = np.maximum((d * d).sum(1), 1e-12)
    w = pts[:, None, :] - segA[None, :, :]
    t = np.clip((w * d[None]).sum(2) / L2[None], 0.0, 1.0)
    proj = segA[None] + t[..., None] * d[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2), t, proj


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _cross(o, a, b):
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - \
           (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


MIN_CROSS_SIN = 0.1        # ~5.7 deg; below this two resampled polylines cannot be told apart


def segs_intersect(p1, p2, p3, p4):
    """Transversal segment intersection, broadcasting over leading axes.

    The strict sign-flip test alone is NOT enough, despite what this docstring used to claim.
    When the two segments are collinear every cross product is ~0 and its sign is decided by
    floating-point noise, so two different resamplings of ONE straight line register as crossing
    each other 99.85% of the time (measured, 2000 trials). That is exactly the situation
    polyline_intersects_route is in for a lane that IS part of the route: the lane centreline and
    the route polyline are the same geometry sampled at different steps. The channel was firing
    on 70% of B2D corridor lanes as pure numerical noise.

    So require the pair to be genuinely transversal: |sin(angle)| = |u x v| / (|u||v|) above
    MIN_CROSS_SIN. Measured effect: the collinear artefact goes 0.9985 -> 0.0000 while real
    crossings at 10 deg and above are kept at 1.0000. Crossings shallower than ~6 deg are given
    up, which is the right trade -- at that angle the crossing is not determined by the polyline
    resampling in the first place.
    """
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    hit = ((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))
    u, v = p2 - p1, p4 - p3
    cr = np.abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])
    lu = np.hypot(u[..., 0], u[..., 1])
    lv = np.hypot(v[..., 0], v[..., 1])
    return hit & (cr > MIN_CROSS_SIN * lu * lv)


def arclen(c):
    c = np.asarray(c, float)
    return 0.0 if len(c) < 2 else float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())


def to_ego(pts, x, y, yaw):
    """nuPlan is right-handed: x forward, y LEFT.  (B2D's source tensor is mirrored; see the
    note in b2d_extract.py -- that clash is recorded, not silently patched here.)"""
    cs, sn = np.cos(yaw), np.sin(yaw)
    d = np.asarray(pts, float) - np.array([x, y])
    return np.c_[cs * d[:, 0] + sn * d[:, 1], -sn * d[:, 0] + cs * d[:, 1]]


# ================================================================ nuPlan access (memoised)
class Maps:
    def __init__(self):
        self.api, self.succ, self.coord, self.rbid, self.rblanes, self.poly = \
            {}, {}, {}, {}, {}, {}

    def get(self, loc):
        if loc not in self.api:
            self.api[loc] = get_maps_api(MAPS_ROOT, 'nuplan-maps-v1.0', loc)
        return self.api[loc]

    def successors(self, loc, o):
        k = (loc, o.id)
        if k not in self.succ:
            self.succ[k] = list(o.outgoing_edges)
        return self.succ[k]

    def centreline(self, loc, o):
        k = (loc, o.id)
        if k not in self.coord:
            self.coord[k] = np.asarray(o.baseline_path.linestring.coords, float)
        return self.coord[k]

    def roadblock_of(self, loc, o):
        k = (loc, o.id)
        if k not in self.rbid:
            self.rbid[k] = o.get_roadblock_id()
        return self.rbid[k]

    def polygon(self, loc, o):
        k = (loc, o.id)
        if k not in self.poly:
            self.poly[k] = o.polygon
        return self.poly[k]

    def roadblock_lanes(self, loc, rid):
        k = (loc, rid)
        if k not in self.rblanes:
            m = self.get(loc)
            o = m.get_map_object(rid, S.ROADBLOCK) or m.get_map_object(rid, S.ROADBLOCK_CONNECTOR)
            self.rblanes[k] = [] if o is None else list(o.interior_edges)
        return self.rblanes[k]


MAPS = Maps()


def lanes_by_centreline(map_api, x, y, radius):
    """verbatim from the task preamble -- polygon proximity is only a cheap superset prefilter,
    the centreline distance is THE rule."""
    pr = map_api.get_proximal_map_objects(Point2D(x, y), radius, [S.LANE, S.LANE_CONNECTOR])
    p = Point(x, y)
    out = []
    for o in pr[S.LANE] + pr[S.LANE_CONNECTOR]:
        d = o.baseline_path.linestring.distance(p)
        if d <= radius:
            out.append((d, o))
    out.sort(key=lambda t: t[0])
    return out


# ================================================================ route polyline
def _chain_window(mp, loc, rbids, x, y):
    """dedup, look up the lanes of every roadblock, and keep only the stretch of the chain
    within ROUTE_SPAN of the ego on each side of the seed roadblock."""
    ids = []
    for r in rbids:
        if not ids or ids[-1] != r:
            ids.append(r)
    layers = [(r, mp.roadblock_lanes(loc, r)) for r in ids]
    layers = [(r, ls) for r, ls in layers if ls]
    if not layers:
        return None, None, None
    best = (np.inf, -1, -1)
    for k, (_, ls) in enumerate(layers):
        for j, e in enumerate(ls):
            c = mp.centreline(loc, e)
            d = float(np.hypot(c[:, 0] - x, c[:, 1] - y).min())
            if d < best[0]:
                best = (d, k, j)
    d_ego, seed, seed_j = best
    lo, acc = seed, 0.0
    while lo > 0 and acc < ROUTE_SPAN:
        lo -= 1
        acc += max(arclen(mp.centreline(loc, e)) for e in layers[lo][1])
    hi, acc = seed, 0.0
    while hi < len(layers) - 1 and acc < ROUTE_SPAN:
        hi += 1
        acc += max(arclen(mp.centreline(loc, e)) for e in layers[hi][1])
    return layers[lo:hi + 1], seed - lo, (seed_j, d_ego)


def _lateral_order(mp, loc, lanes):
    """order the lanes of a roadblock left to right, measured geometrically so we never rely on
    a nuPlan lane_index.  Returns rank per lane index, or None if the bundle is not parallel
    (a roadblock CONNECTOR can bundle diverging manoeuvres, where 'lane change' is meaningless)."""
    ref = densify(mp.centreline(loc, lanes[0]), ROUTE_STEP)
    if len(ref) < 2:
        return None
    segA, segB = ref[:-1], ref[1:]
    th = np.arctan2(segB[:, 1] - segA[:, 1], segB[:, 0] - segA[:, 0])
    mids = []
    for e in lanes:
        mp3 = resample(mp.centreline(loc, e), 3)[0]
        if mp3 is None:
            return None
        mids.append(mp3[1])
    mids = np.array(mids)
    d, _, pr = proj_to_segs(mids, segA, segB)
    k = d.argmin(1)
    dv = mids - pr[np.arange(len(mids)), k]
    off = -np.sin(th[k]) * dv[:, 0] + np.cos(th[k]) * dv[:, 1]
    return {i: r for r, i in enumerate(np.argsort(off))}


def _bridge(mp, loc, src, tgt_ids, maxhop=3):
    """successor-only BFS from src to any lane in tgt_ids; returns the inserted lanes."""
    q = deque([(src, [])])
    seen = {src.id}
    while q:
        w, path = q.popleft()
        if len(path) >= maxhop:
            continue
        for n in mp.successors(loc, w):
            if n.id in tgt_ids:
                return path + [n]
            if n.id not in seen:
                seen.add(n.id)
                q.append((n, path + [n]))
    return None


def build_route(mp, loc, rbids, x, y):
    """Ordered route polyline out of the roadblock bundle.

    A roadblock is a bundle of parallel lanes, not a path, so one lane per roadblock is chosen
    by a DP that (a) forces the roadblock nearest the ego to be entered on the lane the ego is
    actually on and (b) otherwise minimises the number of lane changes.  A lane change inside a
    roadblock is emitted honestly: the first half of the entry lane, then the second half of the
    exit lane, which leaves a join of about one lane width -- exactly what the 5 m failure
    threshold is meant to tolerate.  A stretch that cannot be connected at all is a hole; the
    polyline is cut there rather than being drawn through it.
    """
    layers, seed, seedinfo = _chain_window(mp, loc, rbids, x, y)
    if layers is None:
        return None
    seed_j, d_ego = seedinfo
    n = len(layers)

    # transitions: exit lane index at layer k -> {entry index at layer k+1: [bridge lanes]}
    trans = []
    for k in range(n - 1):
        tgt = {e.id: j for j, e in enumerate(layers[k + 1][1])}
        row = {}
        for i, e in enumerate(layers[k][1]):
            hit = {}
            for s_ in mp.successors(loc, e):
                if s_.id in tgt:
                    hit[tgt[s_.id]] = []
            if not hit:
                br = _bridge(mp, loc, e, set(tgt))
                if br is not None:
                    hit[tgt[br[-1].id]] = br[:-1]
            row[i] = hit
        trans.append(row)

    BIG = 10 ** 6
    INF = float('inf')
    nl = [len(ls) for _, ls in layers]
    # a lane change only makes sense inside a plain roadblock: a roadblock connector bundles
    # alternative manoeuvres, not parallel lanes
    lat = [(_lateral_order(mp, loc, ls) if all(not isinstance(e, NuPlanLaneConnector) for e in ls)
            else None) for _, ls in layers]
    cost = [[INF] * nl[k] for k in range(n)]
    back = [[None] * nl[k] for k in range(n)]
    breaks = []
    for j in range(nl[0]):
        cost[0][j] = 0 if (0 != seed or j == seed_j) else BIG
    for k in range(n - 1):
        if all(c == INF for c in cost[k]):                       # hole: restart the run here
            breaks.append(k)
            for j in range(nl[k]):
                cost[k][j] = 0 if (k != seed or j == seed_j) else BIG
        for entry in range(nl[k]):
            if cost[k][entry] == INF:
                continue
            for ex in range(nl[k]):
                if ex != entry and lat[k] is None:
                    continue
                step = 0 if ex == entry else abs(lat[k][ex] - lat[k][entry])
                c = cost[k][entry] + step
                for nxt, br in trans[k][ex].items():
                    add = 0 if (k + 1 != seed or nxt == seed_j) else BIG
                    if c + add < cost[k + 1][nxt]:
                        cost[k + 1][nxt] = c + add
                        back[k + 1][nxt] = (entry, ex, br)
    if all(c == INF for c in cost[n - 1]):
        breaks.append(n - 1)
        for j in range(nl[n - 1]):
            cost[n - 1][j] = 0

    end = int(np.argmin(cost[n - 1]))
    path = []                                                     # (layer, entry, exit, bridge)
    cur, k = end, n - 1
    while k > 0:
        b = back[k][cur]
        if b is None:                                             # run start (hole above)
            path.append((k, cur, cur, []))
            k -= 1
            cur = int(np.argmin(cost[k]))
            continue
        entry, ex, br = b
        path.append((k, cur, cur, br))
        path.append(('exit', k - 1, entry, ex))
        cur, k = entry, k - 1
    path.append((0, cur, cur, []))

    # rebuild forward: per layer (entry, exit) and the bridge lanes that precede it
    per = {}
    bridges = {}
    for item in path:
        if item[0] == 'exit':
            _, kk, entry, ex = item
            per[kk] = (entry, ex)
        else:
            kk, entry, _, br = item
            per.setdefault(kk, (entry, entry))
            bridges[kk] = br
    pts = []
    lane_changes = 0
    for k in range(n):
        if k not in per:
            continue
        for b in bridges.get(k, []):
            pts.append(densify(mp.centreline(loc, b), ROUTE_STEP))
        entry, ex = per[k]
        ls = layers[k][1]
        if entry == ex or lat[k] is None:
            pts.append(densify(mp.centreline(loc, ls[entry]), ROUTE_STEP))
        else:
            # walk lane by lane so every join is one lane width, never a multi-lane jump
            rank = lat[k]
            inv = {r: i for i, r in rank.items()}
            r0, r1 = rank[entry], rank[ex]
            stepr = 1 if r1 > r0 else -1
            seq = [inv[r] for r in range(r0, r1 + stepr, stepr)]
            lane_changes += len(seq) - 1
            cur = densify(mp.centreline(loc, ls[seq[0]]), ROUTE_STEP)
            i0 = 0
            for j in range(len(seq) - 1):
                cut = int(round((j + 1) / len(seq) * (len(cur) - 1)))
                cut = min(max(cut, i0 + 1), len(cur) - 1)
                pts.append(cur[i0:cut + 1])
                q = cur[cut]
                nxt = densify(mp.centreline(loc, ls[seq[j + 1]]), ROUTE_STEP)
                i0 = int(np.hypot(nxt[:, 0] - q[0], nxt[:, 1] - q[1]).argmin())
                if i0 >= len(nxt) - 1:
                    i0 = max(0, len(nxt) - 2)
                cur = nxt
            pts.append(cur[i0:])
    pts = [p for p in pts if len(p)]
    if not pts:
        return None
    Praw = np.vstack(pts)
    keep = np.r_[True, np.linalg.norm(np.diff(Praw, axis=0), axis=1) > 1e-6]
    Praw = Praw[keep]
    if len(Praw) < 2:
        return None

    # trim to +-ROUTE_SPAN of the ego along the chain before judging the gap: a break far behind
    # the horizon cannot touch route_order_norm (clipped at H_ROUTE) or the 60 m window
    seg = np.linalg.norm(np.diff(Praw, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    j = int(np.hypot(Praw[:, 0] - x, Praw[:, 1] - y).argmin())
    sel = (s >= s[j] - ROUTE_SPAN) & (s <= s[j] + ROUTE_SPAN)
    Praw = Praw[sel]
    if len(Praw) < 2:
        return None
    seg = np.linalg.norm(np.diff(Praw, axis=0), axis=1)
    raw_gap = float(seg.max())
    dense = densify(Praw, ROUTE_STEP)
    dseg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    return dict(route=dense, raw=Praw, raw_gap=raw_gap,
                dense_gap=float(dseg.max()) if len(dseg) else 0.0,
                arclen=float(seg.sum()), n_breaks=len(breaks), lane_changes=lane_changes,
                d_ego=float(d_ego), ok=raw_gap < ROUTE_FAIL_GAP)


# ================================================================ one window
def extract_window(mp, loc, rbids, pose, ag, am):
    """returns a dict of per-window arrays plus diagnostics, or None on route failure."""
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    m_api = mp.get(loc)
    diag = {}

    rt = build_route(mp, loc, rbids, x, y)
    route_broken = rt is None or not rt['ok']
    if route_broken:
        # 2026-08-26: 드롭하지 않고 route_rel_valid=False 로 내보낸다.
        # roadblock_ids 의 연속 roadblock 이 220 m 씩 떨어져 chain 을 못 잇는 경우가
        # 12,146 중 29 개(0.24%) 있다. 직선으로 메우면 계약 위반이므로 order 는 못 준다.
        # 그런데 이 window 의 agent / lane / A2L 그래프는 멀쩡하고, 드롭하면 응답행렬과의
        # 행 정렬까지 깨진다. mask 는 정확히 이 경우를 위해 넣은 것이다.
        # route_rel 5채널 전부 0 으로 두고 valid=False 로 표시한다.
        rt = dict(route=np.array([[x, y], [x + 1.0, y]], float), arclen=0.0, raw_gap=None if rt is None else rt['raw_gap'],
                  dense_gap=None if rt is None else rt['dense_gap'], n_breaks=0,
                  lane_changes=0, d_ego=float('nan'), ok=False)
    rbset = set(rbids)

    # ---- route polyline in the ego frame, and the arclength band it is read in --------------
    # +-120 m of samples around the ego so that a route which doubles back (hairpins, loops)
    # cannot project a lane onto the wrong pass.
    re = to_ego(rt['route'], x, y, yaw)
    rd = np.linalg.norm(re, axis=1)
    rs = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(re, axis=0), axis=1))]
    i_ego = int(np.argmin(rd))
    b0, b1 = max(0, i_ego - ROUTE_BAND), min(len(re), i_ego + ROUTE_BAND + 1)
    rb, sb = re[b0:b1], rs[b0:b1]
    s_ego, rA, rB, rlen = 0.0, None, None, None
    if len(rb) >= 2:
        rA, rB = rb[:-1], rb[1:]
        rlen = np.linalg.norm(rB - rA, axis=1)
        dr, tr, _ = proj_to_segs(np.zeros((1, 2)), rA, rB)
        k0 = int(dr[0].argmin())
        s_ego = sb[k0] + tr[0, k0] * rlen[k0]

    def route_s(pts):
        """arclength of each point's projection on the route, relative to the ego."""
        if rA is None or not len(pts):
            return np.zeros(len(pts))
        dL, tL, _ = proj_to_segs(pts, rA, rB)
        kL = dL.argmin(1)
        return sb[kL] + tL[np.arange(len(pts)), kL] * rlen[kL] - s_ego

    # ---- lanes in the window ------------------------------------------------------------
    # centreline distance is THE rule (KEYS.md); the polygon prefilter inside lanes_by_centreline
    # is a guaranteed superset.  Each lane keeps the contiguous run of samples inside the disc
    # that passes closest to the ego, so a lane that leaves and re-enters is never bridged.
    cand = lanes_by_centreline(m_api, x, y, R_QUERY)
    objs, polys, arcs, corr, junc = [], [], [], [], []
    n_short = 0
    for d, o in cand:
        c = densify(mp.centreline(loc, o), LANE_SAMPLE)
        cd_ = np.hypot(c[:, 0] - x, c[:, 1] - y)
        inw = np.nonzero(cd_ <= R_QUERY)[0]
        if len(inw) < 2:
            n_short += 1
            continue
        runs = np.split(inw, np.nonzero(np.diff(inw) != 1)[0] + 1)
        run = min(runs, key=lambda r: cd_[r].min())
        if len(run) < 2:
            n_short += 1
            continue
        poly, al = resample(to_ego(c[run], x, y, yaw), P)
        if poly is None:
            n_short += 1
            continue
        objs.append(o); polys.append(poly); arcs.append(al)
        corr.append(mp.roadblock_of(loc, o) in rbset)
        junc.append(isinstance(o, NuPlanLaneConnector))
    if not objs:
        return None, dict(no_lane=True)

    poly_c = np.stack(polys)
    arclen_c = np.array(arcs, float)
    junc_c = np.array(junc, bool)
    nc = len(objs)
    segA_c, segB_c = poly_c[:, :-1].reshape(-1, 2), poly_c[:, 1:].reshape(-1, 2)
    dc, tc, prc = proj_to_segs(np.zeros((1, 2)), segA_c, segB_c)
    dc = dc.reshape(nc, P - 1)
    kc = dc.argmin(1)
    d_ego_c = dc[np.arange(nc), kc]
    p_star_c = prc.reshape(nc, P - 1, 2)[np.arange(nc), kc]
    # A roadblock on the mission route is only a corridor member while it is inside the route
    # horizon: KEYS.md defines H_route = 100 m as the extent of the route corridor, and a NavSim
    # roadblock_ids list is a whole ~1 km mission, so a street the route uses again 400 m later
    # is not this window's corridor.  Without this, route_order_norm saturates at +-1 and the
    # corridor splits into disconnected components.  (No-op on B2D, whose routes are ~100 m.)
    ds_c = route_s(p_star_c)
    on_route_c = np.array(corr, bool) & (np.abs(ds_c) <= H_ROUTE)
    if route_broken:
        on_route_c[:] = False        # chain 이 없으니 corridor 도 신뢰할 수 없다
    diag['corridor_beyond_horizon'] = int((np.array(corr, bool) & ~on_route_c).sum())

    n_corr = int(on_route_c.sum())
    if n_corr > M:
        raise RuntimeError(f'corridor lanes {n_corr} exceed M={M}: extractor failure by KEYS.md')
    # priority-retain (KEYS.md): every corridor lane survives, the rest fill by centreline
    # distance.  lane index therefore carries no meaning beyond that and must stay a set axis.
    order = np.lexsort((np.arange(nc), d_ego_c, ~on_route_c))[:M]
    m_ = len(order)
    objs = [objs[i] for i in order]
    poly = poly_c[order]
    on_route, arclen_s = on_route_c[order], arclen_c[order]
    junction = junc_c[order]
    ds_s = ds_c[order]
    dropped_on_route = n_corr - int(on_route.sum())
    assert dropped_on_route == 0, f'dropped_on_route_lane={dropped_on_route}'
    diag.update(n_lane_in_window=nc, n_short=n_short, n_corridor=n_corr, m_used=m_,
                n_corridor_kept=int(on_route.sum()),   # cap 이후. before 와 같은 키를 쓰면
                dropped_on_route=dropped_on_route)     # sidecar 가 자기 자신과 비교하게 된다

    segA = poly[:, :-1].reshape(-1, 2)
    segB = poly[:, 1:].reshape(-1, 2)
    dseg = segB - segA
    seg_th = np.arctan2(dseg[:, 1], dseg[:, 0]).reshape(m_, P - 1)
    seg_len = np.linalg.norm(dseg, axis=1).reshape(m_, P - 1)
    seg_s0 = np.c_[np.zeros((m_, 1)), np.cumsum(seg_len, 1)[:, :-1]]
    ar = np.arange(m_)

    # p_L* : point of each lane closest to the ego anchor (never the polyline start)
    d0, t0, pr0 = proj_to_segs(np.zeros((1, 2)), segA, segB)
    d0 = d0.reshape(m_, P - 1)
    k_star = d0.argmin(1)
    d_ego = d0[ar, k_star]
    p_star = pr0.reshape(m_, P - 1, 2)[ar, k_star]
    s_star = seg_s0[ar, k_star] + t0.reshape(m_, P - 1)[ar, k_star] * seg_len[ar, k_star]

    lanes = np.zeros((M, P, 4), np.float32)
    lane_feat = np.zeros((M, 2), np.float32)
    lane_mask = np.zeros(M, bool)
    lanes[:m_, :, :2] = poly
    lanes[:m_, :-1, 2:] = np.diff(poly, axis=1)
    lanes[:m_, -1, 2:] = 0.0    # row P-1 starts no segment; pts[-1]+seg[-1] == pts[-1]
    lane_feat[:m_, LF_JUNCTION] = junction
    lane_feat[:m_, LF_ARCLEN] = arclen_s / ARCLEN_NORM
    lane_mask[:m_] = True

    # ---- L2L ----------------------------------------------------------------------------
    pos_of = {o.id: i for i, o in enumerate(objs)}
    edges, succ_pairs, succ_guard = [], [], 0
    for i, o in enumerate(objs):
        for s_ in mp.successors(loc, o):
            j = pos_of.get(s_.id)
            if j is None or j == i:
                continue
            # stored polylines are window-clipped; emit the edge only where the two clipped
            # polylines actually meet, which is exactly what the validator measures.
            if np.linalg.norm(poly[i, -1] - poly[j, 0]) > SUCC_GAP:
                succ_guard += 1
                continue
            edges.append((i, j, SUCCESSOR))
            succ_pairs.append((i, j))
    n_dir_violation = 0
    for i, o in enumerate(objs):
        if junction[i]:
            continue                                     # undefined inside a junction
        try:
            left, right = o.adjacent_edges               # nuPlan: same roadblock, so same heading
        except Exception:
            left = right = None
        for nb, tp in ((left, LEFT_NEIGHBOR), (right, RIGHT_NEIGHBOR)):
            if nb is None:
                continue
            j = pos_of.get(nb.id)
            if j is None or j == i or junction[j]:
                continue
            if abs(wrap(seg_th[i, (P - 1) // 2] - seg_th[j, (P - 1) // 2])) > np.pi / 2:
                n_dir_violation += 1
                continue                                 # same travel direction only
            edges.append((i, j, tp))
    diag['neighbor_dir_violation'] = n_dir_violation

    mid = 0.5 * (poly[:, (P - 1) // 2] + poly[:, P // 2])
    mid_th = seg_th[:, (P - 1) // 2]
    dmat = proj_to_segs(mid, segA, segB)[0].reshape(m_, m_, P - 1)
    kmin = dmat.argmin(2)
    dmin = np.take_along_axis(dmat, kmin[..., None], 2)[..., 0]
    th_j = seg_th[np.broadcast_to(ar[None, :], (m_, m_)), kmin]
    opp = (dmin > OPP_MIN) & (dmin < OPP_MAX) & (np.abs(wrap(mid_th[:, None] - th_j)) > OPP_ANGLE)
    np.fill_diagonal(opp, False)
    n_opp_raw = int(opp.sum())
    for i, j in zip(*np.nonzero(opp)):
        edges.append((int(i), int(j), OPPOSITE_DIRECTION))

    n_raw = len(edges)
    if edges:
        ek = np.array(edges, np.int64)
        prio = np.where(on_route[ek[:, 0]] | on_route[ek[:, 1]], 0, 1)
        near = np.minimum(d_ego[ek[:, 0]], d_ego[ek[:, 1]])
        ek = ek[np.lexsort((ek[:, 1], ek[:, 0], ek[:, 2], near, prio))[:E]]
    else:
        ek = np.zeros((0, 3), np.int64)
    l2l_src = np.full(E, -1, np.int64)
    l2l_dst = np.full(E, -1, np.int64)
    l2l_type = np.full(E, -1, np.int64)
    l2l_mask = np.zeros(E, bool)
    l2l_src[:len(ek)], l2l_dst[:len(ek)], l2l_type[:len(ek)] = ek[:, 0], ek[:, 1], ek[:, 2]
    l2l_mask[:len(ek)] = True
    et = ek[:, 2] if len(ek) else np.zeros(0, np.int64)
    diag.update(n_edges=n_raw, n_edge_kept=int(len(ek)),
                n_topo_edges=n_raw - n_opp_raw, n_opp_edges=n_opp_raw,
                edge_overflow=int(n_raw > E), succ_guard=succ_guard,
                dropped_topo=max(0, (n_raw - n_opp_raw) - int((et != OPPOSITE_DIRECTION).sum())))

    # ---- A2L ----------------------------------------------------------------------------
    a2l_idx = np.full((A, K), -1, np.int64)
    a2l_rel = np.zeros((A, K, 4), np.float32)
    a2l_mask = np.zeros((A, K), bool)
    cs_, sn_ = np.cos(yaw), np.sin(yaw)
    live = np.nonzero(am[:, ANCHOR_T])[0]
    n_cand_hist, cand0_moving, contained_kept, contained_tot = [], 0, 0, 0
    dlat_all = []
    if len(live):
        apt = ag[live, ANCHOR_T, :2].astype(np.float64)
        gx = x + cs_ * apt[:, 0] - sn_ * apt[:, 1]
        gy = y + sn_ * apt[:, 0] + cs_ * apt[:, 1]
        apsi = np.arctan2(ag[live, ANCHOR_T, 3].astype(np.float64),
                          ag[live, ANCHOR_T, 2].astype(np.float64))
        veh = ag[live, ANCHOR_T, 7] > 0.5
        spd = ag[live, ANCHOR_T, 4].astype(np.float64)
        n = len(live)
        ad, at, aproj = proj_to_segs(apt, segA, segB)
        ad = ad.reshape(n, m_, P - 1)
        at = at.reshape(n, m_, P - 1)
        aproj = aproj.reshape(n, m_, P - 1, 2)
        ka = ad.argmin(2)
        da = np.take_along_axis(ad, ka[..., None], 2)[..., 0]
        for r in range(n):
            cset = np.nonzero(da[r] <= A2L_RADIUS)[0]
            n_cand_hist.append(len(cset))
            if veh[r] and spd[r] > 0.5:
                diag.setdefault('a2l_cand_moving', []).append(len(cset))
                if len(cset) == 0:
                    cand0_moving += 1
            if len(cset) == 0:
                continue
            kk = ka[r, cset]
            th = seg_th[cset, kk]
            dv = apt[r] - aproj[r, cset, kk]
            dlat = -np.sin(th) * dv[:, 0] + np.cos(th) * dv[:, 1]        # left positive
            dlon = seg_s0[cset, kk] + at[r, cset, kk] * seg_len[cset, kk] - s_star[cset]
            dpsi = wrap(apsi[r] - th)
            # containment tier: B2D uses |d_lat| <= width/2; nuPlan does not expose lane width
            # (get_width_left_right raises), so the same predicate is read off the lane polygon.
            pt = Point(gx[r], gy[r])
            inside = np.array([mp.polygon(loc, objs[int(c)]).contains(pt) for c in cset])
            if inside.any():
                contained_tot += 1
            # tier 2 applies to vehicles only; pedestrians skip it (KEYS.md)
            agree = np.where(np.cos(dpsi) > 0, 0, 1) if veh[r] else np.zeros(len(cset), np.int64)
            o_ = np.lexsort((cset, da[r, cset], agree, np.where(inside, 0, 1)))[:K]
            if inside.any() and inside[o_].any():
                contained_kept += 1
            nk = len(o_)
            g = live[r]
            a2l_idx[g, :nk] = cset[o_]
            a2l_rel[g, :nk, A2L_DLAT] = dlat[o_]
            a2l_rel[g, :nk, A2L_DLON] = dlon[o_]
            a2l_rel[g, :nk, A2L_COS] = np.cos(dpsi[o_])
            a2l_rel[g, :nk, A2L_SIN] = np.sin(dpsi[o_])
            a2l_mask[g, :nk] = True
            dlat_all.extend(dlat[o_].tolist())
            if nk:
                diag.setdefault('a2l_dlat_top1', []).append(float(dlat[o_[0]]))
                if veh[r] and spd[r] > 0.5:
                    diag.setdefault('a2l_dlat_top1_moving', []).append(float(dlat[o_[0]]))
    diag.update(a2l_cand=n_cand_hist, a2l_cand0_moving=cand0_moving,
                a2l_contained_kept=contained_kept, a2l_contained_tot=contained_tot,
                a2l_dlat=dlat_all)

    # ---- route_rel ----------------------------------------------------------------------
    route_rel = np.zeros((M, 5), np.float32)
    route_rel[:m_, R_ON_ROUTE] = on_route
    route_rel[:m_, R_ORDER] = np.where(on_route, np.clip(ds_s / H_ROUTE, -1.0, 1.0), 0.0)
    # crossing is a plain geometric fact, so it is tested against every route segment that comes
    # near this window -- including a second pass of the route that lies outside the band above.
    keep_r = rd < R_QUERY + 5.0
    seg_r = np.nonzero(keep_r[:-1] | keep_r[1:])[0]
    n_int = 0
    if len(seg_r):
        cA, cB = re[seg_r], re[seg_r + 1]
        cross = segs_intersect(segA[:, None, :], segB[:, None, :], cA[None], cB[None])
        route_rel[:m_, R_XSECT] = cross.reshape(m_, P - 1, -1).any((1, 2))
        n_int = int(route_rel[:m_, R_XSECT].sum())

    if on_route.any():
        adj = {}
        for i, j in succ_pairs:
            adj.setdefault(i, []).append(j)
        tgt = set(np.nonzero(on_route)[0].tolist())
        for i in range(m_):
            q, seen = deque((j, 1) for j in adj.get(i, [])), set()
            while q:
                v, h = q.popleft()
                if v in tgt:
                    route_rel[i, R_REACH] = 1.0
                    break
                if h < REACH_HOPS and v not in seen:
                    seen.add(v)
                    q.extend((w2, h + 1) for w2 in adj.get(v, []))
        succ_ids = [[s_.id for s_ in mp.successors(loc, o)] for o in objs]
        owners = {}
        for i in np.nonzero(on_route)[0]:
            for cj in succ_ids[int(i)]:
                owners.setdefault(cj, set()).add(int(i))
        for i in range(m_):
            for cj in succ_ids[i]:
                if owners.get(cj, set()) - {i}:
                    route_rel[i, R_SHARES] = 1.0
                    break

    i_near = int(d_ego.argmin())
    diag['ego_lane_is_corridor'] = int(bool(on_route[i_near]))
    diag['ego_lane_dlat'] = float(d_ego[i_near])
    cset2 = set(np.nonzero(on_route)[0].tolist())
    with_succ = {i for i, _ in succ_pairs}
    diag['corr_with_succ'] = len(cset2 & with_succ)
    diag['corr_linked'] = len({i for i, j in succ_pairs if i in cset2 and j in cset2})
    diag['corr_total'] = len(cset2)
    diag['n_intersect'] = n_int
    diag['route_dir_cos'] = float(np.cos(wrap(np.arctan2(
        *(re[min(i_ego + 1, len(re) - 1)] - re[max(i_ego - 1, 0)])[::-1]))))


    # route_rel_valid: route_rel 의 0 을 "route 와 무관" 으로 읽어도 되는가.
    # corridor lane 이 하나도 없으면 그 0 은 "무관" 이 아니라 "60 m crop 안에 판단 근거가
    # 없음" 이다. 둘을 같은 0 으로 주면 false negative supervision 이 된다.
    # (NavSim 모집단 2.60% — metadata/crop coverage 문제이지 extractor 버그가 아니다.)
    route_valid = bool(on_route.any()) and not route_broken
    if not route_valid:
        # KEYS.md: valid=False 면 5채널 전부 unknown 이고 반드시 0 으로 저장한다.
        # corridor 가 비어도 polyline_intersects_route 는 route polyline 과의 순수 기하라
        # 1 이 될 수 있다 (실측 347 window 중 2 건). 그 1 을 남기면 "route 를 가로지른다"
        # 는 단언이 되어, mask 를 둔 취지와 어긋난다.
        route_rel[:] = 0.0


    return dict(lanes=lanes, lane_feat=lane_feat, lane_mask=lane_mask,
                l2l_src=l2l_src, l2l_dst=l2l_dst, l2l_type=l2l_type, l2l_mask=l2l_mask,
                a2l_idx=a2l_idx, a2l_rel=a2l_rel, a2l_mask=a2l_mask,
                route_rel=route_rel,
                route_rel_valid=np.bool_(route_valid)), dict(diag, route_fail=route_broken, raw_gap=rt['raw_gap'],
                                           dense_gap=rt['dense_gap'], route_arclen=rt['arclen'],
                                           route_breaks=rt['n_breaks'],
                                           route_lane_changes=rt['lane_changes'],
                                           route_d_ego=rt['d_ego'])


# ================================================================ driver
def load_inputs():
    tz = np.load(TENSORS, allow_pickle=True)
    ap = np.load(ANCHORS, allow_pickle=True)
    names = tz['names']
    assert len(set(names.tolist())) == len(names), 'navtest_tensors_v2 tokens are not unique'
    pose_by_token = {tok: pose for tok, pose in zip(ap['token'], ap['anchor_pose_global'])}
    assert len(set(ap['token'].tolist())) == len(ap['token']), 'anchor tokens are not unique'
    assert set(names.tolist()) <= set(pose_by_token), 'token join is not a subset'
    meta_by_token = {tok: (lg, int(fi), ml) for tok, lg, fi, ml in
                     zip(ap['token'], ap['log'], ap['frame_idx'], ap['map_location'])}
    return tz, names, pose_by_token, meta_by_token


_LOGCACHE = {}


def frames_of(log):
    if log not in _LOGCACHE:
        if len(_LOGCACHE) > 6:
            _LOGCACHE.clear()
        with open(f'{META}/{log}.pkl', 'rb') as f:
            _LOGCACHE[log] = pickle.load(f)
    return _LOGCACHE[log]


def write_sidecar(path, ids, diags, get):
    """Per-window audit statistics, in a SEPARATE file so the frozen tensor schema is untouched.

    Lane width is deliberately absent from the input schema, so "is this agent inside that lane"
    cannot be recomputed from the tensor.  Verification therefore has to use the native map
    (CARLA lane_width / nuPlan lane polygon), and that evidence would otherwise live only in a
    log line.  These arrays are what a reviewer asking "how did you verify K truncation preserves
    the assigned lane without representing width?" needs to see.
    """
    import numpy as _np
    cols = ['n_lane_before_M', 'n_lane_after_M', 'n_route_lane_before_M', 'n_route_lane_after_M',
            'edge_count_before_E', 'edge_count_after_E',
            'n_agent_containing_before_K', 'n_agent_containing_after_K']
    out = {c: _np.array([get(d, c) for d in diags], _np.int32) for c in cols}
    out['item_id'] = _np.array(ids, dtype='<U32')
    out['route_fail'] = _np.array([bool(get(d, 'route_fail')) for d in diags], bool)
    _np.savez_compressed(path, **out)
    print(f'wrote {path}  ({len(ids)} windows, {len(cols)} audit columns)')

def run(tokens, out, tz, names, pose_by_token, meta_by_token, quiet=False):
    row_of = {t: i for i, t in enumerate(names.tolist())}
    ag_all, am_all, ego_all, cmd_all = tz['agents'], tz['amask'], tz['ego'], tz['cmd']
    # process grouped by log so the meta pickles are read once, emit in the caller's order
    proc = sorted(tokens, key=lambda t: (meta_by_token[t][0], meta_by_token[t][1]))
    got, diag_of = {}, {}
    t0 = time.time()
    for n, tok in enumerate(proc):
        log, fi, loc = meta_by_token[tok]
        fr = frames_of(log)[fi]
        assert fr['token'] == tok, f'frame/token mismatch {tok}'
        i = row_of[tok]
        pk, dg = extract_window(MAPS, loc, fr['roadblock_ids'], pose_by_token[tok],
                                ag_all[i], am_all[i])
        dg['token'] = tok
        dg['map'] = loc
        diag_of[tok] = dg
        if pk is not None:
            got[tok] = pk
        if not quiet and (n + 1) % 500 == 0:
            print(f'  {n + 1}/{len(tokens)}  {time.time() - t0:.0f}s', flush=True)
    keep = [t for t in tokens if t in got]
    packs = [got[t] for t in keep]
    diags = [diag_of[t] for t in tokens]
    if out is None:
        return keep, packs, diags

    N = len(keep)
    rows = [row_of[t] for t in keep]
    stack = lambda k, dt: np.stack([p[k] for p in packs]).astype(dt)
    np.savez_compressed(
        out,
        lanes=stack('lanes', np.float16),
        lane_feat=stack('lane_feat', np.float16),
        lane_mask=stack('lane_mask', bool),
        l2l_src=stack('l2l_src', np.int16),
        l2l_dst=stack('l2l_dst', np.int16),
        l2l_type=stack('l2l_type', np.int8),
        l2l_mask=stack('l2l_mask', bool),
        a2l_idx=stack('a2l_idx', np.int16),
        a2l_rel=stack('a2l_rel', np.float16),
        a2l_mask=stack('a2l_mask', bool),
        route_rel=stack('route_rel', np.float16),
        route_rel_valid=stack('route_rel_valid', bool),
        agents=ag_all[rows].astype(np.float16),
        agent_mask=am_all[rows].astype(bool),
        ego=ego_all[rows].astype(np.float16),
        command=cmd_all[rows].astype(np.float16),
        item_id=np.array(keep, dtype='<U32'),
        domain=np.array('navsim', dtype='<U8'))
    print(f'wrote {out}  N={N}')
    dg_of = {d['token']: d for d in diags}
    write_sidecar(str(out).replace('.npz', '_audit.npz'), keep, [dg_of[t] for t in keep],
                  lambda dg, c: 0 if SIDECAR_MAP[c] is None else dg.get(SIDECAR_MAP[c], 0))
    return keep, packs, diags


# ---------------------------------------------------------------- debug set
CASES = ['junction_moving_vehicle', 'route_intersecting_candidate', 'straight_non_junction',
         'stopped_waiting', 'dense_agents', 'many_opposite_lanes', 'corridor_near_cap',
         'a2l_zero_or_capped', 'over_M_lanes']

# 2026-08-26: 9번째 case. case 7 은 corridor lane 수로 정렬해서 총 lane 수가 많은 window 를
# 한 번도 못 잡았고, 그 결과 M=128 cap 이 debug set 에서 단 한 번도 걸리지 않아
# corridor_recall_after_M = 1.000 이 공허하게 참이었다. 전수 스캔(scan_lanes_navsim.json)
# 결과 12,146 window 중 lane >= 128 인 것은 11 개뿐이라 무작위/근접 탐색으로는 안 잡힌다.
# 그 11 개를 통째로 넣는다. B2D 는 2,656 window 전수 최대가 68 이라 이 case 가 비어 있다.
HARDCASE_OVER_M = '/data2/jeongtae/relgraph_e16sel/hardcase_navsim_overM.json'


def _over_m_tokens():
    try:
        import json as _j
        return _j.load(open(HARDCASE_OVER_M))
    except Exception:
        return []


def pick_debug(tokens, packs, diags, ego_all, row_of, meta_by_token, per_case=5):
    ok = {d['token']: (p, d) for p, d in zip(packs, [d for d in diags if not d.get('route_fail')
                                                     and not d.get('no_lane')])}
    feats = {}
    for tok, (p, d) in ok.items():
        i = row_of[tok]
        egos = float(ego_all[i, ANCHOR_T, 4])
        lm = p['lane_mask']
        rr = p['route_rel']
        lf = p['lane_feat']
        feats[tok] = dict(
            ego_speed=egos,
            n_agents=int(d.get('a2l_cand') is not None and len(d['a2l_cand'])),
            n_moving_veh=d.get('n_moving_veh', 0),
            n_inter_offroute=int(((rr[:, R_XSECT] > 0.5) & (rr[:, R_ON_ROUTE] < 0.5) & lm).sum()),
            n_junction=int(((lf[:, 0] > 0.5) & lm).sum()),
            n_opp=d.get('n_opp_edges', 0),
            n_corr=d.get('n_corridor', 0),
            cand0=d.get('a2l_cand0_moving', 0),
            capped=int(sum(1 for c in d.get('a2l_cand', []) if c > K)),
            ego_junction=d.get('ego_on_connector', 0),
            loc=d['map'])
    def rank(pred, key, rev=True):
        c = [t for t in feats if pred(feats[t])]
        c.sort(key=lambda t: (-key(feats[t]) if rev else key(feats[t]), t))
        return c
    pools = {
        'junction_moving_vehicle': rank(lambda f: f['ego_junction'] and f['n_moving_veh'] >= 1,
                                        lambda f: f['n_moving_veh']),
        'route_intersecting_candidate': rank(lambda f: f['n_inter_offroute'] >= 1,
                                             lambda f: f['n_inter_offroute']),
        'straight_non_junction': rank(lambda f: not f['ego_junction'] and f['ego_speed'] > 3.0,
                                      lambda f: -f['n_junction']),
        'stopped_waiting': rank(lambda f: f['ego_speed'] < 0.3, lambda f: -f['ego_speed']),
        'dense_agents': rank(lambda f: True, lambda f: f['n_agents']),
        'many_opposite_lanes': rank(lambda f: True, lambda f: f['n_opp']),
        'corridor_near_cap': rank(lambda f: True, lambda f: f['n_corr']),
        'a2l_zero_or_capped': rank(lambda f: f['cand0'] + f['capped'] >= 1,
                                   lambda f: f['cand0'] + f['capped']),
        'over_M_lanes': [t for t in _over_m_tokens() if t in feats],
    }
    out = {}
    for ci, c in enumerate(CASES):
        if c == 'over_M_lanes':
            out[c] = list(pools[c])       # 11 개뿐이고 M 우선보존의 유일한 시험대라 전량 채택
            continue
        by_loc = {}
        for t in pools[c]:
            by_loc.setdefault(feats[t]['loc'], []).append(t)
        picked, locs = [], sorted(by_loc)
        r = ci                       # rotate the start so the spare pick is not always one city
        while len(picked) < per_case and any(by_loc.values()):
            l = locs[r % len(locs)]
            if by_loc[l]:
                picked.append(by_loc[l].pop(0))
            r += 1
            if r > 500:
                break
        out[c] = picked
    return out, feats


def main():
    pa = argparse.ArgumentParser()
    g = pa.add_mutually_exclusive_group(required=True)
    g.add_argument('--debug', action='store_true')
    g.add_argument('--ids')
    g.add_argument('--full', action='store_true')
    pa.add_argument('--out')
    pa.add_argument('--pool', type=int, default=400)
    pa.add_argument('--per-case', type=int, default=5)
    a = pa.parse_args()

    tz, names, pose_by_token, meta_by_token = load_inputs()
    row_of = {t: i for i, t in enumerate(names.tolist())}
    ego_all = tz['ego']

    if a.full:
        toks = names.tolist()
        out = a.out or f'{ROOT}/navsim_relgraph_v2.npz'
    elif a.ids:
        raw = json.load(open(a.ids))
        toks = sorted({t for v in raw.values() for t in v}) if isinstance(raw, dict) else sorted(set(raw))
        out = a.out or f'{ROOT}/navsim_relgraph_debug.npz'
    else:
        toks = None
        out = a.out or f'{ROOT}/navsim_relgraph_debug.npz'

    if toks is None:
        # deterministic stratified probe pool: evenly spaced in sorted-token order per map
        by_loc = {}
        for t in sorted(names.tolist()):
            by_loc.setdefault(meta_by_token[t][2], []).append(t)
        pool = []
        per = max(1, a.pool // len(by_loc))
        for l in sorted(by_loc):
            v = by_loc[l]
            step = max(1, len(v) // per)
            pool += v[::step][:per]
        # M cap 을 실제로 거는 window 를 강제로 넣는다. 층화표본 400 개로는 12,146 중 11 개인
        # lane >= 128 window 가 절대 안 걸리고, 그러면 M 우선보존이 또 미검증으로 남는다.
        forced = [t for t in _over_m_tokens() if t in row_of]
        pool = sorted(set(pool) | set(forced))
        print(f'probe pool {len(pool)} tokens over {len(by_loc)} map locations '
              f'(over-M 강제 포함 {len(forced)})', flush=True)
        keep, packs, diags = run(pool, None, tz, names, pose_by_token, meta_by_token)
        # extra per-window fields the case picker needs
        for d, p in zip([d for d in diags if not d.get('route_fail') and not d.get('no_lane')], packs):
            i = row_of[d['token']]
            ag = tz['agents'][i]; am = tz['amask'][i]
            v = am[:, ANCHOR_T]
            d['n_moving_veh'] = int(((ag[:, ANCHOR_T, 7] > 0.5) & (ag[:, ANCHOR_T, 4] > 2.0) & v).sum())
            d['a2l_cand'] = d.get('a2l_cand', [])
            log, fi, loc = meta_by_token[d['token']]
            x, y, yaw = pose_by_token[d['token']]
            near = lanes_by_centreline(MAPS.get(loc), float(x), float(y), 10.0)
            d['ego_on_connector'] = int(bool(near) and type(near[0][1]).__name__ == 'NuPlanLaneConnector')
        sel, feats = pick_debug(keep, packs, diags, ego_all, row_of, meta_by_token, a.per_case)
        json.dump(sel, open(f'{ROOT}/debug_ids_navsim.json', 'w'), indent=1)
        print(f'wrote {ROOT}/debug_ids_navsim.json')
        for c in CASES:
            print(f'  {c:30s} {len(sel[c])}')
        toks = sorted({t for v in sel.values() for t in v})
        report_probe(diags)

    keep, packs, diags = run(toks, out, tz, names, pose_by_token, meta_by_token)
    report(keep, packs, diags, tz, row_of)


def report_probe(diags):
    tot = len(diags)
    fail = [d for d in diags if d.get('route_fail')]
    print(f'\n[probe pool] route reconstruction {tot - len(fail)}/{tot} ok, {len(fail)} failed on '
          f'the KEYS.md condition (join gap >= {ROUTE_FAIL_GAP} m) = {100*len(fail)/max(tot,1):.2f}%')


def report(keep, packs, diags, tz, row_of):
    print('\n================ report ================')
    tot = len(diags)
    ok = [d for d in diags if not d.get('route_fail') and not d.get('no_lane')]
    fail = [d for d in diags if d.get('route_fail')]
    print(f'windows requested {tot}   emitted {len(keep)}   route failures {len(fail)} '
          f'({100*len(fail)/max(tot,1):.2f}%)')
    if fail:
        print('  failed tokens:', ', '.join(d['token'] for d in fail[:12]))
    rg = np.array([d['raw_gap'] for d in ok if d.get('raw_gap') is not None])
    dg = np.array([d['dense_gap'] for d in ok if d.get('dense_gap') is not None])
    if len(rg):
        print(f'route polyline gap  raw join: med {np.median(rg):.2f} p90 {np.percentile(rg,90):.2f} '
              f'p99 {np.percentile(rg,99):.2f} max {rg.max():.2f} m   >=5 m {int((rg>=5).sum())}')
        print(f'                    after 2 m resample: med {np.median(dg):.2f} max {dg.max():.2f} m '
              f'(this is the B2D-parity number and it is near-vacuous by construction)')
    al = np.array([d['route_arclen'] for d in ok])
    print(f'route arclen (trimmed to +-{ROUTE_SPAN:.0f} m) med {np.median(al):.0f} m   '
          f'lane changes/window med {np.median([d["route_lane_changes"] for d in ok]):.0f}   '
          f'chain breaks/window {np.mean([d["route_breaks"] for d in ok]):.3f}')
    de = np.array([d['route_d_ego'] for d in ok])
    print(f'ego -> nearest route lane  med {np.median(de):.2f} p95 {np.percentile(de,95):.2f} '
          f'max {de.max():.2f} m   (>5 m: {int((de>5).sum())} windows, route does not cover ego)')

    nc = np.array([d['n_corridor'] for d in ok])
    dr = np.array([d['dropped_on_route'] for d in ok])
    print(f'\ncorridor lanes/window  med {np.median(nc):.0f} p99 {np.percentile(nc,99):.0f} '
          f'max {nc.max()}   dropped_on_route_lane total {int(dr.sum())}')
    print(f'corridor lane recall after M={M} truncation  {1.0 - dr.sum()/max(nc.sum(),1):.3f}')
    bh = np.array([d.get('corridor_beyond_horizon', 0) for d in ok])
    print(f'route-roadblock lanes excluded by the H_route={H_ROUTE:.0f} m horizon  '
          f'{int(bh.sum())} total, {100*(bh>0).mean():.1f}% of windows affected')
    nocorr = int(sum(1 for d in ok if d['corr_total'] == 0))
    print(f'windows with no corridor lane at all (roadblock_ids miss the ego)  {nocorr} '
          f'= {100*nocorr/max(len(ok),1):.2f}%')
    nl = np.array([d['m_used'] for d in ok])
    print(f'lanes kept/window  med {np.median(nl):.0f} max {nl.max()}   '
          f'windows at cap {int((nl>=M).sum())}')

    ne = np.array([d['n_edges'] for d in ok])
    nt = np.array([d['n_topo_edges'] for d in ok])
    no = np.array([d['n_opp_edges'] for d in ok])
    print(f'\nL2L edges/window  med {np.median(ne):.0f} p90 {np.percentile(ne,90):.0f} '
          f'p99 {np.percentile(ne,99):.0f} max {ne.max()}')
    print(f'  topological (succ/left/right) med {np.median(nt):.0f} max {nt.max()}   '
          f'opposite-direction med {np.median(no):.0f} max {no.max()}')
    print(f'  E={E} overflow  {int((ne>E).sum())}/{len(ne)} = {100*(ne>E).mean():.2f}%   '
          f'topological edges ever dropped: {int(sum(d["dropped_topo"] for d in ok))}')
    print(f'  LEFT/RIGHT direction violations rejected: '
          f'{sum(d.get("neighbor_dir_violation",0) for d in ok)}')

    cand = np.array([c for d in ok for c in d['a2l_cand']])
    print(f'\nA2L candidates/agent  0: {100*(cand==0).mean():.2f}%  med {np.median(cand):.0f} '
          f'>K={K}: {100*(cand>K).mean():.2f}%  max {cand.max() if len(cand) else 0}')
    cm = np.array([c for d in ok for c in d.get('a2l_cand_moving', [])])
    if len(cm):
        print(f'  moving vehicles (>0.5 m/s) n={len(cm)}   0 candidates '
              f'{100*(cm==0).mean():.2f}% (KEYS reference 0.87%)   >K={K} '
              f'{100*(cm>K).mean():.2f}% (KEYS reference 3.34%)')
    ck, ct = sum(d['a2l_contained_kept'] for d in ok), sum(d['a2l_contained_tot'] for d in ok)
    print(f'  agents inside a lane polygon whose containing lane survived top-K: '
          f'{ck}/{ct} = {ck/max(ct,1):.3f}')
    dl = np.array([v for d in ok for v in d['a2l_dlat']])
    d1 = np.array([v for d in ok for v in d.get('a2l_dlat_top1', [])])
    if len(dl):
        print(f'  d_lat all kept   |.|<0.5 m {100*(np.abs(dl)<0.5).mean():.1f}%  med |.| '
              f'{np.median(np.abs(dl)):.2f}  range {dl.min():.2f}..{dl.max():.2f}')
    if len(d1):
        print(f'  d_lat top-1      |.|<0.5 m {100*(np.abs(d1)<0.5).mean():.1f}%  |.|<1.0 m '
              f'{100*(np.abs(d1)<1.0).mean():.1f}%  med |.| {np.median(np.abs(d1)):.2f}')
    dm = np.array([v for d in ok for v in d.get('a2l_dlat_top1_moving', [])])
    if len(dm):
        print(f'  d_lat top-1, moving vehicles only  |.|<0.5 m {100*(np.abs(dm)<0.5).mean():.1f}%  '
              f'|.|<1.0 m {100*(np.abs(dm)<1.0).mean():.1f}%  med |.| {np.median(np.abs(dm)):.2f}  (n={len(dm)})')

    rc = np.array([d['route_dir_cos'] for d in ok])
    print(f'\nroute polyline direction vs ego heading  cos med {np.median(rc):+.3f}  '
          f'cos<0 {int((rc<0).sum())}/{len(rc)} windows')
    el = np.array([d['ego_lane_is_corridor'] for d in ok])
    print(f'ego\'s own nearest lane is a corridor lane  {el.mean():.3f}   '
          f'|d_lat| to it med {np.median([d["ego_lane_dlat"] for d in ok]):.2f} m')
    cl = sum(d['corr_linked'] for d in ok)
    cw = sum(d['corr_with_succ'] for d in ok)
    ct2 = sum(d['corr_total'] for d in ok)
    print(f'corridor continuity: of the {cw}/{ct2} corridor lanes that have any in-window '
          f'SUCCESSOR, {cl} ({cl/max(cw,1):.3f}) point at another corridor lane '
          f'(the rest end at the {R_QUERY:.0f} m window edge)')

    if packs:
        rr = np.stack([p['route_rel'] for p in packs])
        lm = np.stack([p['lane_mask'] for p in packs])
        for j, nm in enumerate(['on_route_corridor', 'route_reachable_3hop',
                                'shares_downstream', 'polyline_intersects_route',
                                'route_order_norm']):
            v = rr[..., j][lm]
            if j == 4:
                print(f'route_rel[{j}] {nm:26s} mean {v.mean():+.3f} min {v.min():+.3f} max {v.max():+.3f}')
            else:
                print(f'route_rel[{j}] {nm:26s} rate {v.mean():.3f}')
        arr = np.concatenate([np.stack([p['lanes'] for p in packs]).ravel(),
                              np.stack([p['a2l_rel'] for p in packs]).ravel(),
                              rr.ravel()])
        print(f'NaN/Inf in float tensors: {int((~np.isfinite(arr)).sum())}')
        # SUCCESSOR geometric continuity
        gaps = []
        for p in packs:
            ln, ms = p['lanes'], p['l2l_mask']
            for e_ in range(E):
                if not ms[e_] or p['l2l_type'][e_] != SUCCESSOR:
                    continue
                i, j = int(p['l2l_src'][e_]), int(p['l2l_dst'][e_])
                gaps.append(float(np.hypot(*(ln[i, -1, :2] - ln[j, 0, :2]))))
        if gaps:
            g = np.array(gaps)
            print(f'SUCCESSOR end->start distance  med {np.median(g):.2f} p90 {np.percentile(g,90):.2f} '
                  f'max {g.max():.2f} m  (>2 m: {100*(g>2).mean():.1f}%, window clipping shortens lanes)')


if __name__ == '__main__':
    main()

