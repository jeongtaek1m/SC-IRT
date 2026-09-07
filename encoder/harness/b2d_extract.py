#!/usr/bin/env python3
"""Bench2Drive relational scene-graph extractor (relgraph v2).

Emits the key set frozen in KEYS.md.  The NavSim extractor must emit the same key names in the
same order with the same dtypes and the same channel indices, so that one validator runs on both
domains unchanged.

Only PRIMITIVE relations are stored.  There is no conflict / TTC / risk / hardness channel, no
agent_conflicts_ego_route, and no raw CARLA road_id / lane_id anywhere in the tensor: topology
leaves this file only as common relation types (SUCCESSOR / LEFT_NEIGHBOR / RIGHT_NEIGHBOR /
OPPOSITE_DIRECTION) and as the five route_rel flags.  merges_into_route and crosses_route are
not computed anywhere.

Conventions KEYS.md leaves open and that this file fixes -- these are the places where the two
domains can silently drift apart, so the NavSim side must match them:

  ANCHOR TIMESTEP   t = 3.  Verified, not assumed: ego[:, 3, :2] of b2d_tensors.npz is exactly 0
                    for all 2656 windows and ego[:, 3, 2:4] is exactly (1, 0), while every other
                    timestep is nonzero for the majority of windows.  Same index as NavSim.

  GEOMETRY SOURCE   every relation (A2L, route_rel, OPPOSITE_DIRECTION, SUCCESSOR continuity) is
                    computed from the stored P=10 resampled polyline, never from a finer internal
                    one, so a validator can recompute all of it from the tensor alone.

  lanes[..., 2:4]   segment vector from this point to the next, in travel direction.  Row P-1
                    has no following segment and is ZERO, so pts[i] + seg[i] == pts[i+1] holds
                    for every row and pts[-1] + seg[-1] is exactly the end of the polyline.
                    (The NavSim extractor stores the same zero terminator.)

  LATERAL AXIS      RESOLVED 2026-08-26: this extractor now emits y = LEFT by default, so the
                    lateral channels mean the same thing here as in navsim_extract.py.

                    The raw CARLA/UE4 world is LEFT-handed, so applying the plain rotation
                    R(-yaw) to it -- what this file used to do -- yields a frame whose +y
                    points RIGHT.  Measured, not assumed, on both sides:
                      * CARLA.  OpenDRIVE puts lane -2 to the right of lane -1 for travel along
                        +s.  In the un-mirrored frame it landed at y = +3.500 m in 439 of 439
                        lane pairs across Town03/04/05/06.  So +y was RIGHT.
                      * nuPlan.  A Lane's own left_boundary lands at y > 0 in 2283 of 2283
                        samples and right_boundary at y < 0 in 2194 of 2194, over all four map
                        locations; adjacent_edges[0] (left) sits at y = +3.55, [1] at -3.57.
                        So +y is LEFT there.
                    That contradicts SCHEMA.md's coordinate section ("x = forward, y = left",
                    stated for BOTH domains) and KEYS.md's a2l_rel[0] "d_lat ... (left +)", and
                    it is invisible to every shape / dtype / key-order check.

                    KEYS.md also says agents/ego are copied verbatim from the window tensor.
                    Both cannot hold, because the source tensor itself is mirrored.  Verbatim
                    copy loses: it is a provenance rule (take agent states from the window
                    tensor rather than re-deriving them from anno), and the mirror preserves
                    every scalar it names -- speed, half_len, half_wid, is_vehicle, the masks,
                    cos(dpsi) -- negating only the two channels that carry a lateral sign.
                    "Same meaning in both domains" is the claim this file exists to support,
                    so that is what is kept.  --legacy-y-right restores the old frame for
                    reproducing anything built before this date.

                    Do NOT "fix" left_side_correct_frac = 0 by swapping the LEFT_NEIGHBOR /
                    RIGHT_NEIGHBOR labels: the labels are physically right (CARLA
                    get_left_lane really is the driver's left), it is the axis sign that was
                    mirrored, and swapping them would pass the check while making the tensor
                    doubly wrong.

  a2l_rel[..., 1]   d_lon is an arclength coordinate along the lane centreline whose ORIGIN is
                    p_L*, the point of that lane closest to the ego anchor -- the same origin
                    KEYS.md mandates for route_order_norm and for the same reason: any origin
                    taken from the polyline start moves when the window clipping moves.
                    d_lon > 0 means the agent is further along that lane than the ego is.

  OPPOSITE_DIRECTION  evaluated per ORDERED pair exactly as KEYS.md writes it (mid of i against
                    the polyline of j), so a mutually opposite pair yields both i->j and j->i.

  route_reachable_3hop  1..3 SUCCESSOR hops over the lanes selected in this window; a lane never
                    counts as reaching itself in 0 hops.

  shares_downstream_with_route  lane L and some OTHER in-window corridor lane have a common
                    immediate successor lane in the town lane graph.

Failures are loud, never silent: a window whose corridor lanes alone exceed M is failed and
reported rather than truncated, and a route whose reconstruction has a hole or a densified gap
>= 5 m is refused (all 220 currently pass: 0 holes, max gap 2.07 m).

CLI
    python b2d_extract.py --debug        pick the fixed debug set, write debug_ids_b2d.json
                                         and b2d_relgraph_debug.npz
    python b2d_extract.py --ids FILE     extract exactly the item_ids in FILE
    python b2d_extract.py --full         all 2656 windows
"""
import argparse
import gzip
import json
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
from scipy.spatial import cKDTree
import carla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b2d_route import load_routes  # noqa: E402  (mission anchors -> town of each route)

# ---------------------------------------------------------------------------- frozen sizes
A, T, M, P, K, E = 48, 12, 128, 10, 8, 384   # E: 2026-08-26 256 -> 384. hard case 절단 전 최대 수요 265, 위상 edge 까지 잘렸다
R_QUERY = 60.0            # lane query radius, m
H_ROUTE = 100.0           # route_order_norm normaliser, m
ANCHOR_T = 3              # verified against b2d_tensors.npz at startup
A2L_RADIUS = 5.0
OPP_MIN, OPP_MAX = 1.0, 6.0
OPP_ANGLE = np.radians(150.0)
SUCC_GAP = 5.0            # geometric guard on SUCCESSOR (2 m sampling on both sides)
LANE_SAMPLE = 2.0         # centreline sampling step, m
ARCLEN_NORM = 95.0        # lane_feat[1] normaliser
REACH_HOPS = 3
ROUTE_BAND = 60           # +-60 route samples (= +-120 m) searched for a projection

TYPE_SUCCESSOR, TYPE_LEFT, TYPE_RIGHT, TYPE_OPPOSITE = 0, 1, 2, 3

# ---- 채널 심볼 (KEYS.md, 2026-08-26 freeze). 숫자 인덱스를 코드에 쓰지 않는다.
# pick_debug 에서 실제로 rr[:, 4] 를 intersect 로 읽는 버그가 나왔다. 재배열은 조용히 틀린다.
R_ON_ROUTE, R_REACH, R_SHARES, R_XSECT, R_ORDER = 0, 1, 2, 3, 4
LF_JUNCTION, LF_ARCLEN = 0, 1
A2L_DLAT, A2L_DLON, A2L_COS, A2L_SIN = 0, 1, 2, 3

AG_DX, AG_DY, AG_COS, AG_SIN, AG_SPD, AG_HL, AG_HW, AG_ISVEH = range(8)

B2D_TENSORS = '/data2/jeongtae/navsim_interact/b2d_tensors.npz'
ANNO = '/data1/jeongtae/b2d_eval_sensors/route_{rid}/anno/{f:05d}.json.gz'
ROUTE_NPZ = os.path.join(HERE, 'route_{rid}.npz')
XODR = '/data1/jeongtae/carla915/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr'
XODR_ALT = '/data1/jeongtae/carla915/CarlaUE4/Content/Carla/Maps/{town}/OpenDrive/{town}.xodr'
ALT_TOWNS = ('Town11', 'Town12', 'Town13', 'Town15')

SIDECAR_MAP = {'n_lane_before_M': 'n_cand', 'n_lane_after_M': 'n_lane', 'n_route_lane_before_M': 'n_corr', 'n_route_lane_after_M': 'n_corr_sel', 'edge_count_before_E': 'n_edge_raw', 'edge_count_after_E': 'n_edge', 'n_agent_containing_before_K': 'a2l_contained_tot', 'n_agent_containing_after_K': 'a2l_contained_kept', 'route_fail': None}

KEY_ORDER = ['lanes', 'lane_feat', 'lane_mask', 'l2l_src', 'l2l_dst', 'l2l_type', 'l2l_mask',
             'a2l_idx', 'a2l_rel', 'a2l_mask', 'route_rel', 'route_rel_valid',
             'agents', 'agent_mask', 'ego',
             'command', 'item_id', 'domain']
DTYPE = dict(lanes=np.float16, lane_feat=np.float16, lane_mask=bool, l2l_src=np.int16,
             l2l_dst=np.int16, l2l_type=np.int8, l2l_mask=bool, a2l_idx=np.int16,
             a2l_rel=np.float16, a2l_mask=bool, route_rel=np.float16, route_rel_valid=bool,
             agents=np.float16,
             agent_mask=bool, ego=np.float16, command=np.float16)


# ---------------------------------------------------------------------------- geometry
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


def resample(poly, n):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-3:
        return None, 0.0
    si = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(si, s, poly[:, 0]), np.interp(si, s, poly[:, 1])], 1), float(s[-1])


# ---------------------------------------------------------------------------- town lane graph
class Town:
    """Every Driving lane centreline of one town sampled at 2 m, plus the static lane topology.

    Samples are stored sorted by (lane, travel order) so one lane occupies a contiguous,
    travel-ordered index range: window clipping is then a run of consecutive indices.
    """

    def __init__(self, town):
        self.name = town
        path = XODR_ALT.format(town=town) if town in ALT_TOWNS else XODR.format(town=town)
        self.cmap = carla.Map(town, open(path).read())
        keyed, ends = defaultdict(list), {}
        for w in self.cmap.generate_waypoints(LANE_SAMPLE):
            if w.lane_type != carla.LaneType.Driving:
                continue
            k = (w.road_id, w.section_id, w.lane_id)
            tr = w.transform
            keyed[k].append((w.s, tr.location.x, tr.location.y,
                             np.radians(tr.rotation.yaw), w.lane_width, w.is_junction))
            lo, hi = ends.get(k, (None, None))
            if lo is None or w.s < lo.s:
                lo = w
            if hi is None or w.s > hi.s:
                hi = w
            ends[k] = (lo, hi)

        self.keys = sorted(keyed)
        self.code = {k: i for i, k in enumerate(self.keys)}
        xs, ys, yaws, starts, lens, wid, junc = [], [], [], [], [], [], []
        for k in self.keys:
            # lane_id < 0 travels with increasing s, lane_id > 0 against it (verified on
            # Town01/03/12: 100% / 0% of segments agree with the CARLA waypoint yaw).
            v = sorted(keyed[k], key=lambda r: r[0], reverse=(k[2] > 0))
            starts.append(len(xs))
            lens.append(len(v))
            xs += [r[1] for r in v]
            ys += [r[2] for r in v]
            yaws += [r[3] for r in v]
            wid.append(float(np.median([r[4] for r in v])))
            junc.append(bool(v[0][5]))
        self.xy = np.stack([np.asarray(xs, float), np.asarray(ys, float)], 1)
        self.yaw = np.asarray(yaws, float)
        self.start = np.asarray(starts, np.int64)
        self.count = np.asarray(lens, np.int64)
        self.width = np.asarray(wid, float)
        self.is_junction = np.asarray(junc, bool)
        self.corridor_key = [(k[0], k[1], int(np.sign(k[2]))) for k in self.keys]
        self.tree = cKDTree(self.xy)

        # coordinate check: travel-order point sequence must agree with the waypoint yaw,
        # otherwise every tangent, dpsi and OPPOSITE_DIRECTION test is mirrored.
        d = np.diff(self.xy, axis=0)
        same = np.zeros(len(self.xy) - 1, bool)
        for st, n in zip(self.start, self.count):
            if n > 1:
                same[st:st + n - 1] = True
        good = same & (np.linalg.norm(d, axis=1) > 1e-6)
        dot = d[:, 0] * np.cos(self.yaw[:-1]) + d[:, 1] * np.sin(self.yaw[:-1])
        self.tangent_agree = float((dot[good] > 0).mean()) if good.any() else 1.0

        self.succ = [set() for _ in self.keys]
        for k, i in self.code.items():
            lo, hi = ends[k]
            w = hi if k[2] < 0 else lo                    # travel-direction end of the lane
            for dd in (1.0, 2.0, 3.0):                    # samples stop up to 2 m short of it
                for n in (w.next(dd) or []):
                    if n.lane_type != carla.LaneType.Driving:
                        continue
                    kk = (n.road_id, n.section_id, n.lane_id)
                    if kk != k and kk in self.code:
                        self.succ[i].add(self.code[kk])
        self.left = np.full(len(self.keys), -1, np.int64)
        self.right = np.full(len(self.keys), -1, np.int64)
        for k, i in self.code.items():
            lo, hi = ends[k]
            for arr, getter in ((self.left, 'get_left_lane'), (self.right, 'get_right_lane')):
                for w in (lo, hi):
                    r = getattr(w, getter)()
                    if r is None or r.lane_type != carla.LaneType.Driving:
                        continue
                    # get_left_lane/get_right_lane hand back the lane across the centreline
                    # 45.2% of the time (100% in Town01); same road + same lane_id sign is what
                    # makes LEFT/RIGHT_NEIGHBOR mean "same travel direction".
                    if r.road_id != k[0] or np.sign(r.lane_id) != np.sign(k[2]):
                        continue
                    kk = (r.road_id, r.section_id, r.lane_id)
                    if kk in self.code:
                        arr[i] = self.code[kk]
                        break


_town_cache = {}


def get_town(town):
    if town not in _town_cache:
        _town_cache.clear()                # one town at a time; Town12 alone is 335k samples
        _town_cache[town] = Town(town)
    return _town_cache[town]


class WindowFail(Exception):
    pass


# ---------------------------------------------------------------------------- one window
def mirror_yaw_channels(arr, y_ch, sin_ch):
    """y -> -y on a copied agents/ego block: lateral offset and sin(dpsi) both flip."""
    out = arr.copy()
    out[..., y_ch] = -out[..., y_ch]
    out[..., sin_ch] = -out[..., sin_ch]
    return out


def extract_window(tm, corridor, route_g, ego_xy, theta, ag, am, mirror=False):
    """-> (tensors, stats).  Raises WindowFail rather than emitting a contract breach."""
    yaw = theta - np.pi / 2.0                  # x forward; y right unless mirror
    c, s = np.cos(yaw), np.sin(yaw)
    Rg2e = np.array([[c, s], [s, -c]]) if mirror else np.array([[c, s], [-s, c]])
    to_ego = lambda g: (g - ego_xy[None]) @ Rg2e.T

    # -- lanes -------------------------------------------------------------------------
    hit = np.asarray(tm.tree.query_ball_point(ego_xy, R_QUERY), dtype=np.int64)
    if hit.size == 0:
        raise WindowFail('no driving lane centreline within R_query')
    hit.sort()
    codes = np.searchsorted(tm.start, hit, side='right') - 1
    dist_g = np.linalg.norm(tm.xy[hit] - ego_xy[None], axis=1)

    cand, n_short = [], 0
    for cd in np.unique(codes):
        pos = np.nonzero(codes == cd)[0]
        sel, dh = hit[pos], dist_g[pos]
        runs = np.split(np.arange(len(sel)), np.nonzero(np.diff(sel) != 1)[0] + 1)
        run = min(runs, key=lambda r: dh[r].min())       # the pass closest to the ego
        sel = sel[run]
        if len(sel) < 2:
            n_short += 1
            continue                                     # single 2 m sample grazing the rim
        poly, arclen = resample(to_ego(tm.xy[sel]), P)
        if poly is not None:
            cand.append((int(cd), poly, arclen))
    if not cand:
        raise WindowFail('no lane with >= 2 centreline samples inside the query disc')

    codes_c = np.array([x[0] for x in cand], np.int64)
    poly_c = np.stack([x[1] for x in cand])
    arclen_c = np.array([x[2] for x in cand], float)
    on_route_c = np.array([tm.corridor_key[i] in corridor for i in codes_c], bool)
    d_ego_c = proj_to_segs(np.zeros((1, 2)), poly_c[:, :-1].reshape(-1, 2),
                           poly_c[:, 1:].reshape(-1, 2))[0].reshape(len(cand), P - 1).min(1)

    n_corr = int(on_route_c.sum())
    if n_corr > M:
        raise WindowFail('corridor lanes %d exceed M=%d' % (n_corr, M))
    # priority-retain (KEYS.md): every corridor lane survives, the rest fill by centreline
    # distance.  lane index therefore carries no meaning beyond that and must stay a set axis.
    order = np.lexsort((codes_c, d_ego_c, ~on_route_c))[:M]
    m = len(order)
    codes_s, poly = codes_c[order], poly_c[order]
    on_route, arclen = on_route_c[order], arclen_c[order]
    junction, width = tm.is_junction[codes_s], tm.width[codes_s]

    segA = poly[:, :-1].reshape(-1, 2)
    segB = poly[:, 1:].reshape(-1, 2)
    dseg = segB - segA
    seg_th = np.arctan2(dseg[:, 1], dseg[:, 0]).reshape(m, P - 1)
    seg_len = np.linalg.norm(dseg, axis=1).reshape(m, P - 1)
    seg_s0 = np.concatenate([np.zeros((m, 1)), np.cumsum(seg_len, 1)[:, :-1]], 1)

    # p_L* : point of each lane closest to the ego anchor (never the polyline start)
    d0, t0, pr0 = proj_to_segs(np.zeros((1, 2)), segA, segB)
    d0 = d0.reshape(m, P - 1)
    k_star = d0.argmin(1)
    ar = np.arange(m)
    d_ego = d0[ar, k_star]
    p_star = pr0.reshape(m, P - 1, 2)[ar, k_star]
    s_star = seg_s0[ar, k_star] + t0.reshape(m, P - 1)[ar, k_star] * seg_len[ar, k_star]

    lanes = np.zeros((M, P, 4), np.float32)
    lane_feat = np.zeros((M, 2), np.float32)
    lane_mask = np.zeros(M, bool)
    segv = np.diff(poly, axis=1)
    lanes[:m, :, :2] = poly
    lanes[:m, :-1, 2:] = segv
    lanes[:m, -1, 2:] = 0.0    # row P-1 starts no segment; pts[-1]+seg[-1] == pts[-1]
    lane_feat[:m, LF_JUNCTION] = junction
    lane_feat[:m, LF_ARCLEN] = arclen / ARCLEN_NORM
    lane_mask[:m] = True

    # -- L2L ---------------------------------------------------------------------------
    pos_of = {int(x): i for i, x in enumerate(codes_s)}
    edges, succ_pairs, succ_guard = [], [], 0
    for i, cd in enumerate(codes_s):
        for cj in tm.succ[int(cd)]:
            j = pos_of.get(int(cj))
            if j is None or j == i:
                continue
            # stored polylines are window-clipped; emit the edge only where the two clipped
            # polylines actually meet, which is exactly what the validator measures.
            if np.linalg.norm(poly[i, -1] - poly[j, 0]) > SUCC_GAP:
                succ_guard += 1
                continue
            edges.append((i, j, TYPE_SUCCESSOR))
            succ_pairs.append((i, j))
    for i, cd in enumerate(codes_s):
        if junction[i]:
            continue                                     # undefined inside a junction
        for arr, ty in ((tm.left, TYPE_LEFT), (tm.right, TYPE_RIGHT)):
            cj = int(arr[int(cd)])
            j = pos_of.get(cj) if cj >= 0 else None
            if j is not None and j != i and not junction[j]:
                edges.append((i, j, ty))

    mid = 0.5 * (poly[:, (P - 1) // 2] + poly[:, P // 2])
    mid_th = seg_th[:, (P - 1) // 2]
    dmat = proj_to_segs(mid, segA, segB)[0].reshape(m, m, P - 1)
    kmin = dmat.argmin(2)
    dmin = np.take_along_axis(dmat, kmin[..., None], 2)[..., 0]
    th_j = seg_th[np.broadcast_to(ar[None, :], (m, m)), kmin]
    opp = (dmin > OPP_MIN) & (dmin < OPP_MAX) & (np.abs(wrap(mid_th[:, None] - th_j)) > OPP_ANGLE)
    np.fill_diagonal(opp, False)
    for i, j in zip(*np.nonzero(opp)):
        edges.append((int(i), int(j), TYPE_OPPOSITE))

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

    # -- A2L ---------------------------------------------------------------------------
    a2l_idx = np.full((A, K), -1, np.int64)
    a2l_rel = np.zeros((A, K, 4), np.float32)
    a2l_mask = np.zeros((A, K), bool)
    live = np.nonzero(am[:, ANCHOR_T])[0]
    n_zero = n_cap = moving_zero = inside_drop = 0
    contained_tot = contained_kept = 0   # sidecar: 포함 lane 이 top-K 에 살아남았는가
    n_move_veh = 0
    dlat_all = []
    if len(live):
        ap = ag[live, ANCHOR_T, :2].astype(np.float64)
        apsi = np.arctan2(ag[live, ANCHOR_T, AG_SIN].astype(np.float64),
                          ag[live, ANCHOR_T, AG_COS].astype(np.float64))
        veh = ag[live, ANCHOR_T, AG_ISVEH] > 0.5
        spd = ag[live, ANCHOR_T, AG_SPD].astype(np.float64)
        n_move_veh = int((veh & (spd > 0.5)).sum())
        ad, at, aproj = proj_to_segs(ap, segA, segB)
        n = len(live)
        ad = ad.reshape(n, m, P - 1)
        at = at.reshape(n, m, P - 1)
        aproj = aproj.reshape(n, m, P - 1, 2)
        ka = ad.argmin(2)
        da = np.take_along_axis(ad, ka[..., None], 2)[..., 0]
        for r in range(n):
            cs = np.nonzero(da[r] <= A2L_RADIUS)[0]
            if len(cs) == 0:
                n_zero += 1
                moving_zero += int(veh[r] and spd[r] > 0.5)
                continue
            n_cap += int(len(cs) > K)
            kk = ka[r, cs]
            th = seg_th[cs, kk]
            dv = ap[r] - aproj[r, cs, kk]
            dlat = -np.sin(th) * dv[:, 0] + np.cos(th) * dv[:, 1]        # left positive
            dlon = seg_s0[cs, kk] + at[r, cs, kk] * seg_len[cs, kk] - s_star[cs]
            dpsi = wrap(apsi[r] - th)
            inside = np.abs(dlat) <= width[cs] / 2.0
            inside_drop += max(int(inside.sum()) - K, 0)
            # tier 2 applies to vehicles only; pedestrians skip it (KEYS.md)
            agree = np.where(np.cos(dpsi) > 0, 0, 1) if veh[r] else np.zeros(len(cs), np.int64)
            if inside.any():
                contained_tot += 1
            o = np.lexsort((cs, da[r, cs], agree, np.where(inside, 0, 1)))[:K]
            if inside.any() and inside[o].any():
                contained_kept += 1
            nk = len(o)
            g = live[r]
            a2l_idx[g, :nk] = cs[o]
            a2l_rel[g, :nk, A2L_DLAT] = dlat[o]
            a2l_rel[g, :nk, A2L_DLON] = dlon[o]
            a2l_rel[g, :nk, A2L_COS] = np.cos(dpsi[o])
            a2l_rel[g, :nk, A2L_SIN] = np.sin(dpsi[o])
            a2l_mask[g, :nk] = True
            dlat_all.append(dlat[o])

    # -- route_rel ---------------------------------------------------------------------
    route_rel = np.zeros((M, 5), np.float32)
    route_rel[:m, R_ON_ROUTE] = on_route
    re = to_ego(route_g)
    rd = np.linalg.norm(re, axis=1)
    rs = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(re, axis=0), axis=1))])
    # arclength coordinates are read inside a +-120 m band around the ego so that a route which
    # doubles back (hairpins, loops) cannot project a lane onto the wrong pass.
    i_ego = int(np.argmin(rd))
    b0, b1 = max(0, i_ego - ROUTE_BAND), min(len(re), i_ego + ROUTE_BAND + 1)
    rb, sb = re[b0:b1], rs[b0:b1]
    n_int = 0
    if len(rb) >= 2:
        rA, rB = rb[:-1], rb[1:]
        rlen = np.linalg.norm(rB - rA, axis=1)
        dr, tr, _ = proj_to_segs(np.zeros((1, 2)), rA, rB)
        k0 = int(dr[0].argmin())
        s_ego = sb[k0] + tr[0, k0] * rlen[k0]
        idx = np.nonzero(on_route)[0]
        if len(idx):
            dL, tL, _ = proj_to_segs(p_star[idx], rA, rB)
            kL = dL.argmin(1)
            sL = sb[kL] + tL[np.arange(len(idx)), kL] * rlen[kL]
            route_rel[idx, R_ORDER] = np.clip((sL - s_ego) / H_ROUTE, -1.0, 1.0)
    # crossing is a plain geometric fact, so it is tested against every route segment that comes
    # near this window -- including a second pass of the route that lies outside the band above.
    keep_r = rd < R_QUERY + 5.0
    seg_r = np.nonzero(keep_r[:-1] | keep_r[1:])[0]
    if len(seg_r):
        cA, cB = re[seg_r], re[seg_r + 1]
        cross = segs_intersect(segA[:, None, :], segB[:, None, :], cA[None], cB[None])
        route_rel[:m, R_XSECT] = cross.reshape(m, P - 1, -1).any((1, 2))
        n_int = int(route_rel[:m, R_XSECT].sum())

    if on_route.any():
        adj = defaultdict(list)
        for i, j in succ_pairs:
            adj[i].append(j)
        tgt = set(np.nonzero(on_route)[0].tolist())
        for i in range(m):
            q, seen = deque((j, 1) for j in adj[i]), set()
            while q:
                v, h = q.popleft()
                if v in tgt:
                    route_rel[i, R_REACH] = 1.0
                    break
                if h < REACH_HOPS and v not in seen:
                    seen.add(v)
                    q.extend((w2, h + 1) for w2 in adj[v])
        owners = defaultdict(set)
        for i in np.nonzero(on_route)[0]:
            for cj in tm.succ[int(codes_s[i])]:
                owners[int(cj)].add(int(i))
        for i in range(m):
            for cj in tm.succ[int(codes_s[i])]:
                if owners.get(int(cj), set()) - {i}:
                    route_rel[i, R_SHARES] = 1.0
                    break

    et = ek[:, 2] if len(ek) else np.zeros(0, np.int64)
    st = dict(n_cand=len(cand), n_short=n_short, n_lane=m, n_corr=n_corr, n_corr_sel=int(on_route.sum()),
              dropped_corridor=n_corr - int(on_route.sum()), n_junction=int(junction.sum()),
              n_edge_raw=n_raw, n_edge=int(len(ek)), overflow=int(n_raw > E),
              n_succ=int((et == TYPE_SUCCESSOR).sum()), n_left=int((et == TYPE_LEFT).sum()),
              n_right=int((et == TYPE_RIGHT).sum()), n_opp=int((et == TYPE_OPPOSITE).sum()),
              succ_guard=succ_guard, n_agent=int(len(live)), n_move_veh=n_move_veh,
              a2l_zero=n_zero, a2l_cap=n_cap, moving_zero=moving_zero, n_intersect=n_int,
              inside_drop=inside_drop,
              a2l_contained_tot=contained_tot, a2l_contained_kept=contained_kept,
              n_reach=int(route_rel[:m, R_REACH].sum()), n_shares=int(route_rel[:m, R_SHARES].sum()),
              ego_junction=int(junction[int(np.argmin(d_ego))]),
              dlat=np.concatenate(dlat_all) if dlat_all else np.zeros(0))

    # route_rel_valid: route_rel 의 0 을 "route 와 무관" 으로 읽어도 되는가.
    # corridor lane 이 하나도 없으면 그 0 은 "무관" 이 아니라 "60 m crop 안에 판단 근거가
    # 없음" 이다. 둘을 같은 0 으로 주면 false negative supervision 이 된다.
    # (NavSim 모집단 2.60% — metadata/crop coverage 문제이지 extractor 버그가 아니다.)
    route_valid = bool(on_route.any())
    if not route_valid:
        # KEYS.md: valid=False 면 5채널 전부 unknown 이고 반드시 0 으로 저장한다.
        # corridor 가 비어도 polyline_intersects_route 는 route polyline 과의 순수 기하라
        # 1 이 될 수 있다 (실측 347 window 중 2 건). 그 1 을 남기면 "route 를 가로지른다"
        # 는 단언이 되어, mask 를 둔 취지와 어긋난다.
        route_rel[:] = 0.0


    out = dict(lanes=lanes, lane_feat=lane_feat, lane_mask=lane_mask, l2l_src=l2l_src,
               l2l_dst=l2l_dst, l2l_type=l2l_type, l2l_mask=l2l_mask, a2l_idx=a2l_idx,
               a2l_rel=a2l_rel, a2l_mask=a2l_mask, route_rel=route_rel,
               route_rel_valid=np.bool_(route_valid))
    return out, st


# ---------------------------------------------------------------------------- driving it
def load_index():
    d = np.load(B2D_TENSORS, allow_pickle=True)
    item = np.array(['%s_%d' % (r, w) for r, w in zip(d['route'], d['widx'])])
    assert len(set(item.tolist())) == len(item), 'item_id is not unique'
    ego = d['ego'].astype(np.float32)
    zero = [t for t in range(T) if np.abs(ego[:, t, :2]).max() == 0.0]
    assert zero == [ANCHOR_T], 'anchor timestep is %s, not %d' % (zero, ANCHOR_T)
    assert np.all(ego[:, ANCHOR_T, 2] == 1.0) and np.all(ego[:, ANCHOR_T, 3] == 0.0)
    return d, item


_route_cache = {}


def route_data(rid):
    if rid not in _route_cache:
        z = np.load(ROUTE_NPZ.format(rid=rid))
        if int(z['holes']) != 0 or float(z['max_gap']) >= 5.0:
            raise WindowFail('route %s: holes=%d max_gap=%.2f' %
                             (rid, int(z['holes']), float(z['max_gap'])))
        _route_cache.clear()
        _route_cache[rid] = (z['route'].astype(np.float64),
                             {tuple(int(x) for x in c) for c in z['corridor']})
    return _route_cache[rid]


def anchor_pose(rid, widx):
    with gzip.open(ANNO.format(rid=rid, f=20 * widx + 15), 'rt') as fh:
        a = json.load(fh)
    return np.array([a['x'], a['y']], float), float(a['theta'])


def run(items, d, item_all, verbose=True, mirror=False):
    """items: list of item_id.  -> (rows dict keyed by item_id, stats dict, failures)."""
    row_of = {x: i for i, x in enumerate(item_all.tolist())}
    towns = {rid: t for rid, (t, _) in load_routes().items()}
    ag_all, am_all = d['agents'], d['amask']
    if mirror:
        ag_all = mirror_yaw_channels(ag_all.astype(np.float32), AG_DY, AG_SIN)
    grouped = defaultdict(list)
    for it in items:
        rid = it.split('_')[1]
        grouped[(towns[rid], rid)].append(it)
    rows, stats, fails = {}, {}, []
    t0 = time.time()
    for n, (town, rid) in enumerate(sorted(grouped)):
        tm = get_town(town)
        route_g, corridor = route_data(rid)
        for it in sorted(grouped[(town, rid)], key=lambda x: int(x.split('_')[2])):
            widx = int(it.split('_')[2])
            i = row_of[it]
            try:
                exy, th = anchor_pose(rid, widx)
                out, st = extract_window(tm, corridor, route_g, exy, th,
                                         ag_all[i].astype(np.float32), am_all[i], mirror)
            except WindowFail as e:
                fails.append((it, str(e)))
                continue
            st['town'] = town
            st['tangent_agree'] = tm.tangent_agree
            rows[it] = out
            stats[it] = st
        if verbose and (n + 1) % 25 == 0:
            print('  %d/%d routes  %.1fs' % (n + 1, len(grouped), time.time() - t0), flush=True)
    return rows, stats, fails


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

def save(path, items, rows, d, item_all, mirror=False):
    row_of = {x: i for i, x in enumerate(item_all.tolist())}
    keep = [it for it in items if it in rows]
    idx = np.array([row_of[it] for it in keep], np.int64)
    out = {}
    for k in ['lanes', 'lane_feat', 'lane_mask', 'l2l_src', 'l2l_dst', 'l2l_type', 'l2l_mask',
              'a2l_idx', 'a2l_rel', 'a2l_mask', 'route_rel', 'route_rel_valid']:
        out[k] = np.stack([rows[it][k] for it in keep]).astype(DTYPE[k])
    ag, eg = d['agents'][idx].astype(np.float32), d['ego'][idx].astype(np.float32)
    if mirror:
        ag = mirror_yaw_channels(ag, AG_DY, AG_SIN)
        eg = mirror_yaw_channels(eg, 1, 3)
    out['agents'] = ag.astype(DTYPE['agents'])
    out['agent_mask'] = d['amask'][idx].astype(DTYPE['agent_mask'])
    out['ego'] = eg.astype(DTYPE['ego'])
    out['command'] = d['cmd'][idx].astype(DTYPE['command'])
    out['item_id'] = np.array(keep, dtype='<U32')
    out['domain'] = np.array('b2d', dtype='<U8')
    for k in out:
        v = out[k]
        if v.dtype.kind == 'f':
            assert np.isfinite(v.astype(np.float32)).all(), 'non-finite in %s' % k
    np.savez_compressed(path, **{k: out[k] for k in KEY_ORDER})
    return keep, out


# ---------------------------------------------------------------------------- debug set
POOL_PER_ROUTE = 3
PER_CASE = 5
CASES = [
    ('1_junction_moving', lambda s: s['ego_junction'] and s['n_move_veh'] >= 1,
     lambda s: (s['n_move_veh'], s['n_agent'])),
    ('2_intersects_route', lambda s: s['n_intersect'] >= 1, lambda s: (s['n_intersect'],)),
    ('3_straight_nonjunction', lambda s: not s['ego_junction'] and s['n_junction'] == 0,
     lambda s: (s['n_lane'],)),
    ('4_stopped', lambda s: s['ego_speed'] < 0.2, lambda s: (s['n_agent'],)),
    ('5_dense_agents', lambda s: s['n_agent'] >= 15, lambda s: (s['n_agent'],)),
    ('6_many_opposite', lambda s: s['n_opp'] >= 1, lambda s: (s['n_opp'],)),
    ('7_corridor_near_cap', lambda s: s['n_corr'] >= 1, lambda s: (s['n_corr'],)),
    # both flavours of the case: a window that has any zero-candidate agent outranks a
    # window that only hits the K cap, because zero-candidate agents are rare in B2D (0.03%).
    ('8_a2l_zero_or_cap', lambda s: s['a2l_zero'] + s['a2l_cap'] >= 1,
     lambda s: (min(s['a2l_zero'], 1) * 100 + s['a2l_cap'] * 10 + s['a2l_zero'],)),
]


def build_pool(item_all, d):
    by_route = defaultdict(list)
    for it in item_all.tolist():
        by_route[it.rsplit('_', 1)[0]].append(int(it.rsplit('_', 1)[1]))
    pool = []
    for r in sorted(by_route):
        w = sorted(by_route[r])
        take = sorted({w[int(round(x))] for x in
                       np.linspace(0, len(w) - 1, min(POOL_PER_ROUTE, len(w)))})
        pool += ['%s_%d' % (r, x) for x in take]
    return pool


def pick_debug(stats):
    chosen, used_route, out = set(), set(), {}
    for name, ok, score in CASES:
        cands = [it for it, s in stats.items() if ok(s) and it not in chosen]
        cands.sort(key=lambda it: (tuple(-x for x in score(stats[it])), it))
        per_town, take = defaultdict(int), []
        for it in cands:
            if len(take) >= PER_CASE:
                break
            s = stats[it]
            r = it.rsplit('_', 1)[0]
            if r in used_route or per_town[s['town']] >= 2:
                continue
            take.append(it)
            used_route.add(r)
            per_town[s['town']] += 1
            chosen.add(it)
        out[name] = take
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--debug', action='store_true')
    g.add_argument('--ids')
    g.add_argument('--full', action='store_true')
    ap.add_argument('--out')
    ap.add_argument('--legacy-y-right', action='store_false', dest='mirror',
                    help='emit the pre-2026-08-26 CARLA frame (y = RIGHT): a verbatim copy '
                         'of the source agents/ego, but mirrored relative to NavSim, so '
                         'd_lat and the lane geometry mean the opposite of what they mean '
                         'there. For reproducing older artefacts only.')
    ap.set_defaults(mirror=True)
    args = ap.parse_args()

    d, item_all = load_index()
    print('anchor timestep verified: ego[:, %d, :2] == 0 for all %d windows'
          % (ANCHOR_T, len(item_all)))
    print('LATERAL AXIS: %s' % ('y = LEFT, harmonized with NavSim. agents/ego dy and sin are '
                                'negated w.r.t. the source tensor, so this is a mirrored, not '
                                'verbatim, copy -- see the module docstring.'
                                if args.mirror else
                                'LEGACY CARLA frame, y = RIGHT -- MIRRORED relative to NavSim, '
                                'so a2l_rel d_lat and the lane geometry mean the OPPOSITE of '
                                'what they mean in navsim_relgraph. Reproduction only.'))
    ego_speed = d['ego'][:, ANCHOR_T, AG_SPD].astype(np.float32)
    speed_of = {it: float(v) for it, v in zip(item_all.tolist(), ego_speed)}

    if args.debug:
        pool = build_pool(item_all, d)
        print('debug pass 1: %d pool windows over %d routes'
              % (len(pool), len({x.rsplit('_', 1)[0] for x in pool})))
        _, stats, fails = run(pool, d, item_all, mirror=args.mirror)
        for it in stats:
            stats[it]['ego_speed'] = speed_of[it]
        report(stats, fails, 'POOL')
        sel = pick_debug(stats)
        json.dump(sel, open(os.path.join(HERE, 'debug_ids_b2d.json'), 'w'), indent=1)
        items = [x for v in sel.values() for x in v]
        print('\ndebug set: %d windows' % len(items))
        for k, v in sel.items():
            print('  %-24s %d  %s' % (k, len(v), ' '.join(v)))
    elif args.ids:
        j = json.load(open(args.ids))
        items = [x for v in j.values() for x in v] if isinstance(j, dict) else list(j)
        sel = None
    else:
        items = item_all.tolist()
        sel = None

    out_path = args.out or os.path.join(
        HERE, 'b2d_relgraph_v2.npz' if args.full else 'b2d_relgraph_debug.npz')
    rows, stats, fails = run(items, d, item_all, mirror=args.mirror)
    for it in stats:
        stats[it]['ego_speed'] = speed_of[it]
    keep, arrs = save(out_path, items, rows, d, item_all, args.mirror)
    write_sidecar(out_path.replace('.npz', '_audit.npz'), keep, [stats[i] for i in keep],
                  lambda dg, c: 0 if SIDECAR_MAP[c] is None else dg.get(SIDECAR_MAP[c], 0))
    print('\nwrote %s  (%d windows)' % (out_path, len(keep)))
    for k in KEY_ORDER:
        print('  %-11s %-22s %s' % (k, arrs[k].shape, arrs[k].dtype))
    if sel:
        print('\nper-case counts (after extraction)')
        for k, v in sel.items():
            print('  %-24s %d/%d kept' % (k, sum(x in rows for x in v), len(v)))
    report(stats, fails, 'DEBUG' if not args.full else 'FULL')


def report(stats, fails, tag):
    if not stats:
        print('no windows'); return
    g = lambda k: np.array([s[k] for s in stats.values()], float)
    q = lambda a: '%.0f/%.0f/%.0f/%.0f/%.0f' % (a.min(), np.percentile(a, 50),
                                                np.percentile(a, 90), np.percentile(a, 99), a.max())
    print('\n== %s  %d windows, %d towns, %d failed ==' %
          (tag, len(stats), len({s['town'] for s in stats.values()}), len(fails)))
    for it, e in fails[:10]:
        print('   FAIL %s  %s' % (it, e))
    print('  lanes in disc        min/med/p90/p99/max  %s' % q(g('n_cand')))
    print('  lanes kept (M=%d)    %s' % (M, q(g('n_lane'))))
    print('  corridor lanes       %s   dropped by cap: %d' % (q(g('n_corr')), int(g('dropped_corridor').sum())))
    print('  junction lanes       %s' % q(g('n_junction')))
    e_raw, e_kept = g('n_edge_raw'), g('n_edge')
    print('  edges before cap     %s' % q(e_raw))
    print('  edges stored (E=%d)  %s' % (E, q(e_kept)))
    print('  E overflow windows   %d/%d = %.3f%%   dropped edges %d (%.3f%% of raw)'
          % (int(g('overflow').sum()), len(stats), 100 * g('overflow').mean(),
             int((e_raw - e_kept).sum()), 100 * (e_raw - e_kept).sum() / max(e_raw.sum(), 1)))
    print('  edges by type  SUCC %d  LEFT %d  RIGHT %d  OPP %d   succ dropped by 5 m guard %d'
          % (g('n_succ').sum(), g('n_left').sum(), g('n_right').sum(), g('n_opp').sum(),
             int(g('succ_guard').sum())))
    na, az, ac, mz = g('n_agent').sum(), g('a2l_zero').sum(), g('a2l_cap').sum(), g('moving_zero').sum()
    print('  agents at anchor %d   A2L 0 candidates %d (%.2f%%)   > K=%d candidates %d (%.2f%%)'
          % (na, az, 100 * az / max(na, 1), K, ac, 100 * ac / max(na, 1)))
    print('  moving vehicles with 0 candidates %d (%.2f%% of all agents)' % (mz, 100 * mz / max(na, 1)))
    print('  containing lanes pushed out of top-K %d   lanes dropped for < 2 samples in disc %d'
          % (int(g('inside_drop').sum()), int(g('n_short').sum())))
    dl = np.concatenate([s['dlat'] for s in stats.values() if len(s['dlat'])]) if stats else np.zeros(0)
    if len(dl):
        print('  d_lat  |.|<0.5m %.1f%%  |.|<1.5m %.1f%%  median %.2f  p95 |.| %.2f'
              % (100 * (np.abs(dl) < 0.5).mean(), 100 * (np.abs(dl) < 1.5).mean(),
                 np.median(dl), np.percentile(np.abs(dl), 95)))
    print('  route_rel  intersects %s   reachable3 %s   shares_downstream %s'
          % (q(g('n_intersect')), q(g('n_reach')), q(g('n_shares'))))
    print('  lane tangent vs CARLA waypoint yaw agreement: %.4f'
          % min(s['tangent_agree'] for s in stats.values()))


if __name__ == '__main__':
    main()

