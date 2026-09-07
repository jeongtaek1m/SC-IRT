#!/usr/bin/env python3
"""B2D route corridor, reconstructed from the CARLA lane graph -- not from XML interpolation.

The XML waypoint list is a mission anchor, not geometry: measured spacing is 2.00 m median but
the largest gap is 117.4 m, and joining those with straight lines invents road that does not
exist, which would corrupt route_order_norm and polyline_intersects_route. So each waypoint is
map-matched to a lane, consecutive matches are connected through directed lane topology, and the
real centrelines are concatenated and resampled at 2 m.

Corridor semantics (contract A, see KEYS.md): the route is not one exact lane. Every lane in the
same road section travelling in the same direction as a matched route lane is a corridor member,
so that it means the same thing as a nuPlan roadblock, which bundles parallel lanes.

Fails loudly rather than emitting a wrong route: if a stretch cannot be connected through the lane
graph it is left as a hole and reported, and a densified gap of 5 m or more fails the route.
"""
from collections import deque
import numpy as np, carla, xml.etree.ElementTree as ET, os, json

XODR = '/data1/jeongtae/carla915/CarlaUE4/Content/Carla/Maps/OpenDrive'
ALT = {t: f'/data1/jeongtae/carla915/CarlaUE4/Content/Carla/Maps/{t}/OpenDrive/{t}.xodr'
       for t in ('Town11', 'Town12', 'Town13', 'Town15')}
XML = '/home/jeongtae/IRT/repos/carla_garage_b2d/leaderboard/data/bench2drive220.xml'
STEP, MAX_GAP, MAX_LANES = 2.0, 5.0, 400

def load_routes():
    t = ET.parse(XML)
    return {r.attrib['id']: (r.attrib['town'],
            np.array([[float(w.attrib['x']), float(w.attrib['y']), float(w.attrib['z'])]
                      for w in r.find('waypoints')], float))
            for r in t.findall('.//route')}

_maps = {}
def get_map(town):
    if town not in _maps:
        p = ALT.get(town, f'{XODR}/{town}.xodr')
        _maps[town] = carla.Map(town, open(p).read())
    return _maps[town]

def key(w):
    return (w.road_id, w.lane_id, w.section_id)

def _lane_seq(w, max_len=250.0):
    """w 부터 그 lane 을 따라 앞으로 (최대 max_len). next_until_lane_end 는 대형 맵의
    긴 lane 에서 세그폴트를 내므로 next() 로 직접 걷는다. XML waypoint 간격이 최대 117 m
    이므로 250 m 면 목표가 항상 이 안에 들어온다."""
    seq = [w]; cur = w; k = key(w)
    for _ in range(int(max_len / STEP)):
        nx = cur.next(STEP)
        if not nx:
            break
        same = [p for p in nx if key(p) == k]
        if not same:
            break
        cur = same[0]; seq.append(cur)
    return seq

def _cut_at(seq, b):
    """lane 안에서 목표 waypoint b 위치까지만 남긴다"""
    loc = b.transform.location
    d = [ (p.transform.location.x-loc.x)**2 + (p.transform.location.y-loc.y)**2 for p in seq ]
    return seq[:int(np.argmin(d))+1]

def connect(m, a, b, max_lanes=MAX_LANES):
    """a 에서 b 가 놓인 lane 까지의 directed lane-graph 경로; 도달 불가면 None.

    lane 단위로 BFS 한다. lane 하나를 끝까지 걸어간 뒤 그 끝에서 successor 로 넘어가고,
    차선변경이 허용된 경우 인접 lane 도 후보에 넣는다. 2 m 단위로 BFS 하면서 lane 키로
    중복 제거하면 같은 lane 안의 첫 전진이 곧바로 걸려 탐색이 죽는다.
    """
    tgt = key(b)
    q = deque([(a, [])]); seen = {key(a)}
    while q:
        w, path = q.popleft()
        seq = _lane_seq(w)
        if key(w) == tgt:
            return path + _cut_at(seq, b)
        if len(seen) > max_lanes:
            continue
        nxt = list(seq[-1].next(STEP) or [])
        for side in (seq[-1].get_left_lane(), seq[-1].get_right_lane()):
            if side is not None and side.lane_type == carla.LaneType.Driving \
               and side.road_id == seq[-1].road_id and np.sign(side.lane_id) == np.sign(seq[-1].lane_id):
                nxt.append(side)
        for n in nxt:
            k = key(n)
            if k not in seen:
                seen.add(k); q.append((n, path + seq))
    return None

def build(route_id, town, wps):
    m = get_map(town)
    anchors = [m.get_waypoint(carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                              project_to_road=True, lane_type=carla.LaneType.Driving)
               for p in wps]
    pts, lanes, holes = [], [], 0
    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        d = np.hypot(wps[i + 1][0] - wps[i][0], wps[i + 1][1] - wps[i][1])
        if d < 3.0:                                    # dense stretch: the anchors themselves are fine
            pts.append([a.transform.location.x, a.transform.location.y]); lanes.append(key(a))
            continue
        path = connect(m, a, b)                        # sparse stretch: recover through the lane graph
        if path is None:
            holes += 1
            continue
        for w in path:
            pts.append([w.transform.location.x, w.transform.location.y]); lanes.append(key(w))
    if not pts:
        return None
    P = np.array(pts, float)
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]
    P = P[keep]; lanes = [l for l, k in zip(lanes, keep) if k]
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    n = max(int(s[-1] / STEP) + 1, 2)
    si = np.linspace(0, s[-1], n)
    dense = np.c_[np.interp(si, s, P[:, 0]), np.interp(si, s, P[:, 1])]
    gap = float(np.linalg.norm(np.diff(dense, axis=0), axis=1).max()) if len(dense) > 1 else 0.0
    # corridor: same (road, section) and same lane_id sign as any matched route lane
    corridor = set()
    for rid, lid, sec in set(lanes):
        corridor.add((rid, sec, int(np.sign(lid))))
    return dict(route=dense, arclen=float(s[-1]), holes=holes, max_gap=gap,
                lane_keys=sorted(set(lanes)), corridor=sorted(corridor),
                ok=(gap < MAX_GAP and holes == 0))

if __name__ == '__main__':
    R = load_routes()
    rows = []
    for rid, (town, wps) in sorted(R.items()):
        try:
            r = build(rid, town, wps)
        except Exception as e:
            print(f'  ERR {rid} {town} {type(e).__name__} {e}'); continue
        if r is None:
            print(f'  EMPTY {rid}'); continue
        rows.append((rid, town, r['arclen'], r['holes'], r['max_gap'], len(r['lane_keys']),
                     len(r['corridor']), r['ok']))
        np.savez_compressed(f'/data2/jeongtae/relgraph_e16sel/route_{rid}.npz',
                            route=r['route'].astype(np.float32),
                            lane_keys=np.array(r['lane_keys']), corridor=np.array(r['corridor']),
                            arclen=r['arclen'], holes=r['holes'], max_gap=r['max_gap'])
        if len(rows) % 40 == 0: print(f'  {len(rows)}/220', flush=True)
    A = np.array([r[2] for r in rows]); H = np.array([r[3] for r in rows])
    G = np.array([r[4] for r in rows]); OK = np.array([r[7] for r in rows])
    print(f'\n{len(rows)}/220 route')
    print(f'  총연장  중앙값 {np.median(A):.0f} m  범위 {A.min():.0f}~{A.max():.0f}   (XML 직선판은 중앙값 109 m)')
    print(f'  연결 실패 구간(hole)  0개인 route {int((H==0).sum())}/{len(rows)}   평균 {H.mean():.2f}')
    print(f'  densify 후 최대 간격  중앙값 {np.median(G):.2f} m  최대 {G.max():.2f} m   (기준 < {MAX_GAP})')
    print(f'  전체 통과 (hole 0 & gap<{MAX_GAP})  {int(OK.sum())}/{len(rows)}')
    print(f'  corridor (road,section,sign) 수  중앙값 {np.median([r[6] for r in rows]):.0f}')
    json.dump([list(map(lambda x: x.item() if hasattr(x,'item') else x, r)) for r in rows],
              open('/data2/jeongtae/relgraph_e16sel/route_summary.json','w'), indent=1)
