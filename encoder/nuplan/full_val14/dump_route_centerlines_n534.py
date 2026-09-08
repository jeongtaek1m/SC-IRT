"""Dump per-scene route centerlines for the route-conditioned synthetic-ego pipeline (v1).

For each scene token (columns of the response-matrix CSVs) load the nuPlan scenario
and extract the navigation-route centerline the way tuplan_garage's PDM planners do:
route roadblock ids -> route_roadblock_correction -> starting lane near ego ->
Dijkstra over the route lane graph -> concatenated lane baseline paths.
The polyline is resampled to ~1 m spacing and trimmed to [ego-30 m, ego+260 m] arc.

Env: navsim (needs nuplan-devkit + tuplan_garage on PYTHONPATH; also inserted below).

Usage:
  python SC-IRT/scripts/dump_route_centerlines.py --split val14
  python SC-IRT/scripts/dump_route_centerlines.py --split mini

Output: /data1/jeongtae/smart_difficulty/routes/{split}_centerlines.pkl
  {token: {"centerline": float32 [M,2], "ego_xy": float32 [2],
           "ego_heading": float, "ego_speed": float}}
plus {split}_centerlines_skips.json with skip reasons.
"""
import argparse, json, math, os, pickle, sys, traceback

import numpy as np

sys.path.insert(0, "/home/jeongtae/IRT/repos/nuplan-devkit")
sys.path.insert(0, "/home/jeongtae/IRT/repos/tuplan_garage")
sys.path.insert(0, "/home/jeongtae/IRT/SC-IRT/scripts")

from render_bev_video import load_scenarios  # noqa: E402

ANCHOR_ITER = 20        # scenario iteration matching pkl index 10 (t0), 20Hz->10Hz SUB=2
BEHIND_M = 30.0         # keep this much centerline behind ego
AHEAD_M = 260.0         # and this much ahead (>= 120 m required by spec)
MAX_EGO_DIST_M = 10.0   # ego farther than this from centerline => failure
SEARCH_DEPTH = 30       # PDM roadblock window for Dijkstra

SPLITS = {
    "val14n534": {
        "csv": "/data2/jeongtae/nuplan_full_val14/val14n534_token_matrix.csv",
        "data_root": "/data2/jeongtae/nuplan_full_val14/dbs_n534",
    },
    "val14": {
        "csv": "/home/jeongtae/IRT/SC-IRT/result/nuplan_val14_k7_ordered_response_matrix.csv",
        "data_root": "/data1/nuplan/nuplan-v1.1/splits/val",
    },
    "mini": {
        "csv": "/home/jeongtae/IRT/SC-IRT/result/nuplan_mini_kAll_response_matrix.csv",
        "data_root": "/data1/nuplan/nuplan-v1.1/splits/mini",
    },
}
MAP_ROOT = "/nas/datasets_archive/nuplan/maps"
OUT_DIR = "/data1/jeongtae/smart_difficulty/routes"


def load_route_dicts(map_api, route_roadblock_ids):
    """PDM AbstractPDMPlanner._load_route_dicts equivalent."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))
    rb_dict, lane_dict = {}, {}
    for id_ in route_roadblock_ids:
        block = map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
        block = block or map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if block is None:
            continue
        rb_dict[block.id] = block
        for lane in block.interior_edges:
            lane_dict[lane.id] = lane
    return rb_dict, lane_dict


def get_starting_lane(ego_state, route_lane_dict):
    """Nearest on-route lane; prefer close (<2 m) and heading-aligned (<45 deg) lanes.
    Simplified PDM _get_starting_lane (no drivable-area occupancy map needed)."""
    from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        normalize_angle)
    ego_xy = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y])
    ego_h = ego_state.rear_axle.heading
    best_lane, best_key = None, None
    for lane in route_lane_dict.values():
        path = lane.baseline_path.discrete_path
        pts = np.array([[s.x, s.y] for s in path])
        d = np.hypot(pts[:, 0] - ego_xy[0], pts[:, 1] - ego_xy[1])
        i = int(d.argmin())
        he = abs(float(normalize_angle(path[i].heading - ego_h)))
        key = (d[i] > 2.0, he > math.pi / 4, float(d[i]) if d[i] > 2.0 else he)
        if best_key is None or key < best_key:
            best_key, best_lane = key, lane
    return best_lane


def get_discrete_centerline(current_lane, rb_dict, lane_dict, search_depth=SEARCH_DEPTH):
    """PDM AbstractPDMPlanner._get_discrete_centerline equivalent."""
    from tuplan_garage.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import (
        Dijkstra)
    roadblocks = list(rb_dict.values())
    roadblock_ids = list(rb_dict.keys())
    start_idx = int(np.argmax(np.array(roadblock_ids) == current_lane.get_roadblock_id()))
    roadblock_window = roadblocks[start_idx:start_idx + search_depth]
    graph_search = Dijkstra(current_lane, list(lane_dict.keys()))
    route_plan, _ = graph_search.search(roadblock_window[-1])
    discrete = []
    for lane in route_plan:
        discrete.extend(lane.baseline_path.discrete_path)
    return discrete


def resample_and_trim(discrete_path, ego_xy):
    """Concat discrete states -> dedupe -> arc-length resample ~1 m -> trim around ego.
    Returns (centerline float32 [M,2], ego_min_dist, length_ahead)."""
    xy = np.array([[s.x, s.y] for s in discrete_path], np.float64)
    if len(xy) < 2:
        raise ValueError("centerline has <2 points")
    seg = np.hypot(*np.diff(xy, axis=0).T)
    keep = np.r_[True, seg > 1e-3]
    xy = xy[keep]
    if len(xy) < 2:
        raise ValueError("degenerate centerline after dedupe")
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]

    d = np.hypot(xy[:, 0] - ego_xy[0], xy[:, 1] - ego_xy[1])
    i0 = int(d.argmin())
    ego_min_dist = float(d[i0])
    s0 = float(s[i0])

    lo, hi = max(0.0, s0 - BEHIND_M), min(float(s[-1]), s0 + AHEAD_M)
    n = max(int(round(hi - lo)) + 1, 2)
    ss = np.linspace(lo, hi, n)
    cx = np.interp(ss, s, xy[:, 0])
    cy = np.interp(ss, s, xy[:, 1])
    centerline = np.stack([cx, cy], axis=1).astype(np.float32)
    length_ahead = float(s[-1] - s0)
    return centerline, ego_min_dist, length_ahead


def extract_one(scn):
    from tuplan_garage.planning.simulation.planner.pdm_planner.utils.route_utils import (
        route_roadblock_correction)
    ego_state = scn.get_ego_state_at_iteration(ANCHOR_ITER)
    route_ids = scn.get_route_roadblock_ids()
    if not route_ids:
        raise ValueError("empty route_roadblock_ids")
    rb_dict, lane_dict = load_route_dicts(scn.map_api, route_ids)
    if not lane_dict:
        raise ValueError("no route lanes resolved")
    try:
        corrected_ids = route_roadblock_correction(ego_state, scn.map_api, rb_dict)
        rb_dict, lane_dict = load_route_dicts(scn.map_api, corrected_ids)
    except Exception as e:
        print(f"    [warn] route correction failed ({type(e).__name__}: {e}); using raw route",
              flush=True)
    lane = get_starting_lane(ego_state, lane_dict)
    if lane is None:
        raise ValueError("no starting lane found")
    discrete = get_discrete_centerline(lane, rb_dict, lane_dict)
    ego_xy = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y])
    centerline, ego_min_dist, length_ahead = resample_and_trim(discrete, ego_xy)
    if ego_min_dist > MAX_EGO_DIST_M:
        raise ValueError(f"ego {ego_min_dist:.1f} m off centerline (> {MAX_EGO_DIST_M} m)")

    v = ego_state.dynamic_car_state.rear_axle_velocity_2d
    ego_speed = float(math.hypot(v.x, v.y))
    return {
        "centerline": centerline,
        "ego_xy": ego_xy.astype(np.float32),
        "ego_heading": float(ego_state.rear_axle.heading),
        "ego_speed": ego_speed,
    }, ego_min_dist, length_ahead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    args = ap.parse_args()
    cfg = SPLITS[args.split]

    import pandas as pd
    tokens = list(pd.read_csv(cfg["csv"], index_col=0).columns)
    print(f"[{args.split}] {len(tokens)} tokens from {cfg['csv']}", flush=True)

    scenarios = {s.token: s for s in load_scenarios(cfg["data_root"], MAP_ROOT, tokens)}

    out, skips = {}, {}
    n_short = 0
    for k, tok in enumerate(tokens):
        scn = scenarios.get(tok)
        if scn is None:
            skips[tok] = "scenario not found in DBs"
            continue
        try:
            rec, ego_min_dist, length_ahead = extract_one(scn)
            out[tok] = rec
            if length_ahead < 120.0:
                n_short += 1
                print(f"  [{k+1}/{len(tokens)}] {tok[:8]} short route: {length_ahead:.0f} m ahead",
                      flush=True)
        except Exception as e:
            skips[tok] = f"{type(e).__name__}: {e}"
            print(f"  [{k+1}/{len(tokens)}] SKIP {tok[:8]}: {skips[tok]}", flush=True)
            if not isinstance(e, ValueError):
                traceback.print_exc()
        if (k + 1) % 100 == 0:
            print(f"  [{k+1}/{len(tokens)}] ok={len(out)} skip={len(skips)}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{args.split}_centerlines.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    skip_path = os.path.join(OUT_DIR, f"{args.split}_centerlines_skips.json")
    with open(skip_path, "w") as f:
        json.dump(skips, f, indent=1)

    lens = [len(r["centerline"]) for r in out.values()]
    spd = [r["ego_speed"] for r in out.values()]
    print(f"\n[{args.split}] wrote {out_path}")
    print(f"  ok={len(out)} skip={len(skips)} short(<120 m ahead)={n_short}")
    if out:
        print(f"  centerline pts: mean={np.mean(lens):.0f} min={min(lens)} max={max(lens)}")
        print(f"  ego speed m/s: mean={np.mean(spd):.1f} max={max(spd):.1f}")
    print(f"  skips -> {skip_path}")


if __name__ == "__main__":
    main()
