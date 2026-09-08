#!/usr/bin/env python3
"""nuPlan val14 relgraph v2 extraction (PROTOCOL_R section 20.2 S1).

Thin driver around navsim_extract.extract_window -- the map / route / relation code is imported,
not copied, so the nuPlan tensor is produced by the SAME frozen geometry as navsim_relgraph_v2.
navsim_extract.py is not edited.

Inputs
  agents / amask / ego / cmd   /data2/jeongtae/navsim_interact/val14_tensors.npz  (copied verbatim,
                               exactly as NavSim's came from navtest_tensors_v2.npz)
  anchor pose                  the ego pose the tensor was built around: routed_pkls_val14
                               ego at pkl frame 15 (widx 0) / 35 (widx 1).  The agents in the
                               tensor are expressed in THIS frame (interact_data_val14.py), so the
                               lanes must be too.  The logged nuPlan DB pose at the same timestamp
                               is recorded next to it (dx, dy, dyaw) but not used for geometry.
  route roadblock ids          scene.roadblock_ids of the scenario's initial lidar_pc (what the
                               simulated planner is given; == NuPlanScenario.get_route_roadblock_ids)
  map name                     log.map_version of the scenario's DB
  DBs                          /nas/datasets_archive/nuplan/nuplan-v1.1/splits/val/<log>.db

Output
  /data2/jeongtae/relgraph/nuplan_val14_relgraph_v2.npz         KEYS.md schema, domain='nuplan',
                                                                item_id = '<token>_<widx>'
  /data2/jeongtae/relgraph/nuplan_val14_relgraph_v2_audit.npz   sidecar (navsim_extract.write_sidecar)
  transfer/nuplan/nuplan_val14_windows.json                     per-window provenance + diagnostics
  transfer/nuplan/nuplan_val14_failed_windows.json              windows not emitted (never filled)

Run with /home/jeongtae/miniconda3/envs/smart/bin/python.
"""
import csv, json, math, os, pickle, sqlite3, sys, time, traceback

import numpy as np

sys.path.insert(0, '/data2/jeongtae/relgraph')
import navsim_extract as nx                                   # noqa: E402  (sets NUPLAN_MAPS_ROOT)

TENSORS = '/data2/jeongtae/nuplan_full_val14/val14n534_tensors.npz'
PKL_DIR = '/data1/jeongtae/smart_difficulty/pkls/routed_pkls_val14n534'   # the tensor builder's input
LOGMAP = '/data2/jeongtae/nuplan_full_val14/val14n534_scenario_log_map.csv'
DB_DIR = '/nas/datasets_archive/nuplan/nuplan-v1.1/splits/val'
OUT = '/data2/jeongtae/nuplan_full_val14/nuplan_val14n534_relgraph_v2.npz'
RES = '/data2/jeongtae/nuplan_full_val14/res'
ANCHOR_FRAME = {0: 15, 1: 35}          # pkl frame (10 Hz) of the window anchor, S0 verified
ANCHOR_DT = {0: 1.5, 1: 3.5}           # seconds after the scenario's initial lidar_pc


def yaw_of(qw, qx, qy, qz):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def wrapd(a):
    return math.degrees((a + math.pi) % (2 * math.pi) - math.pi)


def db_lookup(con, tok):
    """initial lidar_pc -> (timestamp, route roadblock ids, map name) and the logged ego pose
    at each anchor timestamp (record only)."""
    ts, sc = con.execute('select timestamp, scene_token from lidar_pc where token=?',
                         (bytes.fromhex(tok),)).fetchone()
    rb = con.execute('select roadblock_ids from scene where token=?', (sc,)).fetchone()[0].split()
    loc = con.execute('select map_version from log').fetchone()[0]
    db_pose = {}
    for w, dt in ANCHOR_DT.items():
        t = ts + int(dt * 1e6)
        q = con.execute('select timestamp, scene_token, ego_pose_token from lidar_pc '
                        'where timestamp between ? and ? order by abs(timestamp-?) limit 1',
                        (t - 60000, t + 60000, t)).fetchone()
        if q is None:
            db_pose[w] = None
            continue
        e = con.execute('select x,y,qw,qx,qy,qz from ego_pose where token=?', (q[2],)).fetchone()
        rb2 = con.execute('select roadblock_ids from scene where token=?', (q[1],)).fetchone()[0].split()
        db_pose[w] = dict(x=e[0], y=e[1], yaw=yaw_of(*e[2:]), dt_actual=(q[0] - ts) / 1e6,
                          same_scene=q[1] == sc, same_roadblock_ids=rb2 == rb)
    return ts, rb, loc, db_pose


def pkl_pose(tok):
    d = pickle.load(open(f'{PKL_DIR}/{tok}.pkl', 'rb'))
    ag = d['agent']
    av = int(d['av_index']) if 'av_index' in d else int(ag['av_index'])
    pos = ag['position'][av, :, :2].numpy().astype(np.float64)   # same cast as the tensor builder
    head = ag['heading'][av].numpy()
    val = ag['valid_mask'][av].numpy()
    return {w: (float(pos[f, 0]), float(pos[f, 1]), float(head[f]), bool(val[f]))
            for w, f in ANCHOR_FRAME.items()}


def main():
    t_start = time.time()
    tz = np.load(TENSORS, allow_pickle=True)
    tokens, widx = tz['token'].tolist(), tz['widx'].tolist()
    ag_all, am_all, ego_all, cmd_all = tz['agents'], tz['amask'], tz['ego'], tz['cmd']
    ids = [f'{t}_{w}' for t, w in zip(tokens, widx)]
    assert len(set(ids)) == len(ids)
    log_of = {r['scenario']: r['log_name'] for r in csv.DictReader(open(LOGMAP))}
    assert set(tokens) <= set(log_of), 'token not in scenario->log map'

    # ---- per-scenario lookups, grouped by log so each DB is opened once --------------------
    scen = {}
    by_log = {}
    for t in sorted(set(tokens)):
        by_log.setdefault(log_of[t], []).append(t)
    t0 = time.time()
    for k, (lg, toks) in enumerate(sorted(by_log.items())):
        con = sqlite3.connect(f'file:{DB_DIR}/{lg}.db?mode=ro', uri=True)
        for t in toks:
            ts, rb, loc, db_pose = db_lookup(con, t)
            scen[t] = dict(log=lg, ts=ts, roadblock_ids=rb, map=loc, db_pose=db_pose,
                           pkl_pose=pkl_pose(t))
        con.close()
        if (k + 1) % 50 == 0:
            print(f'  lookup {k + 1}/{len(by_log)} logs  {time.time() - t0:.0f}s', flush=True)
    t_lookup = time.time() - t0
    print(f'lookup done: {len(scen)} scenarios / {len(by_log)} logs in {t_lookup:.0f}s', flush=True)

    # ---- extraction --------------------------------------------------------------------
    got, meta, failed = {}, [], []
    t0 = time.time()
    for i, (tok, w) in enumerate(zip(tokens, widx)):
        s = scen[tok]
        x, y, yaw, valid = s['pkl_pose'][w]
        rec = dict(item_id=ids[i], token=tok, widx=int(w), log=s['log'], map=s['map'],
                   n_roadblock_ids=len(s['roadblock_ids']),
                   anchor_pose_pkl=[x, y, yaw], anchor_frame_pkl=ANCHOR_FRAME[w])
        dp = s['db_pose'][w]
        if dp is not None:
            rec.update(anchor_pose_db=[dp['x'], dp['y'], dp['yaw']], db_dt_actual=dp['dt_actual'],
                       db_same_scene=dp['same_scene'], db_same_roadblock_ids=dp['same_roadblock_ids'],
                       pkl_vs_db_dxy=math.hypot(x - dp['x'], y - dp['y']),
                       pkl_vs_db_dyaw_deg=wrapd(yaw - dp['yaw']))
        if not valid:
            rec['fail'] = 'ego invalid at anchor frame in pkl'
            failed.append(rec); meta.append(rec)
            continue
        try:
            pk, dg = nx.extract_window(nx.MAPS, s['map'], s['roadblock_ids'],
                                       np.array([x, y, yaw], np.float64), ag_all[i], am_all[i])
        except Exception:                                      # noqa: BLE001
            rec['fail'] = 'exception: ' + traceback.format_exc().strip().splitlines()[-1]
            failed.append(rec); meta.append(rec)
            continue
        if pk is None:
            rec['fail'] = 'no_lane' if dg.get('no_lane') else 'extract_window returned None'
            rec['diag'] = {k: v for k, v in dg.items() if not isinstance(v, list)}
            failed.append(rec); meta.append(rec)
            continue
        dg['token'] = ids[i]
        dg['map'] = s['map']
        got[ids[i]] = (pk, dg)
        rec['diag'] = {k: (v if not isinstance(v, (np.integer, np.floating, np.bool_)) else v.item())
                       for k, v in dg.items() if not isinstance(v, list)}
        meta.append(rec)
        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{len(ids)}  {time.time() - t0:.0f}s', flush=True)
    t_extract = time.time() - t0
    print(f'extraction done: {len(got)} emitted / {len(ids)} requested, {len(failed)} failed, '
          f'{t_extract:.0f}s', flush=True)

    # ---- save (key order / dtypes identical to navsim_extract.run) ------------------------
    keep = [d for d in ids if d in got]
    rows = [k for k, d in enumerate(ids) if d in got]
    packs = [got[d][0] for d in keep]
    diags = [got[d][1] for d in keep]
    stack = lambda k, dt: np.stack([p[k] for p in packs]).astype(dt)   # noqa: E731
    np.savez_compressed(
        OUT,
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
        domain=np.array('nuplan', dtype='<U8'))
    print(f'wrote {OUT}  N={len(keep)}', flush=True)
    nx.write_sidecar(OUT.replace('.npz', '_audit.npz'), keep, diags,
                     lambda dg, c: 0 if nx.SIDECAR_MAP[c] is None else dg.get(nx.SIDECAR_MAP[c], 0))

    def jsafe(o):
        if isinstance(o, dict):
            return {k: jsafe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [jsafe(v) for v in o]
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o
    json.dump(jsafe(dict(n_requested=len(ids), n_emitted=len(keep), n_failed=len(failed),
                         t_lookup_s=t_lookup, t_extract_s=t_extract,
                         t_total_s=time.time() - t_start, windows=meta)),
              open(f'{RES}/nuplan_val14_windows.json', 'w'), indent=1)
    json.dump(jsafe(failed), open(f'{RES}/nuplan_val14_failed_windows.json', 'w'), indent=1)

    nx.report(keep, packs, diags, None, None)
    print(f'\nwall time: lookup {t_lookup:.0f}s + extract {t_extract:.0f}s, total '
          f'{time.time() - t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
