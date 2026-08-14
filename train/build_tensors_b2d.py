#!/usr/bin/env python3
"""Bench2Drive rollout annotations -> interaction tensors.

Input is the expert-rollout annotation directory tree (one dir per route, GT 3D
boxes + ego state per frame) produced by the Bench2Drive leaderboard with a
box-recording agent. Output is one window-level npz consumed by
train/train_encoder_b2d.py.

Preprocessing contract (identical across every domain in the paper):
  * 0.5 s grid (annotation frames are 10 Hz; every 5th is kept)
  * 6 s windows: T=12 steps, stride 4 steps (2 s), anchor = 4th step
  * agents: valid at the anchor, within 60 m, nearest 48, ego-anchor rotation
  * agent channels  (48,12,8): [rel_x, rel_y, cos dpsi, sin dpsi, speed,
                                half_len, half_wid, is_vehicle]
  * ego channels    (12,6):    [rel_x, rel_y, cos, sin, speed, is_future(w>3)]
  * command         (4,):      one-hot [left, straight, right, other]

Usage:
  python train/build_tensors_b2d.py --anno_root <rollout_root> \
      --cmd_feats <per-route npz dir with 'cmd' and per-frame 'ego'> \
      --out data/interact/b2d_tensors.npz
"""

import argparse
import glob
import gzip
import json
import os
from multiprocessing import Pool

import numpy as np

A_MAX, T, STRIDE = 48, 12, 4
CMD_MAP = {1: 0, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1}   # -> [left,straight,right,other]


def parse(anno_dir, nf):
    """0.5 s-grid sequence of (ego, {track_id: agent}) from annotation frames.

    rec = (x, y, yaw, speed, half_len, half_wid, is_vehicle); the ego rec uses
    the same layout. This is the exact parser that produced the released
    artifact, unchanged.
    """
    files = sorted(glob.glob(f"{anno_dir}/*.json.gz"))[:nf]
    out = []
    for fi in range(0, len(files), 5):
        try:
            d = json.load(gzip.open(files[fi]))
        except Exception:                                 # noqa: BLE001
            continue
        ego, ags = None, {}
        for b in d.get("bounding_boxes", []):
            cl = b.get("class")
            if cl not in ("vehicle", "walker", "ego_vehicle"):
                continue
            loc, rot, ext = b.get("location"), b.get("rotation"), b.get("extent")
            if loc is None or not np.all(np.isfinite(loc[:2])):
                continue
            yaw = np.deg2rad(rot[2]) if rot and len(rot) > 2 else 0.0
            rec = (loc[0], loc[1], yaw, float(b.get("speed") or 0.0),
                   float(ext[0]) if ext else 2.4, float(ext[1]) if ext else 0.9,
                   1.0 if cl == "vehicle" else 0.0)
            if cl == "ego_vehicle":
                ego = rec
            elif b.get("id") is not None:
                ags[b["id"]] = rec
        if ego is not None:
            out.append((ego, ags))
    return out


def one(args):
    rd, nf, cmd_dir = args
    name = os.path.basename(rd)
    try:
        seq = parse(f"{rd}/anno", nf)
        cmd_arr = np.load(f"{cmd_dir}/{name}.npz")["cmd"]
        out = []
        for a in range(0, len(seq) - T, STRIDE):
            i = a + 3                                     # anchor = 4th step
            ex, ey, eyaw = seq[i][0][0], seq[i][0][1], seq[i][0][2]
            c, s = np.cos(-eyaw), np.sin(-eyaw)
            ids = sorted(seq[i][1], key=lambda t: np.hypot(
                seq[i][1][t][0] - ex, seq[i][1][t][1] - ey))
            ids = [t for t in ids if np.hypot(
                seq[i][1][t][0] - ex, seq[i][1][t][1] - ey) < 60.0][:A_MAX]
            ag = np.zeros((A_MAX, T, 8), np.float16)
            am = np.zeros((A_MAX, T), bool)
            for ai, tid in enumerate(ids):
                for w in range(T):
                    r = seq[a + w][1].get(tid)
                    if r is None:
                        continue
                    dx, dy = r[0] - ex, r[1] - ey
                    ag[ai, w] = [dx * c - dy * s, dx * s + dy * c,
                                 np.cos(r[2] - eyaw), np.sin(r[2] - eyaw),
                                 r[3], r[4], r[5], r[6]]
                    am[ai, w] = True
            eg = np.zeros((T, 6), np.float16)
            for w in range(T):
                e = seq[a + w][0]
                dx, dy = e[0] - ex, e[1] - ey
                eg[w] = [dx * c - dy * s, dx * s + dy * c,
                         np.cos(e[2] - eyaw), np.sin(e[2] - eyaw), e[3],
                         1.0 if w > 3 else 0.0]
            cm = np.zeros(4, np.float16)
            fa = min(i * 5, len(cmd_arr) - 1)
            cm[CMD_MAP.get(int(cmd_arr[fa, 0]), 3)] = 1.0
            out.append((ag, am, eg, cm))
        return name, out, ""
    except Exception as e:                                # noqa: BLE001
        return name, None, f"{type(e).__name__} {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno_root", required=True,
                    help="rollout root containing route_*/anno/")
    ap.add_argument("--cmd_feats", required=True,
                    help="dir of per-route npz with per-frame 'ego' and 'cmd'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    jobs = []
    for rd in sorted(glob.glob(f"{a.anno_root}/route_*")):
        name = os.path.basename(rd)
        if not os.path.isdir(f"{rd}/anno"):
            continue
        nf = len(np.load(f"{a.cmd_feats}/{name}.npz")["ego"])
        jobs.append((rd, nf, a.cmd_feats))
    names, AG, AM, EG, CM, WIX = [], [], [], [], [], []
    with Pool(a.workers) as p:
        for k, (nm, out, err) in enumerate(p.imap_unordered(one, jobs, chunksize=2)):
            if out is None:
                print("  ERR", nm, err, flush=True)
                continue
            for wi, (ag, am, eg, cm) in enumerate(out):
                names.append(nm); WIX.append(wi)
                AG.append(ag); AM.append(am); EG.append(eg); CM.append(cm)
            if (k + 1) % 50 == 0:
                print(f"  ...{k+1}/{len(jobs)} routes, {len(names)} windows", flush=True)
    np.savez_compressed(a.out, route=np.array(names), widx=np.array(WIX),
                        agents=np.stack(AG), amask=np.stack(AM),
                        ego=np.stack(EG), cmd=np.stack(CM))
    print(f"DONE {len(set(names))} routes, {len(names)} windows -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
