"""Make route-conditioned synthetic-ego pkls (v1).

For each original SMART pkl, replace the EGO's future (steps 11..90 @10Hz) with a
synthetic trajectory that follows the scene's route centerline (from
dump_route_centerlines.py) at a plausible speed profile: start at ego's t0 speed,
relax toward min(ego_speed, 13.0 m/s) with |accel| <= 2 m/s^2, advance along the
centerline arc. Heading = local tangent, velocity = speed * tangent. All other
agents and the map are untouched. Tokens without a centerline are copied unmodified.

Env: smart (torch + numpy). CPU only.

Usage:
  python /data1/jeongtae/smart_difficulty/scripts/make_routed_pkls.py --split val14
  python /data1/jeongtae/smart_difficulty/scripts/make_routed_pkls.py --split mini

In:  /data1/jeongtae/smart_difficulty/pkls/smart_pkls_{split}/{token}.pkl
     /data1/jeongtae/smart_difficulty/routes/{split}_centerlines.pkl
Out: /data1/jeongtae/smart_difficulty/pkls/routed_pkls_{split}/{token}.pkl
     /data1/jeongtae/smart_difficulty/pkls/routed_pkls_{split}_skips.json
"""
import argparse, json, os, pickle, shutil

import numpy as np
import torch

BASE = "/data1/jeongtae/smart_difficulty"
DT = 0.1
N_FUT = 80          # steps 11..90
V_CAP = 13.0        # m/s relaxation target cap
A_MAX = 2.0         # m/s^2


def smooth_polyline(cl, window=5):
    """Centered moving average over the ~1 m-spaced polyline (edge-padded).
    nuPlan lane baselines carry meter-scale zigzag (~10 cm lateral, up to ~0.5 rad
    tangent flips) which would inject 10Hz heading jitter into the synthetic ego."""
    if len(cl) < window:
        return cl
    k = np.ones(window) / window
    pad = window // 2
    xs = np.convolve(np.pad(cl[:, 0], pad, mode="edge"), k, "valid")
    ys = np.convolve(np.pad(cl[:, 1], pad, mode="edge"), k, "valid")
    return np.stack([xs, ys], axis=1)


def project_to_polyline(cl, p):
    """Project point p [2] onto polyline cl [M,2]. Returns arc length s0 and distance."""
    a = cl[:-1]
    b = cl[1:]
    ab = b - a
    seg_len2 = (ab ** 2).sum(1)
    seg_len2 = np.maximum(seg_len2, 1e-12)
    t = np.clip(((p - a) * ab).sum(1) / seg_len2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = np.hypot(proj[:, 0] - p[0], proj[:, 1] - p[1])
    i = int(d.argmin())
    seg_len = np.sqrt(seg_len2)
    s = np.r_[0.0, np.cumsum(seg_len)]
    return float(s[i] + t[i] * seg_len[i]), float(d[i])


def synth_future(cl, ego_xy, ego_speed):
    """Synthetic 80-step future along centerline arc.
    Returns xy [80,2], heading [80], speed [80], plus (s0, total arc length, clipped flag)."""
    cl = smooth_polyline(cl.astype(np.float64))
    seg = np.diff(cl, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    s = np.r_[0.0, np.cumsum(seg_len)]
    s0, _ = project_to_polyline(cl, np.asarray(ego_xy, np.float64))

    # speed profile: relax from ego_speed toward min(ego_speed, V_CAP) at |a| <= A_MAX
    v_tgt = min(ego_speed, V_CAP)
    v = np.empty(N_FUT)
    cur = float(ego_speed)
    for k in range(N_FUT):
        dv = np.clip(v_tgt - cur, -A_MAX * DT, A_MAX * DT)
        cur += dv
        v[k] = cur
    arc = s0 + np.cumsum(v * DT)
    clipped = bool(arc[-1] > s[-1])
    arc_c = np.clip(arc, 0.0, s[-1])

    x = np.interp(arc_c, s, cl[:, 0])
    y = np.interp(arc_c, s, cl[:, 1])

    # heading from local tangent: per-vertex segment angle, unwrapped, interpolated
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    ang_v = np.r_[ang, ang[-1]]          # vertex k -> angle of following segment
    ang_v = np.unwrap(ang_v)
    h = np.interp(arc_c, s, ang_v)
    h = (h + np.pi) % (2 * np.pi) - np.pi
    return np.stack([x, y], 1), h, v, s0, float(s[-1]), clipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val14", "mini", "val14n534"], required=True)
    args = ap.parse_args()

    in_dir = os.path.join(BASE, "pkls", f"smart_pkls_{args.split}")
    out_dir = os.path.join(BASE, "pkls", f"routed_pkls_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(BASE, "routes", f"{args.split}_centerlines.pkl"), "rb") as f:
        routes = pickle.load(f)

    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".pkl"))
    skips, n_mod, n_clip = [], 0, 0
    path_lens, disps = [], []
    for fname in files:
        tok = fname[:-4]
        src = os.path.join(in_dir, fname)
        dst = os.path.join(out_dir, fname)
        rec = routes.get(tok)
        if rec is None:
            shutil.copyfile(src, dst)
            skips.append(tok)
            continue

        with open(src, "rb") as f:
            d = pickle.load(f)
        ag = d["agent"]
        av = ag["av_index"]
        t0_xy = ag["position"][av, 10, :2].numpy().astype(np.float64)
        orig_end = ag["position"][av, 90, :2].numpy().astype(np.float64).copy()

        xy, h, v, s0, s_total, clipped = synth_future(rec["centerline"], t0_xy,
                                                      float(rec["ego_speed"]))
        n_clip += int(clipped)

        ag["position"][av, 11:91, 0] = torch.tensor(xy[:, 0], dtype=ag["position"].dtype)
        ag["position"][av, 11:91, 1] = torch.tensor(xy[:, 1], dtype=ag["position"].dtype)
        ag["heading"][av, 11:91] = torch.tensor(h, dtype=ag["heading"].dtype)
        ag["velocity"][av, 11:91, 0] = torch.tensor(v * np.cos(h), dtype=ag["velocity"].dtype)
        ag["velocity"][av, 11:91, 1] = torch.tensor(v * np.sin(h), dtype=ag["velocity"].dtype)
        ag["valid_mask"][av, 11:91] = True

        with open(dst, "wb") as f:
            pickle.dump(d, f)
        n_mod += 1
        path_lens.append(min(s0 + float((v * DT).sum()), s_total) - s0)
        disps.append(float(np.hypot(*(xy[-1] - orig_end))))

    print(f"[{args.split}] modified={n_mod} skipped={len(skips)} "
          f"clipped_at_route_end={n_clip}")
    if path_lens:
        print(f"  synthetic path length m: mean={np.mean(path_lens):.1f} "
              f"min={np.min(path_lens):.1f} max={np.max(path_lens):.1f}")
        print(f"  |synthetic end - original ego end| m: mean={np.mean(disps):.1f} "
              f"median={np.median(disps):.1f} max={np.max(disps):.1f}")
    with open(os.path.join(BASE, "pkls", f"routed_pkls_{args.split}_skips.json"), "w") as f:
        json.dump(skips, f, indent=1)


if __name__ == "__main__":
    main()
