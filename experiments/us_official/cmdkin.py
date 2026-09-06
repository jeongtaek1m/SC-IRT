#!/usr/bin/env python3
"""Reconstructed producer for the in-house `cmdkin` descriptor (Table 3A row
"Kinematics (cmdkin, 25d)").

The original producing script is missing from every repo in this project; only
its output (`data/features/eval_cmdkin_stats.npz`, 220 x 25 float32) and two
downstream slices survive (`eval_cmdkin_kinonly.npz` = columns 0:12,
`eval_cmdkin_cmdonly.npz` = columns 12:25, cut by
b2d_irt/scripts/cmdkin_decompose.py).  This file recomputes all 25 columns
directly from the raw PDM-Lite reference rollout annotations and asserts
bit-for-bit (float32) equality with the shipped npz.

Input  : /data1/jeongtae/b2d_eval_sensors/route_<id>/anno/*.json.gz  (10 Hz)
         fields used: x, y, theta, speed, command_near, command_far,
         x/y_command_near, x/y_command_far   -- ego pose/speed and the route
         command are all PDM-Lite's own rollout record.
Output : /data2/jeongtae/official_baselines/us_features/inhouse_cmdkin.npz
         {stats (220,25) float32, names (220,) 'route_<id>', layout}

Column layout (order fixed; names as in b2d_irt/scripts/cmdkin_decompose.py
and b2d_irt/NAVSIM_KIN12_SPEC.md):
   0 v_mean      5 a_std      10 yaw_mean   12..17 frac_cmd_near_1..6
   1 v_std       6 a_min      11 yaw_max    18..23 frac_cmd_far_1..6
   2 v_max       7 a_max                    24     cmd_switch_rate
   3 v_p10       8 a_p05
   4 frac_stop   9 absa_p95

Numerics are part of the definition: every statistic is evaluated on the
float32 arrays produced by `load_anno` (identical to
b2d_irt/scripts/eval_pipeline.py:load_anno, which is what wrote the
`ego`/`cmd` arrays inside /data1/jeongtae/b2d_jepa/eval_feats/route_*.npz),
and heading differences are wrapped with the float32 modulo form
((dh + pi) % (2 pi) - pi).  Any other wrap (np.angle(exp(i dh)), np.unwrap,
arctan2, or the same modulo in float64) reproduces columns 10-11 only to
~1.8e-07, i.e. not bit-for-bit.
"""
import glob
import gzip
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

EVAL_ROOT = Path('/data1/jeongtae/b2d_eval_sensors')
SHIPPED = Path('/home/jeongtae/SC-IRT/data/features/eval_cmdkin_stats.npz')
OUT = Path('/data2/jeongtae/official_baselines/us_features/inhouse_cmdkin.npz')
DT = 0.1          # 10 Hz annotation rate
V_STOP = 0.5      # m/s, "stopped" threshold
NCMD = 6          # CARLA discrete route commands 1..6

LAYOUT = ('v[mean,std,max,p10,frac_stop<0.5] a[std,min,max,p05,|a|p95] |dyaw|[mean,max] '
          'cmdnear1-6 cmdfar1-6 switch = 25d (time-weighted; all channels read the '
          'PDM-Lite reference rollout: ego speed/heading and the per-frame route command)')


def load_anno(anno_dir):
    """Byte-identical to b2d_irt/scripts/eval_pipeline.py:load_anno."""
    files = sorted(glob.glob(f'{anno_dir}/*.json.gz'))
    n = len(files)
    ego = np.zeros((n, 5), np.float32)
    cmd = np.zeros((n, 7), np.float32)
    for t, f in enumerate(files):
        try:
            d = json.load(gzip.open(f))
        except Exception:
            continue
        row = [d['x'], d['y'], d['theta'] - np.pi / 2, d['speed'], 1.0]
        if np.all(np.isfinite(row)):
            ego[t] = row
        crow = [d.get('command_near', 0), d.get('command_far', 0),
                d.get('x_command_near', 0), d.get('y_command_near', 0),
                d.get('x_command_far', 0), d.get('y_command_far', 0), 1.0]
        if np.all(np.isfinite(crow)):
            cmd[t] = crow
    return ego, cmd


def cmdkin(ego, cmd):
    """The 25 columns, on the valid frames of one route."""
    m = ego[:, 4] > 0
    v, h = ego[m, 3], ego[m, 2]
    a = np.diff(v) / DT                                     # m/s^2, adjacent 10 Hz difference
    dh = np.diff(h)
    yd = np.abs((dh + np.pi) % (2 * np.pi) - np.pi)         # rad per frame, NOT per second
    cn = cmd[m, 0].astype(int)
    cf = cmd[m, 1].astype(int)
    return np.array(
        [v.mean(), v.std(), v.max(), np.percentile(v, 10), (v < V_STOP).mean(),
         a.std(), a.min(), a.max(), np.percentile(a, 5), np.percentile(np.abs(a), 95),
         yd.mean(), yd.max()]
        + [(cn == k).mean() for k in range(1, NCMD + 1)]
        + [(cf == k).mean() for k in range(1, NCMD + 1)]
        + [float((np.diff(cn) != 0).mean())], np.float64)


def job(route_dir):
    return cmdkin(*load_anno(f'{route_dir}/anno'))


def main():
    routes = sorted(EVAL_ROOT.glob('route_*'))
    names = [r.name for r in routes]
    with Pool(12) as p:
        rows = p.map(job, [str(r) for r in routes], chunksize=2)
    st = np.stack(rows).astype(np.float32)

    ref = np.load(SHIPPED, allow_pickle=True)
    assert [str(x) for x in ref['names']] == names, 'route order differs from the shipped npz'
    same = st == ref['stats']
    if not same.all():
        bad = np.where(~same.all(0))[0]
        print('columns NOT bit-identical:', bad.tolist())
        print('max |diff| per bad column:',
              np.abs(st[:, bad].astype(np.float64) - ref['stats'][:, bad].astype(np.float64)).max(0))
    assert same.all(), 'cmdkin reconstruction is not bit-for-bit'
    print(f'bit-for-bit equal to {SHIPPED}: {st.shape}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, stats=st, names=np.array(names), layout=LAYOUT)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
