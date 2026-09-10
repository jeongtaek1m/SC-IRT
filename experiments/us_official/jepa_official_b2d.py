#!/usr/bin/env python3
"""Agent-JEPA through the OFFICIAL code (minDrive-JEPA, github.com/hellojais/mindrive-jepa, Jaiswal 2026,
arXiv 2606.28383) on the Bench2Drive reference rollouts — Table 3A row "Agent-JEPA (official code)".

The official package is used unmodified: its SceneTokenizer builds the [50, 21, 6] scenario tensors, its
scripts/train.py + Trainer train the model (context = first 25 steps, target = last 25, EMA target encoder,
auxiliary position loss, 50 epochs, warmup + cosine, best.pt = lowest validation loss), and its
SurpriseScore computes surprise = ||predicted latent - target latent||_2. What is ours is the data path:

  preprocess   the nuPlan SQLite reader is replaced by a reader of our 10 Hz annotations
               (/data1/jeongtae/b2d_eval_sensors/route_<id>/anno/*.json.gz, the PDM-Lite reference rollout that
               every Table 3A descriptor and the ATDrive encoder read). It emits exactly the scenario dicts the
               official reader emits — ego_trajectory [{timestamp, x, y, vx, vy, heading}], agent_tracks
               [{timestamp, track_token, x, y, vx, vy, heading, label}] — and hands them to the official
               tokenizer (ego-centric origin at t0, /50 m and /10 m/s normalisation, the 20 agents nearest the
               ego at t0 kept by track id for the whole window, label 0 vehicle / 1 pedestrian / 2 cyclist / 3
               other). Conventions of the annotation (established for the Min-TTC rebuild, see
               results/us_official_provenance/min_ttc_provenance.json): heading = radians(rotation[2]) (the CARLA
               yaw; the top-level `theta` is offset by +pi/2), velocity = speed * (cos yaw, sin yaw), timestamps =
               frame index x 100,000 us (10 Hz). A scenario = a 5 s window (50 frames) of a route; windows start
               every 10 frames (1 s), so a route of n frames gives floor((n - 50) / 10) + 1 scenarios (every route
               has >= 66 frames). Class map: vehicle/car, vehicle/truck -> 0, walker -> 1, vehicle/bicycle -> 2,
               anything else -> 3. No response, planner or scenario-type label is read.
  config       configs/b2d.yaml = the official default.yaml with device cuda and our data / checkpoint paths.
  aggregate    the official evaluate.py scores every scenario; a route's difficulty is the MEAN surprise over
               its windows (ours: the paper scores 5 s scenarios and never aggregates to a route). Written to
               data/features/eval_jepa_official.npz (1-d) and eval_jepa_official_3d.npz ([mean, max, p90]).

    $P experiments/us_official/jepa_official_b2d.py preprocess
    (cd /data2/jeongtae/official_baselines/mindrive-jepa && CUDA_VISIBLE_DEVICES=1 $P scripts/train.py --config configs/b2d.yaml)
    (cd ... && CUDA_VISIBLE_DEVICES=1 $P scripts/evaluate.py --config configs/b2d.yaml --checkpoint checkpoints_b2d/best.pt --output outputs/b2d_surprise.csv --top 5)
    $P experiments/us_official/jepa_official_b2d.py aggregate
"""
import csv
import glob
import gzip
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OFFICIAL = Path(os.environ.get('ATDRIVE_JEPA_OFFICIAL', '/data2/jeongtae/official_baselines/mindrive-jepa'))
ANNO = Path('/data1/jeongtae/b2d_eval_sensors')
PROCESSED = OFFICIAL / 'data' / 'b2d_processed'
WIN, STRIDE, HZ_US = 50, 10, 100_000
CLASS_TO_LABEL = {('vehicle', 'car'): 0, ('vehicle', 'truck'): 0, ('vehicle', 'van'): 0, ('vehicle', 'bus'): 0,
                  ('walker', None): 1, ('vehicle', 'bicycle'): 2, ('vehicle', 'motorcycle'): 2}


def route_frames(route_dir):
    fs = sorted(glob.glob(str(route_dir / 'anno' / '*.json.gz')))
    return [json.load(gzip.open(f)) for f in fs]


def scenario_dict(frames, start, rid):
    ego, tracks = [], []
    for k in range(WIN):
        a = frames[start + k]
        ts = (start + k) * HZ_US
        eb = next(b for b in a['bounding_boxes'] if b['class'] == 'ego_vehicle')
        yaw = math.radians(eb['rotation'][2])
        ego.append({'timestamp': ts, 'x': float(a['x']), 'y': float(a['y']),
                    'vx': float(a['speed']) * math.cos(yaw), 'vy': float(a['speed']) * math.sin(yaw), 'heading': yaw})
        for b in a['bounding_boxes']:
            if b['class'] == 'ego_vehicle':
                continue
            yaw_b = math.radians(b['rotation'][2])
            tracks.append({'timestamp': ts, 'track_token': str(b['id']), 'x': float(b['location'][0]),
                           'y': float(b['location'][1]), 'vx': float(b['speed']) * math.cos(yaw_b),
                           'vy': float(b['speed']) * math.sin(yaw_b), 'heading': yaw_b,
                           'label': CLASS_TO_LABEL.get((b['class'], b.get('base_type')), 3)})
    return {'scenario_id': f'route_{rid}_{start}', 'db_path': str(ANNO / f'route_{rid}'),
            'ego_trajectory': ego, 'agent_tracks': tracks}


def preprocess():
    sys.path.insert(0, str(OFFICIAL / 'src'))
    import yaml
    from mindrive_jepa.data.tokenizer import SceneTokenizer                # the official tokenizer, unmodified
    cfg = yaml.safe_load(open(REPO / 'experiments' / 'us_official' / 'jepa_official_b2d.yaml'))
    (OFFICIAL / 'configs').mkdir(exist_ok=True)
    shutil.copy(REPO / 'experiments' / 'us_official' / 'jepa_official_b2d.yaml', OFFICIAL / 'configs' / 'b2d.yaml')
    tok = SceneTokenizer(cfg['data'])
    routes = sorted(ANNO.glob('route_*'))
    scenarios = []
    for rd in routes:
        rid = rd.name.split('_', 1)[1]
        frames = route_frames(rd)
        n = len(frames)
        if n < WIN:
            print(f'route {rid}: {n} frames < {WIN}, skipped')
            continue
        for start in range(0, n - WIN + 1, STRIDE):
            scenarios.append(scenario_dict(frames, start, rid))
    print(f'{len(routes)} routes -> {len(scenarios)} scenarios of {WIN} frames (stride {STRIDE})', flush=True)
    tensors, meta = tok.tokenize_dataset(scenarios)
    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)
    tok.save_processed(tensors, str(PROCESSED), meta)
    x = np.stack([t.numpy() for t in tensors])
    print('tensor', x.shape, 'agents per scenario (nonzero rows at t0) mean %.1f' % (np.abs(x[:, 0, 1:, :2]).sum(-1) > 0).sum(1).mean(),
          '| |x,y| p99 %.2f  |v| p99 %.2f' % (np.percentile(np.abs(x[..., :2]), 99), np.percentile(np.abs(x[..., 2:4]), 99)))


def aggregate(csv_path=None):
    csv_path = Path(csv_path or OFFICIAL / 'outputs' / 'b2d_surprise.csv')
    manifest = json.load(open(PROCESSED / 'manifest.json'))
    sid_of = {Path(s['file']).stem: s['scenario_id'] for s in manifest['scenarios']}
    per_route = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            rid = sid_of[row['scenario']].split('_')[1]
            per_route.setdefault(rid, []).append(float(row['surprise_score']))
    routes = sorted(per_route)
    names = np.array([f'route_{r}' for r in routes])
    mean = np.array([[np.mean(per_route[r])] for r in routes], np.float32)
    three = np.array([[np.mean(per_route[r]), np.max(per_route[r]), np.percentile(per_route[r], 90)] for r in routes], np.float32)
    np.savez(REPO / 'data' / 'features' / 'eval_jepa_official.npz', names=names, stats=mean)
    np.savez(REPO / 'data' / 'features' / 'eval_jepa_official_3d.npz', names=names, stats=three)
    n = [len(per_route[r]) for r in routes]
    print(f'{len(routes)} routes, windows per route min/median/max {min(n)}/{int(np.median(n))}/{max(n)}; '
          f'route mean surprise {mean.mean():.4f} +- {mean.std():.4f}; wrote eval_jepa_official.npz / _3d.npz')


if __name__ == '__main__':
    {'preprocess': preprocess, 'aggregate': aggregate}[sys.argv[1]](*sys.argv[2:])
