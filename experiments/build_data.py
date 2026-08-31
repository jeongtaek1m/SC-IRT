#!/usr/bin/env python3
"""Package the derived data artifacts this release depends on.

Run on the research machine only; the shipped repo already contains every
output, so external users never need this. Each copy is followed by an
equality check against the research-tree original, and the route->type CSV is
verified against the raw CARLA checkpoint JSONs it was derived from (219
routes in the JSONs + the one manual entry 11755 -> EnterActorFlow).
"""
import csv, glob, hashlib, json, re, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FEAT_SRC = Path('/data1/jeongtae/b2d_jepa/features')
IRT = Path('/home/jeongtae/SCIRT/b2d_irt')
CKPT = Path('/data1/jeongtae/b2d_eval_sensors/checkpoints')

FEATURES = ['eval_cmdkin_stats', 'eval_scenparamz', 'eval_gtrisk',
            'eval_routegeom', 'eval_smart_ent', 'eval_agentjepa']


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def copy_checked(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and md5(src) == md5(dst):
        print(f'  = {dst.relative_to(REPO)} (already identical)')
        return
    shutil.copy2(src, dst)
    assert md5(src) == md5(dst)
    print(f'  + {dst.relative_to(REPO)}')


def verify_route_types():
    sn = {}
    for c in glob.glob(str(CKPT / '*.json')):
        for rr in json.load(open(c))['_checkpoint']['records']:
            m = re.search(r'RouteScenario_(\d+)', str(rr.get('route_id', '')))
            if m:
                sn[m.group(1)] = re.sub(r'_\d+$', '', rr.get('scenario_name', 'unknown'))
    sn.setdefault('11755', 'EnterActorFlow')
    csvmap = dict(csv.reader(open(REPO / 'data/matrices/b2d_route_types.csv')))
    assert sn == csvmap, 'route_types.csv no longer matches the checkpoint JSONs'
    print(f'  route_types.csv == checkpoint-derived mapping ({len(sn)} routes)')


def main():
    print('features:')
    for f in FEATURES:
        copy_checked(FEAT_SRC / f'{f}.npz', REPO / 'data/features' / f'{f}.npz')
    print('descriptor tables:')
    copy_checked(IRT / 'b2d_traffic_features_220.csv', REPO / 'data/b2d/traffic_features_220.csv')
    copy_checked(IRT / 'baseline_kin_den.npz', REPO / 'data/b2d/baseline_kin_den.npz')
    print('encoder artifacts (unified split, per-run — no ensembling):')
    for s in (0, 1, 2):
        copy_checked(IRT / f'relgraph_r2_s{s}.npz',
                     REPO / 'data/encoder' / f'relgraph_r2_s{s}.npz')
    print('navhard panel:')
    copy_checked(IRT / 'navhard/navhard_binary_panel.npz',
                 REPO / 'data/navhard/navhard_binary_panel.npz')
    print('checks:')
    verify_route_types()
    src = Path('/home/jeongtae/SCIRT/SC-IRT/result/b2d/b2d_e2e16_response_matrix.csv')
    assert md5(src) == md5(REPO / 'data/matrices/b2d_e2e16_response_matrix.csv')
    print('  response matrix identical to research copy')
    print('done')


if __name__ == '__main__':
    sys.exit(main())
