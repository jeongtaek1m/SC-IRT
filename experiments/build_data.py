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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
FEAT_SRC = Path('/data1/jeongtae/b2d_jepa/features')
IRT = Path('/home/jeongtae/SCIRT/b2d_irt')
RELG_RAW = Path('/data2/jeongtae/relgraph_e16sel')   # the 16-planner RelGraph harness the shipped npz come from:
#   r2_b2d_s{k}.npz (pred (R_DRAWS, 220), routes, sigma (R_DRAWS,)) and the four controls
#   r2noroute_b2d_s{k}, sroute{k}_b2d_s{k}, sa2l{k}_b2d_s{k}, nospeed/r2nospeed_r2_b2d_s{k}
CKPT = Path('/data1/jeongtae/b2d_eval_sensors/checkpoints')
# nuPlan val14 zero-shot inputs (run_nuplan_zeroshot.py): the r0 target npz (tokens, logs, Y,
# fail, b_ref), the stage-2 prediction npz of the encoder arms and the label-shuffle nulls,
# and the 11 x 584 closed-loop score matrix
NUPLAN_TGT = Path('/data2/jeongtae/relgraph/transfer/nuplan/r0_nuplan_oof_s0.npz')
NUPLAN_S2 = Path('/data2/jeongtae/relgraph_e16sel/nuplan_stage2')
NUPLAN_CLS = Path('/home/jeongtae/SCIRT/SC-IRT/result/nuplan_val14_k11_response_matrix.csv')
NUPLAN_ARMS = {'C0e': 3, 'A2e': 3, 'C4r2e': 10, 'C4r2n': 10}   # arm -> number of seeds

FEATURES = ['eval_cmdkin_stats', 'eval_gtrisk',
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


def export_relgraph(src, dst):
    """Raw RelGraph run output -> the shipped format: draw{r}_rt / draw{r}_bt
    (evaluation-type routes of that draw, out-of-fold) + draw{r}_sigma (the
    shared residual SD learned on that draw's calibration block). Verifies
    against the shipped file when it exists."""
    import numpy as np
    from atdrive.b2d import Panel
    from atdrive.splits import unified_split, R_DRAWS
    d = np.load(src, allow_pickle=True)
    routes = [str(r) for r in d['routes']]
    panel = Panel()
    out = {}
    for r in range(R_DRAWS):
        _, ht = unified_split(r, panel.utypes, panel.J)
        te = [i for i, rid in enumerate(routes) if panel.sn.get(rid) in ht]
        out[f'draw{r}_rt'] = np.array([routes[i] for i in te])
        out[f'draw{r}_bt'] = d['pred'][r, te].astype(np.float64)
        out[f'draw{r}_sigma'] = np.float64(d['sigma'][r])
    if dst.exists():
        old = np.load(dst, allow_pickle=True)
        assert set(old.files) == set(out) and all(np.array_equal(old[k], out[k]) for k in out), dst
        print(f'  {dst.name} identical to the raw export')
    else:
        np.savez(dst, **out)
        print(f'  {dst.name} written')


def export_nuplan(dst):
    """The 584-scenario nuPlan val14 panel of run_nuplan_zeroshot.py in one file:
    tok / logs (scene order = the logged-ego readout order, widx == 0), Y (11 x 584
    binary failures, NaN = no record), fail (per-scene failure rate), b_ref (the
    response-calibrated difficulty), cls (11 x 584 closed-loop scores) with the
    planner names, and pred_<arm>_s<k> = the stage-2 `pred_logged` of every encoder
    arm and label-shuffle null at widx == 0. Verifies against the shipped file."""
    import csv
    import numpy as np
    z = np.load(NUPLAN_TGT, allow_pickle=True)
    w0 = np.asarray(z['widx']) == 0
    tok = np.array([str(t) for t in z['tokens'][w0]])
    pos = {t: i for i, t in enumerate(tok)}
    out = {'tok': tok, 'logs': np.array([str(l) for l in z['logs'][w0]]),
           'Y': np.asarray(z['Y'][:, w0], np.float32), 'fail': np.asarray(z['fail'][w0], np.float32),
           'b_ref': np.asarray(z['b_ref'][w0], np.float32)}
    rows = list(csv.reader(open(NUPLAN_CLS)))
    hdr, body = rows[0][1:], rows[1:]
    if len(body) != 11:                                          # stored as scenes x planners
        body = [[hdr[i]] + [r[1 + i] for r in body] for i in range(len(hdr))]
        hdr = [r[0] for r in rows[1:]]
    col = {t: i for i, t in enumerate(hdr)}
    out['planners'] = np.array([r[0] for r in body])
    out['cls'] = np.array([[float(r[1 + col[t]]) if r[1 + col[t]] != '' else np.nan for t in tok]
                           for r in body], np.float64)
    for arm, n in NUPLAN_ARMS.items():
        for k in range(n):
            a = np.load(NUPLAN_S2 / f'{arm}_b2d2nuplan_s{k}.npz', allow_pickle=True)
            wi = np.asarray(a['widx']) == 0
            v = np.full(len(tok), np.nan, np.float32)
            v[[pos[str(t)] for t in a['tgt_groups'][wi]]] = np.asarray(a['pred_logged'], np.float32)[wi]
            out[f'pred_{arm}_s{k}'] = v
    if dst.exists():
        old = np.load(dst, allow_pickle=True)
        assert set(old.files) == set(out) and all(np.array_equal(old[k], out[k], equal_nan=True)
                                                  if out[k].dtype.kind == 'f' else np.array_equal(old[k], out[k])
                                                  for k in out), dst
        print(f'  {dst.name} identical to the raw export')
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, **out)
        print(f'  {dst.name} written')


def main():
    print('features:')
    for f in FEATURES:
        copy_checked(FEAT_SRC / f'{f}.npz', REPO / 'data/features' / f'{f}.npz')
    print('descriptor tables:')
    copy_checked(IRT / 'b2d_traffic_features_220.csv', REPO / 'data/b2d/traffic_features_220.csv')
    copy_checked(IRT / 'baseline_kin_den.npz', REPO / 'data/b2d/baseline_kin_den.npz')
    print('encoder artifacts (unified split, per-run — no ensembling):')
    for s in (0, 1, 2):
        export_relgraph(RELG_RAW / f'r2_b2d_s{s}.npz', REPO / 'data/encoder' / f'relgraph_r2_s{s}.npz')
    for tag, name in (('r2noroute', 'noroute'), ('sroute{s}', 'sroute'), ('sa2l{s}', 'sa2l'),   # structural controls (shuffle seed = model seed)
                      ('nospeed/r2nospeed_r2', 'nospeed')):                                     # channel control (ego speed removed)
        for s in (0, 1, 2):
            src = RELG_RAW / (tag.format(s=s) + f'_b2d_s{s}.npz')
            if src.exists():
                export_relgraph(src, REPO / 'data/encoder' / f'relgraph_r2_{name}_s{s}.npz')
    print('nuPlan val14 zero-shot panel:')
    export_nuplan(REPO / 'data/nuplan/val14_zeroshot.npz')
    print('checks:')
    verify_route_types()
    print('done')


if __name__ == '__main__':
    sys.exit(main())
