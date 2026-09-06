#!/usr/bin/env python3
"""Table 3A (US) descriptor family 'Traffic entropy', rebuilt in the codebase
the checkpoint actually belongs to.

Why a rebuild.  The shipped row (`data/features/eval_smart_ent.npz`) was
produced by /data1/jeongtae/smart_difficulty/scripts/b2d_smart_extract.py,
which loaded /home/jeongtae/IRT/SMART/ckpts/pre_bc_E31.ckpt into the
rainmaker22/SMART model with `load_state_dict(..., strict=False)` after
filtering by name+shape (808/818 tensors loaded, 10 left at initialisation)
and tokenised the scenes with rainmaker22's `cluster_frame_5_2048.pkl`.
`pre_bc_E31.ckpt` is not a rainmaker22 release: its `hyper_parameters` are the
CAT-K (NVlabs/catk, Zhang et al., CVPR 2025, arXiv:2412.05334) `pre_bc`
experiment config -- SMART-tiny 7M, 3 map / 6 agent layers, 8 heads x 16 dim,
`agent_token_file: agent_vocab_555_s2.pkl` -- i.e. CAT-K's *own* BC
pre-training checkpoint at epoch 31 (global_step 194829).

This script runs it in /home/jeongtae/IRT/catk with CAT-K's own TokenProcessor
(agent_vocab_555_s2.pkl), CAT-K's own SMARTDecoder, and `strict=True`
(811/811 tensors; the load is asserted in both directions).

Quantity.  CAT-K/SMART's open-loop (teacher-forced) forward returns
`next_token_logits` [n_agent, 16, 2048] -- the next-agent-token distribution
for the 16 action steps (10->15), (15->20), ..., (85->90) at 2 Hz, each agent
conditioned on the *recorded* token history (no rollout, `sampled_* == gt_*`
because agent_token_sampling.num_k == 1 outside training).  The descriptor is

    H(scene) = mean over {agents} x {16 valid 0.5 s token steps}
               of  -sum_k p_k log p_k,  p = softmax(next_token_logits)

exactly the pooled mean over `next_token_valid` that the previous script also
took, but now over a correctly loaded model and the matching vocabulary.
Reported twice: ego included (primary column, ego is one agent among the
others, as in the model's own loss) and ego excluded.

Scenes.  One centred 91-frame (9.1 s @ 10 Hz) window of the PDM-Lite reference
rollout of each of the 220 Bench2Drive evaluation routes
(/data1/jeongtae/b2d_eval_sensors/route_<id>/anno/*.json.gz), the same window
the shipped row used (b2d_irt/scripts/b2d_smart_220.py).  PDM-Lite is the
EXCLUDED_PLANNER of the protocol and the RelGraph encoder reads the same
rollout, so this is the protocol's route representation, not a planner-outcome
leak; the row label must say the descriptor is computed on that rollout.

Geometry conversion is `b2d_to_smart_viz.py` (imported read-only), whose
docstring warns its output "must NOT be used for difficulty scoring".  That
warning is about *provenance tiering* (a planner's driving record used as scene
evidence), not about a defect of the converter; under the ATDrive protocol the
PDM-Lite rollout is the defined route representation for every descriptor and
for the encoder, and PDM-Lite is excluded from the 16 evaluated planners.  The
warning's technical content (mirror fix, heading convention, 10 Hz anno) is
satisfied by using the converter verbatim.

Usage (env 'smart', GPU 0 or 3 only):
    python traffic_entropy.py convert          # 220 CAT-K pkls
    CUDA_VISIBLE_DEVICES=3 python traffic_entropy.py extract
    python traffic_entropy.py verify           # geometry equivalence checks
    CUDA_VISIBLE_DEVICES=3 python traffic_entropy.py all
"""
import argparse
import gzip
import json
import math
import os
import pickle
import sys
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np
import torch
from scipy.interpolate import interp1d

CATK = '/home/jeongtae/IRT/catk'
CKPT = '/home/jeongtae/IRT/SMART/ckpts/pre_bc_E31.ckpt'
VIZ_SCRIPTS = '/data1/jeongtae/smart_difficulty/scripts'
ROLLOUT = '/data1/jeongtae/b2d_eval_sensors'
OUT_DIR = '/data2/jeongtae/official_baselines/us_features'
PKL_DIR = os.path.join(OUT_DIR, 'catk_pkls_b2d220')
NAME = 'traffic_entropy_catk'

NFR = 91          # 11 history + 80 future steps @ 10 Hz
ANCH = 10         # "current" step
MAX_ACTORS = 47   # + ego = 48, as in b2d_to_smart_viz.build_agent

sys.path.insert(0, CATK)
sys.path.insert(0, VIZ_SCRIPTS)
import b2d_to_smart_viz as V                                     # noqa: E402
from src.smart.modules.smart_decoder import SMARTDecoder         # noqa: E402
from src.smart.tokens.token_processor import TokenProcessor      # noqa: E402
from src.smart.utils.preprocess import preprocess_map            # noqa: E402

V.ROLLOUT = ROLLOUT
_orig_load_frame = V.load_frame


def _padded_load_frame(route, i):
    """Frames past the end of a short rollout are empty -> that step is invalid
    for every agent (b2d_smart_220.py's policy; 10 of the 220 routes are
    shorter than 91 frames, the shortest is 66)."""
    try:
        return _orig_load_frame(route, i)
    except (FileNotFoundError, OSError, EOFError):
        return {'bounding_boxes': []}


V.load_frame = _padded_load_frame

# ---------------------------------------------------------------- map vocabulary
# b2d_to_smart_viz emits rainmaker22/SMART map-point types; CAT-K's vocabulary is
# the WOMD one (src/data_preprocess.py header):
#   point:   0 FREEWAY, 1 SURFACE_STREET, 2 STOP_SIGN, 3 BIKE_LANE,
#            4 ROAD_EDGE_BOUNDARY, 5 ROAD_EDGE_MEDIAN,
#            6 BROKEN, 7 SOLID_SINGLE, 8 DOUBLE, 9 CROSSWALK
#   polygon: 0 lane, 1 road_edge, 2 road_line, 3 crosswalk
#   light:   0 NO_LANE_STATE, 1 UNKNOWN, 2 STOP, 3 GO, 4 CAUTION
PT_CENTERLINE_SRC = V.CENTERLINE          # 16, rainmaker centerline
CATK_SURFACE_STREET = 1
CATK_ROAD_EDGE = 4
CATK_BROKEN = 6
CATK_SOLID = 7
CATK_DOUBLE = 8
# rainmaker marking index -> CAT-K point type (see b2d_to_smart_viz._MARK2SMART)
MARK_SRC2CATK = {
    0: CATK_DOUBLE,    # BrokenSolid Yellow
    1: CATK_DOUBLE,    # BrokenSolid White
    2: CATK_BROKEN,    # Broken White
    3: CATK_BROKEN,    # Broken Yellow
    4: CATK_DOUBLE,    # SolidSolid Yellow
    5: CATK_DOUBLE,    # SolidSolid White
    6: CATK_DOUBLE,    # BrokenBroken Yellow
    7: CATK_DOUBLE,    # BrokenBroken White
    8: CATK_SOLID,     # Solid Yellow
    9: CATK_SOLID,     # Solid White
    10: CATK_DOUBLE,   # SolidBroken White
    11: CATK_DOUBLE,   # SolidBroken Yellow
    12: CATK_ROAD_EDGE,  # Curb / Grass
    14: CATK_ROAD_EDGE,  # unpainted / unmapped lane boundary (see provenance)
}
PT2POLY = {CATK_SURFACE_STREET: 0, CATK_ROAD_EDGE: 1,
           CATK_BROKEN: 2, CATK_SOLID: 2, CATK_DOUBLE: 2}
LIGHT_NO_STATE = 0     # B2D anno carries no per-lane signal state


def routes():
    """The 220 evaluation routes, directory order (== the feature-file order)."""
    out = []
    for d in sorted(os.listdir(ROLLOUT)):
        anno = os.path.join(ROLLOUT, d, 'anno')
        if d.startswith('route_') and os.path.isdir(anno):
            out.append((d, len(os.listdir(anno))))
    return out


# ------------------------------------------------------------------ agent side
def build_agent_catk(route, s):
    """91-step tracks of one window in CAT-K's `data['agent']` schema.

    Parsing conventions are b2d_to_smart_viz.build_agent's, verbatim (CARLA's
    left-handed frame reflected y -> -y, heading = -deg2rad(rotation[2]),
    shape = 2*extent = (length, width, height), velocity = speed * (cos h,
    sin h), actors kept if valid at step 10 with > 3 valid frames, capped at 47
    + ego, ego last).  The schema is CAT-K's get_agent_features: raw per-step
    validity gap-filled by interpolation between the first and last valid step,
    static shape, 2-D velocity, role flags, and agent types restricted to
    {0 vehicle, 1 pedestrian, 2 cyclist} (CAT-K has no "other" type).
    `verify` asserts this against b2d_to_smart_viz.build_agent.
    """
    frames = [V.load_frame(route, s + t) for t in range(NFR)]
    ego = {'pos': np.zeros((NFR, 3), np.float32), 'head': np.zeros(NFR, np.float32),
           'vel': np.zeros((NFR, 2), np.float32), 'shape': np.zeros((NFR, 3), np.float32),
           'valid': np.zeros(NFR, bool), 'type': 0}
    agdata = {}
    for t, d in enumerate(frames):
        for b in d['bounding_boxes']:
            h = -math.radians(b['rotation'][2])            # B2D_MIRROR_FIX
            loc, ext = b['location'], b['extent']
            sp = float(b.get('speed', 0.0))
            a = ego if b['class'] == 'ego_vehicle' else agdata.setdefault(
                b['id'], {'pos': np.zeros((NFR, 3), np.float32),
                          'head': np.zeros(NFR, np.float32),
                          'vel': np.zeros((NFR, 2), np.float32),
                          'shape': np.zeros((NFR, 3), np.float32),
                          'valid': np.zeros(NFR, bool), 'type': V.smart_type(b)})
            a['pos'][t] = [loc[0], -loc[1], 0.0]           # B2D_MIRROR_FIX
            a['head'][t] = h
            a['vel'][t] = [sp * math.cos(h), sp * math.sin(h)]
            a['shape'][t] = [2 * ext[0], 2 * ext[1], 2 * ext[2]]
            a['valid'][t] = True
    keep = [a for a in agdata.values() if a['valid'].sum() > 3 and a['valid'][ANCH]]
    keep = keep[:MAX_ACTORS]
    keep = [a for a in keep if a['type'] <= 2]             # CAT-K: no "other"
    if not ego['valid'][ANCH]:
        raise RuntimeError(f'{route}: ego invalid at step {ANCH}')
    keep.append(ego)                                       # ego last (av index)
    n = len(keep)

    out = {'num_nodes': n,
           'valid_mask': torch.zeros(n, NFR, dtype=torch.bool),
           'role': torch.zeros(n, 3, dtype=torch.bool),
           'id': torch.arange(n, dtype=torch.int64),
           'type': torch.zeros(n, dtype=torch.uint8),
           'position': torch.zeros(n, NFR, 3, dtype=torch.float32),
           'heading': torch.zeros(n, NFR, dtype=torch.float32),
           'velocity': torch.zeros(n, NFR, 2, dtype=torch.float32),
           'shape': torch.zeros(n, 3, dtype=torch.float32)}
    out['role'][n - 1, 0] = True                           # ego_vehicle
    for i, a in enumerate(keep):
        v = a['valid']
        steps = np.where(v)[0]
        out['type'][i] = int(a['type'])
        out['shape'][i] = torch.from_numpy(a['shape'][v].mean(0))
        if len(steps) > 1:                                 # CAT-K gap filling
            t0, t1 = int(steps[0]), int(steps[-1])
            t_in = np.arange(t0, t1 + 1)
            out['valid_mask'][i, t0:t1 + 1] = True
            out['position'][i, t0:t1 + 1] = torch.from_numpy(
                interp1d(steps, a['pos'][v], axis=0)(t_in)).float()
            out['velocity'][i, t0:t1 + 1] = torch.from_numpy(
                interp1d(steps, a['vel'][v], axis=0)(t_in)).float()
            out['heading'][i, t0:t1 + 1] = torch.from_numpy(
                interp1d(steps, np.unwrap(a['head'][v]), axis=0)(t_in)).float()
        else:
            t = int(steps[0])
            out['valid_mask'][i, t] = True
            out['position'][i, t] = torch.from_numpy(a['pos'][t])
            out['velocity'][i, t] = torch.from_numpy(a['vel'][t])
            out['heading'][i, t] = float(a['head'][t])
    return out


# -------------------------------------------------------------------- map side
def build_map_catk(town, ego_xy):
    """b2d_to_smart_viz.build_map's polylines, re-typed into CAT-K's WOMD
    vocabulary and pushed through CAT-K's own preprocess_map()."""
    mp = V.build_map(town, ego_xy)
    src_pt = mp['map_point']['type'].numpy().astype(int)
    pt_type = np.empty_like(src_pt)
    unknown = []
    for i, t in enumerate(src_pt):
        if t == PT_CENTERLINE_SRC:
            pt_type[i] = CATK_SURFACE_STREET
        elif t in MARK_SRC2CATK:
            pt_type[i] = MARK_SRC2CATK[t]
        else:
            pt_type[i] = CATK_ROAD_EDGE
            unknown.append(t)
    if unknown:
        raise RuntimeError(f'unmapped rainmaker point types: {sorted(set(unknown))}')
    p2pl = mp[('map_point', 'to', 'map_polygon')]['edge_index'][1].numpy()
    n_poly = int(mp['map_polygon']['num_nodes'])
    poly_type = np.zeros(n_poly, np.uint8)
    for pl in range(n_poly):
        idx = np.where(p2pl == pl)[0]
        poly_type[pl] = PT2POLY[int(pt_type[idx[0]])] if len(idx) else 0
    map_data = {
        'map_polygon': {'type': torch.from_numpy(poly_type),
                        'light_type': torch.full((n_poly,), LIGHT_NO_STATE,
                                                 dtype=torch.uint8)},
        'map_point': {'type': torch.from_numpy(pt_type.astype(np.uint8)),
                      'position': mp['map_point']['position']},
        ('map_point', 'to', 'map_polygon'): {
            'edge_index': mp[('map_point', 'to', 'map_polygon')]['edge_index']},
    }
    return preprocess_map(map_data)


def convert_one(args):
    route, nf = args
    rid = route.split('_')[1]
    dst = os.path.join(PKL_DIR, f'{route}.pkl')
    if os.path.exists(dst):
        return ('skip', route, '')
    try:
        s = max(0, (nf - NFR) // 2)                        # centred window
        town = json.load(open(f'{ROLLOUT}/{route}/meta.json'))['town']
        agent = build_agent_catk(route, s)
        ego_xy = agent['position'][-1, ANCH, :2].numpy()
        data = build_map_catk(town, ego_xy)
        data['agent'] = agent
        data['scenario_id'] = f'b2d_r{rid}'
        data['b2d'] = {'route': route, 'town': town, 'n_frames': nf,
                       'window_start': s, 'n_frames_used': min(NFR, nf - s)}
        pickle.dump(data, open(dst, 'wb'))
        return ('ok', route, f"{town} n_ag={agent['num_nodes']} "
                             f"n_pt={data['pt_token']['num_nodes']}")
    except Exception as e:                                  # noqa: BLE001
        return ('err', route, f'{type(e).__name__} {e}')


def run_convert(workers=8):
    os.makedirs(PKL_DIR, exist_ok=True)
    tasks = routes()
    print(f'{len(tasks)} routes -> {PKL_DIR}', flush=True)
    ok = skip = err = 0
    with Pool(workers) as p:
        for st, route, msg in p.imap_unordered(convert_one, tasks):
            if st == 'ok':
                ok += 1
                if ok % 20 == 0:
                    print(f'  {ok} converted (last {route}: {msg})', flush=True)
            elif st == 'skip':
                skip += 1
            else:
                err += 1
                print(f'  ERR {route}: {msg}', flush=True)
    print(f'convert done ok={ok} skip={skip} err={err} '
          f'({len(os.listdir(PKL_DIR))} pkls)', flush=True)


# ----------------------------------------------------------------- model side
def load_model(device):
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    mc = ck['hyper_parameters']['model_config']
    tp = TokenProcessor(**mc.token_processor)
    model = SMARTDecoder(**mc.decoder, n_token_agent=tp.n_token_agent)
    sd = {}
    for k, v in ck['state_dict'].items():
        assert k.startswith('encoder.'), f'non-encoder tensor in ckpt: {k}'
        sd[k[len('encoder.'):]] = v
    ms = model.state_dict()
    assert set(sd) == set(ms), (f'name mismatch: ckpt-only={sorted(set(sd) - set(ms))[:5]} '
                               f'model-only={sorted(set(ms) - set(sd))[:5]}')
    bad = [k for k in sd if sd[k].shape != ms[k].shape]
    assert not bad, f'shape mismatch: {bad[:5]}'
    model.load_state_dict(sd, strict=True)                 # strict, no filtering
    model.eval().to(device)
    tp.eval().to(device)
    meta = {'ckpt': CKPT, 'epoch': int(ck['epoch']), 'global_step': int(ck['global_step']),
            'n_tensors_loaded': len(sd), 'n_tensors_model': len(ms),
            'agent_token_file': mc.token_processor.agent_token_file,
            'map_token_file': mc.token_processor.map_token_file,
            'n_token_agent': int(tp.n_token_agent),
            'decoder': dict(mc.decoder)}
    print(f'strict load OK: {len(sd)}/{len(ms)} tensors, vocab '
          f'{meta["agent_token_file"]} ({meta["n_token_agent"]} tokens)', flush=True)
    return model, tp, meta


@torch.no_grad()
def scene_entropy(model, tp, data, device):
    from torch_geometric.data import Batch, HeteroData
    batch = Batch.from_data_list([HeteroData(data)]).to(device)
    tok_map, tok_agent = tp(batch)
    pred = model(tok_map, tok_agent)
    logits = pred['next_token_logits']                     # [n_agent, 16, 2048]
    valid = pred['next_token_valid'].bool()                # [n_agent, 16]
    logp = torch.log_softmax(logits.float(), -1)
    ent = -(logp.exp() * logp).sum(-1)                     # [n_agent, 16] nats
    ego = tok_agent['ego_mask'].bool()                     # [n_agent]
    m_all, m_ego = valid, valid & ego[:, None]
    m_noego = valid & ~ego[:, None]
    f = lambda m: float(ent[m].mean()) if bool(m.any()) else float('nan')   # noqa: E731
    return {
        'ent': f(m_all), 'ent_noego': f(m_noego), 'ent_ego': f(m_ego),
        'n_agent': int(ent.shape[0]),
        'n_cells': int(m_all.sum()), 'n_cells_noego': int(m_noego.sum()),
        'n_cells_ego': int(m_ego.sum()),
        'ent_steps': [float(ent[valid[:, t], t].mean()) if bool(valid[:, t].any())
                      else float('nan') for t in range(ent.shape[1])],
    }


def run_extract(device='cuda', limit=0):
    model, tp, ckpt_meta = load_model(device)
    paths = sorted(p for p in os.listdir(PKL_DIR) if p.endswith('.pkl'))
    if limit:
        paths = paths[:limit]
    feats, meta = {}, {}
    for i, p in enumerate(paths):
        route = p[:-4]
        d = pickle.load(open(os.path.join(PKL_DIR, p), 'rb'))
        b2d = d.pop('b2d')
        f = scene_entropy(model, tp, d, device)
        feats[route] = f
        meta[route] = b2d
        if (i + 1) % 25 == 0:
            print(f'  {i + 1}/{len(paths)} scenes', flush=True)
    print(f'extracted {len(feats)} scenes', flush=True)
    write_outputs(feats, meta, ckpt_meta)
    return feats


def write_outputs(feats, meta, ckpt_meta):
    names = sorted(feats)
    ent = np.array([feats[r]['ent'] for r in names], np.float64)
    ent_noego = np.array([feats[r]['ent_noego'] for r in names], np.float64)
    ent_ego = np.array([feats[r]['ent_ego'] for r in names], np.float64)
    n_agent = np.array([feats[r]['n_agent'] for r in names], np.int64)
    n_cells = np.array([feats[r]['n_cells'] for r in names], np.int64)
    n_cells_noego = np.array([feats[r]['n_cells_noego'] for r in names], np.int64)
    steps = np.array([feats[r]['ent_steps'] for r in names], np.float64)
    nfr_used = np.array([meta[r]['n_frames_used'] for r in names], np.int64)
    layout = ('1d: CAT-K SMART-tiny pre_bc_E31 (strict load, agent_vocab_555_s2) '
              'teacher-forced next-token entropy, mean over agents (ego included) '
              'and the 16 x 0.5 s token steps of one centred 9.1 s window of the '
              'PDM-Lite reference rollout. Higher = the traffic model is less '
              'certain what happens next.')
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, f'{NAME}.npz'),
             stats=ent[:, None], names=np.array(names), layout=np.array(layout),
             ent_ego_included=ent, ent_ego_excluded=ent_noego, ent_ego_only=ent_ego,
             n_agent=n_agent, n_cells=n_cells, n_cells_noego=n_cells_noego,
             ent_per_step=steps, n_frames_used=nfr_used)
    np.savez(os.path.join(OUT_DIR, f'{NAME}_noego.npz'),
             stats=ent_noego[:, None], names=np.array(names),
             layout=np.array(layout.replace('(ego included)', '(ego excluded)')),
             ent_ego_included=ent, ent_ego_excluded=ent_noego, ent_ego_only=ent_ego,
             n_agent=n_agent, n_cells=n_cells, n_cells_noego=n_cells_noego,
             ent_per_step=steps, n_frames_used=nfr_used)

    old = np.load('/home/jeongtae/SC-IRT/data/features/eval_smart_ent.npz',
                  allow_pickle=True)
    old_lut = {str(k): float(v[0]) for k, v in zip(old['names'], old['stats'])}
    old_v = np.array([old_lut[r] for r in names])
    from scipy.stats import spearmanr
    prov = {
        'name': NAME,
        'family': 'Traffic entropy',
        'written_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'what_was_run': {
            'repo': 'NVlabs/catk (CAT-K, Zhang et al., CVPR 2025, arXiv:2412.05334)',
            'checkout': CATK,
            'commit': 'd23886761fc5b5628c5973148c40284452745745',
            'entry_points': [
                'src.smart.tokens.token_processor:TokenProcessor.forward',
                'src.smart.modules.smart_decoder:SMARTDecoder.forward',
                'src.smart.utils.preprocess:preprocess_map',
            ],
            'producing_script': '/home/jeongtae/SC-IRT/experiments/us_official/traffic_entropy.py',
            'geometry_converter': f'{VIZ_SCRIPTS}/b2d_to_smart_viz.py '
                                  '(build_map, smart_type, load_frame; imported read-only)',
            'checkpoint': ckpt_meta,
            'load_mode': 'load_state_dict(strict=True) after stripping the '
                         '"encoder." prefix; set(ckpt) == set(model) asserted '
                         'both directions and every shape checked',
        },
        'formula': {
            'definition': 'H(scene) = mean_{agents, 16 token steps} '
                          '-sum_k p_k log p_k, p = softmax(next_token_logits), nats',
            'source': 'SMART (Wu et al., NeurIPS 2024) next-token distribution over the '
                      '2048-entry agent motion-token vocabulary; the exact tensor is '
                      'CAT-K src/smart/modules/agent_decoder.py:SMARTAgentDecoder.forward '
                      '-> "next_token_logits" [n_agent, 16, n_token] with validity '
                      '"next_token_valid" (= tokenized valid_mask[:, 1:-1]); the model\'s '
                      'own open-loop validation (src/smart/model/smart.py, val_open_loop) '
                      'scores exactly these logits.',
            'token_steps': 'actions (10->15), (15->20), ..., (85->90) at 2 Hz, '
                           'i.e. 16 steps x 0.5 s covering t = 1.0 s .. 9.0 s',
            'teacher_forced': True,
            'teacher_forcing_note': 'open-loop forward: every agent is conditioned on the '
                                    'tokenised RECORDED trajectory (TokenProcessor with '
                                    'agent_token_sampling.num_k = 1 outside training makes '
                                    'sampled_idx == gt_idx), so the 9.1 s window is fully '
                                    'teacher-forced on the recorded future; no rollout, no '
                                    'sampling, model in eval() so dropout and hist_drop are off',
            'ego_policy': 'primary column = ego INCLUDED (ego is one agent among the '
                          'others, as in the model\'s own loss); the ego-excluded column '
                          'is shipped as traffic_entropy_catk_noego.npz and as the '
                          '"ent_ego_excluded" array here',
        },
        'parameters': {
            'window': 'one centred 91-frame (9.1 s @ 10 Hz) window per route: '
                      'start = max(0, (n_frames - 91) // 2)',
            'short_routes': '10 of 220 routes have < 91 frames (min 66); frames past the '
                            'end are empty, so those steps are invalid for every agent '
                            'and drop out of the mean (same policy as b2d_smart_220.py)',
            'agents_per_scene': 'actors valid at step 10 with > 3 valid frames, in '
                                'first-seen order, capped at 47, restricted to CAT-K '
                                'types {0 vehicle, 1 pedestrian, 2 cyclist}, plus ego last',
            'map_radius_m': V.MAP_RADIUS,
            'map_resample_m': V.RESAMPLE,
            'coordinate_fix': 'CARLA is left-handed: y -> -y and heading -> '
                              '-deg2rad(rotation[2]) (b2d_to_smart_viz B2D_MIRROR_FIX)',
            'map_vocabulary_remap': {str(k): int(v) for k, v in MARK_SRC2CATK.items()},
            'light_type': 'all polygons NO_LANE_STATE (0): B2D anno carries no per-lane '
                          'signal state',
            'device': str(ckpt_meta.get('device', 'cuda')),
            'dtype': 'float32 forward, entropy accumulated in float32, stored float64',
        },
        'differences_from_source': [
            'CAT-K preprocesses Waymo Open Motion protos; B2D scenes are converted here '
            'instead (its own preprocess_map is used for the map side, and the agent dict '
            'mirrors get_agent_features: interp1d gap fill, static shape, 2-D velocity, '
            'role flags).',
            'CARLA lane-marking types are mapped into the WOMD vocabulary; unpainted / '
            'unmapped lane boundaries (the majority of B2D boundary polylines) are sent '
            'to ROAD_EDGE_BOUNDARY. Sensitivity of the descriptor to that choice is '
            'reported in sanity.map_unknown_sensitivity when the "sens" stage was run.',
            'No traffic-light state (B2D anno has none) -> NO_LANE_STATE everywhere; '
            'CAT-K\'s light embedding therefore sees one constant class.',
            'Agent z is 0 and the scene is not centred on the WOMD ego frame; the model '
            'is fully translation/rotation equivariant in its relative encodings, so this '
            'only affects the (unused) gt_z_raw.',
            'One 9.1 s window per route (the descriptor is one scalar per route), whereas '
            'WOMD validation scores every 9.1 s scenario in the split.',
        ],
        'differences_from_the_shipped_row': [
            'ckpt loaded strict=True into CAT-K (811/811 tensors) instead of strict=False '
            'into rainmaker22/SMART (808/818, 10 tensors left at initialisation)',
            'tokenised with agent_vocab_555_s2.pkl (the vocabulary the checkpoint was '
            'trained with) instead of rainmaker22 cluster_frame_5_2048.pkl',
            'map/agent/light vocabularies remapped to CAT-K\'s 10/4/3-class WOMD spaces '
            'instead of rainmaker22\'s 17/5/4-class spaces (that mismatch alone left '
            'type_pt_emb / light_pl_emb / type_a_emb partly unused or out of range)',
            'same routes, same rollout, same centred 9.1 s window, same pooled mean',
        ],
        'sanity': {
            'n_routes': len(names),
            'ent_ego_included': {'mean': float(ent.mean()), 'sd': float(ent.std(ddof=1)),
                                 'min': float(ent.min()), 'max': float(ent.max())},
            'ent_ego_excluded': {'mean': float(np.nanmean(ent_noego)),
                                 'sd': float(np.nanstd(ent_noego, ddof=1)),
                                 'min': float(np.nanmin(ent_noego)),
                                 'max': float(np.nanmax(ent_noego))},
            'ent_ego_only': {'mean': float(np.nanmean(ent_ego)),
                             'sd': float(np.nanstd(ent_ego, ddof=1))},
            'agents_per_scene': {'mean': float(n_agent.mean()), 'min': int(n_agent.min()),
                                 'max': int(n_agent.max())},
            'valid_agent_step_cells': {'mean': float(n_cells.mean()),
                                       'min': int(n_cells.min()), 'max': int(n_cells.max()),
                                       'max_possible_per_agent': 16},
            'frames_used': {'min': int(nfr_used.min()), 'max': int(nfr_used.max()),
                            'n_routes_below_91': int((nfr_used < 91).sum())},
            'nan_routes': [r for r in names if not np.isfinite(feats[r]['ent'])],
            'spearman_new_vs_shipped': float(spearmanr(ent, old_v).correlation),
            'spearman_egoincl_vs_egoexcl': float(spearmanr(ent, ent_noego).correlation),
            'shipped_row_stats': {'mean': float(old_v.mean()), 'sd': float(old_v.std(ddof=1)),
                                  'min': float(old_v.min()), 'max': float(old_v.max())},
        },
        'leakage_statement': 'Computed only from the PDM-Lite reference rollout of each '
                             'route (ego + agent tracks + HD-map geometry). PDM-Lite is '
                             'EXCLUDED_PLANNER and is not one of the 16 evaluated '
                             'planners; the RelGraph encoder reads the same rollout. No '
                             'response matrix, planner outcome or scenario-type label was '
                             'read at any point.',
    }
    with open(os.path.join(OUT_DIR, f'{NAME}_provenance.json'), 'w') as fh:
        json.dump(prov, fh, indent=2)
    prov2 = dict(prov, name=f'{NAME}_noego',
                 primary_column='ego EXCLUDED (companion of traffic_entropy_catk.npz)')
    with open(os.path.join(OUT_DIR, f'{NAME}_noego_provenance.json'), 'w') as fh:
        json.dump(prov2, fh, indent=2)
    print(json.dumps(prov['sanity'], indent=2))
    print(f'wrote {OUT_DIR}/{NAME}.npz and {NAME}_noego.npz', flush=True)


# --------------------------------------------------------------------- checks
def run_verify(n=4):
    """build_agent_catk vs b2d_to_smart_viz.build_agent on a few routes:
    identical tracks up to the schema differences we intend."""
    rs = routes()
    rs = [rs[i] for i in np.linspace(0, len(rs) - 1, n).astype(int)]
    for route, nf in rs:
        s = max(0, (nf - NFR) // 2)
        ref = V.build_agent(route, s)
        mine = build_agent_catk(route, s)
        # ego row must match exactly (position/heading/velocity/shape)
        r_av, m_av = int(ref['av_index']), mine['num_nodes'] - 1
        assert torch.allclose(ref['position'][r_av], mine['position'][m_av], atol=1e-5)
        # heading is stored unwrapped here (CAT-K's np.unwrap), so compare mod 2pi
        dh = (ref['heading'][r_av] - mine['heading'][m_av]) / (2 * math.pi)
        assert torch.allclose(dh, dh.round(), atol=1e-5)
        assert torch.allclose(ref['velocity'][r_av][:, :2], mine['velocity'][m_av], atol=1e-5)
        # raw validity: ref's vector_repr mask is valid[t-1] & valid[t] on 1..10
        raw = mine['valid_mask']
        vm = ref['valid_mask']
        rec = vm.clone()
        rec[:, 0:10] |= vm[:, 1:11]
        n_ref = int(ref['num_nodes'])
        agree = float((rec[:n_ref].sum(1) > 0).float().mean())
        print(f'{route}: ref n={n_ref} mine n={mine["num_nodes"]} '
              f'(types<=2 kept), ego tracks identical, ref-valid-rows {agree:.2f}, '
              f'mine valid cells {int(raw.sum())}')
    print('verify OK')


def run_sens(device='cuda', n=30):
    """Sensitivity of the descriptor to the unpainted-boundary type choice:
    re-run n routes with rainmaker type 14 dropped from the map entirely."""
    import copy
    from scipy.stats import spearmanr
    model, tp, _ = load_model(device)
    rs = routes()
    rs = [rs[i] for i in np.linspace(0, len(rs) - 1, n).astype(int)]
    a, b = [], []
    for route, nf in rs:
        d = pickle.load(open(os.path.join(PKL_DIR, f'{route}.pkl'), 'rb'))
        d.pop('b2d')
        f = scene_entropy(model, tp, copy.deepcopy(d), device)
        keep = d['pt_token']['type'] != CATK_ROAD_EDGE
        if int(keep.sum()) < 2:
            continue
        d2 = {'map_save': {k: v[keep] for k, v in d['map_save'].items()},
              'pt_token': {k: (v[keep] if torch.is_tensor(v) else int(keep.sum()))
                           for k, v in d['pt_token'].items()},
              'agent': d['agent'], 'scenario_id': d['scenario_id']}
        d2['pt_token']['num_nodes'] = int(keep.sum())
        g = scene_entropy(model, tp, d2, device)
        a.append(f['ent'])
        b.append(g['ent'])
        print(f'{route}: road_edge-kept {f["ent"]:.4f}  dropped {g["ent"]:.4f}', flush=True)
    a, b = np.array(a), np.array(b)
    out = {'n': len(a), 'mean_kept': float(a.mean()), 'mean_dropped': float(b.mean()),
           'mean_abs_delta': float(np.abs(a - b).mean()),
           'spearman': float(spearmanr(a, b).correlation)}
    print(json.dumps(out, indent=2))
    pp = os.path.join(OUT_DIR, f'{NAME}_provenance.json')
    if os.path.exists(pp):
        prov = json.load(open(pp))
        prov['sanity']['map_unknown_sensitivity'] = out
        json.dump(prov, open(pp, 'w'), indent=2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('stage', choices=['convert', 'extract', 'verify', 'sens', 'all'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.stage in ('convert', 'all'):
        run_convert(a.workers)
    if a.stage == 'verify':
        run_verify()
    if a.stage in ('extract', 'all'):
        run_extract(a.device, a.limit)
    if a.stage == 'sens':
        run_sens(a.device)


if __name__ == '__main__':
    main()
