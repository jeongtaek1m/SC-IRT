#!/usr/bin/env python3
"""Frozen pretrained MOTION features of the reference rollouts (the motion branch of the
foundation-model encoder).

The visual branch (experiments/visual_features.py) is a frozen DINOv3 run over the three
reference-rollout cameras.  This is its motion counterpart: a frozen SMART-tiny 7M traffic
model (Wu et al., NeurIPS 2024) with the CAT-K BC pre-training checkpoint (Zhang et al.,
CVPR 2025; NVlabs/catk, /home/jeongtae/IRT/SMART/ckpts/pre_bc_E31.ckpt, trained on the
Waymo Open Motion Dataset) run over the ego/agent tracks of the PDM-Lite reference rollout
(/data1/jeongtae/b2d_eval_sensors/route_<id>/anno/*.json.gz, 10 Hz).

Scenes.  experiments/us_official/traffic_entropy.py converts ONE centred 9.1 s window per
route into CAT-K's scene format; its converters (`build_agent_catk`, `build_map_catk`, the
short-route padding of `load_frame`) are imported verbatim and the window is SLID along the
whole rollout instead, one window every 20 frames = 2 s (the encoder's window stride).  A
route shorter than 91 frames gets exactly one window, padded the way traffic_entropy.py pads
(frames past the end are empty, so those steps are invalid for every agent).

Quantity.  The AGENT HIDDEN STATES, i.e. the input of the next-token classification head:
CAT-K src/smart/modules/agent_decoder.py:SMARTAgentDecoder.forward computes
`feat_a` [n_agent, 18, hidden_dim] through 6 (temporal, map2agent, agent2agent) attention
layers and then `next_token_logits = self.token_predict_head(feat_a)`, of which the returned
slice `[:, 1:-1]` (16 action steps at 2 Hz) carries the validity `next_token_valid`.  A
forward hook on `token_predict_head` captures exactly that `feat_a`; the same `[:, 1:-1]`
slice is pooled here.  Per window:

    feat = [ mean over the valid (agent, step) cells ; mean over the ego's valid steps ]

(2 x hidden_dim).  The teacher-forced next-token entropy of traffic_entropy.py is stored
alongside as a 1-d fallback.  Nothing is trained, no response is read.

    <out>/route_<id>.npz : feat (Wn, 2*hidden_dim) float16, frame (Wn,) int32 10-Hz window
    start, ent / ent_ego (Wn,), n_agent / n_cells (Wn,), meta (str), plus town / n_frames.

Usage (env 'smart', GPU 2 or 3):
    CUDA_VISIBLE_DEVICES=3 /home/jeongtae/miniconda3/envs/smart/bin/python \
        experiments/motion_fm_features.py --out /data2/jeongtae/motion_feats/smart_catk_2s
"""
import argparse
import functools
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'us_official'))
import traffic_entropy as T                                          # noqa: E402

ROLLOUT = T.ROLLOUT
NFR, ANCH = T.NFR, T.ANCH
STRIDE_FRAMES = 20                       # 2 s, the B2D encoder's window stride


# ---------------------------------------------------------------- in-process caches
# build_map re-reads {town}.pkl (up to 138 MB) and {town}_marks_typed.pkl (up to 61 MB)
# on EVERY call and load_frame re-reads one gzipped json per frame; one centred window per
# route hid that, ~2.3k overlapping windows do not.  Nothing below changes what the
# converters compute -- the cached objects are only read.
class _CachingPickle:
    def __init__(self, real):
        self._real, self._cache = real, {}

    def load(self, f):
        k = getattr(f, 'name', None)
        if k is None:
            return self._real.load(f)
        if k not in self._cache:
            self._cache.clear()                       # one town at a time
            self._cache[k] = self._real.load(f)
        return self._cache[k]

    def __getattr__(self, n):
        return getattr(self._real, n)


T.V.pickle = _CachingPickle(pickle)
T.V._marks_typed = functools.lru_cache(maxsize=1)(T.V._marks_typed)
T.V.load_frame = functools.lru_cache(maxsize=128)(T.V.load_frame)    # padded loader


def routes():
    """The 220 evaluation routes, directory order.  The frame count is the number of anno
    files: route_11755's meta.json says n_frames 0 while 965 frames exist."""
    out = []
    for d in sorted(os.listdir(ROLLOUT)):
        anno = os.path.join(ROLLOUT, d, 'anno')
        if d.startswith('route_') and os.path.isdir(anno):
            out.append((d, len(os.listdir(anno))))
    return out


def window_starts(nf):
    return list(range(0, max(1, nf - NFR + 1), STRIDE_FRAMES))


def convert_route(task):
    """All windows of one route in CAT-K's scene format, spooled to disk (CPU worker).
    Only the path comes back through the pool pipe."""
    route, nf, tmp = task
    try:
        town = json.load(open(f'{ROLLOUT}/{route}/meta.json'))['town']
        scenes = []
        for s in window_starts(nf):
            agent = T.build_agent_catk(route, s)
            ego_xy = agent['position'][-1, ANCH, :2].numpy()
            data = T.build_map_catk(town, ego_xy)
            data['agent'] = agent
            data['scenario_id'] = f"b2d_r{route.split('_')[1]}_w{s}"
            scenes.append((s, data))
        T.V.load_frame.cache_clear()
        dst = os.path.join(tmp, f'{route}.pkl')
        with open(dst, 'wb') as fh:
            pickle.dump(scenes, fh)
        return route, town, nf, dst, ''
    except Exception as e:                                            # noqa: BLE001
        return route, '', nf, '', f'{type(e).__name__} {e}'


@torch.no_grad()
def window_feature(model, tp, data, device, cap):
    from torch_geometric.data import Batch, HeteroData
    batch = Batch.from_data_list([HeteroData(data)]).to(device)
    tok_map, tok_agent = tp(batch)
    cap.clear()
    pred = model(tok_map, tok_agent)
    feat = cap[0][:, 1:-1].float()                       # [n_agent, 16, D] = token_predict_head input
    valid = pred['next_token_valid'].bool()              # [n_agent, 16]
    ego = tok_agent['ego_mask'].bool()                   # [n_agent]
    m_all, m_ego = valid, valid & ego[:, None]
    D = feat.shape[-1]
    zz = torch.zeros(D, device=feat.device)
    h_all = feat[m_all].mean(0) if bool(m_all.any()) else zz
    h_ego = feat[m_ego].mean(0) if bool(m_ego.any()) else zz
    logp = torch.log_softmax(pred['next_token_logits'].float(), -1)
    ent = -(logp.exp() * logp).sum(-1)                   # [n_agent, 16] nats
    f = lambda m: float(ent[m].mean()) if bool(m.any()) else float('nan')   # noqa: E731
    return (torch.cat([h_all, h_ego]).cpu().numpy(), f(m_all), f(m_ego),
            int(feat.shape[0]), int(m_all.sum()))


META = ('CAT-K/SMART-tiny pre_bc_E31 (strict load, agent_vocab_555_s2), teacher-forced '
        'open-loop forward on one 9.1 s window of the PDM-Lite reference rollout every 20 '
        'frames (2 s). feat = concat[ mean over valid (agent, 2 Hz token step) cells of the '
        'agent hidden state feat_a[:, 1:-1] (the input of SMARTAgentDecoder.token_predict_head, '
        'after the 6 temporal/map2agent/agent2agent attention layers), mean over the ego '
        "agent's own valid steps of the same tensor ]; hidden_dim {D} each, 2*{D} total. "
        'Frozen model, no training, no response read.')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default='/data2/jeongtae/motion_feats/smart_catk_2s')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--only', default=None, help='comma-separated route dirs (verification)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tasks = routes()
    if a.only:
        keep = set(a.only.split(','))
        tasks = [t for t in tasks if t[0] in keep]
    if a.limit:
        tasks = tasks[:a.limit]
    tasks = [t for t in tasks if not os.path.exists(os.path.join(a.out, f'{t[0]}.npz'))]
    nwin = sum(len(window_starts(nf)) for _, nf in tasks)
    print(f'{len(tasks)} routes / {nwin} windows -> {a.out}', flush=True)
    if not tasks:
        return
    tmp = os.path.join(a.out, '_scenes')
    os.makedirs(tmp, exist_ok=True)
    tasks = [(r, nf, tmp) for r, nf in tasks]

    pool = Pool(a.workers)                      # fork BEFORE the CUDA context exists
    model, tp, ck = T.load_model(a.device)
    cap = []
    model.agent_encoder.token_predict_head.register_forward_hook(
        lambda m, i, o: cap.append(i[0]))
    D = int(ck['decoder']['hidden_dim'])
    print(f'hidden_dim {D} -> feature {2 * D}-d per window', flush=True)

    t0, done, nw = time.time(), 0, 0
    with pool:
        for route, town, nf, path, err in pool.imap(convert_route, tasks):
            if err:
                print(f'  ERR {route}: {err}', flush=True)
                continue
            with open(path, 'rb') as fh:
                scenes = pickle.load(fh)
            os.remove(path)
            F, ent, ente, na, nc = [], [], [], [], []
            for s, d in scenes:
                h, e, ee, n_a, n_c = window_feature(model, tp, d, a.device, cap)
                F.append(h); ent.append(e); ente.append(ee); na.append(n_a); nc.append(n_c)
            F = np.stack(F)
            np.savez(os.path.join(a.out, f'{route}.npz'),
                     feat=F.astype(np.float16),
                     frame=np.array([s for s, _ in scenes], np.int32),
                     ent=np.array(ent), ent_ego=np.array(ente),
                     n_agent=np.array(na, np.int32), n_cells=np.array(nc, np.int32),
                     meta=np.array(META.format(D=D)), hidden_dim=D, town=np.array(town),
                     n_frames=nf, stride_frames=STRIDE_FRAMES, n_window_frames=NFR,
                     ckpt=np.array(ck['ckpt']))
            done += 1; nw += len(scenes)
            if done % 10 == 0 or done == len(tasks):
                print(f'  {done}/{len(tasks)} routes, {nw} windows '
                      f'({time.time() - t0:.0f} s), last {route}: {len(scenes)} win '
                      f'|feat| {np.abs(F).mean():.4f}', flush=True)
    if not os.listdir(tmp):
        os.rmdir(tmp)
    print(f'done: {done} routes / {nw} windows in {time.time() - t0:.0f} s -> {a.out}',
          flush=True)


if __name__ == '__main__':
    main()
