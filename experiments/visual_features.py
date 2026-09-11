#!/usr/bin/env python3
"""Frozen pretrained visual features of the reference rollouts (the visual branch of the two-branch encoder).

For every evaluation route the PDM-Lite reference rollout was recorded with three cameras (rgb_front,
rgb_front_left, rgb_front_right; 1600 x 900, 10 Hz, /data1/jeongtae/b2d_eval_sensors/route_<id>/camera).
This script runs a frozen DINOv3 ViT-L/16 (timm `vit_large_patch16_dinov3.lvd1689m`, 256 x 256 input, ImageNet
normalisation) on every STRIDE-th frame of each view and stores, per frame and view, the CLS token and the mean
of the patch tokens (1024-d each, float16):

    <out>/route_<id>.npz : cls (L, 3, 1024), patch (L, 3, 1024), frame (L,) 10-Hz frame indices, views (3,)

Nothing is trained here and no response is read: the features are a function of the scene video only. With
STRIDE = 5 the sequence is at 2 Hz, the rate of the motion branch. ~38k images, a few minutes on one GPU.

    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python experiments/visual_features.py --out /data2/jeongtae/visual_feats/dinov3_2hz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SENS = Path('/data1/jeongtae/b2d_eval_sensors')
VIEWS = ('rgb_front', 'rgb_front_left', 'rgb_front_right')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='vit_large_patch16_dinov3.lvd1689m')
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--bs', type=int, default=48)
    a = ap.parse_args()
    import timm
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m = timm.create_model(a.model, pretrained=True, num_classes=0).eval().cuda()
    cfg = timm.data.resolve_data_config({}, model=m)
    tf = timm.data.create_transform(**cfg, is_training=False)
    n_prefix = m.num_prefix_tokens                                   # CLS (+ register tokens): patches follow
    print(f'{a.model}: input {cfg["input_size"]}, prefix tokens {n_prefix}', flush=True)
    routes = sorted(p.name for p in SENS.glob('route_*') if (p / 'meta.json').exists())
    t0 = time.time()
    for k, r in enumerate(routes):
        dst = out / f'{r}.npz'
        if dst.exists():                                              # reuse only a file made with the SAME settings
            z = np.load(dst)
            assert str(z['model']) == a.model and int(z['stride']) == a.stride and tuple(z['views']) == VIEWS, \
                f'{dst}: existing file was written with model {z["model"]} stride {int(z["stride"])} views {tuple(z["views"])}; use another --out'
            continue
        n = min(len(list((SENS / r / 'camera' / v).glob('*.jpg'))) for v in VIEWS)   # meta.json says 0 for route_11755; the frames exist
        frames = list(range(0, n, a.stride))
        cls_all, patch_all = [], []
        for v in VIEWS:
            cls_v, patch_v = [], []
            for i0 in range(0, len(frames), a.bs):
                ims = [tf(Image.open(SENS / r / 'camera' / v / f'{f:05d}.jpg').convert('RGB')) for f in frames[i0:i0 + a.bs]]
                x = torch.stack(ims).cuda()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                    t = m.forward_features(x)                          # (B, prefix + patches, 1024)
                cls_v.append(t[:, 0].float().cpu().numpy())
                patch_v.append(t[:, n_prefix:].float().mean(1).cpu().numpy())
            cls_all.append(np.concatenate(cls_v) if cls_v else np.zeros((0, 1024), np.float32))
            patch_all.append(np.concatenate(patch_v) if patch_v else np.zeros((0, 1024), np.float32))
        np.savez(dst, cls=np.stack(cls_all, 1).astype(np.float16), patch=np.stack(patch_all, 1).astype(np.float16),
                 frame=np.array(frames, np.int32), views=np.array(VIEWS), model=np.array(a.model), stride=a.stride)
        if k % 20 == 0:
            print(f'  {k + 1}/{len(routes)} {r}: {len(frames)} frames x 3 views  ({time.time() - t0:.0f} s)', flush=True)
    print(f'done: {len(routes)} routes in {time.time() - t0:.0f} s -> {out}', flush=True)


if __name__ == '__main__':
    main()
