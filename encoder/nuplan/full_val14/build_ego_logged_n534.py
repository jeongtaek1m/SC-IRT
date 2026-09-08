#!/usr/bin/env python3
"""Logged-ego companion to /data2/jeongtae/navsim_interact/val14_tensors.npz (PROTOCOL_R 20.2 S2).

val14_tensors.npz was built (b2d_irt/scripts/interact_data_val14.py) from routed_pkls_val14,
whose ego frames 11-90 are a synthetic centerline-following trajectory at constant speed
(S0 scouting: 1,165 / 1,168 windows have zero future-speed variance).  This rebuilds ONLY the
ego (N,12,6) and cmd (N,4) arrays with the identical formula from smart_pkls_val14, whose ego is
the logged human drive (frames 0-10 are identical in both pkl sets; verified below).
Windows, anchors, channel layout and dtype are unchanged; agents are not touched.

out: /data2/jeongtae/relgraph/transfer/nuplan/val14_ego_logged.npz  (token, widx, ego, cmd)
"""
import glob, os, pickle
import numpy as np

PKL = '/data1/jeongtae/smart_difficulty/pkls/smart_pkls_val14_missing534'
REF = '/data2/jeongtae/nuplan_full_val14/val14n534_tensors.npz'
OUT = '/data2/jeongtae/nuplan_full_val14/val14n534_ego_logged.npz'
T, STRIDE = 12, 4


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def one(path):
    tok = os.path.basename(path).split('.')[0]
    d = pickle.load(open(path, 'rb'))
    ag = d['agent']
    pos = ag['position'][:, :, :2].numpy().astype(np.float64)
    head = ag['heading'].numpy(); vel = ag['velocity'][:, :, :2].numpy()
    val = ag['valid_mask'].numpy()
    av = int(d['av_index']) if 'av_index' in d else int(ag['av_index'])
    g = np.arange(0, 91, 5)
    spd = np.linalg.norm(vel, axis=-1)
    out = []
    for a in range(0, len(g) - T, STRIDE):
        i = a + 3
        f_anchor = g[i]
        if not val[av, f_anchor]:
            continue
        ex, ey = pos[av, f_anchor]; eyaw = head[av, f_anchor]
        c, s = np.cos(-eyaw), np.sin(-eyaw)
        EGt = np.zeros((T, 6), np.float16)
        for w in range(T):
            f = g[a + w]
            dx, dy = pos[av, f, 0] - ex, pos[av, f, 1] - ey
            EGt[w] = [dx * c - dy * s, dx * s + dy * c,
                      np.cos(head[av, f] - eyaw), np.sin(head[av, f] - eyaw),
                      spd[av, f], 1.0 if w > 3 else 0.0]
        dy_h = wrap(head[av, g[a + T - 1]] - eyaw)
        cm = np.zeros(4, np.float16)
        cm[0 if dy_h > 0.26 else (2 if dy_h < -0.26 else 1)] = 1.0
        out.append((a // STRIDE, EGt, cm))
    return tok, out


def main():
    ref = np.load(REF, allow_pickle=True)
    files = sorted(glob.glob(f'{PKL}/*.pkl'))
    toks, WIX, EG, CM = [], [], [], []
    for p in files:
        tok, out = one(p)
        for wi, eg, cm in out:
            toks.append(tok); WIX.append(wi); EG.append(eg); CM.append(cm)
    toks, WIX, EG, CM = np.array(toks), np.array(WIX), np.stack(EG), np.stack(CM)
    assert np.array_equal(toks, ref['token']) and np.array_equal(WIX, ref['widx']), \
        'window set differs from val14_tensors.npz'
    ego_ref = ref['ego'].astype(np.float32); ego = EG.astype(np.float32)
    # frames 0-10 are shared by both pkl sets -> window 0 steps 0-2 (frames 0,5,10) must agree
    # exactly up to the anchor rotation (anchor frame 15 is already synthetic in the routed pkl,
    # so only speed, which is rotation-free, is compared)
    w0 = WIX == 0
    dsp = np.abs(ego[w0, :3, 4] - ego_ref[w0, :3, 4]).max()
    fut_const_ref = int((ego_ref[:, 4:, 4].std(1) < 1e-3).sum())
    fut_const = int((ego[:, 4:, 4].std(1) < 1e-3).sum())
    print(f'{len(toks)} windows / {len(set(toks))} scenarios; widx0 steps 0-2 max |dspeed| vs '
          f'routed tensor {dsp:.4f} m/s; windows with constant future speed: routed {fut_const_ref}, '
          f'logged {fut_const}; cmd agreement {float((CM.argmax(1) == ref["cmd"].argmax(1)).mean()):.3f}',
          flush=True)
    assert dsp < 1e-2
    np.savez_compressed(OUT, token=toks, widx=WIX, ego=EG, cmd=CM)
    print(f'WROTE {OUT}', flush=True)


if __name__ == '__main__':
    main()
