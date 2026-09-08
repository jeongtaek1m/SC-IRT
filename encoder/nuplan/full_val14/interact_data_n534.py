#!/usr/bin/env python3
"""val14 interaction tensors from routed SMART pkls (91 frames @10Hz, recentered).

Mirrors interact_data_b2d.py schema exactly: 0.5s grid (frames ::5 -> 19 steps,
t=-1.0..+8.0), windows T=12 STRIDE=4 -> 2 windows/scenario, anchor = 4th step,
agents <=48 within 60m sorted by distance, ego-anchor rotation.
agents 8ch [rx,ry,cos,sin,spd,l/2,w/2,isveh] | ego 6ch [rx,ry,cos,sin,spd,is_fut]
cmd 4ch [left,straight,right,unknown] from ego heading change over the window.
Also saves scenario-level kin10 from the full 19-step ego track.
out: /data2/jeongtae/navsim_interact/val14_tensors.npz
"""
import glob, os, pickle
import numpy as np

PKL = '/data1/jeongtae/smart_difficulty/pkls/routed_pkls_val14n534'
OUT = '/data2/jeongtae/nuplan_full_val14'
A_MAX, T, STRIDE = 48, 12, 4


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def one(path):
    tok = os.path.basename(path).split('.')[0]
    d = pickle.load(open(path, 'rb'))
    ag = d['agent']
    pos = ag['position'][:, :, :2].numpy().astype(np.float64)
    head = ag['heading'].numpy(); vel = ag['velocity'][:, :, :2].numpy()
    shp = ag['shape'].numpy(); val = ag['valid_mask'].numpy(); typ = ag['type'].numpy()
    av = int(d['agent'].get('av_index', d.get('av_index', 0))) if isinstance(d['agent'], dict) else 0
    if 'av_index' in d: av = int(d['av_index'])
    g = np.arange(0, 91, 5)                                   # 19 grid steps
    spd = np.linalg.norm(vel, axis=-1)
    out = []
    for a in range(0, len(g) - T, STRIDE):
        i = a + 3
        f_anchor = g[i]
        if not val[av, f_anchor]:
            continue
        ex, ey = pos[av, f_anchor]; eyaw = head[av, f_anchor]
        c, s = np.cos(-eyaw), np.sin(-eyaw)
        cand = [n for n in range(pos.shape[0]) if n != av and val[n, f_anchor]]
        cand.sort(key=lambda n: np.hypot(pos[n, f_anchor, 0] - ex, pos[n, f_anchor, 1] - ey))
        cand = [n for n in cand
                if np.hypot(pos[n, f_anchor, 0] - ex, pos[n, f_anchor, 1] - ey) < 60.0][:A_MAX]
        AGt = np.zeros((A_MAX, T, 8), np.float16); AMt = np.zeros((A_MAX, T), bool)
        for ai, n in enumerate(cand):
            for w in range(T):
                f = g[a + w]
                if not val[n, f]:
                    continue
                dx, dy = pos[n, f, 0] - ex, pos[n, f, 1] - ey
                AGt[ai, w] = [dx * c - dy * s, dx * s + dy * c,
                              np.cos(head[n, f] - eyaw), np.sin(head[n, f] - eyaw),
                              spd[n, f], shp[n, f, 0] / 2, shp[n, f, 1] / 2,
                              1.0 if typ[n] == 0 else 0.0]
                AMt[ai, w] = True
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
        out.append((AGt, AMt, EGt, cm))
    ev = spd[av, g]; dv = np.diff(ev) / 0.5
    yr = np.abs(wrap(np.diff(head[av, g]))) / 0.5
    kin = np.array([ev.mean(), ev.std(), ev.max(), (ev < 0.5).mean(),
                    dv.std(), dv.min(), dv.max(), yr.mean(), yr.max(), ev[2]], np.float32)
    return tok, out, kin


def main():
    files = sorted(glob.glob(f'{PKL}/*'))
    toks, AG, AM, EG, CM, WIX, KIN = [], [], [], [], [], [], {}
    for k, p in enumerate(files):
        tok, out, kin = one(p)
        KIN[tok] = kin
        for wi, (ag, am, eg, cm) in enumerate(out):
            toks.append(tok); WIX.append(wi)
            AG.append(ag); AM.append(am); EG.append(eg); CM.append(cm)
        if (k + 1) % 100 == 0:
            print(f'  ...{k+1}/{len(files)}, {len(toks)} windows', flush=True)
    kt = sorted(KIN)
    np.savez_compressed(f'{OUT}/val14n534_tensors.npz',
                        token=np.array(toks), widx=np.array(WIX),
                        agents=np.stack(AG), amask=np.stack(AM),
                        ego=np.stack(EG), cmd=np.stack(CM),
                        kin_token=np.array(kt), kin=np.stack([KIN[t] for t in kt]))
    print(f'DONE_VAL14_TENSORS {len(set(toks))} scenarios, {len(toks)} windows', flush=True)


if __name__ == '__main__':
    main()
