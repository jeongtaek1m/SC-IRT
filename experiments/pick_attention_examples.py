#!/usr/bin/env python3
"""Rank held-out (draw, route) pairs of the attention dumps as candidates for a qualitative figure.

For every held-out route of every dump: the peak-saliency window w* (argmax ||d f_phi / d z_w||), the
saliency contrast (peak / median over the route's windows), the readout attention in w* (head-mean max
weight, number of live agents, entropy relative to uniform) and the most-attended agent's position,
heading and speed relative to the ego. The score favours a clear moment (contrast) with a peaked
attention on one agent among several (max weight x live agents), on a route whose predicted difficulty
agrees with the observed failure rate. The pick is still a human choice: the table is printed for it.

    python experiments/pick_attention_examples.py --dumps <npz...> [--top 25]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'encoder' / 'harness'))
GRAPH = REPO / 'encoder' / 'harness' / 'b2d_relgraph_v2.npz'
RUN = Path('/data2/jeongtae/relgraph_e16sel/r2nolane_attnviz_b2d_s0.npz')


def candidates(dumps):
    g = np.load(GRAPH, allow_pickle=True)
    run = np.load(RUN, allow_pickle=True)
    from b2d_splits import unified_split
    Y, types = run['Y'], [str(t) for t in run['types']]
    names = [str(r) for r in run['routes']]
    am_any = g['agent_mask'].any(2)                                     # (N, A)
    rows_out = []
    for dump in dumps:
        d = np.load(dump, allow_pickle=True)
        draw = int(d['draw'])
        hp, _ = unified_split(draw, sorted(set(types)), Y.shape[0])
        keepJ = [j for j in range(Y.shape[0]) if j not in hp]
        fail_c = np.nanmean(Y[keepJ], 0)
        attn = {int(r): a for r, a in zip(d['rows'], d['attn'])}
        sal = {int(r): np.asarray(s, float) for r, s in zip(d['sal_route'], d['sal'])}
        wrows = {int(i): np.asarray(w) for i, w in zip(d['heldout'], d['window_rows'])}
        pred = {int(i): float(p) for i, p in zip(d['heldout'], d['pred'])}
        for i in wrows:
            rows, s = wrows[i], sal[i]
            wstar = int(np.argmax(s))
            row = int(rows[wstar])
            w = attn[row].mean(1)
            live = np.where(am_any[row])[0]
            if len(live) < 3:
                continue
            wl = w[live]
            k = live[int(np.argmax(wl))]
            ent = -(wl * np.log(wl + 1e-12)).sum() / np.log(len(live))
            ag, am = g['agents'][row].astype(np.float32), g['agent_mask'][row]
            ego = g['ego'][row].astype(np.float32)
            t_anchor = int(np.argmin(np.abs(ego[:, 0]) + np.abs(ego[:, 1])))
            ts = np.where(am[k])[0]
            t0 = ts[np.argmin(np.abs(ts - t_anchor))]
            x, y = ag[k, t0, 0], ag[k, t0, 1]
            contrast = float(s[wstar] / max(np.median(s), 1e-9))
            second = float(np.sort(wl)[-2]) if len(wl) > 1 else 0.0
            score = contrast * float(wl.max()) * np.sqrt(len(live)) * (1.0 if np.sign(pred[i]) == np.sign(fail_c[i] - 0.5) or abs(pred[i]) < 0.5 else 0.5)
            rows_out.append(dict(draw=draw, route=names[i], type=types[i], pred=pred[i], fail=float(fail_c[i]),
                                 wstar=wstar, nwin=len(rows), t_s=wstar * 2.0, contrast=contrast, n_live=len(live),
                                 wmax=float(wl.max()), w2=second, ent=float(ent), dist=float(np.hypot(x, y)),
                                 ang=float(np.degrees(np.arctan2(y, x))), speed=float(ag[k, t0, 4]),
                                 veh=bool(ag[k, t0, 7] > 0.5), ego_speed=float(ego[t_anchor, 4]), score=float(score)))
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dumps', nargs='+', required=True)
    ap.add_argument('--top', type=int, default=25)
    a = ap.parse_args()
    C = sorted(candidates(a.dumps), key=lambda r: -r['score'])
    print(f'{len(C)} candidates from {len(a.dumps)} dump(s)')
    print(f'{"score":>6} {"draw":>4} {"route":>6} {"type":34} {"b~":>6} {"fail":>5} {"w*/n":>7} {"t(s)":>5} {"contr":>6} {"live":>4} {"wmax":>5} {"w2":>5} {"ent":>5} {"top agent":>28}')
    for r in C[:a.top]:
        print(f'{r["score"]:6.2f} {r["draw"]:4d} {r["route"]:>6} {r["type"]:34} {r["pred"]:+6.2f} {r["fail"]:5.2f} '
              f'{r["wstar"] + 1:3d}/{r["nwin"]:<3d} {r["t_s"]:5.0f} {r["contrast"]:6.2f} {r["n_live"]:4d} {r["wmax"]:5.2f} {r["w2"]:5.2f} {r["ent"]:5.2f} '
              f'{"veh" if r["veh"] else "ped":>4} {r["dist"]:5.1f} m {r["ang"]:+5.0f} deg {r["speed"]:4.1f} m/s')


if __name__ == '__main__':
    main()
