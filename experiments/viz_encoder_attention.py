#!/usr/bin/env python3
"""Attention map of the encoder of record (RelGraph R2-noLane) on held-out routes.

The lane-free encoder has exactly one attention: per window, a query built from the ego
motion and the navigation command softmaxes over the window's agent embeddings (the
command-conditioned readout, encoder/harness/r2_graph.py). Nothing else in the model is
an attention, and the route pooling has no per-window scalar, so the "moment" of a route
is taken from the gradient of the predicted difficulty with respect to each window's
representation, ||d f_phi / d z_w||, computed by the harness with --dump-attn.

Input:  the dump written by
            python r2_graph.py --domain b2d --seed 0 --ablate-lane --draws 1 --dump-attn <npz>
        (draw 0, seed 0: same recipe as the record, one draw), and the graph tensors.
Output: one row per route -- BEV of the peak-saliency window (lanes grey, ego trail black,
        agents as boxes coloured by attention, top-3 numbered), the saliency timeline over
        the route's windows, and the attention over the agents of the peak window.

    python experiments/viz_encoder_attention.py --dump <npz> --out <png>
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import transforms, cm, colors
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'encoder' / 'harness'))

HARNESS = Path('/data2/jeongtae/relgraph_e16sel')
GRAPH = REPO / 'encoder' / 'harness' / 'b2d_relgraph_v2.npz'
STRIDE_S = 4 * 0.5                      # windows: 12 steps at 2 Hz, stride 4 steps = 2 s
EGO_HALF = (2.45, 1.05)                 # Lincoln MKZ footprint, half length / half width (m)


def load(dump):
    d = np.load(dump, allow_pickle=True)
    g = np.load(GRAPH, allow_pickle=True)
    run = np.load(HARNESS / 'r2nolane_attnviz_b2d_s0.npz', allow_pickle=True)
    from b2d_splits import unified_split
    Y, types = run['Y'], [str(t) for t in run['types']]
    hp, _ = unified_split(int(d['draw']), sorted(set(types)), Y.shape[0])
    keepJ = [j for j in range(Y.shape[0]) if j not in hp]
    fail_c = np.nanmean(Y[keepJ], 0)                      # block C: the 12 calibration planners
    attn = {int(r): a for r, a in zip(d['rows'], d['attn'])}   # window row -> (A, heads)
    sal = {int(r): s for r, s in zip(d['sal_route'], d['sal'])}
    wrows = {int(i): np.asarray(w) for i, w in zip(d['heldout'], d['window_rows'])}
    pred = {int(i): float(p) for i, p in zip(d['heldout'], d['pred'])}
    return d, g, run, types, fail_c, attn, sal, wrows, pred


def pick_routes(heldout, pred, types, k=4):
    """Highest, lowest and two intermediate predicted difficulties, distinct scenario types where possible."""
    order = sorted(heldout, key=lambda i: -pred[i])
    want = [order[0], order[len(order) // 3], order[2 * len(order) // 3], order[-1]]
    out, seen = [], set()
    for cand, pool in zip(want, (order, order[len(order) // 4:], order[len(order) // 2:], order[::-1])):
        for i in [cand] + list(pool):
            if i not in out and (types[i] not in seen or i == cand):
                out.append(i); seen.add(types[i]); break
    return out[:k]


def box(ax, x, y, psi, hl, hw, color, alpha=1.0, lw=0.8, ec='k', z=3, hatch=None):
    t = transforms.Affine2D().rotate(psi).translate(x, y) + ax.transData
    ax.add_patch(Rectangle((-hl, -hw), 2 * hl, 2 * hw, transform=t, facecolor=color, edgecolor=ec,
                           linewidth=lw, alpha=alpha, zorder=z, hatch=hatch))


def draw_bev(ax, g, row, a_w, title):
    lanes, lm = g['lanes'][row].astype(np.float32), g['lane_mask'][row]
    for m in np.where(lm)[0]:
        pts = lanes[m, :, :2]
        ax.plot(pts[:, 0], pts[:, 1], color='0.82', lw=1.0, zorder=1)
    ego = g['ego'][row].astype(np.float32)                 # (T, 6) dx dy cos sin speed is_future
    t_anchor = int(np.argmin(np.abs(ego[:, 0]) + np.abs(ego[:, 1])))
    ax.plot(ego[:t_anchor + 1, 0], ego[:t_anchor + 1, 1], color='k', lw=1.4, zorder=4)
    ax.plot(ego[t_anchor:, 0], ego[t_anchor:, 1], color='k', lw=1.0, ls='--', zorder=4)
    box(ax, ego[t_anchor, 0], ego[t_anchor, 1], np.arctan2(ego[t_anchor, 3], ego[t_anchor, 2]),
        *EGO_HALF, color='white', lw=1.4, z=6)
    ax.text(ego[t_anchor, 0], ego[t_anchor, 1], 'ego', ha='center', va='center', fontsize=6, zorder=7)
    ag, am = g['agents'][row].astype(np.float32), g['agent_mask'][row]   # (A, T, 8), (A, T)
    live = np.where(am.any(1))[0]
    w = a_w.mean(1)                                        # head-mean attention per agent slot
    norm = colors.Normalize(vmin=0.0, vmax=max(float(w[live].max()) if len(live) else 1.0, 1e-6))
    cmap = plt.get_cmap('magma_r')
    rank = live[np.argsort(-w[live])]
    for k in live:
        ts = np.where(am[k])[0]
        t0 = ts[np.argmin(np.abs(ts - t_anchor))]           # the valid step nearest the anchor
        x, y = ag[k, t0, 0], ag[k, t0, 1]
        psi = np.arctan2(ag[k, t0, 3], ag[k, t0, 2])
        veh = ag[k, t0, 7] > 0.5
        col = cmap(norm(w[k]))
        for tt in ts[(ts < t0) & (ts >= t0 - 3)]:              # 1.5 s fading trail
            box(ax, ag[k, tt, 0], ag[k, tt, 1], np.arctan2(ag[k, tt, 3], ag[k, tt, 2]),
                max(ag[k, tt, 5], 0.3), max(ag[k, tt, 6], 0.3), col, alpha=0.18 + 0.12 * (tt - (t0 - 3)), lw=0.0, z=2)
        box(ax, x, y, psi, max(ag[k, t0, 5], 0.3), max(ag[k, t0, 6], 0.3), col, lw=0.8 if veh else 0.5,
            ec='k' if veh else '0.35', z=3 + (w[k] > 0.05), hatch=None if veh else '////')
    for n_, k in enumerate(rank[:3]):
        ts = np.where(am[k])[0]; t0 = ts[np.argmin(np.abs(ts - t_anchor))]
        ax.annotate(str(n_ + 1), (ag[k, t0, 0], ag[k, t0, 1]), xytext=(4, 4), textcoords='offset points',
                    fontsize=7, fontweight='bold', color='navy', zorder=8)
    ax.set_xlim(-32, 56); ax.set_ylim(-44, 44)               # ego at the origin, x forward
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7.5)
    return norm, cmap, rank, w, t_anchor


def draw_timeline(ax, s, wstar, wmax):
    t = np.arange(len(s)) * STRIDE_S
    ax.fill_between(t, 0, s, color='#b03060', alpha=0.15)
    ax.plot(t, s, color='#b03060', lw=1.2)
    ax.plot(t, wmax * s.max() / max(wmax.max(), 1e-9), color='0.4', lw=0.8, ls=':')
    ax.axvline(wstar * STRIDE_S, color='r', ls='--', lw=0.9)
    ax.set_xlabel('time along the route (s)', fontsize=7)
    ax.set_ylabel('||d f_phi / d z_w||', fontsize=7)
    ax.set_title('window saliency (solid) and max agent attention (dotted, rescaled)', fontsize=7.5)
    ax.tick_params(labelsize=6)


def draw_bars(ax, g, row, a_w, rank, w, t_anchor, cmap, norm):
    ag, am = g['agents'][row].astype(np.float32), g['agent_mask'][row]
    top = rank[:8]
    labels = []
    for k in top:
        ts = np.where(am[k])[0]; t0 = ts[np.argmin(np.abs(ts - t_anchor))]
        x, y = ag[k, t0, 0], ag[k, t0, 1]
        labels.append(f'{"veh" if ag[k, t0, 7] > 0.5 else "ped/bike"}  {np.hypot(x, y):4.1f} m  {np.degrees(np.arctan2(y, x)):+4.0f} deg  v {ag[k, t0, 4]:.1f} m/s')
    yy = np.arange(len(top))[::-1]
    ax.barh(yy, w[top], color=[cmap(norm(w[k])) for k in top], edgecolor='k', lw=0.4)
    for h in range(a_w.shape[1]):
        ax.plot(a_w[top, h], yy, ls='none', marker='|', color='0.2', ms=5, mew=0.8)
    rest = 1.0 - w[top].sum()
    ax.set_yticks(yy); ax.set_yticklabels([f'#{i + 1}  {l}' for i, l in enumerate(labels)], fontsize=6)
    ax.set_xlim(0, 1.0); ax.tick_params(labelsize=6)
    ax.set_xlabel('readout attention (bar = head mean, ticks = 4 heads)', fontsize=7)
    ax.set_title(f'peak window: {len(np.where(am.any(1))[0])} agents, remaining mass {rest:.2f}', fontsize=7.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--routes', nargs='*', default=None, help='route ids to draw (default: 4 picked by predicted difficulty)')
    a = ap.parse_args()
    d, g, run, types, fail_c, attn, sal, wrows, pred = load(a.dump)
    names = [str(r) for r in run['routes']]
    heldout = [int(i) for i in d['heldout']]
    sel = [names.index(r) for r in a.routes] if a.routes else pick_routes(heldout, pred, types)
    fig, axes = plt.subplots(len(sel), 3, figsize=(16, 3.9 * len(sel)),
                             gridspec_kw={'width_ratios': [1.7, 1.0, 1.15]})
    axes = np.atleast_2d(axes)
    for r_, i in enumerate(sel):
        rows = wrows[i]
        s = np.asarray(sal[i], float)
        wmax = np.array([attn[int(rw)].mean(1).max() for rw in rows])
        wstar = int(np.argmax(s))
        row = int(rows[wstar])
        title = (f'route_{names[i]}  [{types[i]}]\n'
                 f'pred. b {pred[i]:+.2f}  |  fail rate (12 cal. planners) {fail_c[i]:.2f}  |  '
                 f'peak window {wstar + 1}/{len(rows)}, {wstar * STRIDE_S:.0f} s')
        norm, cmap, rank, w, t_anchor = draw_bev(axes[r_, 0], g, row, attn[row], title)
        draw_timeline(axes[r_, 1], s, wstar, wmax)
        draw_bars(axes[r_, 2], g, row, attn[row], rank, w, t_anchor, cmap, norm)
    fig.suptitle('RelGraph R2-noLane (encoder of record), draw 0 seed 0, held-out routes: command-conditioned readout '
                 'attention over agents at the window of highest difficulty saliency (box colour = attention, darker = more)', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f'wrote {a.out}: routes {[names[i] for i in sel]}')


if __name__ == '__main__':
    main()
