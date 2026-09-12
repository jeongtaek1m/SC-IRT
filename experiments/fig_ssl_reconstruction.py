#!/usr/bin/env python3
"""What the SSL pre-training of the encoder actually does: the masking and the reconstruction, on real windows.

Reads the npz written by `r2_graph.py --ssl [--ssl-ego-ctx] --dump-ssl <path>` (the inner-validation routes of
draw 0, the mask drawn with a fixed seed, and the reconstruction at the selected epoch) and draws:
  row 1  agent tracks of three windows in the ego frame: the steps the encoder SEES (grey), the hidden segment
         (red, ground truth) and its reconstruction (blue); the ego box at the window anchor;
  row 2  the same masked agents as time series of the two position channels, the hidden interval shaded;
  row 3  the ego half (with --ssl-ego-ctx): the route's speed sequence with the hidden segments shaded and the
         reconstruction on top, plus the inner-validation curve of both halves with the selected epoch marked.

    python experiments/fig_ssl_reconstruction.py --dump <dir>/ssl_draw0.npz [--ego-dump <dir>/sslego_draw0.npz] \
        --out results/figs/ssl_reconstruction.pdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import transforms
import numpy as np

DX, DY, COS, SIN, V = 0, 1, 2, 3, 4          # the five reconstructed agent channels
EGO_HALF = (2.45, 1.05)


def moving(d, w):
    """Masked agents of window w, ordered by how far they travel inside the window."""
    M, AG, AM, mu, sd = d['mask'], d['agents_std'], d['agent_mask'], d['ag_mean'], d['ag_sd']
    out = []
    for a in np.where(M[w].any(1))[0]:
        v = AM[w, a]
        x = AG[w, a, v, DX] * sd[DX] + mu[DX]
        y = AG[w, a, v, DY] * sd[DY] + mu[DY]
        out.append((float(np.hypot(x.ptp(), y.ptp())), int(a)))
    return [a for _, a in sorted(out, reverse=True)]


def pick_windows(d, k=3):
    """Windows whose masked agents actually move: the score is the travel of the three most mobile of them."""
    score = []
    for w in range(d['mask'].shape[0]):
        ags = moving(d, w)
        if len(ags) < 3:
            continue
        M, AG, AM, mu, sd = d['mask'], d['agents_std'], d['agent_mask'], d['ag_mean'], d['ag_sd']
        tr = 0.0
        for a in ags[:3]:
            v = AM[w, a]
            tr += float(np.hypot((AG[w, a, v, DX] * sd[DX]).ptp(), (AG[w, a, v, DY] * sd[DY]).ptp()))
        score.append((tr, w))
    return [w for _, w in sorted(score, reverse=True)[:k]]


def unstd(x, mu, sd, ch):
    return x * sd[ch] + mu[ch]


def draw_window(ax, d, w):
    M, AG, AM = d['mask'], d['agents_std'], d['agent_mask']
    mu, sd = d['ag_mean'], d['ag_sd']
    b, k, t, pred = d['cell_b'], d['cell_k'], d['cell_t'], d['pred']
    xs = unstd(AG[w, :, :, DX], mu, sd, DX); ys = unstd(AG[w, :, :, DY], mu, sd, DY)
    lim = 0.0
    for a in np.where(AM[w].any(1))[0]:
        vis = AM[w, a] & ~M[w, a]
        if M[w, a].any():
            lim = max(lim, float(np.abs(xs[a][AM[w, a]]).max()), float(np.abs(ys[a][AM[w, a]]).max()))
        hid = M[w, a]
        if vis.any():
            ax.plot(xs[a][vis], ys[a][vis], color='0.55', lw=1.0, marker='o', ms=2.2, zorder=2)
        if hid.any():
            ax.plot(xs[a][hid], ys[a][hid], color='#d62728', lw=1.2, marker='o', ms=3.4, zorder=4, ls='--')
            sel = (b == w) & (k == a)
            px = unstd(pred[sel, DX], mu, sd, DX); py = unstd(pred[sel, DY], mu, sd, DY)
            ax.plot(px, py, color='#1f77b4', lw=1.2, marker='x', ms=4.5, mew=1.2, zorder=5, ls=':')
    t0 = 3                                                    # the window anchor step
    tr = transforms.Affine2D().rotate(0).translate(0, 0) + ax.transData
    ax.add_patch(Rectangle((-EGO_HALF[0], -EGO_HALF[1]), 2 * EGO_HALF[0], 2 * EGO_HALF[1], transform=tr,
                           facecolor='white', edgecolor='k', lw=1.2, zorder=6))
    ax.text(0, 0, 'ego', ha='center', va='center', fontsize=5.5, zorder=7)
    n_a = int(M[w].any(1).sum())
    ax.set_title(f'window {w}: {n_a} of {int((AM[w].sum(1) >= 8).sum())} agents masked', fontsize=7.5)
    pts = np.concatenate([np.stack([xs[a][AM[w, a]], ys[a][AM[w, a]]], 1) for a in np.where(M[w].any(1))[0]] + [np.zeros((1, 2))])
    ctr = (pts.min(0) + pts.max(0)) / 2
    half = max((pts.max(0) - pts.min(0)).max() / 2, 12.0) * 1.12         # equal aspect, cropped to the masked agents
    ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_aspect('equal'); ax.tick_params(labelsize=6)
    ax.set_xlabel('x (m, ego frame)', fontsize=6.5)


def draw_series(ax, d, w):
    M, AG, AM = d['mask'], d['agents_std'], d['agent_mask']
    mu, sd = d['ag_mean'], d['ag_sd']
    b, k, t, pred = d['cell_b'], d['cell_k'], d['cell_t'], d['pred']
    ags = moving(d, w)[:2]
    tt = np.arange(M.shape[2]) * 0.5
    for j, a in enumerate(ags):
        col = ['#333333', '#8c564b'][j]
        vis = AM[w, a]
        y = unstd(AG[w, a, :, DX], mu, sd, DX)
        ax.plot(tt[vis], y[vis], color=col, lw=1.2, marker='o', ms=2.5, label=f'agent {a}, true')
        hid = np.where(M[w, a])[0]
        if len(hid):
            ax.axvspan(tt[hid[0]] - 0.25, tt[hid[-1]] + 0.25, color=col, alpha=0.08, lw=0)
            sel = (b == w) & (k == a)
            ax.plot(tt[t[sel]], unstd(pred[sel, DX], mu, sd, DX), color=col, ls=':', marker='x', ms=6, mew=1.4,
                    lw=1.0, label=f'agent {a}, reconstructed')
    ax.set_xlabel('time in the window (s)', fontsize=6.5); ax.set_ylabel('x (m)', fontsize=6.5)
    ax.tick_params(labelsize=6); ax.legend(fontsize=5.5, frameon=False, loc='best')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--ego-dump', default=None)
    ap.add_argument('--out', default='results/figs/ssl_reconstruction.pdf')
    a = ap.parse_args()
    d = dict(np.load(a.dump, allow_pickle=True))
    de = dict(np.load(a.ego_dump, allow_pickle=True)) if a.ego_dump else None
    ws = pick_windows(d)
    fig = plt.figure(figsize=(12.5, 10.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 0.72, 0.85], hspace=0.40, wspace=0.24,
                          left=0.05, right=0.985, top=0.865, bottom=0.055)
    for c, w in enumerate(ws):
        draw_window(fig.add_subplot(gs[0, c]), d, w)
        draw_series(fig.add_subplot(gs[1, c]), d, w)
    fig.text(0.5, 0.982, 'Track SSL pre-training of the route encoder: what is hidden and what comes back',
             ha='center', va='top', fontsize=12, fontweight='bold')
    fig.text(0.5, 0.955, 'draw 0, seed 0, inner-validation windows at the selected epoch. Row 1: agent tracks in the ego frame; grey = the steps the encoder sees,',
             ha='center', va='top', fontsize=8, color='0.3')
    fig.text(0.5, 0.933, 'red dashed = the hidden 2 s segment, blue dotted = its reconstruction. Row 2: the two most mobile masked agents as x(t), the hidden interval shaded.',
             ha='center', va='top', fontsize=8, color='0.3')
    tgt = d['agents_std'][d['cell_b'], d['cell_k'], d['cell_t'], :5]
    rms = [float(np.sqrt(((d['pred'][:, c] - tgt[:, c]) ** 2).mean()) * d['ag_sd'][c]) for c in (DX, DY, V)]
    fig.text(0.5, 0.911, f'{len(d["cell_b"])} hidden (agent, step) cells over {d["mask"].shape[0]} windows: RMSE {rms[0]:.1f} m in x, {rms[1]:.1f} m in y, {rms[2]:.2f} m/s in speed '
             f'(98% / 97% / 83% of the variance).', ha='center', va='top', fontsize=8, color='0.3')
    # ---- row 3: the ego half and the curves
    axc = fig.add_subplot(gs[2, 2])
    for src, lab, col in ((d, 'agent-track half', '#2ca02c'), (de, 'with the ego half', '#9467bd')):
        if src is None:
            continue
        cv = src['curve']
        axc.plot(cv[:, 0], cv[:, 3], color=col, lw=1.4, label=f'{lab}: inner-val track MSE')
        axc.axvline(int(src['best_epoch']), color=col, ls='--', lw=0.9)
    axc.set_xlabel('SSL epoch', fontsize=6.5); axc.set_ylabel('inner-validation MSE', fontsize=6.5)
    axc.set_title('reconstruction error and the selected epoch (dashed)', fontsize=7.5)
    axc.legend(fontsize=6, frameon=False); axc.tick_params(labelsize=6)
    if de is not None and 'ego_x' in de:
        mu, sd = de['ego_mu'], de['ego_sd']
        val, Me = de['ego_valid'], de['ego_mask']
        r_, t_, pe = de['ego_cell_r'], de['ego_cell_t'], de['ego_pred']
        spd = np.array([np.std(de['ego_x'][r, :int(val[r].sum()), 0]) if Me[r].sum() > 3 else -1 for r in range(len(val))])
        cand = np.argsort(-spd)[:2]                                      # routes whose speed actually varies
        for j, r in enumerate(cand):
            ax = fig.add_subplot(gs[2, j])
            L = int(val[r].sum())
            tt = np.arange(L) * 0.5
            sp = de['ego_x'][r, :L, 0] * sd[0] + mu[0]
            ax.plot(tt, sp, color='0.3', lw=1.2, label='true ego speed')
            hid = np.where(Me[r, :L])[0]
            for h in hid:
                ax.axvspan(tt[h] - 0.25, tt[h] + 0.25, color='#d62728', alpha=0.10, lw=0)
            sel = r_ == r
            ax.plot(de['ego_cell_t'][sel] * 0.5, pe[sel, 0] * sd[0] + mu[0], color='#1f77b4', ls='none',
                    marker='x', ms=4.5, mew=1.2, label='reconstruction')
            ax.set_title(f'ego half, route {int(de["routes_val"][r])}: {len(hid)} of {L} steps hidden', fontsize=7.5)
            ax.set_xlabel('time along the route (s)', fontsize=6.5); ax.set_ylabel('speed (m/s)', fontsize=6.5)
            ax.legend(fontsize=5.5, frameon=False); ax.tick_params(labelsize=6)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200)
    if a.out.endswith('.pdf'):
        fig.savefig(a.out[:-4] + '.png', dpi=170)
    print('wrote', a.out, '| windows', ws, '| track best epoch', int(d['best_epoch']),
          '| ego best epoch', int(de['best_epoch']) if de is not None else '-')


if __name__ == '__main__':
    main()
