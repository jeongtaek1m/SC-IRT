#!/usr/bin/env python3
"""Paper figure: the encoder of record's readout attention at the difficulty-relevant moment of
chosen held-out routes (cherry-picked from experiments/pick_attention_examples.py).

One column per example: BEV of the peak-saliency window (lanes grey, ego trail black with the
future dashed, agents as boxes coloured by the head-mean readout attention on a SHARED scale,
the three most-attended agents numbered with their weights), and under it the saliency
||d f_phi / d z_w|| along the route with the shown window marked.

    python experiments/fig_encoder_attention.py --dump-dir <dir with attn_s0_draw*.npz> \
        --examples 0:26990 3:12345 ... --out fig/encoder_attention.pdf
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import transforms, colors
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'encoder' / 'harness'))
GRAPH = REPO / 'encoder' / 'harness' / 'b2d_relgraph_v2.npz'
RUN = Path('/data2/jeongtae/relgraph_e16sel/r2nolane_attnviz_b2d_s0.npz')
EGO_HALF = (2.45, 1.05)
VMAX = 0.6                                   # shared attention colour scale
NICE = {'NonSignalizedJunctionLeftTurn': 'Non-signalized junction, left turn',
        'NonSignalizedJunctionLeftTurnEnterFlow': 'Non-signalized junction, left turn into flow',
        'NonSignalizedJunctionRightTurn': 'Non-signalized junction, right turn',
        'SignalizedJunctionLeftTurn': 'Signalized junction, left turn',
        'SignalizedJunctionLeftTurnEnterFlow': 'Signalized junction, left turn into flow',
        'SignalizedJunctionRightTurn': 'Signalized junction, right turn',
        'OppositeVehicleRunningRedLight': 'Opposite vehicle running a red light',
        'OppositeVehicleTakingPriority': 'Opposite vehicle taking priority',
        'VehicleTurningRoutePedestrian': 'Turning vehicle, pedestrian',
        'VanillaNonSignalizedTurnEncounterStopsign': 'Non-signalized turn, stop sign',
        'VanillaSignalizedTurnEncounterGreenLight': 'Signalized turn, green light',
        'VanillaSignalizedTurnEncounterRedLight': 'Signalized turn, red light',
        'MergerIntoSlowTrafficV2': 'Merge into slow traffic (v2)',
        'InterurbanAdvancedActorFlow': 'Interurban actor flow (advanced)'}


def nice(t):
    if t in NICE:
        return NICE[t]
    out = ''.join(' ' + c if c.isupper() and i and not t[i - 1].isupper() else c for i, c in enumerate(t))
    return out.strip().replace('Two Ways', '(two-way)').replace('V2', '(v2)')


def box(ax, x, y, psi, hl, hw, color, alpha=1.0, lw=0.6, ec='k', z=3, hatch=None):
    t = transforms.Affine2D().rotate(psi).translate(x, y) + ax.transData
    ax.add_patch(Rectangle((-hl, -hw), 2 * hl, 2 * hw, transform=t, facecolor=color, edgecolor=ec,
                           linewidth=lw, alpha=alpha, zorder=z, hatch=hatch))


def load_dumps(dump_dir):
    out = {}
    for f in sorted(Path(dump_dir).glob('*_draw*.npz')):
        d = np.load(f, allow_pickle=True)
        out[int(d['draw'])] = d
    return out


def example(dumps, g, run, draw, route, fail_c_of):
    d = dumps[draw]
    names = [str(r) for r in run['routes']]
    i = names.index(route)
    heldout = [int(x) for x in d['heldout']]
    assert i in heldout, f'route {route} is not held out in draw {draw}'
    k = heldout.index(i)
    rows = np.asarray(d['window_rows'][k])
    s = np.asarray(d['sal'][list(int(x) for x in d['sal_route']).index(i)], float)
    attn = {int(r): a for r, a in zip(d['rows'], d['attn'])}
    wstar = int(np.argmax(s))
    return dict(i=i, route=route, type=str(run['types'][i]), pred=float(d['pred'][k]), fail=float(fail_c_of(draw)[i]),
                rows=rows, s=s, wstar=wstar, attn=attn[int(rows[wstar])], row=int(rows[wstar]))


def draw_bev(ax, g, ex, cmap, norm, span=70.0, pad=10.0):
    row, a_w = ex['row'], ex['attn']
    lanes, lm = g['lanes'][row].astype(np.float32), g['lane_mask'][row]
    for m in np.where(lm)[0]:
        ax.plot(lanes[m, :, 0], lanes[m, :, 1], color='0.85', lw=0.8, zorder=1)
    ego = g['ego'][row].astype(np.float32)
    ta = int(np.argmin(np.abs(ego[:, 0]) + np.abs(ego[:, 1])))
    ax.plot(ego[:ta + 1, 0], ego[:ta + 1, 1], color='k', lw=1.2, zorder=4)
    ax.plot(ego[ta:, 0], ego[ta:, 1], color='k', lw=0.9, ls='--', zorder=4)
    box(ax, ego[ta, 0], ego[ta, 1], np.arctan2(ego[ta, 3], ego[ta, 2]), *EGO_HALF, color='white', lw=1.2, z=6)
    ax.text(ego[ta, 0], ego[ta, 1], 'ego', ha='center', va='center', fontsize=5, zorder=7)
    ag, am = g['agents'][row].astype(np.float32), g['agent_mask'][row]
    live = np.where(am.any(1))[0]
    w = a_w.mean(1)
    rank = live[np.argsort(-w[live])]
    for k in live:
        ts = np.where(am[k])[0]
        t0 = ts[np.argmin(np.abs(ts - ta))]
        psi = np.arctan2(ag[k, t0, 3], ag[k, t0, 2])
        veh = ag[k, t0, 7] > 0.5
        col = cmap(norm(w[k]))
        if not veh:                                        # pedestrians / bicycles: a 1.2 m disc, drawn on top
            ax.add_patch(Circle((ag[k, t0, 0], ag[k, t0, 1]), 1.2, facecolor=col, edgecolor='k', lw=0.5, zorder=5))
            continue
        for tt in ts[(ts < t0) & (ts >= t0 - 3)]:
            box(ax, ag[k, tt, 0], ag[k, tt, 1], np.arctan2(ag[k, tt, 3], ag[k, tt, 2]),
                max(ag[k, tt, 5], 0.3), max(ag[k, tt, 6], 0.3), col, alpha=0.15 + 0.12 * (tt - (t0 - 3)), lw=0.0, z=2)
        box(ax, ag[k, t0, 0], ag[k, t0, 1], psi, max(ag[k, t0, 5], 0.3), max(ag[k, t0, 6], 0.3), col,
            lw=0.6, ec='k', z=3 + (w[k] > 0.05))
    offsets = ((4, 4), (4, -9), (-22, 4))                   # rank 1 above-right, 2 below-right, 3 above-left
    for n_, k in enumerate(rank[:3]):
        ts = np.where(am[k])[0]; t0 = ts[np.argmin(np.abs(ts - ta))]
        ax.annotate(f'{n_ + 1} ({w[k]:.2f})', (ag[k, t0, 0], ag[k, t0, 1]), xytext=offsets[n_], textcoords='offset points',
                    fontsize=5.5, fontweight='bold', color='navy', zorder=8,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
    pts = np.array([[ego[ta, 0], ego[ta, 1]]] + [[ag[k, ts_[np.argmin(np.abs(ts_ - ta))], 0], ag[k, ts_[np.argmin(np.abs(ts_ - ta))], 1]]
                                                 for k in rank[:3] for ts_ in [np.where(am[k])[0]]])
    lo, hi = pts.min(0) - pad, pts.max(0) + pad                 # frame the ego and the three attended agents,
    ctr, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, span / 2)   # at least `span` m across, equal aspect
    half = np.array([half.max(), half.max()])
    ax.set_xlim(ctr[0] - half[0], ctr[0] + half[0]); ax.set_ylim(ctr[1] - half[1], ctr[1] + half[1])
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-dir', required=True)
    ap.add_argument('--examples', nargs='+', required=True, help='draw:route ...')
    ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=float, default=7.16, help='figure width in inches (IEEE double column)')
    a = ap.parse_args()
    dumps = load_dumps(a.dump_dir)
    g = np.load(GRAPH, allow_pickle=True)
    run = np.load(RUN, allow_pickle=True)
    from b2d_splits import unified_split
    Y, types = run['Y'], [str(t) for t in run['types']]
    _cache = {}

    def fail_c_of(draw):
        if draw not in _cache:
            hp, _ = unified_split(draw, sorted(set(types)), Y.shape[0])
            _cache[draw] = np.nanmean(Y[[j for j in range(Y.shape[0]) if j not in hp]], 0)
        return _cache[draw]

    exs = [example(dumps, g, run, int(e.split(':')[0]), e.split(':')[1], fail_c_of) for e in a.examples]
    n = len(exs)
    cmap, norm = plt.get_cmap('magma_r'), colors.Normalize(0.0, VMAX)
    fig = plt.figure(figsize=(a.width, a.width / n * 1.0 + 0.75))
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 0.28], hspace=0.05, wspace=0.08,
                          left=0.035, right=0.905, top=0.86, bottom=0.11)
    for c, ex in enumerate(exs):
        ax = fig.add_subplot(gs[0, c])
        draw_bev(ax, g, ex, cmap, norm)
        ax.set_title(f'{nice(ex["type"])}\n' + r'$\tilde b$' + f' = {ex["pred"]:+.2f},  fail rate {ex["fail"]:.2f}', fontsize=6.5, pad=3)
        axs = fig.add_subplot(gs[1, c])
        t = np.arange(len(ex['s'])) * 2.0
        axs.fill_between(t, 0, ex['s'], color='#b03060', alpha=0.18)
        axs.plot(t, ex['s'], color='#b03060', lw=0.9)
        axs.axvline(ex['wstar'] * 2.0, color='r', ls='--', lw=0.7)
        axs.set_yticks([]); axs.tick_params(labelsize=5, length=2, pad=1)
        axs.set_xlim(-0.5, max(t[-1], 2.0) + 0.5); axs.set_ylim(0, ex['s'].max() * 1.15)
        axs.set_xlabel('time along the route (s)', fontsize=5.5, labelpad=1)
        if c == 0:
            axs.set_ylabel(r'$\|\partial f_\phi/\partial z_w\|$', fontsize=5.5, labelpad=1)
        for sp in axs.spines.values():
            sp.set_linewidth(0.5)
    cax = fig.add_axes([0.915, 0.40, 0.011, 0.44])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label('attention weight', fontsize=5.5, labelpad=2)
    cb.ax.tick_params(labelsize=5, length=2)
    cb.set_ticks([0, 0.2, 0.4, 0.6]); cb.set_ticklabels(['0', '.2', '.4', r'$\geq$.6'])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
    if a.out.endswith('.pdf'):
        fig.savefig(a.out[:-4] + '.png', dpi=200)
    print('wrote', a.out, '|', ', '.join(f'{e["route"]} [{e["type"]}] w*={e["wstar"] + 1}/{len(e["s"])}' for e in exs))


if __name__ == '__main__':
    main()
