#!/usr/bin/env python3
"""Item characteristic curves of the Bench2Drive route bank.

  figs/fig_icc.{pdf,png}   (a) every route's ICC, coloured by difficulty, with
                               the 16 planner abilities on the axis
                           (b) what the difficulty posterior does: the point
                               curve sigma(theta - b_hat) against the
                               posterior-marginal m_s(theta) for a
                               well-determined, a sparse and an all-fail route
                           (c) the testlet: one scenario type's five routes and
                               the same curves shifted by u = +-sigma_g
                           (d) ICC averaged over the five routes of each
                               scenario type, hardest and easiest labelled

The bank is the whole benchmark calibrated from all 16 planners (the object a
new planner is evaluated against; PROTOCOL section 3).

    python experiments/make_icc_figure.py
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from driveat.b2d import Panel
from driveat.calibration import calibrate
from driveat.curves import XG, BG, curves_from_posterior, sig

FIGS = Path(os.environ.get('DRIVEAT_FIGS_DIR', ROOT / 'figs'))
XLIM = (-4, 4)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    panel = Panel()
    routes = list(panel.allr)
    cols = list(range(panel.J))
    typ = np.array([panel.sn[r] for r in routes])
    f = calibrate(panel.Y, routes, cols, mode='1pl', device='cpu', types=typ)
    b, th, W, sg = f['b'], f['th'], f['W'], f['sigma_g']
    M = curves_from_posterior(W)                       # (361, n) marginal ICCs
    P = sig(XG[:, None] - b[None, :])                  # (361, n) point ICCs
    nobs = np.array([sum((r, c) in panel.Y for c in cols) for r in routes])
    pas = np.array([np.mean([panel.Y[(r, c)] for c in cols if (r, c) in panel.Y]) for r in routes])
    k = (XG >= XLIM[0]) & (XG <= XLIM[1])
    print(f'{len(routes)} routes, {panel.J} planners; b in [{b.min():.2f}, {b.max():.2f}], '
          f'sigma_b {f["sigma_b"]}, sigma_g {sg:.2f}')

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.2))
    norm = colors.Normalize(vmin=np.percentile(b, 2), vmax=np.percentile(b, 98))
    sm = cm.ScalarMappable(norm=norm, cmap='viridis')

    a0 = ax[0, 0]
    for i in np.argsort(b):
        a0.plot(XG[k], M[k, i], lw=.5, alpha=.45, color=sm.to_rgba(b[i]))
    a0.plot(XG[k], sig(XG[k] - np.median(b)), 'k-', lw=2, label=f'median route (b = {np.median(b):+.2f})')
    for t_ in th:
        a0.plot([t_, t_], [-.035, .015], color='crimson', lw=1.1, clip_on=False)
    a0.set_title(f'(a) all {len(routes)} route ICCs, coloured by difficulty\n'
                 f'red ticks: the {len(cols)} calibration planners', fontsize=9)
    a0.legend(fontsize=7, frameon=False, loc='upper left')
    fig.colorbar(sm, ax=a0, label='difficulty b', pad=.01)

    a1 = ax[0, 1]
    sd = np.sqrt(W @ (BG ** 2) - (W @ BG) ** 2)
    mixed = np.where((pas > 0) & (pas < 1))[0]
    well = int(mixed[np.argmin(sd[mixed])])
    allf = int(np.argmax(np.where(pas == 0, sd, -9e9)))
    allp = int(np.argmax(np.where(pas == 1, sd, -9e9)))
    for i, lab, col in ((well, 'mixed responses', 'tab:blue'),
                        (allp, 'every planner passes', 'tab:green'),
                        (allf, 'every planner fails', 'tab:red')):
        a1.plot(XG[k], P[k, i], color=col, ls='--', lw=1.3)
        a1.plot(XG[k], M[k, i], color=col, lw=2.2,
                label=f'{lab}: b_hat {b[i]:+.2f}, SD(b) {sd[i]:.2f}, '
                      f'max gap {np.abs(M[k, i] - P[k, i]).max():.2f}')
    a1.plot([], [], 'k--', lw=1.3, label='point curve sigma(theta - b_hat)')
    a1.plot([], [], 'k-', lw=2.2, label='marginal m_s(theta) (ours)')
    a1.set_title('(b) dashed: point curve at b_hat; solid: marginal over the\n'
                 'difficulty posterior. Saturated routes keep a one-sided tail', fontsize=9)
    a1.legend(fontsize=7, frameon=False, loc='upper left')

    a2 = ax[1, 0]
    tsel = sorted(set(typ))[int(np.argmax([b[typ == t].std() for t in sorted(set(typ))]))]
    ix = np.where(typ == tsel)[0]
    for i in ix:
        a2.plot(XG[k], M[k, i], color='tab:blue', lw=1.2, alpha=.8)
    mt = M[:, ix].mean(1)
    for u, ls in ((+sg, ':'), (-sg, ':')):
        a2.plot(XG[k], np.interp(XG[k] - u, XG, mt), 'k', ls=ls, lw=1.4)
    a2.plot(XG[k], mt[k], 'k-', lw=2)
    a2.set_title(f'(c) testlet: the 5 routes of "{tsel}" (blue)\n'
                 f'black: their mean; dotted: the same mean shifted by a planner\'s\n'
                 f'type effect u = +-sigma_g = {sg:.2f}', fontsize=8)

    a3 = ax[1, 1]
    ts = sorted(set(typ))
    bt = np.array([b[typ == t].mean() for t in ts])
    for t_, bb in zip(ts, bt):
        a3.plot(XG[k], M[np.ix_(k, typ == t_)].mean(1), lw=.9, alpha=.6, color=sm.to_rgba(bb))
    for i in (int(np.argmax(bt)), int(np.argmin(bt))):
        a3.plot(XG[k], M[np.ix_(k, typ == ts[i])].mean(1), lw=2.2, color=sm.to_rgba(bt[i]),
                label=f'{ts[i]} (b = {bt[i]:+.2f})')
    a3.set_title(f'(d) ICC averaged over each of the {len(ts)} scenario types', fontsize=9)
    a3.legend(fontsize=7, frameon=False, loc='upper left')

    for a_ in ax.ravel():
        a_.set_xlim(*XLIM)
        a_.set_ylim(0, 1)
        a_.set_xlabel('planner ability theta')
        a_.set_ylabel('P(success)')
        a_.grid(alpha=.3)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'fig_icc.{ext}', dpi=200, bbox_inches='tight')
    print(f'written: {FIGS / "fig_icc.pdf"} / .png')


if __name__ == '__main__':
    np.random.seed(0)
    main()
