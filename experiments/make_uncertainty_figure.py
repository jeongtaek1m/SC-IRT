#!/usr/bin/env python3
"""Difficulty uncertainty: the marginal ICC against the plug-in point ICC.

  figs/fig_uncertainty.{pdf,png}
    (a) the exact conditional difficulty posterior p(b_s | A, theta_hat) for three
        routes: one with mixed responses, one every planner failed, one every
        planner passed. b_hat marked.
    (b) the ICCs those posteriors imply. Dashed sigma(theta - b_hat) is the
        plug-in curve every baseline uses; solid m_s(theta) = E_b[sigma(theta - b)]
        is what DriveAT uses.
    (c) the consequence for the reported metric: predicted benchmark success rate
        (1/n) sum_s of each curve, as a function of ability, with the calibration
        planners at their fitted ability and observed SR.
    (d) how the gap shrinks as the calibration panel grows, K_cal = 4, 8, 12, 16.

Bank: all 220 routes, calibrated from K_cal planners of draw 0 (DRIVEAT_FIG_KCAL,
default 4 — the sparsest panel the protocol uses, where the difficulty posterior
is widest).

    python experiments/make_uncertainty_figure.py
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from driveat.b2d import Panel
from driveat.splits import up_split
from driveat.calibration import calibrate
from driveat.curves import XG, BG, curves_from_posterior, sig

FIGS = Path(os.environ.get('DRIVEAT_FIGS_DIR', ROOT / 'figs'))
KCAL = int(os.environ.get('DRIVEAT_FIG_KCAL', 4))
KGRID = (4, 8, 12, 16)
DRAW = 0
XLIM = (-4, 4)
BLIM = (-6, 6)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                         'legend.fontsize': 7.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8})

    panel = Panel()
    routes = list(panel.allr)
    typ = np.array([panel.sn[r] for r in routes])

    def panel_of(K):
        """The K_cal calibration planners of draw DRAW, exactly as the UP protocol picks them."""
        if K >= panel.J:
            return list(range(panel.J))
        hp, _ = up_split(DRAW, panel.utypes, panel.J)
        c12 = [c for c in range(panel.J) if c not in hp]
        rs = np.random.RandomState(9000 + DRAW * 100 + K * 10)
        return sorted(np.array(c12)[rs.choice(len(c12), K, replace=False)].tolist())

    cols = panel_of(KCAL)
    f = calibrate(panel.Y, routes, cols, mode='1pl', device='cpu', types=typ)
    b, th, W = f['b'], f['th'], f['W']
    M = curves_from_posterior(W)                  # marginal ICC, (361, n)
    P = sig(XG[:, None] - b[None, :])             # plug-in ICC
    sd = np.sqrt(W @ (BG ** 2) - (W @ BG) ** 2)
    pas = np.array([np.mean([panel.Y[(r, c)] for c in cols if (r, c) in panel.Y]) for r in routes])
    k = (XG >= XLIM[0]) & (XG <= XLIM[1])
    kb = (BG >= BLIM[0]) & (BG <= BLIM[1])

    mixed = np.where((pas > 0.2) & (pas < 0.8))[0]
    sel = [(int(mixed[np.argmin(sd[mixed])]), 'mixed responses', 'tab:blue'),
           (int(np.argmax(np.where(pas == 0, sd, -9e9))), 'all planners fail', 'tab:red'),
           (int(np.argmax(np.where(pas == 1, sd, -9e9))), 'all planners pass', 'tab:green')]

    fig, ax = plt.subplots(1, 4, figsize=(13.6, 2.9))

    for i, lab, col in sel:
        ax[0].plot(BG[kb], W[i][kb] / (BG[1] - BG[0]), color=col, lw=1.8,
                   label=f'{lab} (SD {sd[i]:.2f})')
        ax[0].axvline(b[i], color=col, ls=':', lw=1.1)
    ax[0].set_xlabel(r'route difficulty $b_s$')
    ax[0].set_ylabel('posterior density')
    ax[0].set_title(r'(a) $p(b_s \mid \mathcal{A}, \hat{\theta})$ at $K_{cal}=%d$;  dotted: $\hat b_s$' % KCAL)
    ax[0].set_xlim(*BLIM)
    ax[0].legend(frameon=False, loc='upper left')

    for i, lab, col in sel:
        ax[1].plot(XG[k], P[k, i], color=col, ls='--', lw=1.4)
        ax[1].plot(XG[k], M[k, i], color=col, lw=2.0)
    ax[1].plot([], [], 'k--', lw=1.4, label=r'plug-in $\sigma(\theta-\hat b_s)$')
    ax[1].plot([], [], 'k-', lw=2.0, label=r'marginal $m_s(\theta)$ (ours)')
    ax[1].set_xlabel(r'planner ability $\theta$')
    ax[1].set_ylabel(r'$P(\mathrm{success})$')
    ax[1].set_title('(b) the ICCs they imply')
    ax[1].set_xlim(*XLIM)
    ax[1].set_ylim(0, 1)
    ax[1].legend(frameon=False, loc='upper left')

    srM, srP = M.mean(1), P.mean(1)
    ax[2].plot(XG[k], srP[k], 'k--', lw=1.4, label='plug-in')
    ax[2].plot(XG[k], srM[k], 'k-', lw=2.0, label='marginal (ours)')
    obs = np.array([np.mean([panel.Y[(r, c)] for r in routes if (r, c) in panel.Y]) for c in cols])
    ax[2].plot(th, obs, 'o', ms=4, mfc='none', mec='crimson', mew=1.2,
               label=f'{len(cols)} calibration planners')
    j = np.argmax(np.abs(srM - srP)[k])
    ax[2].annotate(f'max gap {np.abs(srM - srP)[k][j]:.3f} SR',
                   xy=(XG[k][j], srM[k][j]), xytext=(0.05, 0.72), textcoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', lw=.8), fontsize=7.5)
    ax[2].set_xlabel(r'planner ability $\theta$')
    ax[2].set_ylabel('predicted benchmark SR')
    ax[2].set_title('(c) effect on the reported metric')
    ax[2].set_xlim(*XLIM)
    ax[2].set_ylim(0, 1)
    ax[2].legend(frameon=False, loc='upper left')

    med, p90, srg = [], [], []
    for K in KGRID:
        fk = calibrate(panel.Y, routes, panel_of(K), mode='1pl', device='cpu', types=typ)
        Mk = curves_from_posterior(fk['W'])
        Pk = sig(XG[:, None] - fk['b'][None, :])
        g = np.abs(Mk - Pk).max(0)
        med.append(np.median(g))
        p90.append(np.percentile(g, 90))
        srg.append(np.abs(Mk.mean(1) - Pk.mean(1))[k].max())
    ax[3].fill_between(KGRID, med, p90, color='tab:blue', alpha=.18)
    ax[3].plot(KGRID, med, 'o-', color='tab:blue', lw=1.8, ms=4, label='per route, median')
    ax[3].plot(KGRID, p90, 'v--', color='tab:blue', lw=1.0, ms=4, label='per route, 90th pct')
    ax[3].plot(KGRID, srg, 's-', color='k', lw=1.8, ms=4, label='benchmark SR curve')
    ax[3].axvline(KCAL, color='crimson', ls=':', lw=1.1)
    ax[3].set_xticks(KGRID)
    ax[3].set_xlabel(r'calibration planners $K_{cal}$')
    ax[3].set_ylabel(r'$\max_\theta |m_s - \sigma(\theta-\hat b_s)|$')
    ax[3].set_title('(d) the sparser the panel, the larger the gap')
    ax[3].set_ylim(0, None)
    ax[3].legend(frameon=False)

    for a_ in ax[:3]:
        a_.grid(alpha=.3)
    ax[3].grid(alpha=.3)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'fig_uncertainty.{ext}', dpi=300, bbox_inches='tight')

    gap = np.abs(M - P).max(0)
    print(f'routes {len(routes)}, sigma_b {f["sigma_b"]}, sigma_g {f["sigma_g"]:.2f}')
    print(f'posterior SD(b): mixed {np.median(sd[(pas > 0) & (pas < 1)]):.2f}, '
          f'saturated {np.median(sd[(pas == 0) | (pas == 1)]):.2f}')
    print(f'max |marginal - plug-in| per route: median {np.median(gap):.4f}, '
          f'90th pct {np.percentile(gap, 90):.4f}, max {gap.max():.4f}')
    print(f'  on mixed routes {np.median(gap[(pas > 0) & (pas < 1)]):.4f}, '
          f'on saturated routes {np.median(gap[(pas == 0) | (pas == 1)]):.4f} '
          f'({int(((pas == 0) | (pas == 1)).sum())} of {len(routes)} routes)')
    print(f'benchmark-SR curve: max gap {np.abs(srM - srP)[k].max():.4f} SR at theta '
          f'{XG[k][int(np.argmax(np.abs(srM - srP)[k]))]:+.2f}')
    print(f'written: {FIGS / "fig_uncertainty.pdf"} / .png')


if __name__ == '__main__':
    np.random.seed(0)
    main()
