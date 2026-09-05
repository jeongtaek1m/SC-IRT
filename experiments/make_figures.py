#!/usr/bin/env python3
"""Figures from the results jsons (no recomputation).

  figs/fig_cost_error.{pdf,png}   rollouts vs SR-MAE for the four bank orders
                                  under the common readout (continuous curves
                                  from the adaptive tracks) + ATDrive's
                                  risk-target stops (eps = .05, .03), per K_cal
  figs/fig_kb_map.{pdf,png}       K_cal x B map of ATDrive minus the best
                                  native baseline (Table 1)

    python experiments/make_figures.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atdrive.bayes import stop_at

RES = Path(os.environ.get('ATDRIVE_RESULTS_DIR', ROOT / 'results'))
FIGS = Path(os.environ.get('ATDRIVE_FIGS_DIR', ROOT / 'figs'))
KCALS = (4, 8, 12)
BGRID = (30, 55, 110, 165)
ORD = ('ATDrive', 'Fluid', 'metabench', 'Random')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    A = json.load(open(RES / 'adaptive.json'))
    F = json.load(open(RES / 'up_frontier.json'))
    FIGS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(KCALS), figsize=(12, 3.4), sharey=True)
    ts = np.arange(1, min(len(r['ATDrive']['Shat']) for r in A) + 1)   # shortest track: trajectories run through the bank
    for ax, K in zip(axes, KCALS):
        rs = [r for r in A if r['K'] == K]
        for o in ORD:
            ys = [np.mean([abs(r[o]['Shat'][t - 1] - r['SR']) for r in rs]) for t in ts]
            ax.plot(ts, ys, lw=1.8 if o == 'ATDrive' else 1.0, label=o)
        pts = []
        cal = json.load(open(RES / 'risk_cal.json'))
        for eps in (0.05, 0.03):
            st = [stop_at(cal[f"{r['seed']}|{K}|ATDrive"] * np.array(r['ATDrive']['R1']), eps) for r in rs]
            pts.append((np.mean(st), np.mean([abs(r['ATDrive']['Shat'][t - 1] - r['SR']) for r, t in zip(rs, st)])))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'k--', marker='s', ms=4, label='ATDrive, c*R1 <= eps (.05, .03)')
        ax.set_title(f'K_cal = {K}')
        ax.set_xlabel('rollouts')
        ax.grid(alpha=.3)
    axes[0].set_ylabel('SR-MAE')
    axes[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'fig_cost_error.{ext}', dpi=200)
    base = [m for m in F[0]['err'] if m != 'ATDrive']
    Z = np.zeros((len(KCALS), len(BGRID)))
    for i, K in enumerate(KCALS):
        for k, B in enumerate(BGRID):
            ours = np.mean([r['err']['ATDrive'][str(B)] for r in F if r['K'] == K])
            best = min(np.mean([r['err'][m][str(B)] for r in F if r['K'] == K]) for m in base)
            Z[i, k] = ours - best
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    v = np.abs(Z).max()
    im = ax.imshow(Z, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
    ax.set_xticks(range(len(BGRID)))
    ax.set_xticklabels(BGRID)
    ax.set_yticks(range(len(KCALS)))
    ax.set_yticklabels([f'K_cal={K}' for K in KCALS])
    ax.set_xlabel('rollout budget B')
    for i in range(len(KCALS)):
        for k in range(len(BGRID)):
            ax.text(k, i, f'{Z[i, k]:+.3f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, label='ATDrive - best baseline\n(SR-MAE)')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'fig_kb_map.{ext}', dpi=200, bbox_inches='tight')
    print('figures written to', FIGS)


if __name__ == '__main__':
    main()
