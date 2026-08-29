#!/usr/bin/env python3
"""Figures from the results jsons (no recomputation).

  figs/fig_cost_error.{pdf,png}   rollouts vs SR-MAE, J_cal = 13 and 7: fixed-budget
                                  curves (native readouts) + SC-IRT's adaptive R1 sweep
                                  (mean rollouts at each tau)
  figs/fig_jb_map.{pdf,png}       J_cal x B map of SC-IRT minus the best baseline

    python experiments/make_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scirt.bayes import stop_at

RES, FIGS = ROOT / 'results', ROOT / 'figs'
BG = [10, 20, 30, 40, 60, 80, 100, 120]
SHOW = ['Random + IRT', 'Fluid', 'metabench', 'AnchorPoints', 'tinyBenchmarks', 'SC-IRT']
TAUS = (0.06, 0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    F = json.load(open(RES / 'up_frontier.json'))
    A = json.load(open(RES / 'adaptive.json')) if (RES / 'adaptive.json').exists() else []
    FIGS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, J in zip(axes, (13, 7)):
        for m in SHOW:
            ys = [np.mean([r['err'][m][str(B)] for r in F if r['J'] == J]) for B in BG]
            ax.plot(BG, ys, marker='o', ms=3, lw=1.6 if m == 'SC-IRT' else 1.0, label=m)
        if A:
            pts = []
            for tau in TAUS:
                st = [(stop_at(r['SC-IRT']['R1'], tau), abs(r['SC-IRT']['Shat'][stop_at(r['SC-IRT']['R1'], tau) - 1] - r['SR']))
                      for r in A if r['J'] == J]
                pts.append((np.mean([s[0] for s in st]), np.mean([s[1] for s in st])))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], 'k--', marker='s', ms=3, label='SC-IRT, R1 stop (tau sweep)')
        ax.set_title(f'J_cal = {J}')
        ax.set_xlabel('rollouts')
        ax.grid(alpha=.3)
    axes[0].set_ylabel('SR-MAE')
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'fig_cost_error.{ext}', dpi=200)
    # J x B map
    base = [m for m in F[0]['err'] if m != 'SC-IRT']
    JC = (4, 7, 10, 13)
    Z = np.zeros((len(JC), len(BG)))
    for i, J in enumerate(JC):
        for k, B in enumerate(BG):
            ours = np.mean([r['err']['SC-IRT'][str(B)] for r in F if r['J'] == J])
            best = min(np.mean([r['err'][m][str(B)] for r in F if r['J'] == J]) for m in base)
            Z[i, k] = ours - best
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    v = np.abs(Z).max()
    im = ax.imshow(Z, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
    ax.set_xticks(range(len(BG)))
    ax.set_xticklabels(BG)
    ax.set_yticks(range(len(JC)))
    ax.set_yticklabels([f'J={J}' for J in JC])
    ax.set_xlabel('rollout budget B')
    for i in range(len(JC)):
        for k in range(len(BG)):
            ax.text(k, i, f'{Z[i, k]:+.3f}', ha='center', va='center', fontsize=6)
    fig.colorbar(im, ax=ax, label='SC-IRT minus best baseline (SR-MAE)')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'fig_jb_map.{ext}', dpi=200)
    print('figures written to', FIGS)


if __name__ == '__main__':
    main()
