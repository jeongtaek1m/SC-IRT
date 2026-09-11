#!/usr/bin/env python3
"""How the difficulty distribution N(f_phi(x_i), sigma_r^2) is learned, on real routes of one draw.

Reads the --dump-attn file of a draw (in-sample f_phi of the 180 training routes, the draw's Rasch theta_hat and
b_hat, the learned sigma_r) and the response panel. For three training routes (all planners succeed / mixed /
all fail) it plots the response likelihood L_i(b) = prod_j Bern(y_ij | sigmoid(theta_hat_j - b)), the learned
Gaussian, their product, and the per-route objective -log int N(b; f, sigma_r^2) L_i(b) db as a function of f
with the encoder's value marked; then the 180 in-sample f_phi against the Rasch b_hat and the residual spread
against sigma_r.

    python experiments/fig_encoder_learning_example.py --dump <attn_s0_draw0.npz> --out results/figs/encoder_learning_example.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import spearmanr

RUN = Path('/data2/jeongtae/relgraph_e16sel/r2nolane_attnviz_b2d_s0.npz')
BG = np.linspace(-9, 9, 721)


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def loglik(ys, th, b):
    """log prod_j Bern(y_j | sigmoid(theta_j - b)) on the grid b; ys: successes (nan = missing)."""
    out = np.zeros_like(b)
    for y, t in zip(ys, th):
        if np.isnan(y):
            continue
        p = sig(t - b)
        out += y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)
    return out


def objective(ys, th, f, sigma, gx, gw):
    """-log sum_k w_k prod_j Bern(y_j | sigmoid(theta_j - f - sigma xi_k)) for every f in the array."""
    out = np.zeros_like(f)
    for i, fi in enumerate(f):
        ll = loglik(ys, th, fi + sigma * gx)
        m = ll.max()
        out[i] = -(m + np.log((gw * np.exp(ll - m)).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--out', default='results/figs/encoder_learning_example.png')
    a = ap.parse_args()
    d = np.load(a.dump, allow_pickle=True)
    run = np.load(RUN, allow_pickle=True)
    Y = run['Y']                                                 # (J, R): 1 = failure, nan = missing
    names = [str(r) for r in run['routes']]
    types = [str(t) for t in run['types']]
    keepJ, th, bh, tr = d['keepJ'], d['theta'], d['b_hat'], d['train_idx']
    f_tr, sigma = d['pred_train'], float(d['sigma'])
    S = 1 - Y[keepJ]                                             # successes of the 12 calibration planners
    n_succ = np.nansum(S[:, tr], 0); n_obs = (~np.isnan(S[:, tr])).sum(0)
    gx, gw = hermegauss(15); gw = gw / gw.sum()

    # three training routes: all pass, closest to half, all fail (largest |b_hat| among the extremes)
    full = np.where(n_succ == n_obs)[0]; none = np.where(n_succ == 0)[0]
    mixed = np.argsort(np.abs(n_succ / n_obs - 0.5))[:1]
    picks = [full[np.argmin(bh[full])], mixed[0], none[np.argmax(bh[none])]]
    labels = ['all 12 planners succeed', f'{int(n_succ[mixed[0]])} of {int(n_obs[mixed[0]])} succeed', 'all 12 planners fail']

    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), gridspec_kw={'width_ratios': [1, 1, 1, 1.15]})
    for c, (k, lab) in enumerate(zip(picks, labels)):
        i = tr[k]
        ys = S[:, i]
        ll = loglik(ys, th, BG); L = np.exp(ll - ll.max())
        prior = np.exp(-0.5 * ((BG - f_tr[k]) / sigma) ** 2)
        post = L * prior; post = post / post.max()
        ax = axes[0, c]
        ax.fill_between(BG, 0, L, color='0.75', alpha=0.6, label=r'response likelihood $\prod_j \mathrm{Bern}(y_{ij}\,|\,\sigma(\hat\theta_j-b))$ (scaled)')
        ax.plot(BG, prior, color='#1f77b4', lw=1.6, label=r'learned prior $\mathcal{N}(f_\phi(x_i),\sigma_r^2)$ (scaled)')
        ax.plot(BG, post, color='#d62728', lw=1.2, ls='--', label='product (scaled)')
        ax.axvline(bh[k], color='k', lw=0.8, ls=':', label=r'Rasch point estimate $\hat b_i$')
        ax.axvline(f_tr[k], color='#1f77b4', lw=0.8, ls='-.')
        for t in th:
            ax.plot([t], [-0.04], marker='|', color='0.3', ms=6, clip_on=False)
        ax.set_xlim(-8, 8); ax.set_ylim(-0.05, 1.12); ax.set_yticks([])
        ax.set_title(f'route_{names[i]}  [{types[i]}]\n{lab};  $\\hat b_i$ = {bh[k]:+.2f},  $f_\\phi(x_i)$ = {f_tr[k]:+.2f},  $\\sigma_r$ = {sigma:.2f}', fontsize=8)
        ax.set_xlabel('difficulty b   (ticks below the axis: the 12 calibration abilities $\\hat\\theta_j$)', fontsize=7.5)
        ax.tick_params(labelsize=7)
        # objective as a function of f
        F = np.linspace(-8, 8, 321)
        J = objective(ys, th, F, sigma, gx, gw)
        ax2 = axes[1, c]
        ax2.plot(F, J, color='#b03060', lw=1.5)
        ax2.axvline(f_tr[k], color='#1f77b4', lw=0.9, ls='-.', label=r'encoder $f_\phi(x_i)$ after training')
        ax2.axvline(bh[k], color='k', lw=0.8, ls=':', label=r'$\hat b_i$')
        jmin = F[np.argmin(J)]
        ax2.plot([jmin], [J.min()], marker='v', color='#b03060', ms=6, label=f'minimiser of this route alone (f = {jmin:+.1f})' if abs(jmin) < 7.9 else 'no finite minimiser (flattens as f moves out)')
        ax2.set_xlim(-8, 8); ax2.set_xlabel('candidate mean f of the prior', fontsize=7.5)
        ax2.set_ylabel(r'$-\log\int\mathcal{N}(b|f,\sigma_r^2)\,L_i(b)\,db$', fontsize=7.5)
        ax2.tick_params(labelsize=7); ax2.legend(fontsize=6.3, frameon=False, loc='upper center')

    # right column: the whole training block
    ax = axes[0, 3]
    frac = n_succ / np.maximum(n_obs, 1)
    sc = ax.scatter(bh, f_tr, c=frac, cmap='RdYlGn', s=14, edgecolor='k', lw=0.3)
    lim = (-6, 6); ax.plot(lim, lim, color='0.5', lw=0.8, ls='--'); ax.set_xlim(lim); ax.set_ylim(lim)
    rho = spearmanr(bh, f_tr).correlation
    ax.set_xlabel(r'Rasch point difficulty $\hat b_i$ (12 planners)', fontsize=7.5); ax.set_ylabel(r'encoder mean $f_\phi(x_i)$ (in-sample)', fontsize=7.5)
    ax.set_title(f'the 180 training routes of this draw\nSpearman {rho:+.3f};  colour = share of planners that succeed', fontsize=8)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6)
    ax.tick_params(labelsize=7)
    ax2 = axes[1, 3]
    res = f_tr - bh
    ax2.hist(res, bins=25, color='0.7', edgecolor='k', lw=0.4, density=True)
    xx = np.linspace(-5, 5, 300)
    ax2.plot(xx, np.exp(-0.5 * (xx / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)), color='#1f77b4', lw=1.5, label=f'N(0, $\\sigma_r^2$), $\\sigma_r$ = {sigma:.2f}')
    ax2.set_xlabel(r'$f_\phi(x_i)-\hat b_i$ over the training routes', fontsize=7.5); ax2.set_ylabel('density', fontsize=7.5)
    ax2.set_title(f'in-sample residual SD {res.std():.2f} vs the learned shared width {sigma:.2f}\n(held-out residuals are wider: the width is learned on the training block)', fontsize=8)
    ax2.legend(fontsize=6.5, frameon=False); ax2.tick_params(labelsize=7)
    fig.suptitle('How the difficulty distribution is learned (draw 0, seed 0): the encoder places a Gaussian of shared width where the 12 calibration responses put their likelihood',
                 fontsize=9.5)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='lower center', ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160)
    print('wrote', a.out, '| picks', [names[tr[k]] for k in picks], 'sigma', sigma, 'rho(f, b_hat)', round(rho, 3), 'residual SD', round(res.std(), 3))


if __name__ == '__main__':
    main()
