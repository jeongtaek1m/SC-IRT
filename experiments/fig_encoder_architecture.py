#!/usr/bin/env python3
"""Block diagram of the route encoder of record (RelGraph R2-noLane) with the formula each block computes,
drawn from the code in encoder/harness/r2_graph.py and r0_ego.py (shapes and hyper-parameters verbatim).

    python experiments/fig_encoder_architecture.py --out results/figs/encoder_architecture.pdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = {'in': '#f3f3f3', 'ego': '#dbe9f6', 'win': '#e2f0d9', 'pool': '#fff2cc', 'head': '#f8cecc', 'prob': '#e1d5e7', 'eval': '#ffe6cc'}


def box(ax, x, y, w, h, title, lines, fc, fs=7.4, tfs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec='0.25', lw=0.8, zorder=2))
    ax.text(x + w / 2, y + h - 0.13, title, ha='center', va='top', fontsize=tfs, fontweight='bold', zorder=3)
    ax.text(x + 0.12, y + h - 0.42, '\n'.join(lines), ha='left', va='top', fontsize=fs, zorder=3, linespacing=1.35)


def arrow(ax, p, q, text=None, fs=6.8, color='0.2', rad=0.0, lpos=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9, lw=0.9, color=color, zorder=1,
                                 connectionstyle=f'arc3,rad={rad}'))
    if text:
        x, y = lpos if lpos else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.09)
        ax.text(x, y, text, ha='center', va='bottom', fontsize=fs, color=color, zorder=4,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/figs/encoder_architecture.pdf')
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(14, 9.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9.2); ax.axis('off')
    ax.text(7, 9.05, 'ATDrive route encoder of record (RelGraph R2-noLane): what each block computes',
            ha='center', va='top', fontsize=11.5, fontweight='bold')
    ax.text(7, 8.72, 'd = 64, 4 heads of 16, softmin temperature tau = 0.5, 90,177 effective parameters; AdamW lr 1e-3, weight decay 0.1, '
            'batch 64 routes, 30 epochs; trained per draw on the 180 calibration-type routes x 12 calibration planners, 3 seeds',
            ha='center', va='top', fontsize=8, color='0.3')

    box(ax, 0.2, 5.1, 4.1, 3.3, 'Input: reference rollout (PDM-Lite), 2 Hz', [
        r'route ego sequence $x^{ego}_t$, t = 1..L  (L: p50 36, max 396)',
        r'   9-d: $[v,\,a,\,\dot\psi,\,|a|,\,|\dot\psi|,\,\mathrm{cmd}\in\{0,1\}^4]$',
        r'   windows de-duplicated, headings chained',
        r'windows w = 1..W: 12 steps, stride 4  (W: p50 7, max 97)',
        r'   ego steps $e_{w,t}$ (9-d, window-local), command $c_w$',
        r'   agents $a_{w,k,t}$, k $\leq$ 48, t $\leq$ 12, 8-d:',
        r'   $[\Delta x,\Delta y,\cos\Delta\psi,\sin\Delta\psi,v,\ell/2,w/2,\mathrm{veh}]$',
        r'   in the ego frame of the window anchor; masks $m_{w,k,t}$',
        "standardised with the training routes' statistics",
        r'no lane / map tensors (removed before any tensor is built)'], C['in'], fs=7.2)

    box(ax, 4.7, 6.55, 4.4, 1.85, 'Ego branch (shared step MLP + set pooling)', [
        r'$\varphi_e$: Linear(9$\to$64) SiLU Linear(64$\to$64) SiLU, per step',
        r'$z^{ego}=\mathrm{Pool}_t[\varphi_e(x^{ego}_t)]\in\mathbb{R}^{256}$',
        r'Pool = [mean, max, softmin$_\tau$, std] over the valid steps,',
        r'softmin$_\tau(u)=-\tau\log\sum_t e^{-u_t/\tau}$'], C['ego'], fs=7.4)

    box(ax, 4.7, 2.45, 4.4, 3.85, 'Window encoder (64-d vector per window; agents = a set)', [
        r'$\psi_a$: Linear(8$\to$64) SiLU Linear(64$\to$64), per (agent, step)',
        r'   $h_k=\mathrm{LN}\left(W_o[\,\mathrm{mean}_t\psi_a(a_{k,t});\,\mathrm{max}_t\psi_a(a_{k,t})]\right)$, dead agent $\to$ 0',
        r'query from ego motion + command  $u_t=\varphi_q(e_{w,t})$ (own weights)',
        r'   $q=\mathrm{MLP}_q([\,\mathrm{mean}_t u_t;\,\mathrm{max}_t u_t;\,c_w])$,  132$\to$64$\to$64',
        r'readout attention over the live agents, 4 heads (the only attention):',
        r'   $\alpha^{(h)}_k=\mathrm{softmax}_k\left(q^{(h)}\cdot W_K^{(h)}(h_k+\tau_{ag})/\sqrt{16}\right)$',
        r'   $r=\left[\sum_k\alpha^{(h)}_k\,W_V^{(h)}(h_k+\tau_{ag})\right]_{h=1..4}\in\mathbb{R}^{64}$',
        r'window vector  $z_w=\mathrm{MLP}_z([\,r;\,\mathrm{mean}_k(h_k+\tau_{ag})])$,  128$\to$64$\to$64',
        r'(the lane-side blocks of the lane-carrying R2 get all-False masks',
        r' and contribute exactly 0)'], C['win'], fs=7.1)

    box(ax, 9.5, 6.95, 4.3, 1.45, 'Route pooling of the windows', [
        r'$z^{win}=\mathrm{Pool}_w[z_w]\in\mathbb{R}^{256}$',
        r'same [mean, max, softmin$_\tau$, std]'], C['pool'], fs=7.4)
    box(ax, 9.5, 4.85, 4.3, 1.7, 'Head', [
        r'$f_\phi(x)=W_2\,\mathrm{SiLU}\left(W_1\,\mathrm{LN}([z^{ego};z^{win}])\right)$,  512$\to$64$\to$1',
        r'$\log\sigma_r$: one learned scalar, shared by all routes (init $-0.5$)'], C['head'], fs=7.4)

    box(ax, 0.2, 0.2, 8.9, 1.95, r'Difficulty distribution and training objective ($\hat\theta$ fixed from the Rasch fit of block A)', [
        r'$b_i\,|\,x_i\sim\mathcal{N}(f_\phi(x_i),\sigma_r^2)$,    $P(y_{ij}=1\,|\,b_i)=\sigma(\hat\theta_j-b_i)$  (Rasch, calibration planner j)',
        r'$\mathcal{L}=-\frac{1}{N_A}\sum_i\log\int\mathcal{N}(b\,|\,f_\phi(x_i),\sigma_r^2)\prod_{j\in\mathcal{J}_i}\mathrm{Bern}(y_{ij}\,|\,\sigma(\hat\theta_j-b))\,db+\lambda(\log\sigma_r)^2$,  $\lambda=0.05$',
        r'15-point Gauss-Hermite: $\int\approx\sum_{k=1}^{15}w_k\prod_j\mathrm{Bern}\left(y_{ij}\,|\,\sigma(\hat\theta_j-f_\phi(x_i)-\sigma_r\xi_k)\right)$;  $\phi$ and $\sigma_r$ trained jointly'], C['prob'], fs=7.4)

    box(ax, 9.5, 0.2, 4.3, 4.2, 'Use at evaluation (new route, no responses)', [
        r'the predicted prior replaces the response posterior:',
        r'$\tilde m_i(\theta)=\int\sigma(\theta-b)\,\mathcal{N}(b\,|\,f_\phi(x_i),\sigma_r^2)\,db$',
        r'tabulated on the ability grid; $\tilde m_i(\theta+u_g)$ with the',
        r'testlet effect $u_g\sim\mathcal{N}(0,\sigma_g^2)$ of the new type',
        r'US (Table V): $f_\phi$ vs the block-C failure rate (OOF)',
        r'UPS (Table IV): $\tilde m_i$ transports the probe posterior',
        r'$q_t(\theta)$ to the new routes; SR = posterior median',
        r'of the total success count'], C['eval'], fs=7.4)

    arrow(ax, (4.3, 7.5), (4.7, 7.5), r'$x^{ego}_{1..L}$', lpos=(4.5, 7.58))
    arrow(ax, (4.3, 5.6), (4.7, 4.9), r'$e_{w,t},\,c_w,\,a_{w,k,t}$', rad=0.15, lpos=(4.5, 5.0))
    arrow(ax, (9.1, 7.5), (9.5, 6.1), r'$z^{ego}$ (256)', rad=-0.3, lpos=(8.75, 6.36))
    arrow(ax, (9.1, 4.5), (9.5, 7.3), r'$z_w$, w = 1..W', rad=-0.35, lpos=(8.85, 3.05))
    arrow(ax, (11.65, 6.95), (11.65, 6.55), r'$z^{win}$ (256)', lpos=(12.4, 6.62))
    arrow(ax, (11.65, 4.85), (11.65, 4.4), r'$f_\phi(x),\,\sigma_r$  (inference)', lpos=(12.7, 4.5))
    arrow(ax, (9.5, 5.2), (9.1, 1.9), r'training', rad=0.3, lpos=(9.25, 2.72))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches='tight')
    if a.out.endswith('.pdf'):
        fig.savefig(a.out[:-4] + '.png', dpi=170, bbox_inches='tight')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
