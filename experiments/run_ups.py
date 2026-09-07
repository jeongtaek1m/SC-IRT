#!/usr/bin/env python3
"""Table 3B — UPS: unseen planner x unseen scenes on the primary protocol.

The evaluation planner runs probes on the calibrated bank (budgets
B = {30, 55, 110}); its evaluation-scale ability posterior on the theta grid comes from
those probes (12 calibration planners); its success rate on the evaluation-type routes D
(zero rollouts there) is predicted by transporting the probe posterior as
is (atdrive.bayes.transfer) through the scene-conditioned difficulty prior
N(b; b_tilde_s, sigma^2) with the testlet prior on the evaluation types:
the block-D success rate is its posterior median (MAE) and each D cell its
posterior predictive (NLL). b_tilde_s is the out-of-fold prediction of the
ENCODER OF RECORD — RelGraph R2-noLane, which has no lane graph at all (no
lane tokens, lane_feat, L2L edges, A2L candidates or route_rel; ego, command
and agents only) — for that draw, and sigma the per-draw shared residual SD it
learned on the calibration block (data/encoder/relgraph_r2nolane_s*.npz; run
s0 is canonical, runs s1-s2 give the across-run SD).

The table is repeated with two control priors, each also driving its own
Delta-R1 probe rule:
  _lane      the lane-carrying R2 earlier releases shipped as canonical
             (relgraph_r2_s*.npz) -> results/ups_lane.json
  _nospeed   R2-noLane with the ego-speed channel removed from both ego paths
             (relgraph_r2nolane_nospeed_s*.npz) -> results/ups_nospeed.json,
             a clean speed ablation OF THE ENCODER OF RECORD: it differs from
             the canonical prior in the speed channel and nothing else.

Probe policies: naive SR transfer, Random, Delta-R1 on the transported block-D
success rate (canonical — acquire for the quantity that must generalise), and
theta-EIG under the evaluation model (our own acquisition ablation).

    python experiments/run_ups.py     # ~30 min, GPU
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atdrive.b2d import Panel, DATA
from atdrive.splits import unified_split, R_DRAWS
from atdrive.calibration import calibrate
from atdrive.curves import marginal_curves
from atdrive.bayes import Bank, State, bank_from_fit, state_from, transfer
from atdrive.acquisition import eig_pick, r1_pick_transfer
# (no baseline order is used here any more: the UPS table has no comparator row)
from atdrive.metrics import paired_cluster_boot

OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
BP = (30, 55, 110)
T = max(BP)
# No comparator row. UPS asks for the success rate on scenario types the planner was
# never run on; no published method estimates that, so the table is ATDrive, its own
# acquisition ablation (theta-EIG on the same posterior), and two floors (naive SR
# transfer and a random probe order). The 2PL-Fisher row earlier releases carried was
# Fluid's item order borrowed into our transfer machine, i.e. a comparator in name only.
POL = ('Random', 'theta-EIG (abl.)', 'ATDrive (Delta-R1 on D)')
CAN = 'ATDrive (Delta-R1 on D)'
ANCHORS = (('ATDrive (Delta-R1 on D)', 30, .0922), ('ATDrive (Delta-R1 on D)', 55, .0920),
           ('ATDrive (Delta-R1 on D)', 110, .0926), ('Random', 30, .1140), ('theta-EIG (abl.)', 55, .0943))
# pinned on the lane-free run of record (results/ups.json); tolerance .003 = about two across-run SDs
RUNS = (0, 1, 2)
# suffix -> (row label, the shipped npz stem).  The encoder of RECORD is lane-free.
ENC = {'': ('RelGraph R2-noLane (encoder of record)', 'relgraph_r2nolane'),
       '_lane': ('R2 with the lane graph (control)', 'relgraph_r2'),
       '_nospeed': ('R2-noLane, ego speed removed (control)', 'relgraph_r2nolane_nospeed')}


def main():
    panel = Panel()
    RES = {e: {p: {B: {'mae': [], 'nll': []} for B in BP} for p in ('naive',) + POL} for e in ENC}
    RES3 = {e: {run: {p: {B: [] for B in BP} for p in POL} for run in RUNS} for e in ENC}
    JS = []  # planner id per evaluation, parallel to RES[*][*][*]['mae']
    RELG = {(e, run): np.load(DATA / 'encoder' / f'{ENC[e][1]}_s{run}.npz', allow_pickle=True)
            for e in ENC for run in RUNS}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        calR, newR = panel.split_routes(ht)
        typ = np.array([panel.sn[r] for r in calR])
        typD = np.array([panel.sn[r] for r in newR])
        f1 = calibrate(panel.Y, calR, cols, mode='1pl', types=typ)
        f2 = calibrate(panel.Y, calR, cols, mode='2pl', sigma_b=f1['sigma_b'])
        PRI = {}
        for er in RELG:
            pz = RELG[er]
            rt = [str(x) for x in pz[f'draw{seed}_rt']]
            lut = {rt[k]: float(pz[f'draw{seed}_bt'][k]) for k in range(len(rt))}
            PRI[er] = (np.array([lut[r] for r in newR]), float(pz[f'draw{seed}_sigma']))
        for js in hp:
            dj = [i for i, r in enumerate(newR) if (r, js) in panel.Y]
            if len(dj) < 4:
                continue
            yD = np.array([panel.Y[(newR[i], js)] for i in dj], float)
            SRD = float(yD.mean())
            JS.append(int(js))
            bankD = {e: Bank(marginal_curves(PRI[(e, 0)][0][dj], np.full(len(dj), PRI[(e, 0)][1])), typD[dj], f1['sigma_g']) for e in ENC}
            bankDR = {er: Bank(marginal_curves(PRI[er][0][dj], np.full(len(dj), PRI[er][1])), typD[dj], f1['sigma_g']) for er in PRI}
            bi, yb = panel.bank_rows(calR, js)
            n = len(bi)
            bank = bank_from_fit(f1, bi, typ)
            S = {'Random': [int(i) for i in np.random.RandomState(100 + seed * panel.J + js).permutation(n)[:T]]}
            s_, st = [], State(bank, yb)
            for _ in range(T):
                rem = [i for i in range(n) if i not in s_]
                s_.append(eig_pick(st, rem))
                st.add(s_[-1])
            S['theta-EIG (abl.)'] = s_
            for e in ENC:
                sd_, st = [], State(bank, yb)
                for _ in range(T):
                    rem = [i for i in range(n) if i not in sd_]
                    sd_.append(r1_pick_transfer(st, bankD[e], rem))
                    st.add(sd_[-1])
                S[CAN + e] = sd_
            for B in BP:
                for e in ENC:
                    for p in POL:
                        q = state_from(bank, yb, S[p + e if p == CAN else p][:B]).q
                        srD, pD = transfer(q, bankD[e])
                        RES[e][p][B]['mae'].append(abs(srD - SRD))
                        RES[e][p][B]['nll'].append(float(-np.mean(yD * np.log(pD + 1e-9) + (1 - yD) * np.log(1 - pD + 1e-9))))
                        for run in RUNS:
                            RES3[e][run][p][B].append(abs(transfer(q, bankDR[(e, run)])[0] - SRD))
                    nv = float(yb[S['Random'][:B]].mean())
                    RES[e]['naive'][B]['mae'].append(abs(nv - SRD))
                    RES[e]['naive'][B]['nll'].append(float(-np.mean(yD * np.log(nv + 1e-9) + (1 - yD) * np.log(1 - nv + 1e-9))))
        print(f'seed {seed} done', flush=True)
    nev = len(RES['']['naive'][30]['mae'])
    print(f'\n===== Table 3B: UPS, zero rollouts on D ({nev} evaluations) =====')
    for e, lab in ENC.items():
        print(f'\n-- scene prior: {lab} --')
        print(f'{"probe policy":18s} ' + ' '.join(f'{"B=" + str(B) + " MAE":>10s} {"NLL":>7s}' for B in BP))
        for p in ('naive',) + POL:
            print(f'{p:18s} ' + ' '.join(f'{np.mean(RES[e][p][B]["mae"]):10.4f} {np.mean(RES[e][p][B]["nll"]):7.4f}' for B in BP))
        print('across-encoder-run SD of MAE (3 RelGraph runs):')
        for p in POL:
            print(f'  {p:18s} ' + '  '.join('B{}: {:.4f}'.format(B, np.std([np.mean(RES3[e][run][p][B]) for run in RUNS], ddof=1)) for B in BP))
        print(f'paired delta MAE vs {CAN}:')
        for p in ('Random', 'theta-EIG (abl.)'):
            print(f'  {p:18s} ' + '  '.join('B{}: {:+.4f} [{:+.4f},{:+.4f}]'.format(
                B, *paired_cluster_boot(RES[e][p][B]['mae'], RES[e][CAN][B]['mae'], JS)) for B in BP))
    print('\npaired delta MAE (control prior - encoder of record), same policy:')
    for e2 in ('_lane', '_nospeed'):
        print(f'  -- {ENC[e2][0]} --')
        for p in POL:
            print(f'    {p:18s} ' + '  '.join('B{}: {:+.4f} [{:+.4f},{:+.4f}]'.format(
                B, *paired_cluster_boot(RES[e2][p][B]['mae'], RES[''][p][B]['mae'], JS)) for B in BP))
    OUT.mkdir(exist_ok=True)
    def dump(e, path):                                 # ups.json keeps the shipped-prior schema;
        json.dump({p: {str(B): {m: [float(x) for x in v] for m, v in d.items()}                # the
                       for B, d in bb.items()} for p, bb in RES[e].items()}, open(path, 'w'))  # no-
    dump('', OUT / 'ups.json')                         # each control goes to its own file so
    dump('_lane', OUT / 'ups_lane.json')               # nothing downstream of ups.json changes shape
    dump('_nospeed', OUT / 'ups_nospeed.json')
    for (p, B, ref) in ANCHORS:                        # pinned on the lane-free encoder of record
        assert abs(np.mean(RES[''][p][B]['mae']) - ref) < .003, (p, B, np.mean(RES[''][p][B]['mae']))
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    main()
