#!/usr/bin/env python3
"""Table 3A — US: unseen-scene difficulty prediction.

Pooled cell-level evaluation on C = (evaluation-type routes) x (12 calibration
planners), 16 draws x ~40 routes = 640 route evaluations.

Rows
  planner-only null (b = 0), eight descriptor baselines scored through a
  two-stage Ridge plug-in (b_hat ~ x on the calibration types, predict the
  evaluation types), the response-calibrated oracle ceiling, and the
  RelGraph R2 scene encoder from the shipped per-run out-of-fold predictions
  (three independent runs summarised as metric mean +- SD; prediction
  ensembling is banned).

Anchors: null .699/.214; kinematics rho +.497; hand-crafted risk rho +.533;
RelGraph mean AUROC .751 / rho +.490.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atdrive.b2d import Panel, load_features, DATA
from atdrive.splits import unified_split, R_DRAWS
from atdrive.calibration import calibrate_dense, frozen_b_dense
from atdrive.curves import sig

np.random.seed(0)
torch.manual_seed(0)
OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
RUNS = (0, 1, 2)
CONTROLS = {'noroute': 'R2 w/o route relation', 'sroute': 'R2, route correspondence shuffled', 'sa2l': 'R2, agent-lane correspondence shuffled',
            'nospeed': 'R2, speed channel removed'}


def load_descriptor_arms():
    import csv
    ck = load_features('eval_cmdkin_stats')
    gtr = load_features('eval_gtrisk')
    tf = list(csv.reader(open(DATA / 'b2d' / 'traffic_features_220.csv')))
    hdr = tf[0]
    ci = {c: i for i, c in enumerate(hdr)}

    def ff(v):
        try:
            return float(v)
        except ValueError:
            return 0.0

    minttc = {r[0]: np.array([ff(r[ci['ssm_min_ttc']])]) for r in tf[1:]}
    risk = {r[0]: np.array([ff(r[ci[c]]) for c in hdr[2:]]) for r in tf[1:]}
    kd = np.load(DATA / 'b2d' / 'baseline_kin_den.npz', allow_pickle=True)
    kn = [str(x).replace('route_', '') for x in kd['kin_names']]
    dn = [str(x).replace('route_', '') for x in kd['den_names']]
    kin = {kn[i]: kd['kin'][i].astype(np.float64) for i in range(len(kn))}
    den = {dn[i]: kd['den'][i].astype(np.float64) for i in range(len(dn))}
    kinden = {r: np.concatenate([kin[r], den[r]]) for r in kin if r in den}
    return {'Min-TTC': minttc, 'Risk field': risk,
            'Route geometry': load_features('eval_routegeom'),
            'Agent density + kin.': kinden,
            'Traffic entropy': load_features('eval_smart_ent'),
            'Agent-JEPA': load_features('eval_agentjepa'),
            'Kinematics (cmdkin)': ck,
            'Hand-crafted risk (cmdkin+gtrisk)': {k: np.concatenate([ck[k], gtr[k]]) for k in ck if k in gtr}}


def pooled_metrics(d):
    auc = roc_auc_score(d['y'], d['p'])
    mae = float(np.mean(np.abs(np.array(d['rp']) - np.array(d['ro']))))
    rho = float(spearmanr(d['bt'], d['fl']).correlation)
    return auc, mae, rho


def main():
    arms = load_descriptor_arms()
    panel = Panel(extra_feature_dicts=tuple(arms.values()))
    Y0, MK = panel.dense()
    N = len(panel.allr)
    sn, allr = panel.sn, panel.allr
    RELG = {s_: np.load(DATA / 'encoder' / f'relgraph_r2_s{s_}.npz', allow_pickle=True) for s_ in RUNS}
    CTRL = {c: {s_: np.load(DATA / 'encoder' / f'relgraph_r2_{c}_s{s_}.npz', allow_pickle=True) for s_ in RUNS}
            for c in CONTROLS if all((DATA / 'encoder' / f'relgraph_r2_{c}_s{s_}.npz').exists() for s_ in RUNS)}
    ROWS = list(arms) + ['Oracle (resp-calibrated C)'] + [f'RelGraph R2 s{s_}' for s_ in RUNS] \
        + [f'{CONTROLS[c]} s{s_}' for c in CTRL for s_ in RUNS]
    POOL = {a: {'p': [], 'y': [], 'rp': [], 'ro': [], 'bt': [], 'fl': []} for a in ROWS}
    NULLP = {'p': [], 'y': [], 'rp': [], 'ro': []}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        tr = [i for i in range(N) if sn[allr[i]] not in ht]
        te = [i for i in range(N) if sn[allr[i]] in ht]
        bA, th, sb = calibrate_dense(Y0, MK, tr, cols)
        _, th0 = calibrate_dense(Y0, MK, tr, cols, freeze_b0=True)
        obs_fail = np.array([1 - Y0[i, [c for c in cols if MK[i, c]]].mean() if MK[i, cols].any()
                             else 1.0 for i in te])
        for i in te:                                    # planner-only null cells
            js = [c for c in cols if MK[i, c]]
            ps = sig(th0[[cols.index(c) for c in js]])
            ys = Y0[i, js]
            NULLP['p'] += ps.tolist()
            NULLP['y'] += ys.tolist()
            NULLP['rp'].append(float(ps.mean()))
            NULLP['ro'].append(float(ys.mean()))
        bC = frozen_b_dense(Y0, MK, te, cols, th, sb)
        for name in ROWS:
            if name.startswith('Oracle'):
                bte = bC
            elif name.startswith('RelGraph') or name.startswith('R2'):
                pz = RELG[int(name[-1])] if name.startswith('RelGraph') else CTRL[next(c for c in CTRL if name.startswith(CONTROLS[c]))][int(name[-1])]
                rt = [str(x) for x in pz[f'draw{seed}_rt']]
                lut = {rt[k]: float(pz[f'draw{seed}_bt'][k]) for k in range(len(rt))}
                bte = np.array([lut[allr[i]] for i in te])
            else:
                feat = arms[name]
                Z = np.vstack([feat[allr[i]] for i in tr])
                m0, s0 = Z.mean(0), Z.std(0) + 1e-9
                al = 100.0 if Z.shape[1] > 10 else 10.0
                rgm = Ridge(alpha=al).fit((Z - m0) / s0, bA)
                bte = rgm.predict((np.vstack([feat[allr[i]] for i in te]) - m0) / s0)
            for k, i in enumerate(te):
                js = [c for c in cols if MK[i, c]]
                ps = sig(th[[cols.index(c) for c in js]] - bte[k])
                ys = Y0[i, js]
                POOL[name]['p'] += ps.tolist()
                POOL[name]['y'] += ys.tolist()
                POOL[name]['rp'].append(float(ps.mean()))
                POOL[name]['ro'].append(float(ys.mean()))
            POOL[name]['bt'] += bte.tolist()
            POOL[name]['fl'] += obs_fail.tolist()
        print(f'seed {seed} done', flush=True)

    auc0 = roc_auc_score(NULLP['y'], NULLP['p'])
    mae0 = float(np.mean(np.abs(np.array(NULLP['rp']) - np.array(NULLP['ro']))))
    print(f'\n===== Table 3A — US (unified split, pooled {len(POOL[ROWS[0]]["bt"])} route evaluations) =====')
    print(f'Planner-only null: AUROC {auc0:.3f} / Scene-MAE {mae0:.3f}')
    results = {'null': {'auroc': auc0, 'mae': mae0}}
    for name in ROWS:
        auc, mae, rho = pooled_metrics(POOL[name])
        print(f'{name:34s} AUROC {auc:.3f} ({auc - auc0:+.3f})  MAE {mae:.3f} '
              f'({1 - mae / mae0:+.1%})  rho {rho:+.3f}')
        results[name] = {'auroc': auc, 'mae': mae, 'rho': rho}
    rg = [results[f'RelGraph R2 s{s_}'] for s_ in RUNS]
    sd = lambda k: np.std([r[k] for r in rg], ddof=1)
    mn = lambda k: np.mean([r[k] for r in rg])
    print('RelGraph R2 scene encoder (3 runs)  AUROC {:.3f}+-{:.3f}  MAE {:.3f}+-{:.3f}  rho {:+.3f}+-{:.3f}'.format(
        mn('auroc'), sd('auroc'), mn('mae'), sd('mae'), mn('rho'), sd('rho')))
    rho_hc = results['Hand-crafted risk (cmdkin+gtrisk)']['rho']
    d1 = [r['rho'] - rho_hc for r in rg]
    print(f'Delta rho (RelGraph - hand-crafted risk), per run: {np.mean(d1):+.3f}+-{np.std(d1, ddof=1):.3f}')
    if CTRL:
        print('\n===== Table 3A(b) — RelGraph controls (same architecture, recipe and seeds; only the graph tensors or the ego channels differ) =====')
        for c in CTRL:
            rc = [results[f'{CONTROLS[c]} s{s_}'] for s_ in RUNS]
            print('{:44s} AUROC {:.3f}+-{:.3f}  MAE {:.3f}+-{:.3f}  rho {:+.3f}+-{:.3f}   Delta rho vs R2 (paired by seed) {:+.3f}+-{:.3f}'.format(
                CONTROLS[c], np.mean([r['auroc'] for r in rc]), np.std([r['auroc'] for r in rc], ddof=1),
                np.mean([r['mae'] for r in rc]), np.std([r['mae'] for r in rc], ddof=1),
                np.mean([r['rho'] for r in rc]), np.std([r['rho'] for r in rc], ddof=1),
                np.mean([a['rho'] - b['rho'] for a, b in zip(rc, rg)]), np.std([a['rho'] - b['rho'] for a, b in zip(rc, rg)], ddof=1)))

    OUT.mkdir(exist_ok=True)
    json.dump({'table3a': results}, open(OUT / 'us.json', 'w'))

    k, h = results['Kinematics (cmdkin)'], results['Hand-crafted risk (cmdkin+gtrisk)']
    assert abs(auc0 - 0.699) < 0.002 and abs(mae0 - 0.214) < 0.002
    assert abs(k['auroc'] - 0.752) < 0.002 and abs(k['mae'] - 0.180) < 0.002 and abs(k['rho'] - 0.497) < 0.005
    assert abs(h['auroc'] - 0.758) < 0.002 and abs(h['mae'] - 0.175) < 0.002 and abs(h['rho'] - 0.533) < 0.005
    assert abs(mn('auroc') - 0.751) < 0.003 and abs(mn('mae') - 0.192) < 0.003 and abs(mn('rho') - 0.490) < 0.005
    for c, v in (('noroute', 0.520), ('sroute', 0.512), ('sa2l', 0.501), ('nospeed', 0.500)):
        if c not in CTRL:                              # its npz is not in this checkout
            continue
        assert abs(np.mean([results[f'{CONTROLS[c]} s{s_}']['rho'] for s_ in RUNS]) - v) < 0.005, c
    print('anchors OK')


if __name__ == '__main__':
    main()
