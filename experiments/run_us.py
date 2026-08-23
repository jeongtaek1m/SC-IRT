#!/usr/bin/env python3
"""Tables 1 and 2 — US: unseen-scene difficulty prediction.

Pooled cell-level evaluation on C = (held-out-type routes) x (13 calibration
planners), 16 draws x ~40 routes = 640 route evaluations.

Sections
  (desc)  Table 1 descriptor arms: planner-only null, six simple baselines,
          the SC-IRT stack via two-stage Ridge (comparison) and via one-stage
          LLTM+e (canonical), and the response-calibrated oracle ceiling.
  (lltm)  LLTM+e vs two-stage paired delta + the plausible-values
          decomposition of the calibration-noise share.
  (enc)   Table 1 encoder row and Table 2 ablation from the shipped per-run
          prediction artifacts (single runs; seeds summarised as metric
          mean +- SD — prediction ensembling is banned).

Anchors: null 0.710/.207; LLTM+e 0.764/.177/+0.510; 2-stage 0.760/.178/+0.487;
sigma-hat 0.593; PV share 16.4%; hc +0.486; kin +0.428; d64 rho +0.469+-.011.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scirt.b2d import Panel, load_features, DATA
from scirt.splits import unified_split, R_DRAWS
from scirt.calibration import calibrate_dense, calibrate_dense_se, frozen_b_dense
from scirt.curves import sig, GX, GW
from scirt.lltm import lltm_e
from scirt.metrics import cluster_boot_rho_delta

np.random.seed(0)
torch.manual_seed(0)
OUT = Path(__file__).resolve().parents[1] / 'results'


def load_descriptor_arms():
    import csv
    ck = load_features('eval_cmdkin_stats')
    spz = load_features('eval_scenparamz')
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
            'SC-IRT stack (ck+spz)': {k: np.concatenate([ck[k], spz[k]]) for k in ck if k in spz}}


def pooled_metrics(d, auc0=None, mae0=None):
    auc = roc_auc_score(d['y'], d['p'])
    mae = float(np.mean(np.abs(np.array(d['rp']) - np.array(d['ro']))))
    rho = float(spearmanr(d['bt'], d['fl']).correlation) if d.get('bt') else None
    return auc, mae, rho


def main():
    arms = load_descriptor_arms()
    panel = Panel(extra_feature_dicts=tuple(arms.values()))
    Y0, MK = panel.dense()
    N = len(panel.allr)
    sn, allr = panel.sn, panel.allr
    ROWS = list(arms) + ['Oracle (resp-calibrated C)']
    POOL = {a: {'p': [], 'y': [], 'rp': [], 'ro': [], 'bt': [], 'fl': []} for a in ROWS}
    LPOOL = {a: {'p': [], 'y': [], 'rp': [], 'ro': [], 'bt': [], 'fl': []}
             for a in ('lltm', 'lltm-marg')}
    NULLP = {'p': [], 'y': [], 'rp': [], 'ro': []}
    CLREC, SIGS = [], []
    PV = {'rho_m': [], 'rho_between': []}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        tr = [i for i in range(N) if sn[allr[i]] not in ht]
        te = [i for i in range(N) if sn[allr[i]] in ht]
        bA, th = calibrate_dense(Y0, MK, tr, cols)
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
        bC = frozen_b_dense(Y0, MK, te, cols, th)
        for name in ROWS:                               # Table 1 descriptor arms
            if name.startswith('Oracle'):
                bte = bC
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
        # ---- LLTM+e (canonical one-stage) + plausible values --------------
        stack = arms['SC-IRT stack (ck+spz)']
        Z = np.vstack([stack[allr[i]] for i in tr])
        m0, s0 = Z.mean(0), Z.std(0) + 1e-9
        Ztr = (Z - m0) / s0
        Zte = (np.vstack([stack[allr[i]] for i in te]) - m0) / s0
        bA2, thA2, seA = calibrate_dense_se(Y0, MK, tr, cols)
        wL, sgL, thL, _cL = lltm_e(Y0, MK, tr, cols, Ztr)
        bLL = Zte @ wL
        bLL = bLL - ((Ztr @ wL).mean() - bA2.mean())     # location: align to b-hat mean
        SIGS.append(sgL)
        for k, i in enumerate(te):
            js = [c for c in cols if MK[i, c]]
            jj = [cols.index(c) for c in js]
            ys = Y0[i, js]
            ps = sig(thL[jj] - bLL[k])
            LPOOL['lltm']['p'] += ps.tolist()
            LPOOL['lltm']['y'] += ys.tolist()
            LPOOL['lltm']['rp'].append(float(ps.mean()))
            LPOOL['lltm']['ro'].append(float(ys.mean()))
            zz = thL[jj][:, None] - (bLL[k] + sgL * GX[None, :])
            pm = (sig(zz) * GW[None, :]).sum(1)
            LPOOL['lltm-marg']['p'] += pm.tolist()
            LPOOL['lltm-marg']['y'] += ys.tolist()
            LPOOL['lltm-marg']['rp'].append(float(pm.mean()))
            LPOOL['lltm-marg']['ro'].append(float(ys.mean()))
        for a in ('lltm', 'lltm-marg'):
            LPOOL[a]['bt'] += list(bLL)
            LPOOL[a]['fl'] += list(obs_fail)
        CLREC += [f'{seed}:{sn[allr[i]]}' for i in te]
        rngp = np.random.RandomState(400 + seed)
        rhos = []
        for _ in range(20):                              # plausible values, M=20
            bdraw = bA2 + seA * rngp.randn(len(bA2))
            rgm = Ridge(alpha=100.).fit(Ztr, bdraw)
            rhos.append(spearmanr(rgm.predict(Zte), obs_fail).correlation)
        PV['rho_m'].append(float(np.mean(rhos)))
        PV['rho_between'].append(float(np.var(rhos, ddof=1)))
        print(f'seed {seed} done (sigma-hat={sgL:.3f})', flush=True)

    auc0 = roc_auc_score(NULLP['y'], NULLP['p'])
    mae0 = float(np.mean(np.abs(np.array(NULLP['rp']) - np.array(NULLP['ro']))))
    print('\n===== Table 1 — US (unified split, pooled) =====')
    print(f'Planner-only null: AUROC {auc0:.3f} / Scene-MAE {mae0:.3f}')
    results = {'null': {'auroc': auc0, 'mae': mae0}}
    for name in ROWS + ['lltm', 'lltm-marg']:
        d = POOL[name] if name in POOL else LPOOL[name]
        auc, mae, rho = pooled_metrics(d)
        label = {'lltm': 'SC-IRT stack, LLTM+e (canonical)',
                 'lltm-marg': 'LLTM+e, eps-marginalised cells'}.get(name, name)
        print(f'{label:34s} AUROC {auc:.3f} ({auc - auc0:+.3f})  MAE {mae:.3f} '
              f'({1 - mae / mae0:+.1%})  rho {rho:+.3f}')
        results[name] = {'auroc': auc, 'mae': mae, 'rho': rho}
    print(f'\nsigma-hat (LLTM+e residual difficulty SD): '
          f'{np.mean(SIGS):.3f} +- {np.std(SIGS, ddof=1):.3f}')
    d, lo, hi, p = cluster_boot_rho_delta(LPOOL['lltm']['bt'],
                                          POOL['SC-IRT stack (ck+spz)']['bt'],
                                          POOL['SC-IRT stack (ck+spz)']['fl'], CLREC)
    print(f'Delta rho (LLTM+e - two-stage) = {d:+.4f} CI[{lo:+.4f},{hi:+.4f}] P(>0)={p:.3f}')
    wv = np.var(PV['rho_m'], ddof=1)
    bv = np.mean(PV['rho_between']) * (1 + 1 / 20)
    print(f'Plausible values: calibration-noise share of US rho uncertainty = {bv / (wv + bv):.1%}')

    # ---- (enc) encoder rows from the shipped artifacts --------------------
    print('\n===== Encoder rows (shipped per-run artifacts; no ensembling) =====')
    enc = {}
    for dd in (64, 96):
        rows = [json.load(open(DATA / 'encoder' / f'unified_us_encoder_d{dd}s{s}.json'))['unified']
                for s in (0, 1, 2)]
        rhos = [r['rho_pooled'] for r in rows]
        aucs = [r['auroc'] for r in rows]
        maes = [r['mae'] for r in rows]
        enc[dd] = dict(rho=rhos, auroc=aucs, mae=maes)
        print(f'  d{dd}: AUROC {np.mean(aucs):.3f}+-{np.std(aucs, ddof=1):.3f}  '
              f'MAE {np.mean(maes):.3f}+-{np.std(maes, ddof=1):.3f}  '
              f'rho {np.mean(rhos):+.3f}+-{np.std(rhos, ddof=1):.3f}')
    # Table 2 per-run Delta1 (enc - hand-crafted) using the shipped preds ----
    ck = load_features('eval_cmdkin_stats')
    gtr = load_features('eval_gtrisk')
    hc = {k: np.concatenate([ck[k], gtr[k]]) for k in ck if k in gtr}
    BT = {'kin': [], 'hc': []}
    FL = []
    ENC = {(dd, s): [] for dd in (64, 96) for s in (0, 1, 2)}
    P = {(dd, s): np.load(DATA / 'encoder' / f'unified_enc_pred_d{dd}s{s}.npz', allow_pickle=True)
         for dd in (64, 96) for s in (0, 1, 2)}
    for seed in range(R_DRAWS):
        hp, ht = unified_split(seed, panel.utypes, panel.J)
        cols = [c for c in range(panel.J) if c not in hp]
        tr = [i for i in range(N) if sn[allr[i]] not in ht]
        te = [i for i in range(N) if sn[allr[i]] in ht]
        bA, _ = calibrate_dense(Y0, MK, tr, cols)
        obs_fail = np.array([1 - Y0[i, [c for c in cols if MK[i, c]]].mean() for i in te])
        for nm, feat in (('kin', ck), ('hc', hc)):
            Z = np.vstack([feat[allr[i]] for i in tr])
            m0, s0 = Z.mean(0), Z.std(0) + 1e-9
            rgm = Ridge(alpha=100.).fit((Z - m0) / s0, bA)
            BT[nm] += list(rgm.predict((np.vstack([feat[allr[i]] for i in te]) - m0) / s0))
        FL += list(obs_fail)
        for key, pz in P.items():
            rt = [str(x) for x in pz[f'draw{seed}_rt']]
            lut = {rt[k]: pz[f'draw{seed}_bt'][k] for k in range(len(rt))}
            ENC[key] += [lut[allr[i]] for i in te]
    FLa = np.array(FL)
    rho_hc = spearmanr(BT['hc'], FLa).correlation
    rho_kin = spearmanr(BT['kin'], FLa).correlation
    print(f'\n===== Table 2 — ablation (pooled {len(FL)} evaluations) =====')
    print(f'  Kinematics only        rho {rho_kin:+.3f}')
    print(f'  Hand-crafted (ck+gtr)  rho {rho_hc:+.3f}')
    for dd in (64, 96):
        rr = [spearmanr(ENC[(dd, s)], FLa).correlation for s in (0, 1, 2)]
        d1 = [r - rho_hc for r in rr]
        print(f'  Encoder d{dd} (3 runs)   rho {np.mean(rr):+.3f}+-{np.std(rr, ddof=1):.3f}   '
              f'Delta1 vs hc {np.mean(d1):+.3f}+-{np.std(d1, ddof=1):.3f}')

    OUT.mkdir(exist_ok=True)
    json.dump({'table1': results, 'sigma': list(map(float, SIGS)),
               'enc': {str(k): v for k, v in enc.items()}},
              open(OUT / 'us.json', 'w'))

    assert abs(auc0 - 0.710) < 0.002 and abs(mae0 - 0.207) < 0.002
    l = results['lltm']
    assert abs(l['auroc'] - 0.764) < 0.002 and abs(l['rho'] - 0.510) < 0.005
    t = results['SC-IRT stack (ck+spz)']
    assert abs(t['auroc'] - 0.760) < 0.002 and abs(t['rho'] - 0.487) < 0.005
    assert abs(np.mean(SIGS) - 0.593) < 0.01
    assert abs(bv / (wv + bv) - 0.164) < 0.01
    assert abs(rho_hc - 0.486) < 0.005 and abs(rho_kin - 0.428) < 0.005
    assert abs(np.mean(enc[64]['auroc']) - 0.753) < 0.003
    assert abs(np.mean([spearmanr(ENC[(64, s)], FLa).correlation for s in (0, 1, 2)]) - 0.469) < 0.005
    print('anchors OK')


if __name__ == '__main__':
    main()
