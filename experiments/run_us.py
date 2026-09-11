#!/usr/bin/env python3
"""Table 3A — US: unseen-scene difficulty prediction.

Pooled cell-level evaluation on C = (evaluation-type routes) x (12 calibration
planners), 16 draws x ~40 routes = 640 route evaluations.

Rows
  planner-only null (b = 0), eight descriptor baselines scored through a
  two-stage Ridge plug-in (b_hat ~ x on the calibration types, predict the
  evaluation types), the response-calibrated oracle ceiling, and the
  RelGraph R2-noLane scene encoder from the shipped per-run out-of-fold
  predictions (three independent runs summarised as metric mean +- SD;
  prediction ensembling is banned).

THE ENCODER OF RECORD HAS NO LANE GRAPH. R2-noLane is the same R2Net with
the whole map side of the graph removed before any tensor is built (no lane
tokens, no lane_feat, no L2L edges, no A2L candidates, no route_rel); ego,
command and agents remain. The lane-carrying R2 that earlier releases shipped
is now a control: it is the first row of the structural-control block below,
"R2, lane graph kept", next to noroute / sroute / sa2l / nospeed, and the speed
ablation of the encoder of record itself (nlnospeed, relgraph_r2nolane_nospeed_s*.npz).

Anchors: null .699/.214; kinematics rho +.497; hand-crafted risk rho +.533;
R2-noLane mean AUROC .761 / rho +.545 (lane-carrying control .751 / +.490).
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
CONTROLS = {'lane': 'R2, lane graph kept', 'noroute': 'R2 w/o route relation',                 # 'lane' = the encoder earlier releases shipped
            'sroute': 'R2, route correspondence shuffled', 'sa2l': 'R2, agent-lane correspondence shuffled',
            'nospeed': 'R2, speed channel removed', 'nlnospeed': 'R2-noLane, speed channel removed',
            'match0.1': 'R2-noLane + difficulty-matching term, lambda .1 (ablation arm)',
            'match1': 'R2-noLane + difficulty-matching term, lambda 1 (ablation arm)',
            'vis': 'R2-noLane tracks + frozen DINOv3 visual branch (two-branch arm)',
            'visonly': 'frozen DINOv3 visual branch only, without the R2 track branch (two-branch arm)',
            'vis_pca32': 'tracks + DINOv3-L visual, PCA-32 projection (two-branch arm)',
            'visonly_pca32': 'DINOv3-L visual only, PCA-32 projection (two-branch arm)',
            'viswin_pca32': 'window-level fusion: DINOv3-L frames (PCA-32) + Transformer + z_w (visual-window arm)',
            'viswind64l1_pca32': 'window-level fusion, small Transformer (64-d, 1 layer), PCA-32 (visual-window arm)',
            'vis_S': 'tracks + DINOv3-S visual (two-branch arm)',
            'visonly_S': 'DINOv3-S visual only (two-branch arm)',
            'viswin_pca32_S': 'window-level fusion with DINOv3-S frames, PCA-32 (visual-window arm)',
            'vis_pca32_S': 'tracks + DINOv3-S visual, PCA-32 (two-branch arm)',
            'visonly_pca32_S': 'DINOv3-S visual only, PCA-32 (two-branch arm)',
            'viswin_S': 'window-level fusion with DINOv3-S frames, full 1152-d projection (visual-window arm)',
            'viswind64l1_pca32_S': 'window-level fusion, small Transformer (64-d, 1 layer), DINOv3-S PCA-32 (visual-window arm)',
            'viswind32l1_pca32_S': 'window-level fusion, smallest Transformer (32-d, 1 layer), DINOv3-S PCA-32 (visual-window arm)',
            'vis_vpool_S': 'tracks + DINOv3-S visual, cameras mean-pooled to 384-d (two-branch arm)',
            'visonly_vpool_S': 'DINOv3-S visual only, cameras mean-pooled to 384-d (two-branch arm)',
            'viswin_vpool_S': 'window-level fusion, DINOv3-S frames mean-pooled over cameras (visual-window arm)',
            'viswind64l1_vpool_S': 'window-level fusion, small Transformer (64-d, 1 layer), cameras mean-pooled (visual-window arm)',
            'vis_vpoolc8_S': 'tracks + DINOv3-S visual, cameras mean-pooled + 8-channel average-pool to 48-d (two-branch arm)',
            'visonly_vpoolc8_S': 'DINOv3-S visual only, cameras mean-pooled + 8-channel average-pool to 48-d (two-branch arm)',
            'match0.1_vis_vpool_S': 'tracks + DINOv3-S visual (camera-pooled) + matching term lambda .1 (ablation)',
            'match0.1_viswin_vpool_S': 'window-level fusion (camera-pooled) + matching term lambda .1 (ablation)',
            'visonly_vpool_ego_S': 'DINOv3-S visual (camera-pooled) + ego status, no agents (two-branch arm)',
            'vismfm_ego_S': 'DINOv3-S visual + SMART/CAT-K motion + ego status, without the R2 track branch (foundation-model arm)',
            'mfmonly_ego': 'SMART/CAT-K motion + ego status, without the R2 track branch (foundation-model arm)',
            'vis_front_S': 'tracks + DINOv3-S front camera (two-branch arm)',
            'visonly_front_S': 'DINOv3-S front camera only (two-branch arm)',
            'visonly_front_ego_S': 'DINOv3-S front camera + ego status, no agents (foundation-model arm)',
            'viswin_front_S': 'window-level fusion: DINOv3-S front frames + Transformer + z_w (visual-window arm)',
            'viswind64l1_front_S': 'window-level fusion, small Transformer (64-d, 1 layer), front frames (visual-window arm)',
            'viswind128l0_front_S': 'window-level fusion, NO temporal Transformer (P + e_t + GELU, mean), front frames: the temporal ablation of viswin_front_S',
            'ssl': 'R2-noLane, track-SSL initialisation (masked agent-track reconstruction on the training routes, epoch by inner validation) then the same IRT loss (initialisation ablation)',
            'fusewin_vt_front_S': 'token fusion per window: [front visual, track z_w] + 1-layer fusion Transformer, record pooling + ego branch (fusion arm)',
            'fusewin_vs_front_ego_S': 'token fusion per window: [front visual, SMART/CAT-K] + fusion Transformer, record pooling + ego branch, without the R2 track branch (fusion arm)',
            'fusewin_vts_front_S': 'token fusion per window: [front visual, track z_w, SMART/CAT-K] + fusion Transformer, record pooling + ego branch (fusion arm)',
            'vismfm_front_ego_S': 'DINOv3-S front camera + SMART/CAT-K motion + ego status, without the R2 track branch (foundation-model arm)',
            'vismfm_front_S': 'DINOv3-S front camera + SMART/CAT-K motion, without the R2 track branch (foundation-model arm)',
            'match0.1_vis_front_S': 'tracks + front camera + matching term lambda .1 (ablation)',
            'match0.1_viswin_front_S': 'window-level fusion (front) + matching term lambda .1 (ablation)',
            'match0.1_vismfm_front_ego_S': 'front camera + SMART/CAT-K + ego + matching term lambda .1 (ablation)',
            'match0.1_viswin_pca32_S': 'window-level fusion DINOv3-S PCA-32 + matching term lambda .1 (ablation)',
            'match0.1_vis_pca32_S': 'tracks + DINOv3-S PCA-32 + matching term lambda .1 (ablation)',
            'mfmonly': 'frozen SMART/CAT-K features (agent tracks + map through the frozen model), without the R2 track branch (foundation-model arm)',
            'vismfm': 'frozen DINOv3 visual + frozen SMART/CAT-K motion, without the R2 track branch (foundation-model arm)',
            'vismfmtrk': 'tracks + frozen DINOv3 visual + frozen SMART/CAT-K motion (three-branch arm)',
            # the same five branch arms with the harness's difficulty-matching term (--match 0.1)
            'match0.1_vis': 'R2-noLane tracks + frozen DINOv3 visual branch (two-branch arm) + matching term lambda .1 (ablation)',
            'match0.1_visonly': 'frozen DINOv3 visual branch only, without the R2 track branch (two-branch arm) + matching term lambda .1 (ablation)',
            'match0.1_mfmonly': 'frozen SMART/CAT-K motion branch only, without the R2 track branch (foundation-model arm) + matching term lambda .1 (ablation)',
            'match0.1_vismfm': 'frozen DINOv3 visual + frozen SMART/CAT-K motion, without the R2 track branch (foundation-model arm) + matching term lambda .1 (ablation)',
            'match0.1_vismfmtrk': 'tracks + frozen DINOv3 visual + frozen SMART/CAT-K motion (three-branch arm) + matching term lambda .1 (ablation)',
            # the small vision backbone (DINOv3 ViT-S/16, --arm-suffix S) paired with the frozen motion model
            'vismfm_S': 'DINOv3-S visual + frozen SMART/CAT-K motion, without the R2 track branch (foundation-model arm)',
            'vismfmtrk_S': 'tracks + DINOv3-S visual + frozen SMART/CAT-K motion (three-branch arm)',
            'match0.1_vismfm_S': 'DINOv3-S visual + frozen SMART/CAT-K motion, no tracks + matching term lambda .1 (ablation)',
            'match0.1_vismfmtrk_S': 'tracks + DINOv3-S visual + frozen SMART/CAT-K motion + matching term lambda .1 (ablation)',
            # front camera only (--visual-views front) + the frozen motion model ('vismfm_front_S' is above)
            'vismfmtrk_front_S': 'tracks + DINOv3-S front camera + SMART/CAT-K motion (three-branch arm)',
            'match0.1_vismfm_front_S': 'DINOv3-S front camera + SMART/CAT-K motion, no tracks + matching term lambda .1 (ablation)',
            'match0.1_vismfmtrk_front_S': 'tracks + DINOv3-S front camera + SMART/CAT-K motion + matching term lambda .1 (ablation)'}


def ctrl_npz(c, s):
    """The control npz of run s: the lane-carrying model keeps its own file name."""
    if c == 'nlnospeed':                                   # the speed ablation of the encoder of record
        return DATA / 'encoder' / f'relgraph_r2nolane_nospeed_s{s}.npz'
    if c.startswith(('match', 'vis', 'mfm', 'fusewin', 'ssl')):        # --match, --visual, --motion-fm and --fuse-window arms
        return DATA / 'encoder' / f'relgraph_r2nolane_{c}_s{s}.npz'
    return DATA / 'encoder' / (f'relgraph_r2_s{s}.npz' if c == 'lane' else f'relgraph_r2_{c}_s{s}.npz')


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

    risk = {r[0]: np.array([ff(r[ci[c]]) for c in hdr[2:]]) for r in tf[1:]}
    kd = np.load(DATA / 'b2d' / 'baseline_kin_den.npz', allow_pickle=True)
    kn = [str(x).replace('route_', '') for x in kd['kin_names']]
    dn = [str(x).replace('route_', '') for x in kd['den_names']]
    kin = {kn[i]: kd['kin'][i].astype(np.float64) for i in range(len(kn))}
    den = {dn[i]: kd['den'][i].astype(np.float64) for i in range(len(dn))}
    kinden = {r: np.concatenate([kin[r], den[r]]) for r in kin if r in den}
    # Rows of record (rebuilt 2026-09, experiments/us_official/, provenance in results/us_official_provenance/):
    #   Min-TTC              the criticality metric of the literature (Hayward 1972 as reviewed by Westhofen et al.
    #                        2023, min over time and actors, Sec. 5.2), computed from the definition on the reference
    #                        rollout (eval_min_ttc.npz; the earlier row was the in-house ssm_min_ttc column)
    #   EDRF-based           the risk-field equations of Jiang et al. 2024 with the single realised future in place of
    #                        the multimodal predictor, 6 route statistics (eval_edrf.npz)
    #   Agent-JEPA           (a) the OFFICIAL code (github.com/hellojais/mindrive-jepa) retrained on the bank's 10 Hz rollouts
    #                        with its own tokenizer / trainer / surprise score (experiments/us_official/jepa_official_b2d.py;
    #                        best.pt = the paper's best-validation rule), a route = the mean surprise over its 5 s
    #                        windows (1-d) or [mean, max, p90] (3-d); (b) our earlier re-implementation of Sec. 3:
    #                        the paper-literal best-validation checkpoint and the full 50-epoch schedule
    #   Traffic entropy      OUR descriptor: mean next-token entropy of the SMART traffic model run through the official
    #                        CAT-K code and checkpoint (eval_smart_ent_catk.npz); neither paper proposes it
    #   in-house rows        Route geometry, Agent density + kin., Kinematics (cmdkin), Hand-crafted risk (cmdkin+gtrisk),
    #                        and the traffic risk stack of traffic_features_220.csv (the earlier "Risk field" row)
    # Every row goes through the same two-stage Ridge readout below; that readout is ours for every row.
    return {'Min-TTC': load_features('eval_min_ttc'),
            'Agent-JEPA (official code, mean surprise)': load_features('eval_jepa_official'),
            'Agent-JEPA (official code, [mean,max,p90])': load_features('eval_jepa_official_3d'),
            'EDRF-based risk field': load_features('eval_edrf'),
            'Agent-JEPA (best-val ckpt)': load_features('eval_agentjepa_bestval'),
            'Agent-JEPA (full schedule)': load_features('eval_agentjepa_official'),
            'Traffic entropy (ours, SMART/CAT-K)': load_features('eval_smart_ent_catk'),
            'Route geometry': load_features('eval_routegeom'),
            'Agent density + kin.': kinden,
            'In-house traffic risk stack': risk,
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
    RELG = {s_: np.load(DATA / 'encoder' / f'relgraph_r2nolane_s{s_}.npz', allow_pickle=True) for s_ in RUNS}
    CTRL = {c: {s_: np.load(ctrl_npz(c, s_), allow_pickle=True) for s_ in RUNS}
            for c in CONTROLS if all(ctrl_npz(c, s_).exists() for s_ in RUNS)}
    ROWS = list(arms) + ['Oracle (resp-calibrated C)'] + [f'RelGraph R2-noLane s{s_}' for s_ in RUNS] \
        + [f'{CONTROLS[c]} s{s_}' for c in CTRL for s_ in RUNS]
    CROW = {f'{CONTROLS[c]} s{s_}': c for c in CTRL for s_ in RUNS}
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
            elif name.startswith('RelGraph') or name in CROW:
                pz = RELG[int(name[-1])] if name.startswith('RelGraph') else CTRL[CROW[name]][int(name[-1])]
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
    rg = [results[f'RelGraph R2-noLane s{s_}'] for s_ in RUNS]
    sd = lambda k: np.std([r[k] for r in rg], ddof=1)
    mn = lambda k: np.mean([r[k] for r in rg])
    print('RelGraph R2-noLane scene encoder (3 runs, lane-free — the encoder of record)  AUROC {:.3f}+-{:.3f}  MAE {:.3f}+-{:.3f}  rho {:+.3f}+-{:.3f}'.format(
        mn('auroc'), sd('auroc'), mn('mae'), sd('mae'), mn('rho'), sd('rho')))
    rho_hc = results['Hand-crafted risk (cmdkin+gtrisk)']['rho']
    d1 = [r['rho'] - rho_hc for r in rg]
    print(f'Delta rho (R2-noLane - hand-crafted risk), per run: {np.mean(d1):+.3f}+-{np.std(d1, ddof=1):.3f}')
    if CTRL:
        print('\n===== Table 3A(b) — RelGraph controls (same architecture, recipe and seeds; only the graph tensors or the ego channels differ) =====')
        print('the first row is the lane-carrying encoder earlier releases shipped; Delta rho is paired by seed against R2-noLane')
        for c in CTRL:
            rc = [results[f'{CONTROLS[c]} s{s_}'] for s_ in RUNS]
            print('{:44s} AUROC {:.3f}+-{:.3f}  MAE {:.3f}+-{:.3f}  rho {:+.3f}+-{:.3f}   Delta rho vs R2-noLane (paired by seed) {:+.3f}+-{:.3f}'.format(
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
    assert abs(mn('auroc') - 0.761) < 0.003 and abs(mn('mae') - 0.181) < 0.003 and abs(mn('rho') - 0.545) < 0.005
    for c, v in (('lane', 0.490), ('noroute', 0.520), ('sroute', 0.512), ('sa2l', 0.501), ('nospeed', 0.500),
                 ('nlnospeed', 0.547)):
        if c not in CTRL:                              # its npz is not in this checkout
            continue
        assert abs(np.mean([results[f'{CONTROLS[c]} s{s_}']['rho'] for s_ in RUNS]) - v) < 0.005, c
    print('anchors OK')


if __name__ == '__main__':
    main()
