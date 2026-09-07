#!/usr/bin/env python3
"""nuPlan val14 zero-shot retrieval (RESULTS.md, after Table 3A(b)): the DROP IN
PLANNER PERFORMANCE on the predicted-hard subset.

A scene encoder trained on Bench2Drive difficulty is applied zero-shot to the 584
nuPlan val14 scenarios and asked to rank them by predicted difficulty. The question
is not "does the top-q% contain more failures" but "how much worse do real planners
do there": for a higher-is-better planning metric M,

    Delta M_q = M(all 584) - M(top-q% by predicted difficulty),   q in {5, 10}.

M is reported in two forms, both computed from the SAME 11-planner response matrix
(11 planners x 584 scenarios, 6421 finite cells, `data/nuplan/val14_zeroshot.npz`):

  CLS  the nuPlan closed-loop score itself. The matrix is NOT binary — its finite
       cells take about 2.7k distinct values in [0, 1] — so M_CLS is the mean
       closed-loop score, averaged over the 11 planners within a scene and then over
       the scenes of the subset (every scene weighted equally). This is the primary M.
  SR   the binarised success rate, 1 - failure rate, failure = CLS < 0.5. This is the
       binarisation behind the earlier failure-rate-enrichment scoring; `target()`
       checks cell for cell that it reproduces the stored binary matrix Y, which is
       the reconciliation of the two scorings. (Delta SR = base_fail x (enrichment - 1)
       holds by construction of `deltas()` for any subset and is kept only as a
       consistency check.) Enrichment is reported next to Delta M.

Every Delta M is computed identically for (i) the three encoder arms — NLe, THE
ENCODER OF RECORD, which HAS NO LANE GRAPH (R2-noLane: no lane tokens, lane_feat,
L2L edges, A2L candidates or route_rel; ego + command + agents only, applied to the
Bench2Drive source graph and to the nuPlan target graph alike), and the two
lane-carrying controls C0e (speed kept) and A2e (the ego speed removed from both ego
paths), which are what the lane graph cost on transfer — three training seeds each,
trained on the 16-planner panel of record (`b2d_e2e16sel_response_matrix.csv`) with
the repo calibration; (ii) their label-shuffle nulls, 20 permutations x 3 training
seeds per arm, trained under the same ablation on permuted Bench2Drive route labels
(C4nl for NLe, C4r2n for C0e, C4r2e for A2e — every arm is judged against a null
trained exactly like itself); (iii) random q% subsets; (iv) an oracle that ranks by
the response-calibrated difficulty b_ref (in sample).

THE NULL IS MATCHED TO THE ARM STATISTIC AND TO ITS VARIANCE STRUCTURE. An arm is a
mean over three training seeds that all see the SAME labels, so the exchangeable
unit under the null is one labeling with three training seeds. The null family is
therefore 20 fixed permutations x 3 training seeds, the arm is compared with the
20 per-permutation three-seed means (T_null = their 95th percentile; the verdict
is the exact count, p = (#{means >= arm} + 1) / 21, floor .048, clears = p <= .05),
and a one-way variance decomposition of the 60 null runs gives the two components,
SD_perm (between labelings) and
SD_train (between training seeds under one labeling), whose combination
SD_perm^2 + SD_train^2 / 3 is the null SD of a three-seed mean; z = (arm - null
mean) / that SD and its Gaussian tail p are reported for resolution below the
1/21 floor. Averaging three single-seed shuffles that each carry a DIFFERENT
permutation (the earlier C(10, 3) construction) divides the permutation variance
by three as well and understates the threshold; that construction is withdrawn.
Also printed: the single-run 95th percentile of the 60 null runs and how many of
the arm's three seeds exceed it on their own, and a paired cluster bootstrap over
the 218 nuPlan logs of the arm-mean-minus-null-mean contrast (top-q% re-selected
inside every resample), which covers scene-sampling uncertainty only. The same
tests are run on the whole-panel Spearman correlation of predicted difficulty
with the observed failure rate.

Readout: `pred_logged` at widx == 0 — the LOGGED ego trajectory, one window per
scene (the routed ego readout is synthetic and not used).

Uncertainty unit: the 218 nuPlan logs (scenes within a log are not independent).
Planner-side clustering (`atdrive.metrics.paired_cluster_boot`) is not the right
unit here: the sampling unit of this experiment is the scene, and every arm is
scored against the same 11 planners.

    python experiments/run_nuplan_zeroshot.py        # CPU, about a minute; table + json + anchors
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atdrive.b2d import DATA

OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
SRC = DATA / 'nuplan' / 'val14_zeroshot.npz'

QS = (0.05, 0.10)
NBOOT = 1000          # per-run log bootstrap (per-seed intervals in the json)
NBOOT_PAIR = 2000     # paired arm-minus-null log bootstrap
NRAND = 3000
FAIL_THR = 0.5        # CLS < 0.5 counts as a failure (reproduces the stored binary Y)
NSEED = 3             # training seeds per labeling (arms and every null permutation)
NPERM = 20            # fixed label permutations per null family

# arm -> (label, matched null family); NLe is the encoder of record, the other two are controls
ARMS = {'NLe': ('NLe lane-free (canonical)', 'C4nl'),
        'C0e': ('C0e lane graph kept, speed kept', 'C4r2n'),
        'A2e': ('A2e lane graph kept, -speed both egos', 'C4r2e')}
NULLS = {'C4nl': f'NULL C4nl label shuffle, lane-free ({NPERM} perms x {NSEED} seeds)',
         'C4r2n': f'NULL C4r2n label shuffle ({NPERM} perms x {NSEED} seeds)',
         'C4r2e': f'NULL C4r2e label shuffle, -speed ({NPERM} perms x {NSEED} seeds)'}
PERM = np.repeat(np.arange(NPERM), NSEED)         # labeling index of every null run (perm-major order)


def target():
    """The 584-scenario nuPlan val14 panel: logged-ego w0 readout order, the binary
    fail matrix Y, the continuous closed-loop scores, the reference difficulty and
    every run's predicted difficulty (arms: 3 runs; nulls: 60 runs, perm-major)."""
    z = np.load(SRC, allow_pickle=True)
    Y = np.asarray(z['Y'], float)                        # 1 = planner failed the scene
    fail = np.asarray(z['fail'], float)                  # per-scene failure rate over planners
    C = np.asarray(z['cls'], float)                      # 11 x 584 closed-loop scores
    assert np.allclose(np.nanmean(Y, 0), fail, atol=1e-5)
    # the stored binary Y is exactly CLS < 0.5, so SR and enrichment are the same object
    B = np.where(np.isfinite(C), (C < FAIL_THR).astype(float), np.nan)
    ok = np.isfinite(Y) & np.isfinite(B)
    assert (Y[ok] == B[ok]).all() and ((~np.isfinite(Y)) == (~np.isfinite(C))).all()
    preds = {k: [np.asarray(z[f'pred_{k}_s{s}'], float) for s in range(NSEED)] for k in ARMS}
    preds.update({k: [np.asarray(z[f'pred_{k}_p{p}_s{s}'], float) for p in range(NPERM) for s in range(NSEED)]
                  for k in NULLS})
    return dict(tok=np.array([str(t) for t in z['tok']]), logs=np.array([str(l) for l in z['logs']]),
                Y=Y, fail=fail, bref=np.asarray(z['b_ref'], float), cls=np.nanmean(C, 0),
                n_cells=int(np.isfinite(C).sum()), planners=[str(p) for p in z['planners']], pred=preds)


def topq(s, k):
    """Indices of the k scenes with the largest predicted difficulty."""
    ok = np.isfinite(s)
    idx = np.where(ok)[0]
    return idx[np.argsort(-s[ok], kind='stable')[:k]]


def M(sel, cls, fail):
    """The two planning metrics on a subset of scenes (higher is better)."""
    return float(np.mean(cls[sel])), float(1.0 - np.mean(fail[sel]))


def deltas(sel, cls, fail):
    """Both Delta M and the enrichment of the subset, against the panel handed in
    (the full panel, or a bootstrap resample of it)."""
    mc, ms = M(sel, cls, fail)
    fc, fs = M(np.arange(len(cls)), cls, fail)
    return dict(M_cls=mc, M_sr=ms, dM_cls=fc - mc, dM_sr=fs - ms,
                fail=float(np.mean(fail[sel])), enrich=float(np.mean(fail[sel]) / (1 - fs)))


def resample_logs(rng):
    """One cluster-bootstrap resample of the panel: the 218 logs with replacement."""
    return np.concatenate([BYLOG[ULOG[j]] for j in rng.integers(0, len(ULOG), len(ULOG))])


def boot(s, k, B=NBOOT, seed=1):
    """Per-run log bootstrap: 2.5 / 97.5 percentiles of the top-q statistics."""
    rng = np.random.default_rng(seed)
    out = {'dM_cls': [], 'dM_sr': [], 'enrich': []}
    for _ in range(B):
        ii = resample_logs(rng)
        d = deltas(topq(s[ii], int(round(len(ii) * k / len(s)))), T['cls'][ii], T['fail'][ii])
        for key in out:
            out[key].append(d[key])
    return {key: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for key, v in out.items()}


def paired_boot(arm, nul, k, key, B=NBOOT_PAIR, seed=1):
    """Paired log bootstrap of mean_runs(arm) - mean_runs(null), top-q re-selected in
    every resample: (point estimate, [2.5, 97.5] percentiles, P(<= 0)). Scene-sampling
    uncertainty only — the labeling / training-seed variation is the matched null's job."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        ii = resample_logs(rng)
        kb = int(round(len(ii) * k / len(T['tok'])))
        cls_b, fail_b = T['cls'][ii], T['fail'][ii]
        ga = np.mean([deltas(topq(s[ii], kb), cls_b, fail_b)[key] for s in arm])
        gn = np.mean([deltas(topq(s[ii], kb), cls_b, fail_b)[key] for s in nul])
        out.append(ga - gn)
    out = np.array(out)
    pt = float(np.mean([deltas(topq(s, k), T['cls'], T['fail'])[key] for s in arm])
               - np.mean([deltas(topq(s, k), T['cls'], T['fail'])[key] for s in nul]))
    return dict(d=pt, ci=[float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
                p_le0=float(np.mean(out <= 0)))


def matched_test(av, nv, grp=PERM):
    """The arm statistic (mean of its three training seeds under the true labels) against
    the permutation-fixed null: nv holds the 60 null runs, grp their labeling index.
    T_null = 95th percentile of the 20 per-permutation three-seed means; the verdict is
    the exact count, p = (#{means >= arm} + 1) / 21, clears = p <= .05 (i.e. no null
    labeling reaches the arm). Variance components from the 60 runs: SD_train = pooled within-
    permutation SD, SD_perm from the between-permutation variance minus SD_train^2 / 3;
    the null SD of a three-seed mean is sqrt(SD_perm^2 + SD_train^2 / 3), z and its
    Gaussian tail p give resolution below the 1/21 floor. T95_single = 95th percentile
    of the 60 single runs; n_seed_clear = arm seeds above it on their own."""
    av, nv, grp = np.asarray(av, float), np.asarray(nv, float), np.asarray(grp)
    am = float(av.mean())
    G = np.unique(grp)
    pm = np.array([nv[grp == g].mean() for g in G])                      # per-permutation 3-seed means
    within = float(np.mean([nv[grp == g].var(ddof=1) for g in G]))       # SD_train^2 (pooled)
    between = float(pm.var(ddof=1))                                       # SD_perm^2 + SD_train^2 / 3
    var_perm = max(between - within / len(av), 0.0)
    sd_mean = float(np.sqrt(var_perm + within / len(av)))
    t95 = float(np.percentile(pm, 95))
    n_ge = int(np.sum(pm >= am))
    p_exact = float((n_ge + 1) / (len(pm) + 1))
    z = (am - pm.mean()) / sd_mean
    t95_single = float(np.percentile(nv, 95))
    return dict(arm=am, arm_sd=float(av.std(ddof=1)), null_mean=float(pm.mean()),
                sd_train=float(np.sqrt(within)), sd_perm=float(np.sqrt(var_perm)), sd_mean=sd_mean,
                T_null=t95, clears=bool(p_exact <= 0.05), n_perm=int(len(pm)), n_ge=n_ge,
                p=p_exact, z=float(z), p_gauss=float(norm.sf(z)),
                T95_single=t95_single, n_seed_clear=int(np.sum(av > t95_single)),
                perm_means=[float(v) for v in pm])


def spearman(a, b):
    g = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(rankdata(a[g]), rankdata(b[g]))[0, 1])


def main():
    OUT.mkdir(exist_ok=True)
    res = {'panel': dict(n_scenes=len(T['tok']), n_planners=T['Y'].shape[0],
                         n_cells=T['n_cells'], n_logs=len(ULOG), base_fail=BASE,
                         M_cls_full=float(T['cls'].mean()), M_sr_full=float(1 - BASE),
                         planners=T['planners'], fail_threshold=FAIL_THR,
                         readout='pred_logged @ widx==0 (logged ego)',
                         source_labels='b2d_e2e16sel_response_matrix.csv (panel of record), atdrive calibration'),
           'notes': {'null': f'{NPERM} fixed label permutations x {NSEED} training seeds per family; '
                             'T_null = 95th percentile of the per-permutation three-seed means; verdict = exact count, '
                             f'p = (#{{means >= arm}} + 1) / {NPERM + 1}, clears = p <= .05; z uses sqrt(SD_perm^2 + SD_train^2 / 3); '
                             f'T95_single = 95th percentile of the {NPERM * NSEED} single runs (not the threshold)',
                     'enrichment': 'Delta SR = base_fail x (enrichment - 1) by construction; the '
                                   'reconciliation of the two scorings is the cell-for-cell Y == (CLS < .5) '
                                   'check in target()'},
           'by_q': {}, 'spearman': {}}

    print(f"panel: {len(T['tok'])} nuPlan val14 scenarios x {T['Y'].shape[0]} planners, "
          f"{T['n_cells']} finite cells, {len(ULOG)} logs")
    print(f"M_CLS(full) = {T['cls'].mean():.4f} (mean closed-loop score)   "
          f"M_SR(full) = {1 - BASE:.4f} (CLS >= {FAIL_THR})   base failure {BASE:.4f}\n")

    # ---- per-run sign diagnostics ----------------------------------------------
    S = T['pred']
    LAB = {**{k: v[0] for k, v in ARMS.items()}, **NULLS}
    RHO = {k: [spearman(s, T['fail']) for s in ss] for k, ss in S.items()}
    RHOB = {k: [spearman(s, T['bref']) for s in ss] for k, ss in S.items()}
    print(f"{'run':44}{'rho(pred,fail)':>16}{'rho(pred,b_ref)':>17}")
    for k in ARMS:
        for sd in range(NSEED):
            print(f"{LAB[k] + f' seed {sd}':44}{RHO[k][sd]:+16.4f}{RHOB[k][sd]:+17.4f}")
    for k in NULLS:
        for p in range(NPERM):
            ii = np.where(PERM == p)[0]
            print(f"{LAB[k][:30] + f' perm {p}':44}"
                  f"{'  '.join(f'{RHO[k][i]:+.3f}' for i in ii):>24}   (three training seeds)")

    for q in QS:
        k = int(round(len(T['tok']) * q))
        blk = {'k': k, 'arms': {}}
        orc = deltas(topq(T['bref'], k), T['cls'], T['fail'])
        orc['ci'] = boot(T['bref'], k)
        blk['oracle_bref'] = orc
        rng = np.random.default_rng(0)
        R = [deltas(rng.choice(len(T['tok']), k, replace=False), T['cls'], T['fail']) for _ in range(NRAND)]
        blk['random'] = {key: float(np.mean([r[key] for r in R])) for key in
                         ('M_cls', 'M_sr', 'dM_cls', 'dM_sr', 'fail', 'enrich')}
        blk['random'].update({f'{key}_p95': float(np.percentile([r[key] for r in R], 95))
                              for key in ('dM_cls', 'dM_sr', 'enrich')})
        for name, ss in S.items():
            pts = [deltas(topq(s, k), T['cls'], T['fail']) for s in ss]
            e = {'label': LAB[name], 'n': len(pts), 'rho_fail_mean': float(np.mean(RHO[name]))}
            for key in ('M_cls', 'M_sr', 'dM_cls', 'dM_sr', 'fail', 'enrich'):
                e[f'{key}_mean'] = float(np.mean([p[key] for p in pts]))
                e[f'{key}_sd'] = float(np.std([p[key] for p in pts], ddof=1))
            for key in ('dM_cls', 'dM_sr', 'enrich'):
                e[f'{key}_per_run'] = [p[key] for p in pts]                 # run order (nulls: perm-major)
                e[f'{key}_p95_single'] = float(np.percentile([p[key] for p in pts], 95))
            e['ci_per_run'] = [boot(s, k) for s in ss]
            e['frac_of_oracle'] = float(e['dM_cls_mean'] / orc['dM_cls'])
            blk['arms'][name] = e
        # verdict vs the permutation-fixed label-shuffle null, in the arm's own record
        for name, (_, nul) in ARMS.items():
            v = {}
            for key in ('dM_cls', 'dM_sr', 'enrich'):
                v[key] = dict(null=nul, **matched_test(blk['arms'][name][f'{key}_per_run'],
                                                       blk['arms'][nul][f'{key}_per_run']))
                if key != 'dM_sr':
                    v[key]['paired'] = paired_boot(S[name], S[nul], k, key)
            blk['arms'][name]['verdict'] = v
        res['by_q'][f'{q:.2f}'] = blk

    for name, (_, nul) in ARMS.items():
        res['spearman'][name] = dict(null=nul, **matched_test(RHO[name], RHO[nul]))

    # ---------------------------- report -----------------------------------------
    for q in QS:
        blk = res['by_q'][f'{q:.2f}']
        k = blk['k']
        print(f"\n=== top {q * 100:.0f}%   k = {k} of {len(T['tok'])} scenes "
              f"| M_CLS(full) {T['cls'].mean():.4f}  M_SR(full) {1 - BASE:.4f} ===")
        print(f"{'subset':44}{'n':>3}{'M_CLS':>8}{'dM_CLS':>9}{'T_null':>9}{'p':>7}{'z':>7}"
              f"{'enrich':>8}{'T_null':>8}{'p':>7}   verdict (perm-fixed null, {NPERM} x {NSEED}, clears = p <= .05)")
        o = blk['oracle_bref']
        print(f"{'ORACLE b_ref (ceiling, in sample)':44}{1:3d}{o['M_cls']:8.4f}{o['dM_cls']:+9.4f}"
              f"{'':23}{o['enrich']:8.3f}")
        r = blk['random']
        print(f"{'random q% (3000 draws)':44}{1:3d}{r['M_cls']:8.4f}{r['dM_cls']:+9.4f}{'':23}{r['enrich']:8.3f}")
        for name in ARMS:
            a = blk['arms'][name]
            vc, vs = a['verdict']['dM_cls'], a['verdict']['enrich']
            print(f"{a['label']:44}{a['n']:3d}{a['M_cls_mean']:8.4f}{a['dM_cls_mean']:+9.4f}"
                  f"{vc['T_null']:+9.4f}{vc['p']:7.3f}{vc['z']:+7.2f}"
                  f"{a['enrich_mean']:8.3f}{vs['T_null']:8.3f}{vs['p']:7.3f}   "
                  f"dM_CLS {'CLEARS' if vc['clears'] else 'DOES NOT CLEAR'} | "
                  f"enrich {'CLEARS' if vs['clears'] else 'DOES NOT CLEAR'} | "
                  f"{a['frac_of_oracle']:.1%} of the oracle")
            print(f"{'':47}dM_CLS: SD_perm {vc['sd_perm']:.4f} SD_train {vc['sd_train']:.4f} "
                  f"(arm seeds SD {vc['arm_sd']:.4f}); Gaussian p {vc['p_gauss']:.3f}; "
                  f"single-run T95 {vc['T95_single']:+.4f}, arm seeds above it {vc['n_seed_clear']}/{NSEED}")
        for name in NULLS:
            a = blk['arms'][name]
            pr = a['dM_cls_per_run']
            print(f"{a['label']:44}{a['n']:3d}{a['M_cls_mean']:8.4f}{a['dM_cls_mean']:+9.4f}"
                  f"{'':23}{a['enrich_mean']:8.3f}   per-run dM_CLS {min(pr):+.3f} .. {max(pr):+.3f}; "
                  f"{a['frac_of_oracle']:.1%} of the oracle")
        print('paired log-cluster bootstrap of arm mean - shuffle mean (top-q re-selected per resample; scene sampling only):')
        for name in ARMS:
            for key in ('dM_cls', 'enrich'):
                pb = blk['arms'][name]['verdict'][key]['paired']
                print(f"   {name} {key:7}: {pb['d']:+.4f} [{pb['ci'][0]:+.4f},{pb['ci'][1]:+.4f}]  P(<=0) {pb['p_le0']:.3f}")
    print('\n=== whole-panel Spearman(predicted difficulty, observed failure rate) ===')
    for name, v in res['spearman'].items():
        print(f"   {name}: {v['arm']:+.4f} (seeds SD {v['arm_sd']:.4f}) vs null perm-means {v['null_mean']:+.4f} "
              f"(SD_perm {v['sd_perm']:.4f}, SD_train {v['sd_train']:.4f}); T_null {v['T_null']:+.4f}; "
              f"p {v['p']:.3f} ({v['n_ge']}/{v['n_perm']} perm means >= arm); z {v['z']:+.2f} (Gaussian p {v['p_gauss']:.4f}); "
              f"single-run T95 {v['T95_single']:+.4f}, arm seeds above it {v['n_seed_clear']}/{NSEED}")

    json.dump(res, open(OUT / 'nuplan_zeroshot.json', 'w'), indent=1, default=float)
    print(f"\nwrote {OUT / 'nuplan_zeroshot.json'}")

    # ---------------------------- anchors ----------------------------------------
    p = res['panel']
    assert (p['n_scenes'], p['n_planners'], p['n_cells'], p['n_logs']) == (584, 11, 6421, 218)
    assert abs(p['base_fail'] - 0.17846) < 1e-4, p['base_fail']
    assert abs(p['M_cls_full'] - 0.78082) < 1e-4, p['M_cls_full']
    assert abs(p['M_sr_full'] - 0.82154) < 1e-4, p['M_sr_full']
    for q in QS:
        b = res['by_q'][f'{q:.2f}']
        for nm, e in b['arms'].items():                       # the construction identity (consistency only)
            assert abs(e['dM_sr_mean'] - p['base_fail'] * (e['enrich_mean'] - 1)) < 1e-9, (q, nm)
    q5, q10 = res['by_q']['0.05'], res['by_q']['0.10']
    assert (q5['k'], q10['k']) == (29, 58)
    for blk, v in ((q5, ANC5_ORC), (q10, ANC10_ORC)):
        assert abs(blk['oracle_bref']['dM_cls'] - v) < 5e-4, (blk['k'], blk['oracle_bref']['dM_cls'])
    if ANCHORS is None:
        print('anchors: panel and oracle only (arm / null anchors not yet pinned)')
        return
    for blk, nm, v in ((q5, 'NLe', ANCHORS['q5_NLe']), (q5, 'C0e', ANCHORS['q5_C0e']),
                       (q5, 'A2e', ANCHORS['q5_A2e']), (q10, 'NLe', ANCHORS['q10_NLe']),
                       (q10, 'C0e', ANCHORS['q10_C0e']), (q10, 'A2e', ANCHORS['q10_A2e'])):
        assert abs(blk['arms'][nm]['dM_cls_mean'] - v) < 5e-4, (blk['k'], nm, blk['arms'][nm]['dM_cls_mean'])
    V = lambda blk, nm, key: blk['arms'][nm]['verdict'][key]
    for (qk, nm, key), (t, ge, clears) in ANCHORS['verdicts'].items():
        v = V(q5 if qk == 5 else q10, nm, key)
        assert abs(v['T_null'] - t) < 5e-4 and v['n_ge'] == ge and v['clears'] == clears, (qk, nm, key, v['T_null'], v['n_ge'])
    for nm, (rho, ge) in ANCHORS['spearman'].items():
        v = res['spearman'][nm]
        assert abs(v['arm'] - rho) < 1e-3 and v['n_ge'] == ge, (nm, v['arm'], v['n_ge'])
    print('anchors OK')


ANC5_ORC, ANC10_ORC = 0.4486, 0.3631               # oracle Delta M_CLS (labels of the target panel; unchanged)
ANCHORS = dict(                                    # panel of record, 20 permutations x 3 seeds (RESULTS.md)
    q5_NLe=0.2351, q5_C0e=0.1914, q5_A2e=0.2275,   # NLe (lane-free) is the arm of record
    q10_NLe=0.1973, q10_C0e=0.1376, q10_A2e=0.1590,
    verdicts={                                     # (q, arm, key): (T_null, #null perm means >= arm, clears)
        (5, 'NLe', 'dM_cls'): (0.1871, 0, True),
        (5, 'NLe', 'enrich'): (1.9686, 0, True),
        (10, 'NLe', 'dM_cls'): (0.1582, 0, True),
        (10, 'NLe', 'enrich'): (1.7857, 0, True),
        (5, 'A2e', 'dM_cls'): (0.1972, 0, True),
        (5, 'A2e', 'enrich'): (2.0570, 0, True),
        (5, 'C0e', 'dM_cls'): (0.2052, 2, False),
        (5, 'C0e', 'enrich'): (2.1024, 2, False),
        (10, 'A2e', 'dM_cls'): (0.1598, 1, False),
        (10, 'A2e', 'enrich'): (1.8357, 1, False),
        (10, 'C0e', 'dM_cls'): (0.1515, 3, False),
        (10, 'C0e', 'enrich'): (1.7998, 3, False),
    },
    spearman={'NLe': (0.4254, 1), 'A2e': (0.3016, 1), 'C0e': (0.2485, 1)},
)

if __name__ == '__main__':
    np.random.seed(0)
    T = target()
    BASE = float(T['fail'].mean())
    ULOG = np.unique(T['logs'])
    BYLOG = {l: np.where(T['logs'] == l)[0] for l in ULOG}
    main()
