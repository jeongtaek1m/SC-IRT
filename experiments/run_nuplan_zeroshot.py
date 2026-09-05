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

Every Delta M is computed identically for (i) the two encoder arms — C0e, the
canonical encoder, and A2e, the ego speed removed from both ego paths — three
seeds each; (ii) their label-shuffle nulls, TEN encoders per arm trained on
permuted Bench2Drive labels under the same ablation (C4r2n for C0e, C4r2e for
A2e); (iii) random q% subsets; (iv) an oracle that ranks by the response-
calibrated difficulty b_ref (in sample).

THE NULL IS MATCHED TO THE ARM STATISTIC. An arm is a mean over three seeds, so
its threshold is the 95th percentile of the C(10, 3) = 120 three-seed means of
its shuffle family (T_null), with p = (#{three-seed means >= arm} + 1) / 121.
Comparing the three-seed mean against the 95th percentile of the ten single
seeds — the single-seed threshold, still printed — inflates the threshold by
about sqrt(3) (null per-seed SD .088 against .051 for a mean of three) and was
the source of the withdrawn "neither arm clears" reading. Two threshold-free
tests are given beside it: the exact two-sample permutation over the 13 seeds
(3 arm + 10 null; 286 relabelings; p = fraction with mean difference >= the
observed, observed included), and a paired cluster bootstrap over the 218
nuPlan logs of the arm-mean-minus-null-mean contrast, the top-q% re-selected
inside every resample. The same three tests are run on the whole-panel
Spearman correlation of predicted difficulty with the observed failure rate,
the statistic with the power on this panel.

Readout: `pred_logged` at widx == 0 — the LOGGED ego trajectory, one window per
scene (the routed ego readout is synthetic and not used).

Uncertainty unit: the 218 nuPlan logs (scenes within a log are not independent).
Planner-side clustering (`atdrive.metrics.paired_cluster_boot`) is not the right
unit here: the sampling unit of this experiment is the scene, and every arm is
scored against the same 11 planners.

    python experiments/run_nuplan_zeroshot.py        # CPU, about a minute; table + json + anchors
"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atdrive.b2d import DATA

OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
SRC = DATA / 'nuplan' / 'val14_zeroshot.npz'

QS = (0.05, 0.10)
NBOOT = 1000          # per-arm log bootstrap (per-seed intervals in the json)
NBOOT_PAIR = 2000     # paired arm-minus-null log bootstrap
NRAND = 3000
FAIL_THR = 0.5        # CLS < 0.5 counts as a failure (reproduces the stored binary Y)

# arm -> (label, seeds, matched null family)
ARMS = {'C0e': ('C0e speed kept (canonical)', 3, 'C4r2n'),
        'A2e': ('A2e -speed both ego paths', 3, 'C4r2e')}
NULLS = {'C4r2n': ('NULL C4r2n label shuffle (n=10)', 10),
         'C4r2e': ('NULL C4r2e label shuffle, -speed (n=10)', 10)}


def target():
    """The 584-scenario nuPlan val14 panel: logged-ego w0 readout order, the binary
    fail matrix Y, the continuous closed-loop scores, the reference difficulty and
    every arm's predicted difficulty."""
    z = np.load(SRC, allow_pickle=True)
    Y = np.asarray(z['Y'], float)                        # 1 = planner failed the scene
    fail = np.asarray(z['fail'], float)                  # per-scene failure rate over planners
    C = np.asarray(z['cls'], float)                      # 11 x 584 closed-loop scores
    assert np.allclose(np.nanmean(Y, 0), fail, atol=1e-5)
    # the stored binary Y is exactly CLS < 0.5, so SR and enrichment are the same object
    B = np.where(np.isfinite(C), (C < FAIL_THR).astype(float), np.nan)
    ok = np.isfinite(Y) & np.isfinite(B)
    assert (Y[ok] == B[ok]).all() and ((~np.isfinite(Y)) == (~np.isfinite(C))).all()
    preds = {k: [np.asarray(z[f'pred_{k}_s{s}'], float) for s in range(n)]
             for k, n in [(k, v[1]) for k, v in ARMS.items()] + [(k, v[1]) for k, v in NULLS.items()]}
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
    """Per-arm-seed log bootstrap: 2.5 / 97.5 percentiles of the top-q statistics."""
    rng = np.random.default_rng(seed)
    out = {'dM_cls': [], 'dM_sr': [], 'enrich': []}
    for _ in range(B):
        ii = resample_logs(rng)
        d = deltas(topq(s[ii], int(round(len(ii) * k / len(s)))), T['cls'][ii], T['fail'][ii])
        for key in out:
            out[key].append(d[key])
    return {key: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for key, v in out.items()}


def paired_boot(arm, nul, k, key, B=NBOOT_PAIR, seed=1):
    """Paired log bootstrap of mean_seeds(arm) - mean_seeds(null), top-q re-selected
    in every resample: (point estimate, [2.5, 97.5] percentiles, P(<= 0))."""
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


def matched_test(av, nv):
    """The arm statistic (mean of its seeds) against its shuffle family.
    T_null = 95th percentile of the C(n_null, n_arm) null means; p_trip = fraction of
    those means >= the arm mean (+1 / (N + 1)); p_perm = the exact two-sample
    permutation over all seeds; T95_single = the single-seed 95th percentile."""
    av, nv = np.asarray(av, float), np.asarray(nv, float)
    am = float(av.mean())
    trip = np.array([nv[list(c)].mean() for c in itertools.combinations(range(len(nv)), len(av))])
    t95 = float(np.percentile(trip, 95))
    allv = np.concatenate([av, nv])
    obs = av.mean() - nv.mean()
    cnt = tot = 0
    for c in itertools.combinations(range(len(allv)), len(av)):
        rest = [i for i in range(len(allv)) if i not in c]
        tot += 1
        cnt += (allv[list(c)].mean() - allv[rest].mean()) >= obs - 1e-12
    return dict(arm=am, arm_sd=float(av.std(ddof=1)), null_mean=float(nv.mean()), null_sd=float(nv.std(ddof=1)),
                T_null=t95, clears=bool(am > t95), n_trip=int(len(trip)), n_trip_ge=int(np.sum(trip >= am)),
                p_trip=float((np.sum(trip >= am) + 1) / (len(trip) + 1)),
                p_perm=float(cnt / tot), n_perm=int(tot), n_perm_ge=int(cnt),
                T95_single=float(np.percentile(nv, 95)), n_single_ge=int(np.sum(nv >= am)))


def spearman(a, b):
    g = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(rankdata(a[g]), rankdata(b[g]))[0, 1])


def main():
    OUT.mkdir(exist_ok=True)
    res = {'panel': dict(n_scenes=len(T['tok']), n_planners=T['Y'].shape[0],
                         n_cells=T['n_cells'], n_logs=len(ULOG), base_fail=BASE,
                         M_cls_full=float(T['cls'].mean()), M_sr_full=float(1 - BASE),
                         planners=T['planners'], fail_threshold=FAIL_THR,
                         readout='pred_logged @ widx==0 (logged ego)'),
           'notes': {'null': 'T_null = 95th percentile of the 120 three-seed means of the shuffle family '
                             '(matched to the three-seed arm mean); T95_single = the single-seed 95th '
                             'percentile (not the threshold)',
                     'enrichment': 'Delta SR = base_fail x (enrichment - 1) by construction; the '
                                   'reconciliation of the two scorings is the cell-for-cell Y == (CLS < .5) '
                                   'check in target()'},
           'by_q': {}, 'spearman': {}}

    print(f"panel: {len(T['tok'])} nuPlan val14 scenarios x {T['Y'].shape[0]} planners, "
          f"{T['n_cells']} finite cells, {len(ULOG)} logs")
    print(f"M_CLS(full) = {T['cls'].mean():.4f} (mean closed-loop score)   "
          f"M_SR(full) = {1 - BASE:.4f} (CLS >= {FAIL_THR})   base failure {BASE:.4f}\n")

    # ---- per-seed sign diagnostics ----------------------------------------------
    S = T['pred']
    LAB = {k: v[0] for k, v in list(ARMS.items()) + list(NULLS.items())}
    print(f"{'arm':42}{'seed':>5}{'rho(pred,fail)':>16}{'rho(pred,b_ref)':>17}")
    RHO = {}
    for k, ss in S.items():
        RHO[k] = [spearman(s, T['fail']) for s in ss]
        for sd, s in enumerate(ss):
            print(f"{LAB[k]:42}{sd:5d}{RHO[k][sd]:+16.4f}{spearman(s, T['bref']):+17.4f}")

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
                e[f'{key}_per_seed'] = sorted(p[key] for p in pts)
                e[f'{key}_p95_single'] = float(np.percentile([p[key] for p in pts], 95))
            e['ci_per_seed'] = [boot(s, k) for s in ss]
            e['frac_of_oracle'] = float(e['dM_cls_mean'] / orc['dM_cls'])
            blk['arms'][name] = e
        # verdict vs the matched label-shuffle null, in the arm's own record
        for name, (_, _, nul) in ARMS.items():
            v = {}
            for key in ('dM_cls', 'dM_sr', 'enrich'):
                v[key] = dict(null=nul, **matched_test(blk['arms'][name][f'{key}_per_seed'],
                                                       blk['arms'][nul][f'{key}_per_seed']))
                if key != 'dM_sr':
                    v[key]['paired'] = paired_boot(S[name], S[nul], k, key)
            blk['arms'][name]['verdict'] = v
        res['by_q'][f'{q:.2f}'] = blk

    for name, (_, _, nul) in ARMS.items():
        res['spearman'][name] = dict(null=nul, **matched_test(RHO[name], RHO[nul]))

    # ---------------------------- report -----------------------------------------
    for q in QS:
        blk = res['by_q'][f'{q:.2f}']
        k = blk['k']
        print(f"\n=== top {q * 100:.0f}%   k = {k} of {len(T['tok'])} scenes "
              f"| M_CLS(full) {T['cls'].mean():.4f}  M_SR(full) {1 - BASE:.4f} ===")
        print(f"{'subset':42}{'n':>3}{'M_CLS':>8}{'dM_CLS':>9}{'T_null':>9}{'p_trip':>8}{'p_perm':>8}"
              f"{'enrich':>8}{'T_null':>8}{'p_trip':>8}   verdict (matched three-seed null)")
        o = blk['oracle_bref']
        print(f"{'ORACLE b_ref (ceiling, in sample)':42}{1:3d}{o['M_cls']:8.4f}{o['dM_cls']:+9.4f}"
              f"{'':25}{o['enrich']:8.3f}")
        r = blk['random']
        print(f"{'random q% (3000 draws)':42}{1:3d}{r['M_cls']:8.4f}{r['dM_cls']:+9.4f}{'':25}{r['enrich']:8.3f}")
        for name in ARMS:
            a = blk['arms'][name]
            vc, vs = a['verdict']['dM_cls'], a['verdict']['enrich']
            print(f"{a['label']:42}{a['n']:3d}{a['M_cls_mean']:8.4f}{a['dM_cls_mean']:+9.4f}"
                  f"{vc['T_null']:+9.4f}{vc['p_trip']:8.4f}{vc['p_perm']:8.4f}"
                  f"{a['enrich_mean']:8.3f}{vs['T_null']:8.3f}{vs['p_trip']:8.4f}   "
                  f"dM_CLS {'CLEARS' if vc['clears'] else 'DOES NOT CLEAR'} | "
                  f"enrich {'CLEARS' if vs['clears'] else 'DOES NOT CLEAR'} | "
                  f"{a['frac_of_oracle']:.1%} of the oracle")
        for name in NULLS:
            a = blk['arms'][name]
            print(f"{a['label']:42}{a['n']:3d}{a['M_cls_mean']:8.4f}{a['dM_cls_mean']:+9.4f}"
                  f"{'':25}{a['enrich_mean']:8.3f}   per-seed dM_CLS {a['dM_cls_per_seed'][0]:+.3f} .. "
                  f"{a['dM_cls_per_seed'][-1]:+.3f}; single-seed T95 {a['dM_cls_p95_single']:+.4f}; "
                  f"{a['frac_of_oracle']:.1%} of the oracle")
        print('paired log-cluster bootstrap of arm mean - shuffle mean (top-q re-selected per resample):')
        for name in ARMS:
            for key in ('dM_cls', 'enrich'):
                pb = blk['arms'][name]['verdict'][key]['paired']
                print(f"   {name} {key:7}: {pb['d']:+.4f} [{pb['ci'][0]:+.4f},{pb['ci'][1]:+.4f}]  P(<=0) {pb['p_le0']:.3f}")
    print('\n=== whole-panel Spearman(predicted difficulty, observed failure rate) ===')
    for name, v in res['spearman'].items():
        print(f"   {name}: {v['arm']:+.4f} (seeds SD {v['arm_sd']:.4f}) vs shuffle mean {v['null_mean']:+.4f}; "
              f"single-seed T95 {v['T95_single']:+.4f}; matched T_null {v['T_null']:+.4f}; "
              f"p_trip {v['p_trip']:.4f}; exact permutation p {v['p_perm']:.4f} ({v['n_perm_ge']}/{v['n_perm']})")

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
    for blk, nm, v in ((q5, 'C0e', ANC5_C0E), (q5, 'A2e', ANC5_A2E), (q10, 'C0e', ANC10_C0E), (q10, 'A2e', ANC10_A2E)):
        assert abs(blk['arms'][nm]['dM_cls_mean'] - v) < 5e-4, (blk['k'], nm, blk['arms'][nm]['dM_cls_mean'])
    for blk, nm, v in ((q5, 'C4r2n', ANC5_TN), (q5, 'C4r2e', ANC5_TE), (q10, 'C4r2n', ANC10_TN), (q10, 'C4r2e', ANC10_TE)):
        assert abs(blk['arms'][nm]['dM_cls_p95_single'] - v) < 5e-4, (blk['k'], nm)   # single-seed p95, not the threshold
    assert abs(q5['arms']['C0e']['enrich_mean'] - 1.8696) < 5e-3
    assert abs(q5['arms']['A2e']['enrich_mean'] - 1.9569) < 5e-3
    # the finding, against the matched three-seed-mean null (T_null, p_trip), the exact permutation
    # (p_perm) and the paired log bootstrap: A2e clears at q = 5% and is marginal at q = 10%; C0e is
    # borderline at q = 5% (arm .1611 vs T_null .1610, p .058, paired CI spans 0) and does not clear at
    # q = 10%; both arms clear on the whole-panel rank correlation.
    V = lambda blk, nm, key: blk['arms'][nm]['verdict'][key]
    for blk, nm, key, t, ge, perm, clears in ((q5, 'A2e', 'dM_cls', .1584, 1, 11, True),
                                             (q5, 'A2e', 'enrich', 1.804, 0, None, True),
                                             (q10, 'A2e', 'dM_cls', .1394, 6, 32, True),
                                             (q10, 'A2e', 'enrich', 1.689, 4, None, True),
                                             (q5, 'C0e', 'dM_cls', .1610, 6, None, True),
                                             (q10, 'C0e', 'dM_cls', .1508, 37, None, False)):
        v = V(blk, nm, key)
        assert abs(v['T_null'] - t) < 5e-4 and v['n_trip_ge'] == ge and v['clears'] == clears, (blk['k'], nm, key, v)
        assert perm is None or v['n_perm_ge'] == perm, (blk['k'], nm, key, v['n_perm_ge'])
    for blk, nm, lo_pos in ((q5, 'A2e', True), (q10, 'A2e', True), (q5, 'C0e', False), (q10, 'C0e', False)):
        ci = V(blk, nm, 'dM_cls')['paired']['ci']
        assert (ci[0] > 0) == lo_pos and ci[1] > 0, (blk['k'], nm, ci)
    for nm, rho, perm in (('A2e', .3108, 1), ('C0e', .2875, 3)):
        v = res['spearman'][nm]
        assert abs(v['arm'] - rho) < 1e-3 and v['n_perm_ge'] == perm and v['arm'] > v['T95_single'], (nm, v)
    print('anchors OK')


ANC5_ORC, ANC10_ORC = 0.4486, 0.3631               # oracle Delta M_CLS
ANC5_C0E, ANC5_A2E = 0.1611, 0.1783                # arm means
ANC10_C0E, ANC10_A2E = 0.1142, 0.1395
ANC5_TN, ANC5_TE = 0.1839, 0.1977                  # single-seed 95th percentiles of the shuffle families
ANC10_TN, ANC10_TE = 0.1915, 0.1878

if __name__ == '__main__':
    np.random.seed(0)
    T = target()
    BASE = float(T['fail'].mean())
    ULOG = np.unique(T['logs'])
    BYLOG = {l: np.where(T['logs'] == l)[0] for l in ULOG}
    main()
