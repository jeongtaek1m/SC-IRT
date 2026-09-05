#!/usr/bin/env python3
"""Adaptive policies under one IRT (RESULTS.md, after Table 2): complete adaptive
POLICIES — selection rule x stopping rule — scored on the ATDrive model.

The missing middle between Table 2 and the complete-system comparison.
`run_cat_objective.py` gives every order the SAME stopping rule (ours) and
varies only selection; `run_system_comparison.py` gives every system its OWN
IRT as well, so a row there mixes model, selection and stopping. This table
holds the ATDrive IRT inference and readout FIXED (exact difficulty posterior,
planner x type testlet, posterior-median SR readout, the same bank of all 220
routes) and varies BOTH the selection rule and the stopping rule together, so
each row is a published policy run as a policy on our model:

  ATDrive      Delta-R1 selection   stop c * R1_t <= eps, eps in {.05, .03}
  ATLAS-style  Fisher selection     stop SE(theta)_t <= tau, tau in {.1, .2, .3},
                                    minimum 30 routes      (ATLAS's published settings)
  Fluid-style  Fisher selection     fixed length: B = 100 (Fluid's published n_max)
                                    and B = the cost match to ATDrive at eps = .05
  theta-EIG    ability-EIG select.  stop SE(theta)_t <= tau (the ATLAS-style rule)
  Random       random order         stop c * R1_t <= eps with its own LOO c

Under a common IRT, ATLAS-style and Fluid-style share one selection rule, so
those rows differ only in where they stop; the factorial block below isolates
the two factors. "Fisher" here is `atdrive.acquisition.fisher_pick` — the 1PL
information at the ATDrive posterior ability — not `baselines.fluid_order`'s
2PL rule, so the Fluid-style rows are policies on our model, not the Table 1 /
Table 2 Fluid rows. The Random order is `run_cat_objective.py`'s draw,
RandomState(100 + 16 draw + slot) with slot the 0-3 index of the held-out
planner, a different permutation from Table 2's Random row.

THE FACTORIAL. Two more arms are built from the same trajectories — Delta-R1
with a fixed length (B = match) and Delta-R1 with an ability-SD stop whose
threshold is matched on the Delta-R1 LOO tracks — so that each factor can be
swapped alone: selection only (Fisher vs Delta-R1 at an identical fixed
budget, or under the SE stop at matched cost; theta-EIG vs Delta-R1 likewise)
and stopping only (fixed length or SE stop vs the risk stop, Delta-R1 fixed).
The mixed contrast of the policy table (Fisher + B = match vs ATDrive) swaps
both at once and is printed for reference.

FAIRNESS. Every threshold is published (ATLAS's tau and its 30-route minimum,
Fluid's n_max = 100, our eps) or fixed on the CALIBRATION planners only, from
the leave-one-planner-out records of `run_cat_objective.py` (r["loo"]): the
risk scale c per (draw, K_cal, order) is the 90th percentile of
|SR_hat_t - SR| / R1_t over t in [10, bank], and the two cost-matched
operating points (marked *, our construction) are set so that their mean LOO
cost equals ATDrive's LOO cost at eps = .05. No threshold sees an evaluation
planner. The two starred rows are ours, not the methods'.

CAVEAT, printed with the table: SE(theta) here is ATDrive's posterior SD
(241-point grid, N(0, 1) prior, testlet marginalised), not ATLAS's
1 / sqrt(sum I + 1). ATLAS's published tau are transplanted onto that scale,
where the whole bank only drives SE down to about .20-.24, so tau = .1 (and,
at K_cal >= 8, tau = .2) is unreachable and those rows exhaust the benchmark.

Nothing is recomputed: every trajectory comes from results/cat_objective.json
(Shat, R1, SE for Delta-R1 / theta-EIG / Fisher / Random, evaluation and LOO
records, written by `run_cat_objective.py --merge`). Metrics per row: mean
routes, routes / 220, the fraction that exhausts the bank, SR-MAE at the
stop, IES = (SR-MAE / SR-MAE_ref) x (routes / 55) with the reference the
uniform random order read at 55 routes with the same readout (PROTOCOL
section 7; the reference at 110 routes is printed beside it), and the paired
planner-cluster delta of SR-MAE against ATDrive at eps = .05 (clusters = the
16 planner ids, per PROTOCOL section 7).

    python experiments/run_policy_matrix.py        # table + results/policy_matrix.json + anchors
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atdrive.metrics import paired_cluster_boot, ies
from atdrive.splits import R_DRAWS

OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
KCALS = tuple(int(x) for x in os.environ.get('ATDRIVE_KCALS', '4,8,12').split(','))
NROUTES = 220               # the benchmark; a planner's own bank is its recorded subset (210-220)
T0 = 10                     # no stop before 10 routes (PROTOCOL section 4)
QUANT = 90                  # percentile of the LOO |err| / R1 ratio
EPS = (0.05, 0.03)          # ATDrive's published risk targets
ATLAS_TAUS = (0.1, 0.2, 0.3)
ATLAS_MIN = 30              # ATLAS's published minimum item count
FLUID_NMAX = 100            # Fluid's published default n_max
MATCH_EPS = 0.05            # the ATDrive operating point the starred rows match
IES_REF = 55                # IES reference budget: the uniform random order at 55 routes (PROTOCOL section 7)
IES_REF2 = 110              # printed beside it: the reference at half the bank
R1S, EIG, FIS, RND = 'Delta-R1 (metric)', 'theta-EIG (ability)', 'Fisher (ability)', 'Random'
BASE = f'ATDrive     dR1 + c*R1<={EPS[0]:.2f}'
FIXED_B = (30, 55, 70, 78, 80)   # fixed budgets of the selection-only contrast (the 30-80 route operating range)


def first_le(v, thr, tmin=1):
    """First 1-based t >= tmin with v[t-1] <= thr; the bank size if never."""
    v = np.asarray(v, float)
    tmin = min(max(tmin, 1), len(v))
    hit = np.where(v[tmin - 1:] <= thr)[0]
    return int(hit[0]) + tmin if len(hit) else len(v)


def risk_scale(loo, o):
    """c = QUANT-th percentile of |SR_hat_t - SR| / R1_t over the left-out
    calibration planners of one draw and t in [T0, bank] (R1 <= 1e-6 skipped)."""
    rr = []
    for r in loo:
        a = np.abs(np.array(r['tr'][o]['Shat'])[T0 - 1:] - r['SR'])
        p = np.array(r['tr'][o]['R1'])[T0 - 1:]
        rr += list(a[p > 1e-6] / p[p > 1e-6])
    return float(np.percentile(rr, QUANT))


def loo_cost(loo, c, eps):
    """Mean cost of the ATDrive rule on the LOO calibration tracks of one draw."""
    return float(np.mean([first_le(np.array(r['tr'][R1S]['R1']) * c, eps, T0) for r in loo]))


def tau_match(loo, target, o=FIS):
    """The SE threshold whose mean LOO cost (order o, ATLAS's 30-route minimum) is
    closest to `target`, on a grid of the LOO tracks' own SE values. Calibration
    side only; our construction."""
    cand = np.unique(np.concatenate([np.array(r['tr'][o]['SE'])[T0 - 1:] for r in loo]))
    cand = cand[::max(1, len(cand) // 240)]
    cost = np.array([np.mean([first_le(r['tr'][o]['SE'], t, ATLAS_MIN) for r in loo]) for t in cand])
    return float(cand[int(np.argmin(np.abs(cost - target)))])


def policies(E, loo_by_seed):
    """{label: (selection order, stop index per evaluation record)} for one K_cal."""
    C = {o: {s: risk_scale(L, o) for s, L in loo_by_seed.items()} for o in (R1S, RND)}
    tgt = {s: loo_cost(L, C[R1S][s], MATCH_EPS) for s, L in loo_by_seed.items()}
    BM = {s: int(round(v)) for s, v in tgt.items()}
    TM = {s: tau_match(L, tgt[s]) for s, L in loo_by_seed.items()}
    P = {}
    for eps in EPS:
        P[f'ATDrive     dR1 + c*R1<={eps:.2f}'] = (
            R1S, [first_le(np.array(r['tr'][R1S]['R1']) * C[R1S][r['seed']], eps, T0) for r in E])
    for tau in ATLAS_TAUS:
        P[f'ATLAS-style Fisher + SE<={tau:.1f}'] = (
            FIS, [first_le(r['tr'][FIS]['SE'], tau, ATLAS_MIN) for r in E])
    P['ATLAS-style Fisher + SE<=tau_m *'] = (
        FIS, [first_le(r['tr'][FIS]['SE'], TM[r['seed']], ATLAS_MIN) for r in E])
    P[f'Fluid-style Fisher + fixed B={FLUID_NMAX}'] = (
        FIS, [min(FLUID_NMAX, len(r['tr'][FIS]['Shat'])) for r in E])
    P['Fluid-style Fisher + fixed B=match *'] = (
        FIS, [min(BM[r['seed']], len(r['tr'][FIS]['Shat'])) for r in E])
    for tau in ATLAS_TAUS:
        P[f'theta-EIG   EIG + SE<={tau:.1f}'] = (
            EIG, [first_le(r['tr'][EIG]['SE'], tau, ATLAS_MIN) for r in E])
    for eps in EPS:
        P[f'Random      order + c*R1<={eps:.2f}'] = (
            RND, [first_le(np.array(r['tr'][RND]['R1']) * C[RND][r['seed']], eps, T0) for r in E])
    return P, C, BM, TM


def factorial(E, loo_by_seed, P, BM):
    """The arms that swap one factor at a time (RESULTS.md, 'the factorial that
    carries the claim'): {arm label: (order, stop index per evaluation)}."""
    tgt = {s: loo_cost(L, risk_scale(L, R1S), MATCH_EPS) for s, L in loo_by_seed.items()}
    TMR = {s: tau_match(L, tgt[s], R1S) for s, L in loo_by_seed.items()}      # matched on the Delta-R1 SE track
    fixed = lambda o: [min(BM[r['seed']], len(r['tr'][o]['Shat'])) for r in E]
    A = {'A  dR1 + c*R1<=.05 (ATDrive)': P[BASE],
         'B  dR1 + fixed B=match': (R1S, fixed(R1S)),
         'C  Fisher + fixed B=match': P['Fluid-style Fisher + fixed B=match *'],
         'E  dR1 + SE<=tau_m': (R1S, [first_le(r['tr'][R1S]['SE'], TMR[r['seed']], ATLAS_MIN) for r in E]),
         'F  Fisher + SE<=tau_m': P['ATLAS-style Fisher + SE<=tau_m *'],
         'G  EIG + fixed B=match': (EIG, fixed(EIG))}
    for B in FIXED_B:
        for o, lab in ((R1S, 'dR1'), (FIS, 'Fisher'), (EIG, 'EIG')):
            A[f'fixed{B} {lab}'] = (o, [min(B, len(r['tr'][o]['Shat'])) for r in E])
    return A


def cell(E, o, ts):
    """(routes, |err| at the stop, bank-exhausted flag) over the evaluations."""
    t = np.array(ts, float)
    e = np.array([abs(r['tr'][o]['Shat'][int(x) - 1] - r['SR']) for r, x in zip(E, ts)])
    cap = np.array([float(int(x) == len(r['tr'][o]['Shat'])) for r, x in zip(E, ts)])
    return t, e, cap


def alt_ref(alt, K, B=IES_REF):
    """The same reference read on the OTHER saved random-order draw (the Random
    order of results/adaptive.json, a different permutation per evaluation).
    The reference is one order per evaluation, so this is the honest error bar
    on the IES denominator."""
    v = [abs(r['Random']['Shat'][B - 1] - r['SR']) for r in alt if r['K'] == K]
    return float(np.mean(v)) if v else None


def report(recs, alt):
    LOO = [r for r in recs if r['loo']]
    EVA = [r for r in recs if not r['loo']]
    seeds = sorted(set(r['seed'] for r in recs))
    print(f'\n{len(EVA)} planner evaluations ({len(KCALS)} K_cal x {len(seeds)} draws x 4 evaluation '
          f'planners) + {len(LOO)} leave-one-out calibration tracks')
    print('\n===== threshold provenance =====')
    print(f'  ATLAS-style tau in {ATLAS_TAUS}, minimum {ATLAS_MIN} routes : ATLAS\'s published settings, a priori')
    print(f'  Fluid-style fixed B = {FLUID_NMAX}                            : Fluid\'s published default n_max')
    print(f'  ATDrive eps in {EPS}                          : published, a priori')
    print(f'  ATDrive / Random risk scale c                        : LOO {QUANT}th pct |err|/R1, CALIBRATION planners')
    print(f'  * B=match, tau_m                                     : LOO mean cost matched to ATDrive at '
          f'eps={MATCH_EPS} on the CALIBRATION planners (our construction, not the methods\' own)')
    print('  No threshold on this table was tuned on an evaluation planner.')
    print('\n===== ability-SD floor: SE(theta) after the WHOLE bank (why the small tau are unreachable) =====')
    for K in KCALS:
        mn = np.array([min(r['tr'][FIS]['SE']) for r in EVA if r['K'] == K])
        print(f'  K_cal {K:2d}: min SE over the bank  mean {mn.mean():.3f}  range [{mn.min():.3f}, {mn.max():.3f}]  '
              + '  '.join(f'reach {t:.1f}: {np.mean(mn <= t):3.0%}' for t in ATLAS_TAUS))
    res, pooled, fpool = {}, {}, {}
    for K in KCALS:
        E = sorted([r for r in EVA if r['K'] == K], key=lambda r: (r['seed'], r['js']))
        cl = [r['j'] for r in E]                      # the 16 unique planner ids (PROTOCOL section 7)
        lbs = {s: [r for r in LOO if r['seed'] == s and r['K'] == K] for s in seeds}
        P, C, BM, TM = policies(E, lbs)
        ref = float(np.mean([abs(r['tr'][RND]['Shat'][IES_REF - 1] - r['SR']) for r in E]))
        ref2 = float(np.mean([abs(r['tr'][RND]['Shat'][IES_REF2 - 1] - r['SR']) for r in E]))
        alt1 = alt_ref(alt, K)
        base = cell(E, *P[BASE])[1]
        print(f'\n===== K_cal = {K} =====  IES reference (uniform Random read at {IES_REF} routes, PROTOCOL section 7) '
              f'SR-MAE {ref:.4f}; at {IES_REF2} routes {ref2:.4f}')
        if alt1 is not None:                          # the reference is ONE random order per evaluation
            print(f'   reference sensitivity: the independent random-order draw of results/adaptive.json reads '
                  f'{alt1:.4f} at {IES_REF} routes,\n   which would scale every IES in this cell by '
                  f'{ref / alt1:.2f} — the IES column is only as stable as its single-draw denominator')
        print(f'   c median: dR1 {np.median(list(C[R1S].values())):.2f}  Random {np.median(list(C[RND].values())):.2f}'
              f'   B=match median {np.median(list(BM.values())):.0f}   tau_m median {np.median(list(TM.values())):.3f}')
        print(f'   {"policy":36s} {"routes":>7s} {"/220":>5s} {"cap":>4s} {"SR-MAE":>8s} {"IES":>6s} {"IES@110":>8s}'
              f'   delta of SR-MAE vs ATDrive eps={EPS[0]:.2f}')
        for lab, (o, ts) in P.items():
            t, e, cap = cell(E, o, ts)
            d, lo, hi = (0.0, 0.0, 0.0) if lab == BASE else paired_cluster_boot(e, base, cl)
            print(f'   {lab:36s} {t.mean():7.1f} {t.mean() / NROUTES:5.0%} {cap.mean():4.0%} {e.mean():8.4f} '
                  f'{ies(e.mean(), t.mean(), ref, IES_REF):6.2f} {ies(e.mean(), t.mean(), ref2, IES_REF2):8.2f}   '
                  + ('      —' if lab == BASE else f'{d:+.4f} [{lo:+.4f},{hi:+.4f}]'))
            res[f'K{K}|{lab.strip()}'] = {'routes': float(t.mean()), 'frac': float(t.mean() / NROUTES),
                                          'cap': float(cap.mean()), 'mae': float(e.mean()),
                                          'ies': float(ies(e.mean(), t.mean(), ref, IES_REF)),
                                          'ies110': float(ies(e.mean(), t.mean(), ref2, IES_REF2)),
                                          'delta_vs_atdrive05': [float(d), float(lo), float(hi)],
                                          'ies_ref_mae': ref, 'ies_ref_mae_110': ref2, 'ies_ref_mae_alt_draw': alt1}
            pooled.setdefault(lab, {'t': [], 'e': [], 'cl': []})
            pooled[lab]['t'] += list(t)
            pooled[lab]['e'] += list(e)
            pooled[lab]['cl'] += cl
        for lab, (o, ts) in factorial(E, lbs, P, BM).items():
            t, e, _ = cell(E, o, ts)
            fpool.setdefault(lab, {'t': [], 'e': [], 'cl': []})
            fpool[lab]['t'] += list(t)
            fpool[lab]['e'] += list(e)
            fpool[lab]['cl'] += cl
    print(f'\n===== the equal-cost rows, pooled over K_cal {KCALS} ({len(EVA)} evaluations, '
          f'clusters = the {len(set(r["j"] for r in EVA))} planner ids) =====')
    print('   (the starred rows are the only ones in the table above that spend what ATDrive spends; every'
          '\n    other delta above is confounded with cost, which is what the IES column prices in)')
    bp = np.array(pooled[BASE]['e'])
    for lab in (BASE, 'Fluid-style Fisher + fixed B=match *', 'ATLAS-style Fisher + SE<=tau_m *'):
        t, e = np.array(pooled[lab]['t']), np.array(pooled[lab]['e'])
        d, lo, hi = (0.0, 0.0, 0.0) if lab == BASE else paired_cluster_boot(e, bp, pooled[lab]['cl'])
        print(f'   {lab:36s} routes {t.mean():5.1f}  SR-MAE {e.mean():.4f}'
              + ('' if lab == BASE else f'   d {d:+.4f} [{lo:+.4f},{hi:+.4f}]'))
        res[f'pooled|{lab.strip()}'] = {'routes': float(t.mean()), 'mae': float(e.mean()),
                                        'delta_vs_atdrive05': [float(d), float(lo), float(hi)]}
    print('\n===== the factorial: one factor swapped at a time, pooled over K_cal (same clusters) =====')
    D = lambda x, y: paired_cluster_boot(fpool[x]['e'], fpool[y]['e'], fpool[x]['cl'])
    res['factorial'] = {}
    for lab in ('A  dR1 + c*R1<=.05 (ATDrive)', 'B  dR1 + fixed B=match', 'C  Fisher + fixed B=match',
                'E  dR1 + SE<=tau_m', 'F  Fisher + SE<=tau_m', 'G  EIG + fixed B=match'):
        t, e = np.array(fpool[lab]['t']), np.array(fpool[lab]['e'])
        print(f'   {lab:32s} routes {t.mean():5.1f}  SR-MAE {e.mean():.4f}')
        res['factorial'][lab.strip()] = {'routes': float(t.mean()), 'mae': float(e.mean())}
    print('   contrast (+ = the first arm worse):')
    for x, y, what in (('C', 'B', 'selection only: Fisher vs dR1, identical fixed budget'),
                       ('F', 'E', 'selection only: Fisher vs dR1, SE stop at matched cost'),
                       ('G', 'B', 'selection only: EIG vs dR1, identical fixed budget'),
                       ('B', 'A', 'stopping only: fixed length vs risk stop, dR1 fixed'),
                       ('E', 'A', 'stopping only: SE stop vs risk stop, dR1 fixed'),
                       ('F', 'C', 'stopping only: SE stop vs fixed length, Fisher fixed'),
                       ('C', 'A', 'both swapped at once (the mixed contrast of the table above)')):
        kx = next(k for k in fpool if k.startswith(x + '  '))
        ky = next(k for k in fpool if k.startswith(y + '  '))
        d, lo, hi = D(kx, ky)
        print(f'   {x} - {y}  {d:+.4f} [{lo:+.4f},{hi:+.4f}]   {what}')
        res['factorial'][f'{x}-{y}'] = [float(d), float(lo), float(hi)]
    print('   selection at exactly fixed budgets (no stopping rule), Fisher - dR1 and EIG - dR1:')
    for B in FIXED_B:
        d, lo, hi = D(f'fixed{B} Fisher', f'fixed{B} dR1')
        d2, lo2, hi2 = D(f'fixed{B} EIG', f'fixed{B} dR1')
        print(f'   B = {B:3d}  Fisher - dR1 {d:+.4f} [{lo:+.4f},{hi:+.4f}]   EIG - dR1 {d2:+.4f} [{lo2:+.4f},{hi2:+.4f}]'
              f'   (dR1 {np.mean(fpool[f"fixed{B} dR1"]["e"]):.4f}, Fisher {np.mean(fpool[f"fixed{B} Fisher"]["e"]):.4f})')
        res['factorial'][f'fixed{B}|Fisher-dR1'] = [float(d), float(lo), float(hi)]
        res['factorial'][f'fixed{B}|EIG-dR1'] = [float(d2), float(lo2), float(hi2)]
    print('\n   IES note: a policy that exhausts the bank reports SR-MAE 0 by construction (every route is'
          '\n   observed, so the readout is the success rate itself) and therefore IES 0; for those rows the'
          '\n   routes and cap columns, not IES, are the honest description.')
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = OUT / 'cat_objective.json'                      # the trajectories of record (run_cat_objective.py --merge)
    assert src.exists(), f'{src} not found; run experiments/run_cat_objective.py --merge first'
    recs = json.load(open(src))
    assert len(set(r['seed'] for r in recs)) == R_DRAWS, len(set(r['seed'] for r in recs))
    ap = OUT / 'adaptive.json'
    alt = json.load(open(ap)) if ap.exists() else []      # the second random-order draw, for the IES denominator
    res = report(recs, alt)
    json.dump(res, open(OUT / 'policy_matrix.json', 'w'), indent=1)
    print(f'\nwritten: {OUT / "policy_matrix.json"}')

    # anchors: the 16-draw run of record. The ATDrive rows are also the cross-check
    # against RESULTS.md Table 2 / syscmp (83.5 / .0272, 79.0, 70.3 / .0207 at eps = .05),
    # reproduced here from a different results file and an independently recomputed c.
    for key, field, v, tol in ((f'K4|ATDrive     dR1 + c*R1<=0.05', 'routes', 83.5, 0.5),
                               (f'K4|ATDrive     dR1 + c*R1<=0.05', 'mae', .0272, .002),
                               (f'K8|ATDrive     dR1 + c*R1<=0.05', 'routes', 79.0, 0.5),
                               (f'K12|ATDrive     dR1 + c*R1<=0.05', 'routes', 70.3, 0.5),
                               (f'K12|ATDrive     dR1 + c*R1<=0.05', 'mae', .0207, .002),
                               ('K12|ATLAS-style Fisher + SE<=0.1', 'routes', 217.4, 0.5),
                               ('K12|ATLAS-style Fisher + SE<=0.1', 'cap', 1.00, .01),
                               ('K12|ATLAS-style Fisher + SE<=0.3', 'routes', 99.9, 1.0),
                               ('K12|ATLAS-style Fisher + SE<=0.3', 'mae', .0199, .002),
                               ('K12|ATLAS-style Fisher + SE<=tau_m *', 'routes', 68.9, 1.0),
                               ('K12|Fluid-style Fisher + fixed B=match *', 'routes', 69.7, 0.5),
                               ('K12|Fluid-style Fisher + fixed B=match *', 'mae', .0270, .002),
                               ('K12|Fluid-style Fisher + fixed B=100', 'mae', .0203, .002),
                               ('K4|theta-EIG   EIG + SE<=0.3', 'mae', .0274, .002),
                               ('K12|Random      order + c*R1<=0.05', 'routes', 112.3, 1.0),
                               ('K12|Random      order + c*R1<=0.05', 'mae', .0199, .002),
                               ('K12|ATDrive     dR1 + c*R1<=0.05', 'ies_ref_mae', .0393, .001),
                               ('pooled|Fluid-style Fisher + fixed B=match *', 'mae', .0304, .002),
                               ('pooled|ATDrive     dR1 + c*R1<=0.05', 'mae', .0253, .002)):
        got = res[key][field]
        assert abs(got - v) < tol, (key, field, got, v)
    d, lo, hi = res['pooled|Fluid-style Fisher + fixed B=match *']['delta_vs_atdrive05']
    assert abs(d - .0051) < .002 and lo < 0 < hi, (d, lo, hi)      # the mixed contrast (selection and stopping swapped at once)
    # the factorial: selection separates, stopping does not
    F = res['factorial']
    for arm, routes, mae in (('B  dR1 + fixed B=match', 77.5, .0242), ('C  Fisher + fixed B=match', 77.5, .0304),
                             ('E  dR1 + SE<=tau_m', 76.2, .0242), ('F  Fisher + SE<=tau_m', 76.3, .0308),
                             ('G  EIG + fixed B=match', 77.5, .0274)):
        assert abs(F[arm]['routes'] - routes) < .5 and abs(F[arm]['mae'] - mae) < .002, (arm, F[arm])
    for key, v, sign in (('C-B', .0062, 1), ('F-E', .0066, 1), ('G-B', .0031, 1),
                         ('B-A', -.0011, 0), ('E-A', -.0011, 0), ('F-C', .0004, 0),
                         ('fixed55|Fisher-dR1', .0100, 1), ('fixed78|Fisher-dR1', .0063, 1)):
        d, lo, hi = F[key]
        assert abs(d - v) < .002 and ((lo > 0) if sign else (lo < 0 < hi)), (key, d, lo, hi)
    print('anchors OK')


if __name__ == '__main__':
    np.random.seed(0)
    main()
