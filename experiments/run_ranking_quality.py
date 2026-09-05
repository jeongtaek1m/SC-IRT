#!/usr/bin/env python3
"""Ranking quality of the complete-system trajectories (RESULTS.md, a subsection
of the complete-system comparison; results/ranking_quality.json).

The complete-system table (`run_system_comparison.py`, results/syscmp.json)
reports cost and SR-MAE only. SR-MAE says how far an estimate is from the
truth on the rate scale; it does not say whether the estimate would *place*
the new planner correctly on the leaderboard, which is the decision the tool
is actually used for. This script scores the ranking metrics on the saved
trajectories — nothing is refitted here. Three readings:

  INSERTION (the primary table).
  (a) INSERTION ACCURACY.  For a held-out planner j*, its estimated SR_hat is
      compared with the TRUE full-benchmark SR of each of the 12 calibration
      planners of that draw, and the fraction of those 12 comparisons that
      agree with the truth is recorded. Averaged over the 64 evaluations of a
      cell (16 draws x 4 held-out planners). This is NOT RESULTS.md's
      "pairwise rank correct" of Table 1, which orders the four held-out
      planners of a draw against each other with both sides estimated; the
      two differ by up to 6 points on identical rows.
  (b) ABSOLUTE RANK ERROR.  j* is inserted into the descending-SR ranking of
      those same 12 planners, once at SR_hat and once at its true SR; the
      score is |r_hat - r|, r = 1 + #{k : SR_k > SR} on 1..13.
  With no truth tie, |r_hat - r| = 12 x (1 - insertion accuracy) identically,
  so the two are one number in two units, and both are a deterministic
  coarsening of the SR error that SR-MAE averages.

  CO-ESTIMATED (16-planner board).  All four held-out planners of a draw carry
  their own estimates on the 16-planner leaderboard; held-out-vs-held-out
  pairs are estimate-vs-estimate. The identity above no longer holds.

  WITHIN-DRAW (Table 1's definition, "pairwise rank correct").  Each evaluation
  is scored on its 3 pairs with the other held-out planners of its draw, both
  sides estimated. The ATDrive fixed-budget rows reproduce Table 1's column
  (91.7 / 90.6 / 89.6 / 100.0 at K4 B30 / K8 B55 / K12 B30 / K12 B165), which
  is asserted.

  For each reading the separation cross-tab against ATDrive eps = .05 is
  reported: does the ranking metric separate rows that SR-MAE ties?

TIES (declared, not implicit).
  * Truth tie, SR_k == SR (exact): the comparison carries no information, so
    it scores 0.5 in (a) and does not increment either rank in (b) — the new
    planner is placed at the top of a tied block on both sides, so a truth tie
    can never by itself create rank error. On this panel the count is 0: the
    16 true SRs are distinct (closest pair .6727 / .6636, gap .0091), which is
    why the choice of convention does not move any number below. The count is
    reported per cell so this stays checkable on another panel.
  * Estimate tie, SR_hat == SR_k exactly with SR != SR_k: the estimator
    declines to order the pair, scored 0.5 in (a); in (b) the tied incumbent
    does not increment r_hat (same top-of-block convention). Counted and
    reported.

COMPARISON SET — which routes each true SR is averaged over.
  Every planner's true SR is its success rate over ITS OWN recorded routes
  within the 220-route benchmark (3,482 of 3,520 cells observed, so 210-220
  routes per planner). That is the right set because it is the same estimand
  on both sides of every comparison: the readout of each system reconstructs
  the held-out planner's success rate over exactly that planner's recorded
  bank (`atlas_pirt` / `fluid_pirt` divide by n = len(bank), and ATDrive's
  posterior median is the median of sum_obs + T over the same n), and the
  number a calibration planner is published with on this benchmark is its
  success rate over the routes it has a recorded outcome for. Comparing a
  reconstruction of j*'s own-bank rate against the incumbents' own-bank rates
  is therefore comparing like with like, and it is the comparison the user of
  the tool makes when reading a leaderboard. The alternative — restricting
  both planners of a pair to the routes recorded for both — is reported as a
  robustness check ('robust_intersection'): it flips at most 3 of the 768
  comparisons of a cell (max shift .004 in pairwise accuracy, .05 of a rank).

  The comparison set is the 12 calibration planners of the draw for every
  K_cal, including at K_cal = 4 where the model was fitted from only 4 of
  them: the 12 true SRs are published quantities, not estimates, so all 12 are
  available to place a new planner regardless of how many were used to
  calibrate the bank. Holding the set fixed also keeps the K_cal = 4 / 8 / 12
  rows on one scale.

OPERATING POINTS. Every system is scored at its own stops, taken from
`run_system_comparison.rows_for` so they are the same stops the complete-system
table reports (ATLAS at tau in {.1, .2, .3}, Fluid at B = 100 / B = match /
SE <= delta*, ATDrive at eps in {.05, .03}), plus the ATDrive order at the
fixed budgets B in {30, 55, 110, 165} for context.

    python experiments/run_ranking_quality.py       # CPU; table + json + anchors
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from atdrive.b2d import Panel
from atdrive.splits import up_split
from atdrive.metrics import paired_cluster_boot
from run_system_comparison import rows_for            # the complete-system operating points, verbatim

OUT = Path(os.environ.get('ATDRIVE_RESULTS_DIR', Path(__file__).resolve().parents[1] / 'results'))
KCALS = (4, 8, 12)
BGRID = (30, 55, 110, 165)
NCAL = 12                     # calibration planners per draw = the leaderboard a new planner is placed on
BASE = 'ATDrive eps=0.05'     # the row every delta is taken against, as in the complete-system table


def true_sr(panel):
    """{planner index: success rate over that planner's own recorded routes}."""
    return {k: float(panel.bank_rows(panel.allr, k)[1].mean()) for k in range(panel.J)}


def cal_true(panel, seed):
    """(the 12 calibration planner ids of the draw, their true SR) — the leaderboard."""
    hp, _ = up_split(seed, panel.utypes, panel.J)
    cs = [c for c in range(panel.J) if c not in hp]
    T = true_sr(panel)
    return cs, np.array([T[c] for c in cs])


def pair_acc(shat, strue, sk):
    """(a) insertion accuracy: fraction of the 12 comparisons ordered as the truth; ties per the docstring."""
    tie_t = sk == strue
    tie_e = (~tie_t) & (sk == shat)
    ok = (np.sign(shat - sk) == np.sign(strue - sk)).astype(float)
    return float(np.where(tie_t | tie_e, 0.5, ok).mean()), int(tie_t.sum()), int(tie_e.sum())


def rank_err(shat, strue, sk):
    """(b) |r_hat - r| with r = 1 + #{k : SR_k > SR}; a tie does not increment."""
    return float(abs(int((sk > shat).sum()) - int((sk > strue).sum())))


def score(recs, sysname, stops, LB):
    """Per-evaluation (|err|, pairwise accuracy, |rank error|) at one operating point."""
    e, a, d, tt, te = [], [], [], 0, 0
    for r, t in zip(recs, stops):
        sh, st, sk = r[sysname]['Shat'][int(t) - 1], r['SR'], LB[r['seed']]
        acc, n_tt, n_te = pair_acc(sh, st, sk)
        e.append(abs(sh - st))
        a.append(acc)
        d.append(rank_err(sh, st, sk))
        tt += n_tt
        te += n_te
    return np.array(e), np.array(a), np.array(d), tt, te


def rows_all(rs):
    """{label: (system, stop array)} — the complete-system stops plus the fixed-budget ATDrive rows."""
    out = {lab: (lab.split()[0], t) for lab, (t, _e, _c) in rows_for(rs).items()}
    for B in BGRID:
        out[f'ATDrive B={B}'] = ('ATDrive', np.array([min(B, r['n']) for r in rs]))
    return out


def coestimated(recs, T, LBI, res):
    """The same trajectories with the held-out planners at their own estimates:
    (i) on the 16-planner board, (ii) within the draw (Table 1's definition).
    Returns the two result blocks (per-row accuracies, deltas, cross-tabs)."""
    J = len(T)
    true = np.array([T[k] for k in range(J)])
    out = {}
    for reading in ('coestimated', 'withindraw'):
        blk, sepr, macro = {}, [], {}
        for K in KCALS:
            rs = [r for r in recs if r['K'] == K]
            js = [r['js'] for r in rs]
            HP = {s: [r['js'] for r in rs if r['seed'] == s] for s in set(r['seed'] for r in rs)}
            R = rows_all(rs)
            A = {}
            for lab, (sysname, t) in R.items():
                sh = {(r['seed'], r['js']): r[sysname]['Shat'][int(ti) - 1] for r, ti in zip(rs, t)}
                acc = []
                for r in rs:
                    s, j = r['seed'], r['js']
                    if reading == 'withindraw':
                        oth = [q for q in HP[s] if q != j]
                        est = {q: sh[(s, q)] for q in HP[s]}
                    else:
                        oth = [q for q in range(J) if q != j]
                        est = {q: (sh[(s, q)] if q in HP[s] else true[q]) for q in range(J)}
                    acc.append(float(np.mean([(est[j] > est[q]) == (true[j] > true[q]) for q in oth])))
                A[lab] = np.array(acc)
            base = A[BASE]
            for lab in R:
                d, lo, hi = (0.0, 0.0, 0.0) if lab == BASE else paired_cluster_boot(A[lab], base, js)
                blk[f'K{K}|{lab}'] = {'acc': float(A[lab].mean()), 'd_acc': [float(d), float(lo), float(hi)]}
                macro.setdefault(lab, []).append(float(A[lab].mean()))
                if lab != BASE:
                    elo, ehi = res[f'K{K}|{lab}']['d_mae'][1:]
                    sepr.append((K, lab, elo * ehi > 0, lo * hi > 0))
        both = sum(1 for _K, _l, m, r in sepr if m and r)
        monly = [f'K{K}|{l}' for K, l, m, r in sepr if m and not r]
        ronly = [f'K{K}|{l}' for K, l, m, r in sepr if r and not m]
        neither = sum(1 for _K, _l, m, r in sepr if not m and not r)
        name = ('co-estimated: all four held-out planners at their own estimates on the 16-planner board'
                if reading == 'coestimated' else
                'within-draw: the four held-out planners ordered against each other (Table 1\'s "pairwise rank correct")')
        print(f'\n===== {name} =====')
        print(f'   {"row":20s} ' + ' '.join(f'{"K" + str(K):>8s}' for K in KCALS) + f'   d(acc) vs {BASE} at K_cal = 4 / 8 / 12')
        for lab in macro:
            print(f'   {lab:20s} ' + ' '.join(f'{v:8.4f}' for v in macro[lab]) + '   '
                  + ('   —' if lab == BASE else '  '.join('{:+.4f} [{:+.4f},{:+.4f}]'.format(*blk[f'K{K}|{lab}']['d_acc'])
                                                          for K in KCALS)))
        print(f'   cross-tab vs SR-MAE: both separate {both}, SR-MAE only {len(monly)}, ranking only {len(ronly)} '
              f'({", ".join(ronly) if ronly else "none"}), both tie {neither}')
        out[reading] = {'rows': blk, 'macro': {l: float(np.mean(v)) for l, v in macro.items()},
                        'both': both, 'mae_only': monly, 'rank_only': ronly, 'neither': neither}
    return out['coestimated'], out['withindraw']


def report(recs, panel):
    recs = sorted(recs, key=lambda r: (r['seed'], r['K'], r['js']))
    T = true_sr(panel)
    for r in recs:                                  # the saved SR is the own-bank rate this table ranks on
        assert abs(r['SR'] - T[r['js']]) < 1e-12, (r['seed'], r['js'])
    LB = {s: cal_true(panel, s)[1] for s in sorted(set(r['seed'] for r in recs))}
    LBI = {s: cal_true(panel, s)[0] for s in LB}

    print(f'\n{len(recs)} planner evaluations; leaderboard = the {NCAL} calibration planners of each draw')
    print('true SR of every planner over ITS OWN recorded routes of the 220-route benchmark '
          f'({sum(len(panel.bank_rows(panel.allr, k)[0]) for k in range(panel.J))} of {220 * panel.J} cells observed):')
    print('   ' + '  '.join(f'{panel.names[k]} {T[k]:.4f}' for k in np.argsort([-T[k] for k in range(panel.J)])[:8]))
    print('   ' + '  '.join(f'{panel.names[k]} {T[k]:.4f}' for k in np.argsort([-T[k] for k in range(panel.J)])[8:]))
    print(f'   exact ties among the {panel.J} true SR: {panel.J - len(set(T.values()))}   '
          f'(closest pair gap {min(np.diff(sorted(T.values()))):.4f})')

    res, acc_by, sepr, pooled = {}, {}, [], []
    for K in KCALS:
        rs = [r for r in recs if r['K'] == K]
        js = [r['js'] for r in rs]
        R = rows_all(rs)
        base = score(rs, 'ATDrive', R[BASE][1], LB)
        print(f'\n-- K_cal = {K} --   ({len(rs)} evaluations)')
        print(f'   {"row":20s} {"routes":>7s} {"SR-MAE":>8s} {"ins-acc":>9s} {"|drank|":>8s} {"rank=0":>7s} '
              f'{"rank<=1":>8s}   d(ins-acc) vs {BASE}')
        for lab, (sysname, t) in R.items():
            e, a, d, tt, te = score(rs, sysname, t, LB)
            isb = lab == BASE
            da, alo, ahi = (0.0, 0.0, 0.0) if isb else paired_cluster_boot(a, base[1], js)
            dd, dlo, dhi = (0.0, 0.0, 0.0) if isb else paired_cluster_boot(d, base[2], js)
            de, elo, ehi = (0.0, 0.0, 0.0) if isb else paired_cluster_boot(e, base[0], js)
            if lab != 'ATLAS  tau=0.1':               # exhausts the bank: |err| = 0 by construction, not a ranking result
                pooled.append(np.stack([e, d]))
            if not isb:
                sepr.append((K, lab, elo * ehi > 0, dlo * dhi > 0))
            print(f'   {lab:20s} {np.mean(t):7.1f} {e.mean():8.4f} {a.mean():9.4f} {d.mean():8.3f} '
                  f'{np.mean(d == 0):7.0%} {np.mean(d <= 1):8.0%}   '
                  + ('   —' if isb else f'{da:+.4f} [{alo:+.4f},{ahi:+.4f}]'))
            res[f'K{K}|{lab}'] = {'rollouts': float(np.mean(t)), 'mae': float(e.mean()),
                                  'pair_acc': float(a.mean()), 'rank_err': float(d.mean()),
                                  'rank_exact': float(np.mean(d == 0)), 'rank_le1': float(np.mean(d <= 1)),
                                  'ties_truth': tt, 'ties_estimate': te,
                                  'd_pair_acc': [float(da), float(alo), float(ahi)],
                                  'd_rank_err': [float(dd), float(dlo), float(dhi)],
                                  'd_mae': [float(de), float(elo), float(ehi)]}
            acc_by.setdefault(lab, []).append((a.mean(), d.mean(), e.mean()))
    print(f'\n===== macro average over K_cal in {KCALS} =====')
    print(f'   {"row":20s} {"SR-MAE":>8s} {"ins-acc":>9s} {"|drank|":>8s}')
    for lab, v in acc_by.items():
        v = np.array(v)
        print(f'   {lab:20s} {v[:, 2].mean():8.4f} {v[:, 0].mean():9.4f} {v[:, 1].mean():8.3f}')
        res[f'macro|{lab}'] = {'mae': float(v[:, 2].mean()), 'pair_acc': float(v[:, 0].mean()),
                               'rank_err': float(v[:, 1].mean())}

    # the two metrics are one metric: with no truth tie, the k that sit between SR_hat and SR are
    # exactly the discordant pairs, so |r_hat - r| = NCAL x (1 - pairwise accuracy) evaluation by evaluation.
    worst = max(abs(res[f'K{K}|{lab}']['rank_err'] - NCAL * (1 - res[f'K{K}|{lab}']['pair_acc']))
                for K in KCALS for lab in acc_by)
    print(f'\n|drank| = {NCAL} x (1 - ins-acc) exactly on this panel (max deviation over all cells {worst:.2e}): '
          'the two metrics are one number in two units, because no true SR is tied.')
    res['identity_max_dev'] = float(worst)

    # ---- does ranking separate anything SR-MAE ties? (insertion reading) ---------------------
    both = sum(1 for _K, _l, m, r in sepr if m and r)
    monly = [(K, l) for K, l, m, r in sepr if m and not r]
    ronly = [(K, l) for K, l, m, r in sepr if r and not m]
    neither = sum(1 for _K, _l, m, r in sepr if not m and not r)
    P = np.concatenate(pooled, axis=1)
    gaps = np.concatenate([np.diff(np.sort(v)) for v in LB.values()])
    print(f'\n===== does ranking quality separate the systems where SR-MAE did not? (insertion reading) =====')
    print(f'   {len(sepr)} (row x K_cal) paired comparisons against {BASE}, same cluster bootstrap over planner ids:')
    print(f'      SR-MAE separates and ranking separates : {both}')
    print(f'      SR-MAE separates, ranking does NOT     : {len(monly)}   '
          + ', '.join(f'K{K} {l.strip()}' for K, l in monly))
    print(f'      SR-MAE ties, ranking SEPARATES         : {len(ronly)}   '
          + (', '.join(f'K{K} {l.strip()}' for K, l in ronly) if ronly else '(none)'))
    print(f'      both tie                               : {neither}')
    print(f'   -> {"NO" if not ronly else "YES"}: under insertion scoring ranking separates no pair that SR-MAE called a tie'
          f'{"" if not ronly else " — see the rows above"}, and it loses {len(monly)} separations SR-MAE made'
          f' ({len(monly)} is 5 or 6 by bootstrap draw: K12 ATLAS tau=0.2 sits on the 95% boundary).')
    print(f'   why: the leaderboard is sparse next to the estimator error. Adjacent-SR gaps of the {NCAL}-planner '
          f'leaderboards: median {np.median(gaps):.4f}, 10th pct {np.percentile(gaps, 10):.4f}, so an SR error has '
          'to clear about half a gap to move a rank.')
    print(f'   mean |err| where the rank is exact: {P[0][P[1] == 0].mean():.4f} '
          f'(pooled {int((P[1] == 0).sum())} of {P.shape[1]} scored evaluations, ATLAS tau=0.1 excluded) vs '
          f'{P[0][P[1] > 0].mean():.4f} where it is not; the rank metric quantises the rest of the error away.')
    res['separation'] = {'both': both, 'mae_only': [f'K{K}|{l}' for K, l in monly],
                         'rank_only': [f'K{K}|{l}' for K, l in ronly], 'neither': neither,
                         'gap_median': float(np.median(gaps)), 'gap_p10': float(np.percentile(gaps, 10)),
                         'mae_rank_exact': float(P[0][P[1] == 0].mean()), 'mae_rank_moved': float(P[0][P[1] > 0].mean()),
                         'n_rank_exact': int((P[1] == 0).sum()), 'n_scored': int(P.shape[1]),
                         'boundary_cell': {'cell': f'K12|ATLAS  tau=0.2', 'd_mae': res['K12|ATLAS  tau=0.2']['d_mae']}}
    res['eps05_rank_exact'] = {f'K{K}': res[f'K{K}|{BASE}']['rank_exact'] for K in KCALS}
    res['eps05_rank_le1'] = {f'K{K}': res[f'K{K}|{BASE}']['rank_le1'] for K in KCALS}
    res['fluid_se_stop_macro'] = {'rollouts': float(np.mean([res[f'K{K}|Fluid  SE<=delta*']['rollouts'] for K in KCALS])),
                                  'rank_err': float(np.mean([res[f'K{K}|Fluid  SE<=delta*']['rank_err'] for K in KCALS]))}
    print(f'   at {BASE} the rank is exact in ' + ' / '.join(f'{v:.0%}' for v in res['eps05_rank_exact'].values())
          + f' of evaluations (macro {np.mean(list(res["eps05_rank_exact"].values())):.1%}) and within one rung in '
          + ' / '.join(f'{v:.1%}' for v in res['eps05_rank_le1'].values())
          + f' (macro {np.mean(list(res["eps05_rank_le1"].values())):.1%}); Fluid SE<=delta* macro: '
          f'{res["fluid_se_stop_macro"]["rollouts"]:.1f} rollouts, {res["fluid_se_stop_macro"]["rank_err"]:.2f} rungs')

    # ---- the two co-estimated readings ------------------------------------------------------
    res['coestimated'], res['withindraw'] = coestimated(recs, T, LBI, res)

    # robustness: score each pair on the routes recorded for BOTH planners instead of each own bank
    Yd, MK = panel.dense()
    rob = {}
    for K in KCALS:
        rs = [r for r in recs if r['K'] == K]
        R = rows_all(rs)
        for lab, (sysname, t) in R.items():
            a, d = [], []
            for r, ti in zip(rs, t):
                sh, m = r[sysname]['Shat'][int(ti) - 1], MK[:, r['js']]
                sk = np.array([float(Yd[m & MK[:, c], c].mean()) for c in LBI[r['seed']]])
                st = np.array([float(Yd[m & MK[:, c], r['js']].mean()) for c in LBI[r['seed']]])
                a.append(np.where(sk == st, 0.5, np.where(sk == sh, 0.5,
                         (np.sign(sh - sk) == np.sign(st - sk)).astype(float))).mean())
                d.append(abs(int((sk > sh).sum()) - int((sk > st).sum())))
            rob[f'K{K}|{lab}'] = {'pair_acc': float(np.mean(a)), 'rank_err': float(np.mean(d))}
    res['robust_intersection'] = rob
    shift = max(abs(rob[k]['pair_acc'] - res[k]['pair_acc']) for k in rob)
    print(f'robustness — pairs scored on the routes recorded for both planners instead of each own bank: '
          f'max shift in pairwise accuracy {shift:.4f}')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    recs = json.load(open(OUT / 'syscmp.json'))
    res = report(recs, Panel())
    json.dump(res, open(OUT / 'ranking_quality.json', 'w'), indent=1)
    print(f'\nwritten: {OUT / "ranking_quality.json"}')
    assert len(recs) == len(KCALS) * 64
    for K, lab, field, v, tol in ANCHORS:
        got = res[f'K{K}|{lab}'][field]
        assert abs(got - v) < tol, (K, lab, field, got, v)
    assert res['identity_max_dev'] < 1e-9                    # |drank| = 12 x (1 - ins-acc), no truth tie
    assert res['separation']['rank_only'] == []              # insertion reading: no pair SR-MAE ties is separated ...
    assert 5 <= len(res['separation']['mae_only']) <= 6      # ... and 5-6 SR-MAE calls are lost (K12 ATLAS tau=0.2
    d, lo, hi = res['separation']['boundary_cell']['d_mae']  #     sits on the 95% boundary: 5 or 6 by bootstrap draw)
    assert abs(d - .0107) < .001 and abs(lo) < .001, (d, lo, hi)
    assert abs(res['separation']['mae_rank_exact'] - .0203) < .001 and res['separation']['n_rank_exact'] == 1417
    assert res['separation']['n_scored'] == 2112
    assert abs(res['eps05_rank_le1']['K8'] - 0.953125) < 1e-9 and abs(res['fluid_se_stop_macro']['rank_err'] - 1.391) < 5e-3
    # the co-estimated readings: 2-3 cells separate that SR-MAE ties; at K_cal = 8 they favour Fluid
    assert res['coestimated']['rank_only'] == ['K8|Fluid  fixed B=100', 'K12|ATDrive B=110'], res['coestimated']['rank_only']
    assert res['withindraw']['rank_only'] == ['K8|Fluid  fixed B=100', 'K8|Fluid  fixed B=match', 'K12|ATDrive B=110']
    W = res['withindraw']['rows']
    for K, B, v in ((4, 30, .9167), (8, 55, .9063), (12, 30, .8958), (12, 165, 1.0)):   # = Table 1's column
        assert abs(W[f'K{K}|ATDrive B={B}']['acc'] - v) < 5e-4, (K, B, W[f'K{K}|ATDrive B={B}']['acc'])
    d, lo, hi = W['K8|Fluid  fixed B=match']['d_acc']
    assert abs(d - .0521) < .002 and lo > 0, (d, lo, hi)
    assert (res['withindraw']['both'], len(res['withindraw']['mae_only']), res['withindraw']['neither']) == (15, 8, 7)
    print('anchors OK')


# Pinned on results/syscmp.json (16 draws, 192 planner evaluations). The
# ATDrive B rows reproduce run_adaptive.py's fixed-budget anchors exactly
# (.0450 / .0332 / .0202 / .0081), i.e. the ATDrive trajectory saved by the
# complete-system script is the trajectory Table 2 scores.
# (K_cal, row, field, value, tolerance).
ANCHORS = ((4, 'ATLAS  tau=0.1', 'pair_acc', 1.0, 1e-9),       # bank exhausted: the ranking must be exact
           (4, 'ATLAS  tau=0.3', 'pair_acc', .9492, .005),
           (8, 'ATLAS  tau=0.3', 'rank_err', .641, .05),
           (4, 'Fluid  SE<=delta*', 'rank_err', 2.203, .05),
           (12, 'Fluid  fixed B=100', 'pair_acc', .9727, .005),
           (4, 'ATDrive eps=0.05', 'pair_acc', .9727, .005),
           (4, 'ATDrive eps=0.05', 'rank_err', .328, .05),
           (12, 'ATDrive eps=0.05', 'pair_acc', .9779, .005),
           (12, 'ATDrive eps=0.03', 'rank_err', .094, .05),
           (8, 'ATDrive B=165', 'rank_err', .031, .05),
           (4, 'ATDrive B=30', 'mae', .0450, .002),           # = run_adaptive.py's anchor
           (12, 'ATDrive eps=0.05', 'ties_truth', 0, .5),
           (12, 'ATDrive eps=0.05', 'ties_estimate', 0, .5))

if __name__ == '__main__':
    main()
