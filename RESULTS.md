# Results of record

Every number here is printed (and asserted, `anchors OK`) by the script named
in each section. Conventions: K = planners, S = scenes, K_cal = calibration
panel size, B = number of routes rolled out, SR-MAE = |SR_hat - SR| averaged
over 64 planner evaluations per cell (16 draws x 4 evaluation planners).
`*` marks a paired-bootstrap 95% CI vs ATDrive that excludes zero; the
bootstrap resamples the 16 unique planners (the same planners recur across
draws; n_eff = 16 planner clusters behind every 64-evaluation cell). Stars are uncorrected
paired-bootstrap 95% intervals; no multiplicity correction is applied across cells. Acquisition ties
(routes of one type with identical posteriors are exchangeable) are broken
by bank order — a documented, deterministic choice. Its sensitivity was measured on the K_cal = 12
cell by breaking exact ties uniformly at random (3 seeds): B = 30 moves from .0477 to .034-.041,
B = 55 from .0231 to .0226-.0265, B = 110 from .0160 to .0141-.0156; 4-34 of the first 55 routes
change. The one Table 1 cell a baseline wins outright (K12 B30, Fluid .0404) is inside that
tie-break range. K_cal = 4 / 8 were not measured and can move more (PROTOCOL section 4).

**Estimand.** SR is the planner's success rate over the routes the simulator completed for
it — the routes with a recorded outcome in the response matrix (210-220 of the 220 Bench2Drive
routes; 3,482 of 3,520 cells observed) — and the bank on which routes are administered, every
estimator, and the true value it is scored against all use that same route set. The 38 missing
cells (TCP-traj 10, LEAD-tfv6 9, VAD 7, UniAD-Tiny 5, ORION 4, MindDrive-3B 2, SimLingo-IVL2-1B 1)
lie on 25 routes concentrated in HighwayExit, MergerIntoSlowTraffic, EnterActorFlow and
InterurbanAdvancedActorFlow; those routes have an observed fail rate of .588 against .475 for
complete routes (full-panel Rasch b_hat +0.51 vs -0.10), so missingness is not at random.
Calibration drops missing cells from the likelihood, i.e. assumes MAR. Under the alternative
convention pass/220 with unrecorded routes counted as failures (the published Bench2Drive number
for TCP-traj, VAD, UniAD-Tiny), SR_220 = SR_rec x n_rec/220 for the truth and the estimate alike,
so every error in this document scales by n_rec/220 in [0.955, 1]: the published SR-MAE is an
upper bound and no cell moves by more than 4.6% (K12 B55 .0231 -> .0227; eps = .05 .0207 -> .0204).

**The UP bank is the whole benchmark.** A new planner is placed on all 220
Bench2Drive routes of all 44 scenario types, and the item bank is calibrated
from the 12 calibration planners on those same routes; B is how many of the
220 are actually executed. Only US and UPS hold scenario types out (36 : 8),
because only they need routes with no calibrated difficulty.

## Table 1 — UP at fixed budgets (`run_up_frontier.py`)

Unseen-planner SR reconstruction on the 16 x 220 Bench2Drive panel (one
planner per model family, PROTOCOL section 1), random 12:4 planner split,
B routes rolled out (30/55/110/165 = 5 x {6, 11, 22, 33} = 14/25/50/75% of
the benchmark). Each method uses its native readout.

| method | K4 B30 | B55 | B110 | B165 | K8 B30 | B55 | B110 | B165 | K12 B30 | B55 | B110 | B165 | macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random (IRT-free) | .0623* | .0397 | .0245 | .0144* | .0623* | .0397 | .0245* | .0144* | .0623* | .0397* | .0245* | .0144* | .0352 |
| Random + IRT | .0536 | .0368 | .0232 | .0138 | .0536 | .0362 | .0232 | .0133* | .0534 | .0362* | .0227* | .0132* | .0316 |
| Random-strat + IRT | .0594* | .0353 | .0210 | .0122 | .0568 | .0331 | .0194 | .0113* | .0565 | .0332* | .0195 | .0112 | .0307 |
| DISCO-sel + IRT | .0614* | .0374 | .0299 | .0140 | .0538 | .0423 | .0264* | .0134* | .0594 | .0417* | .0263* | .0110* | .0348 |
| AnchorPoints | .1290* | .1290* | .1290* | .1290* | .0552 | .0485* | .0397* | .0397* | .0583 | .0445* | .0286* | .0167* | .0706 |
| Total-Fisher | .0604* | .0450* | .0306* | .0209* | .0620* | .0401 | .0215 | .0117 | .0585 | .0431* | .0225* | .0136* | .0358 |
| Marginal-Fisher | .0560 | .0473* | .0328* | .0207* | .0625 | .0407 | .0240 | .0108 | .0583 | .0390* | .0237* | .0133* | .0358 |
| tinyBenchmarks-lite | .0755* | .0621* | .0334* | .0184* | .0472 | .0378 | .0168 | .0139* | .0508 | .0334* | .0235* | .0112 | .0353 |
| metabench-lite | .0615 | .0430* | .0250 | .0157* | .0497 | .0433 | .0266 | .0142* | .0643 | .0427* | .0221 | .0130* | .0351 |
| Fluid-style | .0653* | .0385 | .0312* | .0166* | .0499 | .0364 | .0210 | .0105 | .0404 | .0281 | .0243* | .0119* | .0312 |
| **ATDrive** | **.0450** | **.0332** | .0223 | **.0116** | **.0448** | .0337 | .0202 | **.0082** | .0477 | **.0231** | **.0160** | **.0081** | **.0262** |

The three random-policy rows are the expected error over five independent
orders per evaluation; every other row is deterministic. AnchorPoints is
degenerate at K_cal = 4: four binary responses admit only 14-23 distinct
route patterns, the correlation distance collapses and the same anchor
estimate comes out at every budget.

Reading. ATDrive has the lowest error in 8 of 12 cells and the lowest macro
average by a clear margin (.0262 against .0307 for type-stratified Random,
.0312 for Fluid-style and .0316 for Random + IRT). The four cells it does
not win are ties inside the intervals: K4 B110 and K8 B55/B110, where the
type-stratified order or tinyBenchmarks is lower by .001-.003, and K12 B30,
where Fluid is lower by .007. The margin grows with the budget and with the
calibration panel — at K_cal = 12 ATDrive is .0231 / .0160 / .0081 at
B = 55 / 110 / 165 while the best baseline sits at .0281 / .0195 / .0110 —
because a better-calibrated bank makes the Delta-R1 score sharper.

**What each budget buys** (`run_adaptive.py --merge`). SR-MAE is an average;
the decision a user makes needs the distribution. Over the same 64
evaluations, ATDrive's estimate and the pairwise ranking of the four
evaluation planners within a draw (the 6 pairs of estimates ordered as their
true SRs; PROTOCOL section 7):

| K_cal | B | SR-MAE | within 1 pt | within 2 pt | within 3 pt | pairwise rank correct |
|---|---|---|---|---|---|---|
| 4 | 30 | .0450 | 20% | 27% | 38% | 91.7% |
| 4 | 55 | .0332 | 17% | 33% | 55% | 92.7% |
| 4 | 110 | .0223 | 25% | 50% | 70% | 93.8% |
| 4 | 165 | .0116 | 52% | 84% | 98% | 99.0% |
| 8 | 30 | .0448 | 11% | 31% | 44% | 93.8% |
| 8 | 55 | .0337 | 19% | 36% | 56% | 90.6% |
| 8 | 110 | .0202 | 20% | 55% | 80% | 95.8% |
| 8 | 165 | .0082 | 70% | 94% | 97% | 100.0% |
| 12 | 30 | .0477 | 11% | 22% | 36% | 89.6% |
| 12 | 55 | .0231 | 31% | 48% | 70% | 95.8% |
| 12 | 110 | .0160 | 36% | 70% | 84% | 97.9% |
| 12 | 165 | .0081 | 72% | 94% | 98% | 100.0% |

Reading. The two questions separate. Ordering planners is nearly settled at
14% of the cost: 90-94% of the within-draw pairs are already in the right
order at B = 30. Reporting an SR number is not: at B = 30 only 36-44% of
estimates land within 3 SR points, which is the resolution a leaderboard
entry needs, and that only reaches 70-84% at B = 110 and 97-98% at B = 165.
The low budgets are therefore a ranking regime, not a reporting regime, and
the stopping rule below never selects one — it spends 70-129 routes.

## Route-level discrimination at fixed budget (`run_route_discrimination.py`)

Table 1 scores one aggregate per evaluation. This diagnostic asks how well
the posterior predicts the routes it did *not* buy: after B rollouts the
ATDrive posterior gives P(y_s = 1 | D_B) for every bank route, the B
administered routes are removed, and the rest are scored against the
planner's true outcomes with AUROC and Brier. All five bank orders run under
the common ATDrive readout, so only the purchased routes differ; the ATDrive
order is Table 1's stored Delta-R1 selection (reuse verified by recomputing
`r1_traj` on three records, 165 / 165 routes each). 16 draws x K_cal in
{4, 8, 12} x 4 held-out planners = 192 evaluations, 64 per cell. An
evaluation whose unobserved outcomes are all one class has no AUROC and is
dropped from the AUROC average only: 36 of 3,840 scorings (0.9%), all at
B >= 110, all all-failure residuals (the order bought every success),
17 for ATDrive and 19 for Fluid, on the two weakest planners (SR .135 and
.155).

AUROC on each order's own unobserved routes (`*` = paired planner-cluster
95% CI vs ATDrive excludes zero):

| order | K4 B30 | B55 | B110 | B165 | K8 B30 | B55 | B110 | B165 | K12 B30 | B55 | B110 | B165 | macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ATDrive | .7316 | .7470 | .7549 | .7105 | .7706 | .7887 | .8021 | .7976 | .7842 | .8011 | .8050 | .8281 | .7768 |
| Fluid | .7224* | .7292* | .7392 | .7402 | .7556* | .7631* | .7860 | .8010 | .7730* | .7826* | .7986 | .8155 | .7672 |
| metabench | .7288 | .7297* | .7409 | .7236 | .7502* | .7591* | .7591* | .7704 | .7621* | .7638* | .7663* | .7436* | .7498 |
| Random | .7247 | .7375 | .7593 | .7680 | .7570* | .7687* | .7856 | .8009 | .7695* | .7805* | .7957 | .8110 | .7715 |
| Random-strat | .7221* | .7439 | .7690 | .7750* | .7559* | .7752 | .7964 | .8086 | .7674* | .7844* | .8040 | .8181 | .7767 |

Macro over the 12 cells, with the two controls that decide how to read it.
The zero-rollout row scores the same residual sets with the posterior
predictive before any rollout (calibration only, no evaluation-planner
data), so it cannot differ between orders through posterior quality:

| statistic | ATDrive | Fluid | metabench | Random | Random-strat |
|---|---|---|---|---|---|
| AUROC, own residual set | .7768 | .7672 | .7498 | .7715 | .7767 |
| Brier, own residual set | .1341 | .1453 | .1805 | .1717 | .1687 |
| Brier, zero-rollout predictor, same residual sets | .1815 | .1864 | .2297 | .2241 | .2238 |
| AUROC, zero-rollout predictor, same residual sets | .7473 | .7339 | .7113 | .7321 | .7345 |
| SR-MAE, common readout (Table 2 machine) | .0262 | .0284 | .0335 | .0334 | .0288 |
| SD of predicted p on the residual set | 0.1784 | 0.1575 | 0.1676 | 0.1821 | 0.1830 |

| control | result |
|---|---|
| ATDrive - Random-strat AUROC, pooled over the 751 paired evaluations | +.0009, 95% CI [-.0189, +.0192]; ATDrive ahead in 9 of 12 cells; the .0001 macro gap comes from one cell, K4 B165 (Random-strat - ATDrive +.0675 [+.0068, +.1391] on 59 records), without which the macro is .7828 vs .7768; on the records where all five orders have an AUROC the macro is .7773 vs .7764 |
| Zero-rollout predictor vs SR-MAE | macro Brier order identical to the SR-MAE order (rho +1.00) with a larger between-order spread (.0482 vs .0464 for the real posteriors); AUROC macro rho +0.90 (real posteriors +0.70) |
| Skill over the zero-rollout predictor, Random-strat - ATDrive | Brier skill +.0076 [+.0020, +.0146]; AUROC gain +.0118 [+.0006, +.0232] (Random - ATDrive: +.0049 [-.0012, +.0118] and +.0096 [-.0012, +.0204]; Fluid - ATDrive: -.0063 [-.0080, -.0050] and +.0017 [-.0036, +.0084]) |
| Common evaluation set (routes no order administered; B = 30 and 55 only, 112.1 and 60.8 routes, both classes present in 100% / 99.5% of evaluations) | AUROC ATDrive .7658, Fluid .7586, metabench .7586, Random .7638, Random-strat .7645; Random-strat - ATDrive -.0013 [-.0072, +.0055], Random -.0020 [-.0088, +.0050], Fluid -.0072 [-.0106, -.0036], metabench -.0073 [-.0127, -.0013]; Brier .1596 / .1617 / .1635 / .1617 / .1615, spread across orders .0175 -> .0039; per-cell Spearman with SR-MAE falls from +0.52 to +0.10 (AUROC) and +0.88 to +0.67 (Brier) |
| Rank agreement with SR-MAE, own residual sets | mean per-cell Spearman: AUROC +0.53 (common readout) / +0.64 (native Table 1); Brier +0.70 / +0.59. Same cell winner: AUROC 7 of 12, Brier 6 of 12. Identical 5-way order: AUROC 0 of 12, Brier 1 of 12. Brier's macro order matches SR-MAE only by placing Random (.0334) below metabench (.0335), a .0001 gap this document reads as a tie |

Reading. AUROC and SR-MAE agree on the extremes (ATDrive best, metabench
worst) and correlate positively but loosely. Route-level AUROC cannot
resolve ATDrive from the type-stratified order: the pooled difference is
+.0009 with an interval two hundred times wider, so the two are unresolved,
not tied. Brier's apparent agreement with SR-MAE is not evidence about the
posteriors: a predictor that has seen no evaluation-planner data reproduces
the Brier ranking exactly on the same residual sets, and once that baseline
is subtracted the type-stratified order is ahead of ATDrive on both scores.
Both route-level scores are dominated by which routes an order leaves
behind; with the evaluation set held fixed at B = 30 and 55 the five orders
barely separate. The residual-flatness explanation is a correlation only
and fails where it was invoked: at K4 B165 Fluid's residual set is flatter
than ATDrive's (SD .1178 vs .1336) yet its AUROC is higher (.7402 vs .7105).
Neither route-level score should be read as a method comparison without
the common-set or skill-corrected version beside it.

## Table 2 — risk-target stopping (`run_tau_calibration.py`, `run_adaptive.py`)

The configuration a user of the tool is actually in: 12 calibration
planners, a new planner, the whole 220-route benchmark, and no budget
chosen in advance. Each order stops at the first t with c * R1_t <= eps,
where R1 is the posterior L1 risk of the reported SR under the common
readout and c is that order's calibration-fixed risk scale
(leave-one-planner-out on the calibration panel, 90th percentile of
realised / predicted error over t in [10, 220]; never selected on
evaluation planners; ATDrive: c = 1.97 / 2.12 / 1.95 at K_cal = 4 / 8 / 12).
eps is an *error* target. No trajectory is censored: every order may run to
route 220, and none did (cap 0% everywhere). Columns: mean rollouts spent
and the same as a fraction of the benchmark, SR-MAE at the stop, the
calibration gap (mean realised error minus mean c * R1 at the stop; negative =
conservative), and the paired delta of the SR-MAE vs ATDrive.

| K_cal | eps | method | rollouts | of 220 | SR-MAE | gap | d vs ATDrive |
|---|---|---|---|---|---|---|---|
| 4 | .05 | **ATDrive** | **83.5** | **38%** | .0272 | -.023 | — |
| | | Fluid | 99.1 | 45% | .0282 | -.021 | +.0011 |
| | | metabench | 108.8 | 49% | .0265 | -.023 | -.0007 |
| | | Random | 98.7 | 45% | .0267 | -.023 | -.0005 |
| | | Random-strat | 87.9 | 40% | .0219 | -.028 | -.0053 |
| 4 | .03 | **ATDrive** | **128.9** | **59%** | .0164 | -.013 | — |
| | | Fluid | 142.3 | 65% | .0194 | -.010 | +.0031 |
| | | metabench | 160.1 | 73% | .0138 | -.016 | -.0026 |
| | | Random | 150.5 | 68% | .0154 | -.014 | -.0009 |
| | | Random-strat | 139.9 | 64% | .0130 | -.017 | -.0033 |
| 8 | .05 | **ATDrive** | **79.0** | **36%** | .0281 | -.022 | — |
| | | Fluid | 98.8 | 45% | .0216 | -.028 | -.0065 |
| | | metabench | 118.5 | 54% | .0212 | -.029 | -.0069 |
| | | Random | 93.3 | 42% | .0260 | -.024 | -.0021 |
| | | Random-strat | 87.7 | 40% | .0208 | -.029 | -.0073 |
| 8 | .03 | **ATDrive** | **124.9** | **57%** | .0174 | -.013 | — |
| | | Fluid | 142.0 | 65% | .0172 | -.013 | -.0002 |
| | | metabench | 167.4 | 76% | .0141 | -.016 | -.0033 |
| | | Random | 146.2 | 66% | .0165 | -.013 | -.0008 |
| | | Random-strat | 140.3 | 64% | .0122 | -.018 | -.0051* |
| 12 | .05 | **ATDrive** | **70.3** | **32%** | **.0207** | -.029 | — |
| | | Fluid | 88.4 | 40% | .0283 | -.021 | +.0076* |
| | | metabench | 107.6 | 49% | .0229 | -.027 | +.0022 |
| | | Random | 91.2 | 41% | .0294 | -.020 | +.0087* |
| | | Random-strat | 87.4 | 40% | .0221 | -.028 | +.0015 |
| 12 | .03 | **ATDrive** | **114.8** | **52%** | .0138 | -.016 | — |
| | | Fluid | 132.0 | 60% | .0190 | -.011 | +.0052* |
| | | metabench | 159.7 | 73% | .0136 | -.016 | -.0002 |
| | | Random | 144.1 | 65% | .0177 | -.012 | +.0039 |
| | | Random-strat | 140.8 | 64% | .0112 | -.019 | -.0026 |

Reading. For an error target of .05, ATDrive stops after 70-84 of the 220
routes (32-38% of the benchmark); the same target costs
Fluid 88-99, Random 91-99, the type-stratified order 87-88 and metabench
108-119 routes. For .03 it needs 115-129 routes (52-59%) against 132-167 for
the others. The saving is the acquisition, not the scale: every order
carries its own leave-one-planner-out c, and the orders that keep going
arrive at a similar error. Only at K_cal = 12 does the error difference
reach significance, and there ATDrive is the *better* one (-.009 vs Random,
-.008 vs Fluid at eps = .05); at K_cal = 8 the type-stratified order reaches
eps = .03 with a lower error (-.005*) for 15 more routes. The negative gaps
say the scaled risk over-states the realised error at the stop, as a 90th-percentile scale (about
twice the median ratio) must; c at K_cal = 4 rests on four LOO trajectories per draw and ranges
1.25-3.42 across draws (medians 1.97 / 2.12 / 1.95). The raw R1 tracks the
realised error: pooled over the LOO tracks, the decile means of raw R1 and
of |SR_hat - SR| agree within .005 at K_cal = 8 / 12 and within .012 at
K_cal = 4 (K_cal = 8: .0025 / .0021
in the lowest decile, .0147 / .0159 in the middle, .0601 / .0652 in the
highest).

**Matched cost (appendix).** The earlier rule — tau_hat chosen so that the
LOO mean rollouts hit a target — is kept as the cost-matched comparison. On
a tau sweep, ATDrive's adaptive stop matches its own fixed-budget curve at
the same mean cost within .005 in every cell (K_cal = 12, tau = .040: 37.5
rollouts, .0344 adaptive vs .0391 fixed), so the stopping rule spends the
budget as well as a fixed budget of the same mean length: at matched mean cost the adaptive stop's
error equals the fixed-budget error (paired deltas -.0013 to +.0025; the stop is a budget-selection
device that converts an SR-unit error target into a stopping time, not an accuracy gain).

## Adaptive policies under one IRT (`run_cat_objective.py`, `run_policy_matrix.py`)

Complete policies (selection rule x stopping rule) scored on the saved
`results/cat_objective.json` trajectories with the ATDrive IRT, bank and
posterior-median readout held fixed. 192 evaluations plus 384
leave-one-planner-out calibration tracks. Thresholds are published (ATLAS's
tau and 30-route minimum, Fluid's n_max = 100, our eps) or fixed on the
calibration planners only; the two starred rows are our constructions,
matched on the leave-one-out tracks to ATDrive's cost at eps = .05. Fisher
here is the 1PL information at the ATDrive posterior ability
(`atdrive.acquisition.fisher_pick`), so the ATLAS-style and Fluid-style
rows share one selection rule and differ in where they stop; they are not
the Table 1 / Table 2 Fluid rows. The Random order is `run_cat_objective.py`'s
draw (seeded with the 0-3 index of the held-out planner), a different
permutation from Table 2's Random row. IES = (SR-MAE / SR-MAE_ref) x
(routes / 55) with the reference the uniform random order read at 55 routes
(PROTOCOL section 7; reference SR-MAE .0402 / .0375 / .0393 at K_cal = 4 / 8 / 12).

| policy | K4 routes | K4 SR-MAE | K4 IES | K8 routes | K8 SR-MAE | K8 IES | K12 routes | K12 SR-MAE | K12 IES |
|---|---|---|---|---|---|---|---|---|---|
| ATDrive Delta-R1 + c R1 <= .05 | 83.5 | .0272 | 1.02 | 79.0 | .0281 | 1.07 | 70.3 | .0207 | 0.67 |
| ATDrive Delta-R1 + c R1 <= .03 | 128.9 | .0164 | 0.95 | 124.9 | .0174 | 1.05 | 114.8 | .0138 | 0.73 |
| ATLAS-style Fisher + SE <= 0.1 | 217.4 | .0000 | 0.00 | 217.4 | .0000 | 0.00 | 217.4 | .0000 | 0.00 |
| ATLAS-style Fisher + SE <= 0.2 | 212.2 | .0015 | 0.15 | 217.4 | .0000 | 0.00 | 217.4 | .0000 | 0.00 |
| ATLAS-style Fisher + SE <= 0.3 | 100.8 | .0289 | 1.32 | 103.2 | .0228 | 1.14 | 99.9 | .0199 | 0.92 |
| ATLAS-style Fisher + SE <= tau_m * | 79.5 | .0332 | 1.19 | 80.4 | .0325 | 1.27 | 68.9 | .0269 | 0.86 |
| Fluid-style Fisher + fixed B = 100 | 100.0 | .0263 | 1.19 | 100.0 | .0270 | 1.31 | 100.0 | .0203 | 0.94 |
| Fluid-style Fisher + fixed B = match * | 82.9 | .0329 | 1.23 | 80.0 | .0313 | 1.21 | 69.7 | .0270 | 0.87 |
| theta-EIG + SE <= 0.1 | 217.4 | .0000 | 0.00 | 217.4 | .0000 | 0.00 | 217.4 | .0000 | 0.00 |
| theta-EIG + SE <= 0.2 | 212.1 | .0017 | 0.17 | 217.4 | .0000 | 0.00 | 217.4 | .0000 | 0.00 |
| theta-EIG + SE <= 0.3 | 97.0 | .0274 | 1.20 | 96.5 | .0263 | 1.23 | 93.8 | .0217 | 0.94 |
| Random order + c R1 <= .05 | 110.4 | .0195 | 0.97 | 95.1 | .0217 | 1.00 | 112.3 | .0199 | 1.03 |
| Random order + c R1 <= .03 | 160.5 | .0104 | 0.76 | 147.5 | .0123 | 0.88 | 162.0 | .0113 | 0.85 |

ATLAS's tau = 0.1, and tau = 0.2 at K_cal >= 8, are unreachable on this
ability scale: after the whole bank the ATDrive posterior SD of theta only
falls to 0.233 / 0.244 / 0.242, so those rows exhaust the benchmark and
report SR-MAE 0 by construction (SE here is ATDrive's posterior SD, not
ATLAS's 1 / sqrt(sum I + 1); on its own scale ATLAS tau = 0.2 stops at 66.4
routes at K_cal = 8, see the complete-system comparison below).

The factorial that carries the claim. Pooled over K_cal (192 evaluations,
clusters = the 16 planner ids), every arm at 76-78 routes; B = match is the
same integer budget per draw for arms B, C, G, so those arms administer
identical lengths record by record; tau_m for arm E is matched on the
Delta-R1 track's own ability SD:

| arm | selection | stopping | routes | SR-MAE |
|---|---|---|---|---|
| A | Delta-R1 | c R1 <= .05 (ATDrive) | 77.6 | .0253 |
| B | Delta-R1 | fixed B = match | 77.5 | .0242 |
| C | Fisher | fixed B = match | 77.5 | .0304 |
| E | Delta-R1 | SE <= tau_m | 76.2 | .0243 |
| F | Fisher | SE <= tau_m | 76.3 | .0309 |
| G | theta-EIG | fixed B = match | 77.5 | .0274 |

| contrast | what changes | Delta SR-MAE [95% CI] |
|---|---|---|
| C - B | selection only (Fisher vs Delta-R1), identical fixed budget | +.0062 [+.0006, +.0123] |
| F - E | selection only (Fisher vs Delta-R1), SE stop at matched cost | +.0066 [+.0011, +.0128] |
| G - B | selection only (theta-EIG vs Delta-R1), identical fixed budget | +.0031 [+.0003, +.0062] |
| B - A | stopping only (fixed length vs risk stop), Delta-R1 fixed | -.0011 [-.0037, +.0012] |
| E - A | stopping only (ability-SD stop vs risk stop), Delta-R1 fixed | -.0010 [-.0046, +.0023] |
| F - C | stopping only (ability-SD stop vs fixed length), Fisher fixed | +.0005 [-.0022, +.0030] |
| C - A | both rules swapped at once (the only equal-cost row of the table above) | +.0051 [-.0004, +.0112] |
| Fisher - Delta-R1 at exactly fixed budgets | no stopping rule | +.0103 [+.0014, +.0197] at B = 30; +.0100 [+.0019, +.0192] at 55; +.0081 [+.0024, +.0144] at 70; +.0063 [+.0012, +.0118] at 78; +.0057 [+.0008, +.0109] at 80 |

Reading. With the IRT, the bank and the readout held fixed, the selection
rule is what separates these policies and the stopping rule is not:
swapping Fisher for Delta-R1 at an identical budget costs about .006 SR-MAE
with an interval that excludes zero, at every fixed budget in the 30-80
route operating range and under either stopping rule, while swapping the
stopping rule with the selection held fixed moves the error by .001 or less
with intervals that include zero. The mixed contrast C - A is that
selection effect partly cancelled by the stopping effect and blurred by
pairing an adaptive-length arm with a fixed-length one, which is why it
alone does not clear zero. Two nulls follow. ATDrive's adaptive risk stop
is not shown to beat spending its own mean budget as a fixed length (.0253
at 77.6 routes vs .0242 at 77.5). Every non-matched delta in the policy
table is a cost difference: the Random order under our risk stop reaches a
lower SR-MAE than ATDrive eps = .05 in all three cells while spending
+32% / +20% / +60% more routes, and on the cost-adjusted IES it is ahead of
ATDrive at K_cal = 4 and 8 (0.97 vs 1.02, 1.00 vs 1.07) and behind at
K_cal = 12 (1.03 vs 0.67). The ATLAS-style tau_m row at K_cal = 4 spends
79.5 routes against ATDrive's 83.5 and is not cost-matched there.

## Complete-system comparison (`run_system_comparison.py`)

Each published method run as its own system on the protocol split
(PROTOCOL section 8): ATLAS-style (3PL with a bank-wide guessing constant,
EAP ability, top-5 randomesque Fisher, SE <= tau after a 30-item minimum,
p-IRT readout), Fluid-style (2PL, Newton-MAP ability, greedy Fisher, fixed
length or a precision stop built from its own MAP standard error), ATDrive
as in Table 2. Nothing is shared between the rows except the response
matrix, the split and the metric. 192 evaluations; `*` = paired
planner-cluster 95% CI vs ATDrive eps = .05 excludes zero; IES against the
uniform random order read at 55 routes with each system's own readout.

| K_cal | row | rollouts | of 220 | SR-MAE | IES | d vs ATDrive eps = .05 |
|---|---|---|---|---|---|---|
| 4 | ATLAS tau = 0.1 | 217.4 | 99% | .0000 | 0.00 | -.0272* |
| | ATLAS tau = 0.2 | 77.7 | 35% | .0370 | 1.33 | +.0098* |
| | ATLAS tau = 0.3 | 32.5 | 15% | .0514 | 0.77 | +.0242* |
| | Fluid fixed B = 100 | 100.0 | 45% | .0305 | 1.41 | +.0034 |
| | Fluid fixed B = match * | 82.9 | 38% | .0318 | 1.21 | +.0047 |
| | Fluid SE <= delta* * | 3.2 | 1% | .1406 | 0.21 | +.1135* |
| | **ATDrive eps = .05** | **83.5** | 38% | .0272 | 1.02 | — |
| | ATDrive eps = .03 | 128.9 | 59% | .0164 | 0.95 | -.0108* |
| 8 | ATLAS tau = 0.1 | 217.4 | 99% | .0000 | 0.00 | -.0281* |
| | ATLAS tau = 0.2 | 66.4 | 30% | .0311 | 1.02 | +.0030 |
| | ATLAS tau = 0.3 | 30.0 | 14% | .0424 | 0.63 | +.0143* |
| | Fluid fixed B = 100 | 100.0 | 45% | .0230 | 1.13 | -.0051 |
| | Fluid fixed B = match * | 80.0 | 36% | .0273 | 1.07 | -.0008 |
| | Fluid SE <= delta* * | 8.8 | 4% | .0857 | 0.37 | +.0576* |
| | **ATDrive eps = .05** | **79.0** | 36% | .0281 | 1.05 | — |
| | ATDrive eps = .03 | 124.9 | 57% | .0174 | 1.02 | -.0107* |
| 12 | ATLAS tau = 0.1 | 217.4 | 99% | .0000 | 0.00 | -.0207* |
| | ATLAS tau = 0.2 | 56.9 | 26% | .0314 | 0.85 | +.0107 |
| | ATLAS tau = 0.3 | 30.0 | 14% | .0437 | 0.62 | +.0230* |
| | Fluid fixed B = 100 | 100.0 | 45% | .0255 | 1.19 | +.0048 |
| | Fluid fixed B = match * | 69.7 | 32% | .0264 | 0.86 | +.0058 |
| | Fluid SE <= delta* * | 19.0 | 9% | .0545 | 0.48 | +.0338* |
| | **ATDrive eps = .05** | **70.3** | 32% | .0207 | 0.67 | — |
| | ATDrive eps = .03 | 114.8 | 52% | .0138 | 0.74 | -.0069* |

Reading. This is not an equal-budget comparison: every system stops where
its own rule stops, so most rows differ from ATDrive in cost as well as in
error, and only the starred Fluid B = match row spends what ATDrive spends
(there it is within .006 of ATDrive, inside the intervals at every K_cal).
ATLAS at tau = 0.1 exhausts the bank — every route is observed, its SR-MAE
of 0 is the success rate read off the benchmark, and its IES of 0 says
nothing about ATLAS; only tau = 0.3 (and tau = 0.2 at K_cal >= 8) gives a
genuine stop, at 30-78 routes and a higher error than ATDrive's stop.
Fluid's own precision stop halts after 3-19 routes and is far off. IES
prices the cost in but rewards an early stop with a large error (ATLAS
tau = 0.3 and Fluid's precision stop have the lowest IES of the genuine
stops because they halt at 3-33 routes), so it is read with the SR-MAE
column beside it.

### Ranking quality of the complete-system trajectories (`run_ranking_quality.py`)

Pure scoring of the saved `results/syscmp.json` trajectories at every
system's own stop plus the ATDrive order at fixed budgets; 192 evaluations,
no refit. Three ranking readings (PROTOCOL section 7). *Insertion accuracy*
places one held-out planner's estimate among the 12 calibration planners of
its draw at their published success rates (fraction of the 12 comparisons
ordered as the truth; |Delta rank| = |r_hat - r| on the 13-rung ranking; no
true SR is tied on this panel, closest pair .0091). *Pairwise rank correct*
is the metric of Table 1: the four held-out planners of a draw ordered
against each other, both sides estimated. The two are different numbers on
the same rows (e.g. K4 B30: 95.8% insertion vs 91.7% pairwise).

| row | routes K4 / K8 / K12 | SR-MAE (macro) | insertion accuracy (macro) | \|Delta rank\| (macro) | pairwise rank correct K4 / K8 / K12 |
|---|---|---|---|---|---|
| ATLAS tau = 0.1 (exhausts the bank; not a ranking result) | 217.4 | .0000 | 1.0000 | .000 | 100.0 / 100.0 / 100.0 |
| ATLAS tau = 0.2 | 77.7 / 66.4 / 56.9 | .0331 | .9640 | .432 | 93.8 / 93.8 / 92.7 |
| ATLAS tau = 0.3 | 32.5 / 30.0 / 30.0 | .0458 | .9484 | .620 | 88.5 / 93.8 / 91.7 |
| Fluid fixed B = 100 | 100 | .0263 | .9709 | .349 | 91.7 / 95.8 / 95.8 |
| Fluid fixed B = match | 82.9 / 80.0 / 69.7 | .0285 | .9666 | .401 | 90.6 / 96.9 / 96.9 |
| Fluid SE <= delta* | 3.2 / 8.8 / 19.0 | .0936 | .8841 | 1.391 | 66.1 / 84.4 / 90.1 |
| **ATDrive eps = .05** | 83.5 / 79.0 / 70.3 | .0253 | .9718 | .339 | 92.7 / 91.7 / 95.8 |
| ATDrive eps = .03 | 128.9 / 124.9 / 114.8 | .0158 | .9848 | .182 | 94.8 / 96.9 / 97.9 |
| ATDrive B = 30 | 30 | .0458 | .9562 | .526 | 91.7 / 93.8 / 89.6 |
| ATDrive B = 55 | 55 | .0300 | .9701 | .359 | 92.7 / 90.6 / 95.8 |
| ATDrive B = 110 | 110 | .0195 | .9805 | .234 | 93.8 / 95.8 / 97.9 |
| ATDrive B = 165 | 165 | .0093 | .9952 | .057 | 99.0 / 100.0 / 100.0 |

At its published eps = .05 stop ATDrive places the new planner at exactly
the right rung in 70 / 64 / 75% of evaluations (macro 69.8%) and within one
rung in 96.9 / 95.3 / 98.4% (macro 96.9%). Fluid's SE <= delta* stop is off
by 2.20 rungs at K_cal = 4 (3.2 rollouts) and 1.39 on macro (10.3 rollouts).

Does ranking separate anything SR-MAE does not? 33 paired comparisons
against ATDrive eps = .05, cluster bootstrap over the 16 planner ids:

| reading | both separate | SR-MAE only | ranking only | both tie |
|---|---|---|---|---|
| insertion (12 published SRs) | 18 | 5-6 | 0 | 9-10 |
| co-estimated, all four held-out planners at their own estimates on the 16-planner board | 15 | 8 | 2 (K8 Fluid B = 100 +.0156 [+.0020, +.0304]; K12 ATDrive B = 110 +.0104 [+.0023, +.0187]) | 8 |
| pairwise rank correct (Table 1 definition, within draw) | 15 | 8 | 3 (K8 Fluid B = 100 +.0417 [+.0051, +.0773]; K8 Fluid B = match +.0521 [+.0222, +.0808]; K12 ATDrive B = 110 +.0208 [+.0050, +.0398]) | 7 |

Reading. Under insertion scoring the answer is no, and that answer is
designed in: with the incumbents entered at their exact published SRs,
|Delta rank| = 12 x (1 - insertion accuracy) identically (maximum deviation
8.9e-16), so both are a deterministic coarsening of the SR error that
SR-MAE averages and cannot separate what that statistic ties. The count of
separations SR-MAE makes and ranking loses is 5 or 6 depending on the
bootstrap draw (K12 ATLAS tau = 0.2 sits on the boundary, +.0107
[-.0001, +.0217]). With both sides estimated the identity no longer holds
and 2-3 cells separate that SR-MAE ties; the cells at K_cal = 8 go against
ATDrive's eps = .05 stop, and one of them is cost-matched (Fluid B = match,
80.0 routes vs 79.0; pairwise rank correct 96.9% vs 91.7%). Under seed
clustering those intervals narrow to touching zero, so the defensible
statement is "not established as ties", not "Fluid ranks better". On this
panel the rank metric is a worse test statistic than SR-MAE for
single-planner placement: the 12-rung leaderboard has a median adjacent-SR
gap of .0318 against an estimator error of .02-.03, so the rank is exact in
1,417 of 2,112 evaluations (excluding the bank-exhausting ATLAS tau = 0.1
row) and moves only when the error is large (mean |err| .0203 where the
rank is exact vs .0618 where it moves). What it adds is the failure rate
above, which the SR scale hides.

## Full-system ablation (`run_system_ablation.py`)

ATDrive is two IRT pieces (the exact difficulty posterior, the planner x type
testlet) and two CAT pieces (the Delta-R1 acquisition, the LOO-calibrated
risk scale c). Each is switched off alone and scored twice: at the fixed
budget B = 55, and under the risk-target rule. Removing c means stopping on
the raw R1 (c = 1), which changes nothing at a fixed budget.

| K_cal | arm | B55 SR-MAE | eps=.05 roll / MAE | eps=.03 roll / MAE |
|---|---|---|---|---|
| 4 | **ATDrive (full)** | .0332 | **83.5** / .0272 | **128.9** / .0164 |
| | w/o b posterior | .0333 (+.0001) | 87.7 / .0266 | 131.2 / .0183 |
| | w/o testlet | .0477 (+.0145*) | 96.3 / .0283 | 139.2 / .0174 |
| | w/o Delta-R1 acquisition | .0405 (+.0074) | 98.7 / .0267 | 150.5 / .0154 |
| | w/o LOO calibration of c | .0332 (=) | 29.0 / .0423 | 63.7 / .0299 |
| 8 | **ATDrive (full)** | .0337 | **79.0** / .0281 | **124.9** / .0174 |
| | w/o b posterior | .0344 (+.0007) | 78.4 / .0263 | 123.3 / .0162 |
| | w/o testlet | .0383 (+.0046) | 102.3 / .0273 | 144.7 / .0147 |
| | w/o Delta-R1 acquisition | .0385 (+.0048) | 93.3 / .0260 | 146.2 / .0165 |
| | w/o LOO calibration of c | .0337 (=) | 26.8 / .0501 | 58.7 / .0365 |
| 12 | **ATDrive (full)** | .0231 | **70.3** / .0207 | **114.8** / .0138 |
| | w/o b posterior | .0270 (+.0039) | 72.1 / .0254 | 116.4 / .0136 |
| | w/o testlet | .0414 (+.0184*) | 94.4 / .0255 | 136.9 / .0169 |
| | w/o Delta-R1 acquisition | .0393 (+.0162*) | 91.2 / .0294 | 144.1 / .0177 |
| | w/o LOO calibration of c | .0231 (=) | 25.9 / .0535 | 57.4 / .0251 |

Reading. The four pieces fail in different ways, which is why all four are
in the method. Dropping the LOO calibration is the one that breaks the
guarantee: the raw risk stops after 26-29 routes, less than half the cost,
but the realised error at that stop is .042-.054 against the .05 target it
claims to have met, so the saving is not real. Dropping the testlet or the
acquisition keeps the error at the target but pays for it in routes, 13-24
more at eps = .05 and 10-30 more at eps = .03, and costs .005-.018 SR-MAE at
a fixed budget. The difficulty posterior is the smallest piece: it is
neutral at K_cal = 4 / 8 and worth .004 SR-MAE at K_cal = 12, where the bank
is sharp enough for the shape of the posterior to matter.

## Component ablation at fixed budgets (`run_ablation.py`)

The same three switchable components across the whole budget grid; paired
deltas vs full:

| cell | full | w/o b-uncertainty | w/o testlet | w/o risk acquisition |
|---|---|---|---|---|
| K4 B30 | .0450 | .0454 (+.0004) | .0595 (+.0145) | .0551 (+.0101) |
| K4 B55 | .0332 | .0333 (+.0001) | .0477 (+.0145*) | .0405 (+.0074) |
| K4 B110 | .0223 | .0238 (+.0014) | .0241 (+.0017) | .0243 (+.0020) |
| K4 B165 | .0116 | .0107 (-.0009) | .0126 (+.0010) | .0134 (+.0017) |
| K8 B30 | .0448 | .0398 (-.0050) | .0585 (+.0137) | .0567 (+.0119) |
| K8 B55 | .0337 | .0344 (+.0007) | .0383 (+.0046) | .0385 (+.0048) |
| K8 B110 | .0202 | .0193 (-.0009) | .0252 (+.0050) | .0239 (+.0036) |
| K8 B165 | .0082 | .0074 (-.0008*) | .0101 (+.0019) | .0133 (+.0051*) |
| K12 B30 | .0477 | .0467 (-.0010) | .0595 (+.0118) | .0577 (+.0100) |
| K12 B55 | .0231 | .0270 (+.0039) | .0414 (+.0184*) | .0393 (+.0162*) |
| K12 B110 | .0160 | .0161 (+.0000) | .0238 (+.0078*) | .0244 (+.0084*) |
| K12 B165 | .0081 | .0076 (-.0006*) | .0110 (+.0029*) | .0134 (+.0052*) |

Reading. The testlet and the acquisition carry the fixed-budget error, most
clearly at K_cal = 12 where both are significant at every budget from 55 up
(+.003 to +.018). The difficulty posterior is neutral for the point
estimate — every cell is inside +-.005 and the two significant ones favour
the point curves by .0006-.0008 at B = 165 — because its job is the risk,
not the estimate.

## What the testlet does (`run_ablation.py`, `run_tau_calibration.py` / `run_adaptive.py` with `ATDRIVE_NO_TESTLET=1`)

The same ATDrive with sigma_g fixed to 0 (routes of one type conditionally
independent), everything else unchanged:

| | with testlet | without (sigma_g = 0) |
|---|---|---|
| ATDrive B30 / B55 / B110 / B165 at K4 | .0450 / .0332 / .0223 / .0116 | .0595 / .0477* / .0241 / .0126 |
| at K8 | .0448 / .0337 / .0202 / .0082 | .0585 / .0383 / .0252 / .0101 |
| at K12 | .0477 / .0231 / .0160 / .0081 | .0595 / .0414* / .0238* / .0110* |
| raw R1 vs realised error, LOO deciles 1 / 5 / 10 at K4 | .0030/.0029, .0157/.0174, .0627/.0749 | .0032/.0032, .0159/.0197, .0631/.0749 |
| at K8 | .0025/.0021, .0147/.0159, .0601/.0652 | .0025/.0024, .0145/.0216, .0594/.0762 |
| at K12 | .0022/.0019, .0140/.0144, .0585/.0579 | .0022/.0021, .0134/.0190, .0568/.0640 |
| risk scale c (ATDrive) at K4 / 8 / 12 | 1.97 / 2.12 / 1.95 | 2.30 / 2.88 / 2.77 |
| eps = .05: rollouts, SR-MAE at K4 | 83.5, .0272 | 96.3, .0283 |
| at K8 | 79.0, .0281 | 102.3, .0273 |
| at K12 | 70.3, .0207 | 94.4, .0255 |
| eps = .03: rollouts at K4 / 8 / 12 | 128.9 / 124.9 / 114.8 | 139.2 / 144.7 / 136.9 |

Reading. Without the dependence structure the posterior L1 risk under-states
the realised error in the middle decile by 24-49% (K8 middle decile .0145
vs .0216; |err| / raw R1 - 1 at K_cal = 4 / 8 / 12) and in the top decile by
13-28%, so the calibration has to inflate
it (c 2.3-2.9 instead of 1.9-2.1), the error target of .05 costs 94-102
routes instead of 70-84, and the fixed-budget error rises in every cell. The
testlet is not a small-panel patch: it is what makes R1 a risk. The grouping
it uses is the benchmark's own scenario-type annotation, entered only as
"these routes share an offset"; no difficulty or feature is read from it
(PROTOCOL section 1).

## Table 3A — US: unseen scenes (`run_us.py`)

Predict scene difficulty (and per-cell outcomes) for the 8 evaluation
scenario types from the scene alone; pooled over 16 draws (640 route
evaluations). Descriptor rows are scored through a two-stage Ridge plug-in
fitted on the calibration types; the encoder row is the RelGraph R2
out-of-fold prediction (trained per draw on the 36 calibration types of
the 12 calibration planners). Planner-only null: AUROC .699 / scene-MAE
.214.

| difficulty source | AUROC | scene-MAE | rho(b_tilde, fail rate) |
|---|---|---|---|
| Min-TTC | .692 | .219 (-2.6%) | -.061 |
| Risk field | .705 | .213 (+0.3%) | +.091 |
| Route geometry | .719 | .201 (+6.1%) | +.268 |
| Agent density + kin. | .715 | .212 (+0.9%) | +.220 |
| Traffic entropy | .706 | .212 (+0.7%) | +.097 |
| Agent-JEPA | .696 | .217 (-1.5%) | +.001 |
| Kinematics (cmdkin, 25d) | .752 | .180 (+15.6%) | +.497 |
| Hand-crafted risk (cmdkin+gtrisk, 73d) | .758 | .175 (+18.0%) | +.533 |
| **ATDrive: RelGraph R2 scene encoder (3 runs)** | **.751 +- .003** | **.192 +- .003** | **+.490 +- .016** |
| Oracle (response-calibrated) | .870 | .037 | +.995 |

Reading. The learned relational encoder and the two hand-crafted stacks
clear every single-descriptor baseline by +.03-.07 AUROC (16-18 points of
scene-MAE for the hand-crafted stacks, 10 for the encoder); between them
the encoder is tied on AUROC and behind on scene-MAE and rank correlation
(RelGraph minus hand-crafted risk: Delta rho -.043 +- .016 across runs,
i.e. about three run-SDs). The encoder buys no difficulty signal beyond
well-chosen rollout descriptors on this bank; what it offers is the same
signal from the raw scene graph without feature engineering. The oracle
(.870) is the ceiling of any scene-only predictor: roughly half the
difficulty variance is not visible from the scene. (Earlier versions
reported a descriptor stack that included the scenario-definition
parameters; it was removed because those values are the benchmark's own
construction parameters, not observable scene content.)

### Table 3A(b) — structural controls of the encoder (`run_us.py`)

Same architecture, recipe, seeds and calibration; only the graph tensors
differ for the three structural controls, and only the ego channels for
the fourth (three runs each, Delta rho paired by seed against the R2 runs):

| variant | AUROC | scene-MAE | rho | Delta rho vs R2 |
|---|---|---|---|---|
| RelGraph R2 (shipped) | .751 +- .003 | .192 +- .003 | +.490 +- .016 | — |
| R2 without the ego-route relation | .756 +- .002 | .184 +- .005 | +.520 +- .014 | +.030 +- .019 |
| R2, route correspondence shuffled | .754 +- .007 | .189 +- .009 | +.512 +- .024 | +.022 +- .015 |
| R2, agent-lane correspondence shuffled | .753 +- .006 | .191 +- .007 | +.501 +- .035 | +.011 +- .019 |
| R2, ego-speed channel removed (channel control) | .753 +- .005 | .191 +- .005 | +.500 +- .020 | +.010 +- .004 |

Reading. The relational machinery is inert on this bank: removing the
ego-route relation helps in all three runs (+.030 rho), shuffling the
route correspondence helps in all three as well (+.022), and shuffling the
agent-lane correspondence changes nothing beyond seed noise. What the
encoder learns is carried by the ego and agent tracks, not by the
lane-graph relations. Removing the ego-speed channel changes nothing on
this bank either (+.010 rho, within seed noise); that arm matters only on
nuPlan, where it is the one that clears its label-shuffle null (next
section). R2 stays the shipped encoder because it was fixed before these
controls were scored; the paper's claim for the encoder is the
learned-from-raw-tracks difficulty prior and its transport to UPS, not the
graph structure.

## nuPlan val14 zero-shot retrieval (`run_nuplan_zeroshot.py`)

A scene encoder trained on Bench2Drive difficulty ranks the 584 nuPlan
val14 scenarios zero-shot; the score is the drop in planner performance on
the predicted-hard top-q%, Delta M_CLS = M(all 584) - M(top-q%), on the
11-planner closed-loop score (6,421 finite cells, 218 logs; the matrix is
not binary, so M_CLS is the primary metric; M_CLS(full) = .7808, base
failure rate 17.85% with failure = CLS < .5, which reproduces the stored
binary matrix cell for cell). Enrichment of the failure rate is reported
beside it. Two encoder arms (C0e: the canonical encoder; A2e: speed removed
from both ego paths), three seeds each, are each tested against ten
label-shuffled encoders trained under the same ablation. The null
threshold is matched to the arm statistic: T95 is the 95th percentile of
the 120 three-seed means of the ten shuffles, p is the fraction of those
means at or above the arm mean (plus one, over 121), and an exact
two-sample permutation over the 13 seeds (286 relabelings) is given beside
it. The oracle ranks by the response-calibrated difficulty b_ref (in
sample) and random q% subsets give the floor.

| q | arm | Delta M_CLS | matched T95 | p (120 triples) | exact permutation p | enrichment | enrichment T95 | verdict |
|---|---|---|---|---|---|---|---|---|
| 5% | oracle b_ref | +.4486 | — | — | — | 3.619 | — | ceiling |
| 5% | random 5% (3,000 draws) | +.0005 | — | — | — | 1.003 | — | floor |
| 5% | A2e, -speed | +.1783 | +.1584 | .0165 | .0385 | 1.957 | 1.804 (p .0083) | clears |
| 5% | C0e, speed kept | +.1611 | +.1610 | .0579 | .1154 | 1.870 | 1.816 (p .0331) | borderline |
| 5% | shuffle mean, -speed family (10) | +.0897 | — | — | — | 1.455 | — | — |
| 5% | shuffle mean, speed-kept family (10) | +.1056 | — | — | — | 1.523 | — | — |
| 10% | oracle b_ref | +.3631 | — | — | — | 3.057 | — | ceiling |
| 10% | random 10% | -.0001 | — | — | — | 1.000 | — | floor |
| 10% | A2e, -speed | +.1395 | +.1394 | .0579 | .1119 | 1.750 | 1.689 (p .0413) | marginal |
| 10% | C0e, speed kept | +.1142 | +.1508 | .3140 | .3531 | 1.617 | 1.769 (p .2397) | does not clear |
| 10% | shuffle mean, -speed family | +.0764 | — | — | — | 1.381 | — | — |
| 10% | shuffle mean, speed-kept family | +.0957 | — | — | — | 1.473 | — | — |

Threshold-free checks. Paired cluster bootstrap over the 218 logs of the
arm-minus-shuffle contrast (top-q re-selected inside each resample):
A2e - shuffles +.0886 [+.0160, +.1682] at q = 5% and +.0631 [+.0222, +.1097]
at q = 10%; C0e - shuffles +.0555 [-.0247, +.1314] and +.0185
[-.0195, +.0650]. Whole-panel Spearman of predicted difficulty with the
observed failure rate: A2e +.311 and C0e +.288 against shuffle means of
+.005 and +.037; both clear even the single-seed 95th-percentile threshold
(+.2305 and +.2436), exact permutation p = .0035 (the design floor) and
.0105. The arms recover 39.7% / 38.4% (A2e) and 35.9% / 31.5% (C0e) of the
oracle ceiling at q = 5% / 10%; the label-shuffled encoders already recover
20-26%.

Reading. The -speed encoder retrieves genuinely harder scenes at q = 5%
(p about .02-.04 by three routes), and at q = 10% the drop is marginal
while the enrichment and the paired contrast still clear. The canonical
speed-kept encoder does not clear on the top-q drop at either q (at q = 5%
it sits on its threshold, p .06, with a paired contrast that spans zero).
Comparing a three-seed arm mean against the 95th percentile of single
shuffle seeds inflates the threshold by about the square root of three
(null per-seed SD .088 vs .051 for a mean of three) and produced the earlier
"neither arm clears" reading, which is withdrawn. Top-q retrieval is a
low-power statistic here — per-seed shuffle drops span -.056 to +.200, and
three arm seeds against ten shuffles floor the achievable permutation p at
.0035 — and the whole-panel rank correlation is the statistic with the
power, where both arms clear.

## Table 3B — UPS: unseen planner x unseen scenes (`run_ups.py`)

Predict an unseen planner's behaviour on unseen scenario types with zero
rollouts on the target block: probe the planner on B calibration-type
routes, transport the ability posterior through the RelGraph difficulty
prior N(b_tilde_s, sigma^2) with the testlet prior on the (unobserved)
evaluation types. The MAE scores the posterior median of the block-D
success rate, the NLL scores the per-cell posterior predictive. 64
evaluations, RelGraph run s0 (the across-run SD of every MAE cell is
.002-.005). The canonical probe rule is Delta-R1 on the block-D success
rate — the UP acquisition with its risk evaluated on the target block;
theta-EIG and the 2PL Fisher rule are ablations. naive = the planner's
success rate on the probed routes, used as-is for block D.

| probe policy | B30 MAE | NLL | B55 MAE | NLL | B110 MAE | NLL |
|---|---|---|---|---|---|---|
| naive (no IRT) | .1007 | .6414 | .0900 | .6262 | .0865 | .6203 |
| Random | .1129 | .6021 | .1057 | .5932 | .0985 | .5872 |
| theta-EIG (abl.) | .0966 | .5868 | .0949 | .5852 | .0953 | .5865 |
| 2PL Fisher (abl.) | .0997 | .5889 | .0986 | .5876 | .0953 | .5873 |
| **ATDrive (Delta-R1 on D)** | **.0950** | **.5857** | .0936 | **.5848** | .0950 | **.5863** |

The scene prior's contribution to the transported block is isolated in the full-220 UPS section
below (calC + priorT(marg) vs calC + sceneT: B55 full-benchmark .0330 vs .0319, target block .0914 vs
.0936; per-cell AUROC on target cells .760-.764 scene vs .706-.710 scene-free, NLL .585 vs .62-.64).

Reading. The transport is a per-cell result, not a block-SR result: it
lowers the predictive NLL of the unseen cells from .62-.64 to .585-.587
at every budget, but its block-SR error sits on a floor of about .095
from B = 30 on — the floor of the scene prior (rho about .5), which more
probes cannot lower — while the planner's own success rate on the probed
calibration routes reaches .090 / .087 at B = 55 / 110. The naive error
is the calibration-vs-evaluation gap itself: |SR_A - SR_D| averages .085
over the 64 evaluations of this panel, and the transported estimate's
floor is not below that gap here. The
block-SR claim for UPS is therefore withdrawn on this panel; what the
transport delivers is the per-cell predictive. Among probe rules the
target-aligned Delta-R1 is ahead of Random by .018* / .012* / .004 and of
the ablations by .000-.005 (ns); the probe placement matters at low
budgets, and only through the prior.

**With the speed-ablated prior** (`results/ups_nospeed.json`, the fourth
control of Table 3A(b) driving both the transport and the Delta-R1 probe
rule; the shipped R2 prior stays canonical):

| probe policy | B30 MAE | NLL | B55 MAE | NLL | B110 MAE | NLL |
|---|---|---|---|---|---|---|
| naive (no IRT) | .1007 | .6414 | .0900 | .6262 | .0865 | .6203 |
| Random | .1087 | .6022 | .1028 | .5953 | .0925 | .5881 |
| theta-EIG (abl.) | .0922 | .5878 | .0936 | .5879 | .0936 | .5877 |
| 2PL Fisher (abl.) | .0973 | .5911 | .0985 | .5895 | .0934 | .5875 |
| ATDrive (Delta-R1 on D) | .0926 | .5881 | .0926 | .5879 | .0936 | .5878 |

The paired delta of the Delta-R1 row against the shipped prior is
-.0024 [-.0144, +.0087] / -.0009 [-.0119, +.0094] / -.0014 [-.0132, +.0092]:
the two priors are indistinguishable on the block-SR scale, and the reading
above does not change.

## UPS retargeted to the full 220-route SR (`run_ups_full.py`)

Table 3B predicts only the 40-route target block. Here the estimand is the
held-out planner's success rate on the whole benchmark, I = S_t u (C \ S_t)
u T: the probed calibrated routes are used as observed, the remaining
calibrated routes are inferred from their response-calibrated difficulty
posteriors, and the 40 held-out-type routes from the scene-conditioned
prior N(b_tilde_s, sigma^2). 64 evaluations, 36 : 8 type split, K_cal = 12,
RelGraph run s0; per evaluation |C| = 177.9 calibrated + |T| = 39.5 target
routes, so T is 18.2% of the benchmark; mean |SR_C - SR_T| = .0848. The
readout arms differ only in the difficulty model of the unobserved routes:
naive (the probed success rate for everything), calC + priorT(marg)
(T from b ~ N(0, sigma_b^2), encoder off), calC + priorT(const) (T from one
constant b = mean b_tilde, encoder level kept, per-route signal removed),
calC + sceneT (canonical), and two oracles that reveal C or T. Probe
orders: Random, Delta-R1 on D (the canonical UPS rule, acquisition driven
by the scene prior), Delta-R1 on D with a scene-free target bank
(encoder-free acquisition), and Delta-R1 on the full bank. The
target-block diagnostic reproduces Table 3B element-wise.

Full-benchmark SR-MAE:

| probe order | arm | B = 30 | B = 55 | B = 110 |
|---|---|---|---|---|
| Random | naive | .0698 | .0455 | .0273 |
| Random | calC + priorT(marg) | .0603 | .0422 | .0269 |
| Random | calC + priorT(const) | .0617 | .0424 | .0266 |
| Random | calC + sceneT | .0627 | .0438 | .0276 |
| Random | trueC + sceneT | .0205 | .0192 | .0179 |
| Random | calC + trueT | .0483 | .0313 | .0179 |
| Delta-R1 on D | naive | .0556 | .0609 | .0427 |
| Delta-R1 on D | calC + priorT(marg) | .0456 | .0330 | .0200 |
| Delta-R1 on D | calC + priorT(const) | .0467 | .0312 | .0213 |
| **Delta-R1 on D** | **calC + sceneT (canonical)** | **.0467** | **.0319** | **.0211** |
| Delta-R1 on D | trueC + sceneT | .0172 | .0170 | .0173 |
| Delta-R1 on D | calC + trueT | .0387 | .0234 | .0098 |
| Delta-R1 on D, scene-free bank | calC + priorT(marg) (encoder-free end to end) | .0454 | .0326 | .0205 |
| Delta-R1 on D, scene-free bank | calC + sceneT | .0465 | .0315 | .0216 |
| Delta-R1 on full I | calC + priorT(marg) | .0406 | .0243 | .0200 |
| Delta-R1 on full I | calC + sceneT | .0393 | .0234 | .0203 |
| Delta-R1 on full I | trueC + sceneT | .0169 | .0164 | .0172 |
| Delta-R1 on full I | calC + trueT | .0333 | .0190 | .0104 |

What the scene encoder buys (+ = the scene prior is worse; paired
planner-cluster bootstrap):

| probe order | contrast | B = 30 | B = 55 | B = 110 |
|---|---|---|---|---|
| Random | sceneT vs priorT(marg) | +.0024 [-.0009, +.0059] | +.0016 [-.0020, +.0055] | +.0007 [-.0027, +.0038] |
| Delta-R1 on D | sceneT vs priorT(marg) | +.0011 [-.0019, +.0039] | -.0011 [-.0037, +.0019] | +.0011 [-.0015, +.0042] |
| Delta-R1 on D | sceneT vs priorT(const) | -.0000 [-.0013, +.0013] | +.0008 [-.0009, +.0025] | -.0002 [-.0017, +.0015] |
| Delta-R1 on full I | sceneT vs priorT(marg) | -.0013 [-.0048, +.0023] | -.0009 [-.0044, +.0027] | +.0003 [-.0028, +.0038] |
| encoder-free acquisition + readout vs the canonical cell | | -.0013 [-.0049, +.0025] | +.0007 [-.0025, +.0038] | -.0006 [-.0039, +.0023] |
| acquisition channel alone (scene-free vs scene-driven order, both read out scene-free) | | -.0003 [-.0020, +.0012] | -.0004 [-.0015, +.0007] | +.0006 [-.0001, +.0016] |
| Delta-R1 on D | headroom: sceneT vs trueT | +.0080 [+.0038, +.0121] | +.0085 [+.0043, +.0127] | +.0113 [+.0088, +.0139] |
| naive on its own Random order vs the canonical cell | | +.0231 [+.0025, +.0437] | +.0136 [+.0026, +.0239] | +.0062 [-.0012, +.0134] |

Where the error lives (posterior-mean parts, canonical order): the C-part
error falls .0388 -> .0234 -> .0098 from B = 30 to 110 while the T-part
error is flat at .0173 / .0170 / .0173 (Random order: .0487 / .0316 / .0180
and .0205 / .0192 / .0179); the two parts reproduce the median-based
SR-MAE to within .0007. Pooled AUROC of the per-cell predictive on the T
cells is .760-.764 for the scene prior against .706-.710 scene-free under
the Delta-R1 orders (.744-.760 vs .690-.702 under Random); the
within-evaluation AUROC of .688 is the raw RelGraph out-of-fold ranking
(predicted p is a monotone function of -b_tilde_s within an evaluation) and
is identical at every budget and under every probe order.

Reading. With 180 of 220 routes response-calibrated, the scene encoder
contributes nothing measurable to the full-benchmark SR: swapping the
per-route RelGraph prior for a scene-free prior moves the SR-MAE by at most
.0024 and all nine paired intervals contain zero, and removing the encoder
from the acquisition as well leaves the error unchanged. That null rests on
the planner-cluster intervals (half-width about .003) alone: Table 3B's
across-run SD of .002-.005 is in target-block units and enters the full SR
scaled by n_T / N = .182, i.e. .0004-.0009, so the deltas are above the
encoder's run-to-run noise, not below it. The test is weakly powered: the
trueT oracle headroom is .008-.011, so an encoder capturing more than about
a quarter to a third of the available T information is excluded and a
small real contribution is not. Against the naive baseline on its own
random probe order the model is better at B = 30 and 55 and tied at
B = 110. The T-part error is a floor no probe budget lowers — it is the
calibration-vs-target block gap of .0848 — and at B = 110 the 18% of routes
the scene prior has to guess carry more of the error than the 82% the
responses cover. Arithmetically the T channel of this error is Table 3B's
already-withdrawn block-SR null multiplied by .182; the encoder still ranks
unseen routes within a planner (AUROC .76 vs .71 scene-free), which is the
per-cell result Table 3B reports.

## Readout drop-in (`run_readout_dropin.py`)

Analysis, not a headline: every baseline's selected subset re-scored with
the ATDrive readout (exact posteriors, testlet, posterior median). SR-MAE:

| selector (subset) | K4 B30 | B55 | B110 | B165 | K12 B30 | B55 | B110 | B165 |
|---|---|---|---|---|---|---|---|---|
| Fluid | .0536 | .0359 | .0247 | .0135 | .0410 | .0290 | .0225 | .0113 |
| Total-Fisher | .0570 | .0419 | .0254 | .0170 | .0496 | .0367 | .0209 | .0147 |
| metabench | .0613 | .0420 | .0242 | .0136 | .0617 | .0372 | .0220 | .0119 |
| tinyBenchmarks | .0754 | .0608 | .0275 | .0163 | .0498 | .0359 | .0222 | .0104 |
| AnchorPoints | .0687 | .0493 | .0313 | .0166 | .0524 | .0368 | .0218 | .0101 |
| Random | .0551 | .0405 | .0243 | .0134 | .0577 | .0393 | .0244 | .0134 |
| ATDrive (own order, Table 1) | .0450 | .0332 | .0223 | .0116 | .0477 | .0231 | .0160 | .0081 |

Reading. Under one readout the remaining differences are selection. At
K_cal = 12 the ATDrive order beats every re-scored subset at B = 55 / 110 by
.005-.016 and at B = 165 by .002-.007; at K_cal = 4 it leads the non-Fluid
subsets by .007-.030 at B = 30 / 55 and by .002-.009 at B = 110 / 165, and
the Fluid subset by .002-.009 at every budget. The
readout itself is a drop-in improvement for selectors whose native readout is
a plug-in on a scarce panel (compare the native Table 1 rows: AnchorPoints
.1290 -> .0687 at K4 B30, tinyBenchmarks .0621 -> .0608 at K4 B55).

## Model adequacy (`run_model_adequacy.py`)

Appendix. On the UP calibration block (12 calibration planners x 220
routes per draw) 10% of the observed cells are held out and predicted by
the 1PL, 2PL and 3PL fits of the rest; 16 draws. Held-out cell NLL:
1PL .5284 +- .0110, 2PL .5358 +- .0123, 3PL .5323 +- .0104 (2PL - 1PL
+.0074 +- .0029 across draws); the split-half reliability of the fitted
log-discrimination across two random 6 / 6-planner halves is +.062 +- .022.
Neither richer model predicts held-out responses better than the Rasch
model and the discrimination it would add is not reproducible across
halves of the panel, so the 1PL is the adequate model here, not a
simplification.

## Inside a real evaluation (`tools/b2d_adaptive_eval.py`)

The same posterior drives an actual Bench2Drive run: the bank is the full
22 x 220 panel (excluding the planner under test if it is in it), the risk
scale c is fixed by leave-one-planner-out on that bank (c = 2.36 for the
22-planner bank; 2.55 / 2.32 / 2.16 for the three held-out banks below),
and the driver picks one route at a time, runs it through the leaderboard
evaluator, reads the outcome and stops at c * R1 <= eps — there is no
route cap; the run ends when the risk target is met or the bank is
exhausted. Simulated from the matrix (`--dry-run`, planner held out of the
bank, eps = .03; the bank is the planner's recorded routes, 211-220 of
220):

| planner | true SR | routes run | stopped by | SR_hat | abs err | types covered |
|---|---|---|---|---|---|---|
| VAD | .155 | 106 | risk (c * R1 = .0296) | .170 | .015 | 37 / 44 |
| HiP-AD | .664 | 134 | risk (c * R1 = .0298) | .656 | .007 | 42 / 44 |
| LEAD-tfv6 | .777 | 99 | risk (c * R1 = .0300) | .814 | .036 | 38 / 44 |

Every run stopped on its own: half the bank for VAD and LEAD-tfv6, 61% for
HiP-AD, with errors of .015 and .007 for VAD and HiP-AD. The LEAD-tfv6
stop (99 routes) missed its .03 target with a realised error of .036 —
the kind of miss the risk target admits for roughly one run in ten. The weakest planner remains the hardest case: an all-fail
record only bounds theta from above, so its estimate leans on the prior
early (SR_hat .22 after 40 routes) and settles as the acquisition finds
the routes it can pass.
