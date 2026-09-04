# Results of record

Every number here is printed (and asserted, `anchors OK`) by the script named
in each section. Conventions: K = planners, S = scenes, K_cal = calibration
panel size, B = number of routes rolled out, SR-MAE = |SR_hat - SR| averaged
over 64 planner evaluations per cell (16 draws x 4 evaluation planners).
`*` marks a paired-bootstrap 95% CI vs DriveAT that excludes zero; the
bootstrap resamples the 16 unique planners (the same planners recur across
draws), so differences below about .005 are read as ties. Acquisition ties
(routes of one type with identical posteriors are exchangeable) are broken
by bank order — a documented, deterministic choice that moves individual
cells by up to .004 (PROTOCOL section 4).

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
| **DriveAT** | **.0450** | **.0332** | .0223 | **.0116** | **.0448** | .0337 | .0202 | **.0082** | .0477 | **.0231** | **.0160** | **.0081** | **.0262** |

The three random-policy rows are the expected error over five independent
orders per evaluation; every other row is deterministic. AnchorPoints is
degenerate at K_cal = 4: four binary responses admit at most 16 distinct
route patterns, the correlation distance collapses and the same anchor
estimate comes out at every budget.

Reading. DriveAT has the lowest error in 8 of 12 cells and the lowest macro
average by a clear margin (.0262 against .0307 for type-stratified Random,
.0312 for Fluid-style and .0316 for Random + IRT). The four cells it does
not win are ties inside the intervals: K4 B110 and K8 B55/B110, where the
type-stratified order or tinyBenchmarks is lower by .001-.003, and K12 B30,
where Fluid is lower by .007. The margin grows with the budget and with the
calibration panel — at K_cal = 12 DriveAT is .0231 / .0160 / .0081 at
B = 55 / 110 / 165 while every baseline sits at .0195-.0427 — because a
better-calibrated bank makes the Delta-R1 score sharper.

**What each budget buys.** SR-MAE is an average; the decision a user makes
needs the distribution. Over the same 64 evaluations, DriveAT's estimate and
the pairwise ranking of the four evaluation planners within a draw:

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

## Table 2 — risk-target stopping (`run_tau_calibration.py`, `run_adaptive.py`)

The configuration a user of the tool is actually in: 12 calibration
planners, a new planner, the whole 220-route benchmark, and no budget
chosen in advance. Each order stops at the first t with c * R1_t <= eps,
where R1 is the posterior L1 risk of the reported SR under the common
readout and c is that order's calibration-fixed risk scale
(leave-one-planner-out on the calibration panel, 90th percentile of
realised / predicted error over t in [10, 220]; never selected on
evaluation planners; DriveAT: c = 1.97 / 2.12 / 1.95 at K_cal = 4 / 8 / 12).
eps is an *error* target. No trajectory is censored: every order may run to
route 220, and none did (cap 0% everywhere). Columns: mean rollouts spent
and the same as a fraction of the benchmark, SR-MAE at the stop, the
calibration gap (mean realised error minus mean c * R1 at the stop; negative =
conservative), and the paired delta of the SR-MAE vs DriveAT.

| K_cal | eps | method | rollouts | of 220 | SR-MAE | gap | d vs DriveAT |
|---|---|---|---|---|---|---|---|
| 4 | .05 | **DriveAT** | **83.5** | **38%** | .0272 | -.023 | — |
| | | Fluid | 99.1 | 45% | .0282 | -.021 | +.0011 |
| | | metabench | 108.8 | 49% | .0265 | -.023 | -.0007 |
| | | Random | 98.7 | 45% | .0267 | -.023 | -.0005 |
| | | Random-strat | 87.9 | 40% | .0219 | -.028 | -.0053 |
| 4 | .03 | **DriveAT** | **128.9** | **59%** | .0164 | -.013 | — |
| | | Fluid | 142.3 | 65% | .0194 | -.010 | +.0031 |
| | | metabench | 160.1 | 73% | .0138 | -.016 | -.0026 |
| | | Random | 150.5 | 68% | .0154 | -.014 | -.0009 |
| | | Random-strat | 139.9 | 64% | .0130 | -.017 | -.0033 |
| 8 | .05 | **DriveAT** | **79.0** | **36%** | .0281 | -.022 | — |
| | | Fluid | 98.8 | 45% | .0216 | -.028 | -.0065 |
| | | metabench | 118.5 | 54% | .0212 | -.029 | -.0069 |
| | | Random | 93.3 | 42% | .0260 | -.024 | -.0021 |
| | | Random-strat | 87.7 | 40% | .0208 | -.029 | -.0073 |
| 8 | .03 | **DriveAT** | **124.9** | **57%** | .0174 | -.013 | — |
| | | Fluid | 142.0 | 65% | .0172 | -.013 | -.0002 |
| | | metabench | 167.4 | 76% | .0141 | -.016 | -.0033 |
| | | Random | 146.2 | 66% | .0165 | -.013 | -.0008 |
| | | Random-strat | 140.3 | 64% | .0122 | -.018 | -.0051* |
| 12 | .05 | **DriveAT** | **70.3** | **32%** | **.0207** | -.029 | — |
| | | Fluid | 88.4 | 40% | .0283 | -.021 | +.0076* |
| | | metabench | 107.6 | 49% | .0229 | -.027 | +.0022 |
| | | Random | 91.2 | 41% | .0294 | -.020 | +.0087* |
| | | Random-strat | 87.4 | 40% | .0221 | -.028 | +.0015 |
| 12 | .03 | **DriveAT** | **114.8** | **52%** | .0138 | -.016 | — |
| | | Fluid | 132.0 | 60% | .0190 | -.011 | +.0052* |
| | | metabench | 159.7 | 73% | .0136 | -.016 | -.0002 |
| | | Random | 144.1 | 65% | .0177 | -.012 | +.0039 |
| | | Random-strat | 140.8 | 64% | .0112 | -.019 | -.0026 |

Reading. For an error target of .05, DriveAT stops after 70-84 of the 220
routes (32-38% of the benchmark); the same target costs
Fluid 88-99, Random 91-99, the type-stratified order 87-88 and metabench
108-119 routes. For .03 it needs 115-129 routes (52-59%) against 132-167 for
the others. The saving is the acquisition, not the scale: every order
carries its own leave-one-planner-out c, and the orders that keep going
arrive at a similar error. Only at K_cal = 12 does the error difference
reach significance, and there DriveAT is the *better* one (-.008 vs Random,
-.008 vs Fluid at eps = .05); at K_cal = 8 the type-stratified order reaches
eps = .03 with a lower error (-.005*) for 15 more routes. The negative gaps
say the scaled risk is conservative in the mean. The raw R1 tracks the
realised error: pooled over the LOO tracks, the decile means of raw R1 and
of |SR_hat - SR| agree within .012 at every K_cal (K_cal = 8: .0025 / .0021
in the lowest decile, .0147 / .0159 in the middle, .0601 / .0652 in the
highest).

**Matched cost (appendix).** The earlier rule — tau_hat chosen so that the
LOO mean rollouts hit a target — is kept as the cost-matched comparison. At
its own calibration-fixed tau, DriveAT's adaptive stop matches its own
fixed-budget curve at the same mean cost within .005 in every cell
(K_cal = 12, tau = .040: 37.5 rollouts, .0344 adaptive vs .0391 fixed), so
the stopping rule spends the budget as well as an oracle that had been told
the budget in advance.

## Full-system ablation (`run_system_ablation.py`)

DriveAT is two IRT pieces (the exact difficulty posterior, the planner x type
testlet) and two CAT pieces (the Delta-R1 acquisition, the LOO-calibrated
risk scale c). Each is switched off alone and scored twice: at the fixed
budget B = 55, and under the risk-target rule. Removing c means stopping on
the raw R1 (c = 1), which changes nothing at a fixed budget.

| K_cal | arm | B55 SR-MAE | eps=.05 roll / MAE | eps=.03 roll / MAE |
|---|---|---|---|---|
| 4 | **DriveAT (full)** | .0332 | **83.5** / .0272 | **128.9** / .0164 |
| | w/o b posterior | .0333 (+.0001) | 87.7 / .0266 | 131.2 / .0183 |
| | w/o testlet | .0477 (+.0145*) | 96.3 / .0283 | 139.2 / .0174 |
| | w/o Delta-R1 acquisition | .0405 (+.0074) | 98.7 / .0267 | 150.5 / .0154 |
| | w/o LOO calibration of c | .0332 (=) | 29.0 / .0423 | 63.7 / .0299 |
| 8 | **DriveAT (full)** | .0337 | **79.0** / .0281 | **124.9** / .0174 |
| | w/o b posterior | .0344 (+.0007) | 78.4 / .0263 | 123.3 / .0162 |
| | w/o testlet | .0383 (+.0046) | 102.3 / .0273 | 144.7 / .0147 |
| | w/o Delta-R1 acquisition | .0385 (+.0048) | 93.3 / .0260 | 146.2 / .0165 |
| | w/o LOO calibration of c | .0337 (=) | 26.8 / .0501 | 58.7 / .0365 |
| 12 | **DriveAT (full)** | .0231 | **70.3** / .0207 | **114.8** / .0138 |
| | w/o b posterior | .0270 (+.0039) | 72.1 / .0254 | 116.4 / .0136 |
| | w/o testlet | .0414 (+.0184*) | 94.4 / .0255 | 136.9 / .0169 |
| | w/o Delta-R1 acquisition | .0393 (+.0162*) | 91.2 / .0294 | 144.1 / .0177 |
| | w/o LOO calibration of c | .0231 (=) | 25.9 / .0535 | 57.4 / .0251 |

Reading. The four pieces fail in different ways, which is why all four are
in the method. Dropping the LOO calibration is the one that breaks the
guarantee: the raw risk stops after 26-29 routes, less than half the cost,
but the realised error at that stop is .042-.054 against the .05 target it
claims to have met, so the saving is not real. Dropping the testlet or the
acquisition keeps the error at the target but pays for it in routes, 16-24
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

## What the testlet does (`run_ablation.py`, `run_tau_calibration.py` / `run_adaptive.py` with `DRIVEAT_NO_TESTLET=1`)

The same DriveAT with sigma_g fixed to 0 (routes of one type conditionally
independent), everything else unchanged:

| | with testlet | without (sigma_g = 0) |
|---|---|---|
| DriveAT B30 / B55 / B110 / B165 at K4 | .0450 / .0332 / .0223 / .0116 | .0595 / .0477* / .0241 / .0126 |
| at K8 | .0448 / .0337 / .0202 / .0082 | .0585 / .0383 / .0252 / .0101 |
| at K12 | .0477 / .0231 / .0160 / .0081 | .0595 / .0414* / .0238* / .0110* |
| raw R1 vs realised error, LOO deciles 1 / 5 / 10 at K4 | .0030/.0029, .0157/.0174, .0627/.0749 | .0032/.0032, .0159/.0197, .0631/.0749 |
| at K8 | .0025/.0021, .0147/.0159, .0601/.0652 | .0025/.0024, .0145/.0216, .0594/.0762 |
| at K12 | .0022/.0019, .0140/.0144, .0585/.0579 | .0022/.0021, .0134/.0190, .0568/.0640 |
| risk scale c (DriveAT) at K4 / 8 / 12 | 1.97 / 2.12 / 1.95 | 2.30 / 2.88 / 2.77 |
| eps = .05: rollouts, SR-MAE at K4 | 83.5, .0272 | 96.3, .0283 |
| at K8 | 79.0, .0281 | 102.3, .0273 |
| at K12 | 70.3, .0207 | 94.4, .0255 |
| eps = .03: rollouts at K4 / 8 / 12 | 128.9 / 124.9 / 114.8 | 139.2 / 144.7 / 136.9 |

Reading. Without the dependence structure the posterior L1 risk under-states
the realised error in the middle deciles by 26-47% (K8 middle decile .0145
vs .0216) and in the top decile by 12-19%, so the calibration has to inflate
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
| **DriveAT: RelGraph R2 scene encoder (3 runs)** | **.751 +- .003** | **.192 +- .003** | **+.490 +- .016** |
| Oracle (response-calibrated) | .870 | .037 | +.995 |

Reading. The learned relational encoder and the two hand-crafted stacks
clear every single-descriptor baseline by +.03-.06 AUROC (16-18 points of
scene-MAE for the hand-crafted stacks, 11 for the encoder); between them
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
differ (three runs each, Delta rho paired by seed against the R2 runs):

| variant | AUROC | scene-MAE | rho | Delta rho vs R2 |
|---|---|---|---|---|
| RelGraph R2 (shipped) | .751 +- .003 | .192 +- .003 | +.490 +- .016 | — |
| R2 without the ego-route relation | .756 +- .002 | .184 +- .005 | +.520 +- .014 | +.030 +- .019 |
| R2, route correspondence shuffled | .754 +- .007 | .189 +- .009 | +.512 +- .024 | +.022 +- .015 |
| R2, agent-lane correspondence shuffled | .753 +- .006 | .191 +- .007 | +.501 +- .035 | +.011 +- .019 |

Reading. The relational machinery is inert on this bank: removing the
ego-route relation helps in all three runs (+.030 rho), shuffling the
route correspondence helps in all three as well (+.022), and shuffling the
agent-lane correspondence changes nothing beyond seed noise. What the
encoder learns is carried by the ego and agent tracks, not by the
lane-graph relations. R2 stays the shipped encoder because it was fixed
before these controls were scored; the paper's claim for the encoder is
the learned-from-raw-tracks difficulty prior and its transport to UPS, not
the graph structure.

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
| **DriveAT (Delta-R1 on D)** | **.0950** | **.5857** | .0936 | **.5848** | .0950 | **.5863** |

Reading. The transport is a per-cell result, not a block-SR result: it
lowers the predictive NLL of the unseen cells from .62-.64 to .585-.587
at every budget, but its block-SR error sits on a floor of about .095
from B = 30 on — the floor of the scene prior (rho about .5), which more
probes cannot lower — while the planner's own success rate on the probed
calibration routes reaches .090 / .087 at B = 55 / 110. The naive error
is the calibration-vs-evaluation gap itself: |SR_A - SR_D| averages .085
over the 64 evaluations of this panel (.102 on the 22-planner panel,
where the naive row was .12 and every transport beat it by .02-.03), and
the transported estimate's floor is not below that gap here. The
block-SR claim for UPS is therefore withdrawn on this panel; what the
transport delivers is the per-cell predictive. Among probe rules the
target-aligned Delta-R1 is ahead of Random by .018* / .012* / .004 and of
the ablations by .000-.005 (ns); the probe placement matters at low
budgets, and only through the prior.

## Readout drop-in (`run_readout_dropin.py`)

Analysis, not a headline: every baseline's selected subset re-scored with
the DriveAT readout (exact posteriors, testlet, posterior median). SR-MAE:

| selector (subset) | K4 B30 | B55 | B110 | B165 | K12 B30 | B55 | B110 | B165 |
|---|---|---|---|---|---|---|---|---|
| Fluid | .0536 | .0359 | .0247 | .0135 | .0410 | .0290 | .0225 | .0113 |
| Total-Fisher | .0570 | .0419 | .0254 | .0170 | .0496 | .0367 | .0209 | .0147 |
| metabench | .0613 | .0420 | .0242 | .0136 | .0617 | .0372 | .0220 | .0119 |
| tinyBenchmarks | .0754 | .0608 | .0275 | .0163 | .0498 | .0359 | .0222 | .0104 |
| AnchorPoints | .0687 | .0493 | .0313 | .0166 | .0524 | .0369 | .0219 | .0101 |
| Random | .0551 | .0405 | .0243 | .0134 | .0577 | .0393 | .0244 | .0134 |
| DriveAT (own order, Table 1) | .0450 | .0332 | .0223 | .0116 | .0477 | .0231 | .0160 | .0081 |

Reading. Under one readout the remaining differences are selection. At
K_cal = 12 the DriveAT order beats every re-scored subset at B = 55 / 110 by
.005-.014 and at B = 165 by .002-.007; at K_cal = 4 it leads at every budget
by .009-.030 except against the Fluid subset, which is within .002-.009. The
readout itself is a drop-in improvement for selectors whose native readout is
a plug-in on a scarce panel (compare the native Table 1 rows: AnchorPoints
.1290 -> .0687 at K4 B30, tinyBenchmarks .0621 -> .0608 at K4 B55).

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
