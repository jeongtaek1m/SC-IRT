# Results of record

Every number here is printed (and asserted, `anchors OK`) by the script named
in each section. Conventions: K = planners, S = scenes, K_cal = calibration
panel size, B = number of routes rolled out, SR-MAE = |SR_hat - SR| averaged
over 64 planner evaluations per cell (16 draws x 4 evaluation planners).
`*` marks a paired-bootstrap 95% CI vs SC-IRT that excludes zero; the
bootstrap resamples the 16 unique planners (the same planners recur across
draws), so differences below about .005 are read as ties. Acquisition ties
(routes of one type with identical posteriors are exchangeable) are broken
by bank order — a documented, deterministic choice that moves individual
cells by up to .004 (PROTOCOL section 4).

## Table 1 — UP at fixed budgets (`run_up_frontier.py`)

Unseen-planner SR reconstruction on the 16 x 220 Bench2Drive panel (one
planner per model family, PROTOCOL section 1), random 12:4 planner split,
B routes rolled out (30/55/110 = 5 x {6, 11, 22}).

| method | K4 B30 | K4 B55 | K4 B110 | K8 B30 | K8 B55 | K8 B110 | K12 B30 | K12 B55 | K12 B110 | macro |
|---|---|---|---|---|---|---|---|---|---|---|
| Random (IRT-free) | .0650* | .0420* | .0213* | .0650* | .0420* | .0213* | .0650* | .0420* | .0213* | .0428 |
| Random + IRT | .0572 | .0384 | .0202* | .0569* | .0383* | .0208* | .0572* | .0377* | .0205* | .0386 |
| Random-strat + IRT | .0545 | .0360 | .0168 | .0548 | .0364 | .0170 | .0537* | .0359* | .0171* | .0358 |
| DISCO-sel + IRT | .0666* | .0408 | .0210 | .0553* | .0396 | .0237* | .0557* | .0400* | .0212* | .0404 |
| AnchorPoints | .1447* | .1447* | .1447* | .0694* | .0523* | .0422* | .0602* | .0466* | .0245* | .0810 |
| Total-Fisher | .0576 | .0422 | .0258* | .0606 | .0409 | .0183 | .0559 | .0399* | .0214* | .0403 |
| Marginal-Fisher | .0556 | .0479* | .0249* | .0609 | .0440* | .0207* | .0578* | .0378* | .0217* | .0413 |
| tinyBenchmarks-lite | .0748* | .0465* | .0281* | .0513 | .0400 | .0172 | .0493 | .0374* | .0189* | .0404 |
| metabench-lite | .0555 | .0335 | .0202 | .0533 | .0416 | .0204* | .0554 | .0381* | .0205* | .0376 |
| Fluid-style | .0460 | .0380 | .0220* | .0500 | .0334 | .0200* | .0398 | .0282 | .0174* | .0327 |
| **SC-IRT** | .0484 | **.0344** | **.0161** | **.0450** | **.0312** | **.0147** | .0410 | **.0232** | **.0125** | **.0296** |

The three random-policy rows are the expected error over five independent
orders per evaluation (the single-order value moves by up to .004 per cell
between seeds); every other row is deterministic. AnchorPoints is
degenerate at K_cal = 4: four binary responses admit at most 16 distinct
route patterns, the correlation distance collapses and the same anchor
estimate comes out at every budget.

Reading. SC-IRT has the lowest error in 7 of 9 cells; at B = 30 for
K_cal = 4 and 12 the Fluid order is lower by .001-.002 (inside the
intervals). The macro average is .0296 against .0327 for Fluid-style, the
best baseline, .0358 for type-stratified Random (which uses the same
scenario grouping) and .0376 for metabench-lite. The gap opens with the
calibration panel: at K_cal = 12 SC-IRT is .0232 / .0125 at B = 55 / 110
against .0282 / .0174 for Fluid and .0359* / .0171* for the stratified
order, and at B = 110 every baseline is significantly worse for every
K_cal except the stratified order and tinyBenchmarks / Total-Fisher at
K_cal = 4 / 8. At B = 30 no method separates from SC-IRT except the
IRT-free / uniform Random rows and the degenerate ones.

## Table 2 — risk-target stopping (`run_tau_calibration.py`, `run_adaptive.py`)

Each order stops at the first t with c * R1_t <= eps, where R1 is the
posterior L1 risk of the reported SR under the common readout and c is
that order's calibration-fixed risk scale (leave-one-planner-out on the
calibration panel, 90th percentile of realised / predicted error over the
whole bank; never selected on evaluation planners; SC-IRT: c = 2.09 /
2.12 / 1.94 at K_cal = 4 / 8 / 12). eps is an *error* target. Every
trajectory runs through the whole per-draw bank (180 routes), so no stop
is censored by a cap. Columns: mean rollouts spent, SR-MAE at the stop,
coverage P(|SR_hat - SR| <= eps), and the calibration gap (mean realised
error minus mean c * R1 at the stop; negative = conservative). d = paired
delta of the SR-MAE vs SC-IRT.

| K_cal | eps | method | rollouts | SR-MAE | coverage | gap | d vs SC-IRT |
|---|---|---|---|---|---|---|---|
| 4 | .05 | **SC-IRT** | **75.7** | .0245 | .86 | -.025 | — |
| | | Fluid | 86.3 | .0268 | .88 | -.023 | +.0023 |
| | | metabench | 97.1 | .0236 | .91 | -.026 | -.0009 |
| | | Random | 83.3 | .0248 | .86 | -.025 | +.0003 |
| | | Random-strat | 89.2 | .0222 | .94 | -.028 | -.0023 |
| 4 | .03 | **SC-IRT** | **113.5** | .0148 | .84 | -.015 | — |
| | | Fluid | 121.5 | .0186 | .84 | -.011 | +.0038 |
| | | metabench | 137.0 | .0135 | .91 | -.016 | -.0013 |
| | | Random | 124.9 | .0156 | .91 | -.014 | +.0008 |
| | | Random-strat | 130.1 | .0128 | .94 | -.017 | -.0020 |
| 8 | .05 | **SC-IRT** | **74.2** | .0236 | .94 | -.026 | — |
| | | Fluid | 89.9 | .0239 | .95 | -.026 | +.0003 |
| | | metabench | 103.7 | .0218 | .98 | -.028 | -.0018 |
| | | Random | 89.0 | .0263 | .88 | -.023 | +.0026 |
| | | Random-strat | 89.4 | .0204 | .95 | -.029 | -.0033 |
| 8 | .03 | **SC-IRT** | **111.3** | .0141 | .94 | -.016 | — |
| | | Fluid | 123.2 | .0152 | .91 | -.015 | +.0011 |
| | | metabench | 141.5 | .0131 | .97 | -.017 | -.0010 |
| | | Random | 130.1 | .0150 | .84 | -.015 | +.0009 |
| | | Random-strat | 130.5 | .0130 | .94 | -.017 | -.0011 |
| 12 | .05 | **SC-IRT** | **65.3** | .0219 | .94 | -.028 | — |
| | | Fluid | 77.6 | .0269 | .91 | -.023 | +.0050 |
| | | metabench | 94.1 | .0244 | .94 | -.025 | +.0026 |
| | | Random | 89.1 | .0254 | .88 | -.024 | +.0035 |
| | | Random-strat | 84.9 | .0209 | .94 | -.029 | -.0010 |
| 12 | .03 | **SC-IRT** | **102.1** | .0146 | .91 | -.015 | — |
| | | Fluid | 113.9 | .0171 | .78 | -.013 | +.0025 |
| | | metabench | 135.5 | .0129 | .91 | -.017 | -.0017 |
| | | Random | 130.5 | .0153 | .88 | -.014 | +.0007 |
| | | Random-strat | 127.4 | .0137 | .95 | -.016 | -.0009 |

Reading. For an error target of .05, SC-IRT stops after 65-76 rollouts
(36-42% of the per-draw bank, 30-34% of the 220-route benchmark) with
86-94% coverage; the same target costs Fluid 78-90, Random 83-89, the
type-stratified order 85-89 and metabench 94-104 rollouts. For .03 every
order needs more than half the bank; SC-IRT reaches it earliest (102-114
rollouts) with 84-94% coverage, the others 114-142. No paired difference
in the error at the stop is significant: the orders that spend 12-40 more
rollouts arrive at a similar or slightly lower error (Random-strat -.001
to -.003), i.e. the rule trades rollouts against error through the
acquisition, which makes R1 fall faster, not through the scale. The
negative gaps say the scaled risk is conservative in the mean; the
90th-percentile scale fixed on the calibration panel transfers to the
evaluation planners at the nominal 90% or above for K_cal = 8 / 12 and at
84-86% for K_cal = 4, where the four-planner calibration gives the
error / risk ratio a heavier tail. The raw R1 itself tracks the realised
error: pooled over the LOO tracks, the decile means of raw R1 and of
|SR_hat - SR| agree within .011 at every K_cal (e.g. K_cal = 8: .0028 /
.0025 in the lowest decile, .0162 / .0176 in the middle, .0622 / .0704 in
the highest).

**Matched cost (appendix).** The earlier rule — tau_hat chosen so that the
LOO mean rollouts hit 30 / 55 — is kept as the cost-matched comparison;
IES = (SR-MAE / Random-at-fixed-55) x (rollouts / 55):

| cell | SC-IRT | Fluid | metabench | Random | Random-strat |
|---|---|---|---|---|---|
| K4 target 30 | .0449 (29.8, IES .65) | .0483 (+.0034) | .0558 (+.0109) | .0626 (+.0177*) | .0527 (+.0078) |
| K4 target 55 | .0341 (54.3, .91) | .0362 (+.0021) | .0354 (+.0013) | .0373 (+.0032) | .0358 (+.0018) |
| K8 target 30 | .0456 (29.5, .62) | .0442 (-.0014) | .0532 (+.0076) | .0677 (+.0221*) | .0534 (+.0078) |
| K8 target 55 | .0347 (53.6, .85) | .0283 (-.0064) | .0423 (+.0075) | .0415 (+.0068) | .0321 (-.0027) |
| K12 target 30 | .0428 (30.0, .61) | .0379 (-.0049) | .0498 (+.0070) | .0659 (+.0231*) | .0496 (+.0068) |
| K12 target 55 | .0262 (55.4, .69) | .0284 (+.0022) | .0335 (+.0073) | .0392 (+.0130*) | .0331 (+.0069) |

Realised budgets land on target without post-hoc adjustment (29.5-30.0,
53.6-55.4). SC-IRT is best or tied-best in four cells; Fluid is lower at
K8 / K12 target 30 and at K8 target 55 (-.0014 to -.0064, all ns) and the
type-stratified order at K8 target 55 (-.0027, ns).

## Component ablation (`run_ablation.py`)

Three components sit on top of a plug-in Rasch: the exact difficulty
posterior (off = point curves at b_hat), the planner x type testlet (off =
sigma_g = 0) and the risk acquisition (off = a random order under the same
posterior). Each is switched off alone; paired deltas vs full:

| cell | full | w/o b-uncertainty | w/o testlet | w/o risk acquisition |
|---|---|---|---|---|
| K4 B30 | .0484 | .0551 (+.0067) | .0612 (+.0128*) | .0566 (+.0082) |
| K4 B55 | .0344 | .0327 (-.0017) | .0407 (+.0062) | .0371 (+.0027) |
| K4 B110 | .0161 | .0174 (+.0013) | .0175 (+.0014) | .0208 (+.0047) |
| K8 B30 | .0450 | .0462 (+.0012) | .0519 (+.0069) | .0589 (+.0139) |
| K8 B55 | .0312 | .0300 (-.0013) | .0383 (+.0071*) | .0398 (+.0086*) |
| K8 B110 | .0147 | .0136 (-.0011) | .0198 (+.0050*) | .0221 (+.0074*) |
| K12 B30 | .0410 | .0370 (-.0040) | .0547 (+.0137*) | .0591 (+.0180*) |
| K12 B55 | .0232 | .0249 (+.0017) | .0325 (+.0092*) | .0382 (+.0150*) |
| K12 B110 | .0125 | .0119 (-.0006) | .0176 (+.0051*) | .0219 (+.0094*) |

Reading. The risk acquisition carries the fixed-budget error (+.003 to
+.018 when removed, significant at K8 B >= 55 and in every K12 cell). The
testlet is the second load-bearing piece: removing it costs +.001 to +.014,
significantly at K4 B30, K8 B >= 55 and every K12 cell — the
independent-item model re-samples scenario types it has already seen and
mis-prices the remaining ones. The difficulty posterior is neutral for the
point estimate at fixed budgets (all cells inside +-.007, none significant):
its role is the risk — it is what makes R1 track the realised error
(Table 2) and what gives all-pass / all-fail routes a one-sided difficulty
instead of a symmetric one.

## What the testlet does (`run_ablation.py`, `run_tau_calibration.py` / `run_adaptive.py` with `SCIRT_NO_TESTLET=1`)

The same SC-IRT with sigma_g fixed to 0 (routes of one type conditionally
independent), everything else unchanged:

| | with testlet | without (sigma_g = 0) |
|---|---|---|
| Table 1, SC-IRT B30 / B55 / B110 at K4 | .0484 / .0344 / .0161 | .0612* / .0407 / .0175 |
| at K8 | .0450 / .0312 / .0147 | .0519 / .0383* / .0198* |
| at K12 | .0410 / .0232 / .0125 | .0547* / .0325* / .0176* |
| raw R1 vs realised error, LOO deciles 1 / 5 / 10 at K4 | .0034/.0032, .0177/.0193, .0657/.0761 | .0037/.0035, .0181/.0211, .0664/.0795 |
| at K8 | .0028/.0025, .0162/.0176, .0622/.0704 | .0028/.0027, .0159/.0224, .0618/.0734 |
| at K12 | .0024/.0021, .0153/.0162, .0603/.0606 | .0024/.0023, .0147/.0196, .0590/.0624 |
| risk scale c (SC-IRT) at K4 / 8 / 12 | 2.09 / 2.12 / 1.94 | 2.27 / 2.78 / 2.55 |
| eps = .05: rollouts, SR-MAE, coverage at K4 | 75.7, .0245, .86 | 85.8, .0302, .78 |
| at K8 | 74.2, .0236, .94 | 90.3, .0254, .88 |
| at K12 | 65.3, .0219, .94 | 80.1, .0249, .86 |
| eps = .03: coverage at K4 / 8 / 12 | .84 / .94 / .91 | .84 / .89 / .83 |

Reading. Without the dependence structure the posterior L1 risk
under-states the realised error in the middle deciles by 17-41% (K8 middle
decile .0159 vs .0224) and in the top decile by 6-19%, so the calibration
has to inflate it (c 2.3-2.8 instead of 1.9-2.1), the error target of .05
costs 80-90 rollouts instead of 65-76 with lower coverage (.78-.88 instead
of .86-.94), and the fixed-budget error rises in every cell. The testlet is
not a patch for one K_cal: it is what makes R1 a risk. The grouping it
uses is the benchmark's own scenario-type annotation, entered only as
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
| **SC-IRT: RelGraph R2 scene encoder (3 runs)** | **.751 +- .003** | **.192 +- .003** | **+.490 +- .016** |
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
| **SC-IRT (Delta-R1 on D)** | **.0950** | **.5857** | .0936 | **.5848** | .0950 | **.5863** |

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

## Table 4 — the two-stage panel (`run_navhard.py`)

The same UP protocol on a panel with a different simulator, metric and
evaluation stage: the NAVSIM navhard leaderboard (two-stage
pseudo-closed-loop EPDMS), 87 unique submissions x 225 scored units,
pass = EPDMS >= 0.5. Per draw, 6 evaluation planners (96 evaluations per
cell); K_cal subsampled from the remaining 81 ({4, 8, 12} as on
Bench2Drive, plus the full 81); budgets are unit counts and there is no
scenario-type structure (sigma_g = 0, i.e. the independent-item special
case). SR-MAE; the strongest competitor per cell in parentheses:

| K_cal | B=30 | B=55 | B=110 |
|---|---|---|---|
| 4 | .0584 (Random+IRT .0523) | .0386 (Random+IRT .0385) | **.0231** (Random+IRT .0245) |
| 8 | .0515 (metabench .0453) | .0373 (Random+IRT .0369) | **.0216** (Random+IRT .0236) |
| 12 | .0514 (Fluid .0408) | .0367 (Fluid .0315) | **.0215** (Random+IRT .0233) |
| 81 | .0464 (Fluid .0304*) | .0360 (DISCO-sel .0293*) | .0169 (Fluid .0150) |

Reading. On a panel SC-IRT was never tuned on and where it has no
grouping to exploit, it is best at B = 110 for every Bench2Drive-sized
calibration panel, tied at B = 55 for K_cal = 4 / 8 (within .0004 of
Random + IRT) and behind at B = 30 and at K12 B55 — by .006 to Random +
IRT / metabench at K_cal = 4 / 8 and by .011 / .005 to Fluid at K_cal = 12,
none significant. Random + IRT (expected error over five orders) is the
strongest competitor: a plug-in IRT readout on a random subset is hard to
beat when the units are exchangeable and the calibration panel is a
handful of submissions drawn from a heterogeneous leaderboard. Several
published orderings are significantly worse (K_cal = 4: both Fisher
orders, tinyBenchmarks and Fluid at B >= 55; AnchorPoints everywhere).
The picture inverts with the full 81-planner calibration panel at the
low budgets, where 2PL discrimination is precisely estimated and Fluid /
Fisher orderings win by a margin (.0304* vs .0464 at B30) — a regime that
does not exist on Bench2Drive (16 planners in total) and that the 1PL
evaluation model does not try to exploit. Caveats (REPRODUCIBILITY.md):
the leaderboard is dominated by a few teams' submission sweeps and
near-duplicates can straddle the planner split; this affects all methods
identically but limits reading the panel as 87 independent planners.

## Readout drop-in (`run_readout_dropin.py`)

Analysis, not a headline: every baseline's selected subset re-scored with
the SC-IRT readout (exact posteriors, testlet, posterior median). SR-MAE:

| selector (subset) | K4 B30 | K4 B55 | K4 B110 | K12 B30 | K12 B55 | K12 B110 |
|---|---|---|---|---|---|---|
| Fluid | .0453 | .0360 | .0177 | .0398 | .0316 | .0158 |
| Total-Fisher | .0563 | .0411 | .0223 | .0487 | .0336 | .0188 |
| metabench | .0544 | .0346 | .0192 | .0520 | .0362 | .0201 |
| tinyBenchmarks | .0684 | .0455 | .0245 | .0497 | .0373 | .0187 |
| AnchorPoints | .0675 | .0499 | .0238 | .0563 | .0396 | .0201 |
| Random | .0566 | .0371 | .0208 | .0591 | .0382 | .0219 |
| SC-IRT (own order, Table 1) | .0484 | .0344 | .0161 | .0410 | .0232 | .0125 |

Reading. Under one readout the remaining differences are selection: the
SC-IRT order beats every re-scored subset at K12 B55 / B110 by .008-.016
and at K4 B110 by .002-.008, and ties the Fluid subset at B = 30 (.0453 /
.0398 vs .0484 / .0410) and the Fluid / metabench subsets at K4 B55. The
readout itself is a drop-in improvement for selectors whose native readout
is a plug-in on a scarce panel (compare the native Table 1 rows:
AnchorPoints .1447 -> .0675 at K4 B30, Total-Fisher .0409 -> .0341 at K8
B55, Fluid .0220 -> .0177 at K4 B110).

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
the kind of miss the 84-94% coverage of Table 2 predicts for roughly one
run in ten. The weakest planner remains the hardest case: an all-fail
record only bounds theta from above, so its estimate leans on the prior
early (SR_hat .22 after 40 routes) and settles as the acquisition finds
the routes it can pass.
