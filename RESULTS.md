# Results of record

Every number here is printed (and asserted, `anchors OK`) by the script named
in each section. Conventions: K = planners, S = scenes, K_cal = calibration
panel size, B = rollout budget, SR-MAE = |SR_hat - SR| averaged over 96
planner evaluations per cell (16 draws x 6 evaluation planners). `*` marks a
paired-bootstrap 95% CI vs SC-IRT that excludes zero. Differences below
about .005 are inside the intervals at this replication and are read as ties.

## Table 1 — UP at fixed budgets (`run_up_frontier.py`)

Unseen-planner SR reconstruction on the 22 x 220 Bench2Drive panel,
random 16:6 planner split, B counted in scenario types (5 routes each).

| method | K7 B30 | K7 B55 | K7 B110 | K10 B30 | K10 B55 | K10 B110 | K16 B30 | K16 B55 | K16 B110 | macro |
|---|---|---|---|---|---|---|---|---|---|---|
| Random (IRT-free) | .0670* | .0405 | .0224 | .0670* | .0405* | .0224* | .0670 | .0405 | .0224 | .0433 |
| Random + IRT | .0602 | .0401 | .0216 | .0571* | .0369 | .0193 | .0587 | .0377 | .0191 | .0390 |
| Random-strat + IRT | .0589 | .0365 | .0175* | .0554 | .0358 | .0175 | .0554 | .0353* | .0174 | .0366 |
| DISCO | .0657* | .0466* | .0315* | .0547 | .0398 | .0253* | .0455 | .0389 | .0249* | .0414 |
| AnchorPoints | .0805* | .0669* | .0649* | .0559* | .0387 | .0264* | .0554 | .0470 | .0243* | .0511 |
| Total-Fisher | .0581 | .0406 | .0268 | .0549 | .0411 | .0260* | .0507 | .0358 | .0208 | .0394 |
| Marginal-Fisher | .0549 | .0471* | .0297* | .0551 | .0409 | .0241* | .0521 | .0379 | .0234* | .0406 |
| tinyBenchmarks | .0626* | .0475 | .0411* | .0529 | .0384 | .0214 | .0588 | .0344* | .0222 | .0421 |
| metabench | .0554 | .0424 | .0236 | .0472 | .0361 | .0224 | .0460* | .0375 | .0232 | .0371 |
| Fluid | .0486 | .0421 | .0258 | .0495 | .0436* | .0214* | .0472* | .0414 | .0219* | .0379 |
| **SC-IRT** | **.0497** | **.0375** | .0248 | **.0453** | **.0339** | **.0189** | .0571 | .0419 | .0197 | **.0365** |

Reading. SC-IRT is best or tied-best in every K_cal <= 10 cell except
K7 B110 and holds the best macro average. The advantage concentrates where
evaluation is hard — small calibration panels at low-to-medium budgets
(e.g. K10 B30: .0453 vs Fluid .0495, Random .0670*). At K_cal = 16 with
B >= 55 (a rich panel plus 31-61% of the per-draw 180-route bank) the
IRT-based orderings stop paying for themselves and stratified random
sampling is the frontier (.0353* at K16 B55); that saturation column is
part of the result, not an omission. AnchorPoints and DISCO, which pick
by response-similarity clustering, are the weakest orderings on a
220-scene bank at these budgets.

## Table 2 — adaptive stopping (`run_tau_calibration.py`, `run_adaptive.py`)

Each method stops at its own calibration-fixed threshold tau_hat
(leave-one-planner-out on the calibration panel, tuned to a mean-budget
target — a cost target, never an accuracy target, so evaluation SR-MAE is
measured, not selected). IES = (SR-MAE / Random-at-fixed-55) x (mean
rollouts / 55); lower is better, d = paired delta vs SC-IRT.

Thresholds are stable across draws (SC-IRT median tau_hat: .039-.041 for
target 30, .026-.027 for target 55, IQR width <= .002 at every K_cal) and
the realized budgets land on target without post-hoc adjustment (29.8-31.1
and 54.0-56.3 across all methods).

| cell | method | rollouts | SR-MAE | IES | d vs SC-IRT |
|---|---|---|---|---|---|
| K7 target 30 | **SC-IRT** | 30.2 | .0471 | .63 | — |
| | Fluid | 30.2 | .0422 | .57 | -.0049 |
| | metabench | 30.1 | .0540 | .72 | +.0069 |
| | Random | 31.0 | .0590 | .81 | +.0118* |
| K7 target 55 | **SC-IRT** | 54.2 | **.0342** | **.82** | — |
| | Fluid | 54.0 | .0409 | .98 | +.0067* |
| | metabench | 55.8 | .0434 | 1.08 | +.0092* |
| | Random | 56.3 | .0398 | 1.00 | +.0056 |
| K10 target 30 | **SC-IRT** | 30.4 | **.0409** | **.61** | — |
| | Fluid | 31.0 | .0483 | .73 | +.0073 |
| | metabench | 30.6 | .0511 | .77 | +.0102 |
| | Random | 31.1 | .0556 | .85 | +.0147* |
| K10 target 55 | **SC-IRT** | 55.2 | **.0349** | .94 | — |
| | Fluid | 56.1 | .0424 | 1.17 | +.0075* |
| | metabench | 55.9 | .0360 | .99 | +.0011 |
| | Random | 56.0 | .0384 | 1.05 | +.0034 |
| K16 target 30 | SC-IRT | 30.3 | .0543 | .80 | — |
| | Fluid | 30.4 | .0447 | .66 | -.0097* |
| | metabench | 29.8 | .0461 | .66 | -.0082* |
| | Random | 30.7 | .0570 | .85 | +.0027 |
| K16 target 55 | SC-IRT | 54.7 | .0399 | 1.06 | — |
| | Fluid | 55.3 | .0395 | 1.05 | -.0004 |
| | metabench | 55.8 | .0374 | 1.01 | -.0025 |
| | Random | 55.8 | .0374 | 1.01 | -.0025 |

Reading. At K_cal <= 10 the full pipeline (risk acquisition + risk
stopping) is best or tied-best in every cell — at K7 target 55 it beats
every competitor's own adaptive variant (Fluid +.0067*, metabench
+.0092*), and at K10 target 30 it does better than fixed-55 Random using
30 rollouts (IES .61). At K16 the panel is rich enough that 2PL-based
orderings pay off and SC-IRT's ordering advantage inverts (target 30:
Fluid -.0097*, metabench -.0082*) — the same saturation boundary as
Table 1 and the component ablation, reported as-is.

Adaptive vs fixed at matched mean rollouts (SC-IRT against its own
fixed-budget curve): -.0018 to -.0046 for tau <= .040 at K7/K10 — the
stopping rule redistributes budget across evaluation planners at no
accuracy cost. R1 is a usable risk: thresholded on left-out calibration
planners it transfers to unseen planners with realized budgets on target.

## Component ablation (`run_ablation.py`)

The model has exactly two components on top of plug-in Rasch: (i)
b-uncertainty — the conditional-Laplace difficulty posterior marginalised
into every probability; (ii) risk acquisition — the Delta-R1 rollout order.
Each is switched off independently (off-acquisition = per-evaluation random
permutation; off-uncertainty = point curves at s -> 0). Paired deltas vs full,
`*` = 95% CI excludes zero:

| cell | full | w/o b-uncertainty | w/o risk acquisition | w/o both |
|---|---|---|---|---|
| K7 B30 | .0497 | .0493 (-.0005) | .0617 (+.0120*) | .0601 (+.0103) |
| K7 B55 | .0375 | .0392 (+.0017) | .0408 (+.0033) | .0404 (+.0029) |
| K10 B30 | .0453 | .0450 (-.0003) | .0586 (+.0133*) | .0571 (+.0119*) |
| K10 B55 | .0339 | .0360 (+.0022) | .0371 (+.0032) | .0373 (+.0034) |
| K16 B30 | .0571 | .0563 (-.0008) | .0597 (+.0026) | .0588 (+.0017) |
| K16 B55 | .0419 | .0379 (-.0040) | .0376 (-.0043) | .0375 (-.0044) |

Reading. The fixed-budget error is carried by the risk acquisition: removing
it costs +.0120*/+.0133* at the small-panel low-budget cells and the loss
shrinks toward zero as K_cal and B grow (at K16 B55 the panel saturates and
the ordering no longer matters — the same reversal as Table 1). Removing the
b-uncertainty leaves fixed-budget SR-MAE unchanged everywhere (all cells
inside the intervals): its role is not the point estimate but the risk
calibration — without it R1 is not a posterior risk and the stopping rule of
Table 2 has no calibrated trigger. The w/o-both column tracks
w/o-risk-acquisition, i.e. the two components do not interact at fixed budgets.

## Table 3A — US: unseen scenes (`run_us.py`)

Predict scene difficulty (and per-cell outcomes) for the 8 evaluation
scenario types from the scene alone; pooled over 16 draws (640 route
evaluations). Descriptor rows are scored through a two-stage Ridge plug-in
fitted on the calibration types; the encoder row is the RelGraph R2
out-of-fold prediction. Planner-only null: AUROC .694 / scene-MAE .199.

| difficulty source | AUROC | scene-MAE | rho(b_tilde, fail rate) |
|---|---|---|---|
| Min-TTC | .689 | .205 (-3.2%) | -.146 |
| Risk field | .710 | .199 (+0.1%) | +.125 |
| Route geometry | .720 | .196 (+1.6%) | +.353 |
| Agent density + kin. | .723 | .195 (+1.9%) | +.308 |
| Traffic entropy | .706 | .198 (+0.6%) | +.122 |
| Agent-JEPA | .696 | .201 (-1.1%) | +.006 |
| Kinematics (cmdkin, 25d) | .753 | .173 (+12.8%) | +.514 |
| Hand-crafted risk (cmdkin+gtrisk, 73d) | .751 | .175 (+12.0%) | +.541 |
| **SC-IRT: RelGraph R2 scene encoder (3 runs)** | **.754 +- .004** | **.178 +- .003** | **+.532 +- .019** |
| Oracle (response-calibrated) | .863 | .000 | +.997 |

Reading. The learned relational encoder and the two hand-crafted stacks
are statistically tied (RelGraph minus hand-crafted risk: Delta rho
-.008 +- .019 across runs) and all three clear the single-descriptor
baselines by +.03 AUROC and 10 points of scene-MAE. The encoder buys no
difficulty signal beyond well-chosen rollout descriptors on this bank, but
it needs no feature engineering — it consumes the raw scene graph. The
oracle gap (.863) is the ceiling any scene-only predictor faces: roughly
half the difficulty variance is not visible from the scene. Earlier
versions reported a descriptor stack that included the scenario-definition
parameters (scenparamz); it was removed because those values are the
benchmark's own construction parameters, not observable scene content.

## Table 3B — UPS: unseen planner x unseen scenes (`run_ups.py`)

Predict an unseen planner's behaviour on unseen scenario types with zero
rollouts on the target block: probe the planner on B calibration-type
rollouts, transport theta_hat through the RelGraph difficulty prior
N(b_tilde_s, sigma^2) (sigma = the residual SD the encoder learned on the
calibration block, ~.65). 96 evaluations, RelGraph run s0; MAE on the
target-block SR, mean per-cell NLL. Across the three encoder runs the MAE
cells move by .002-.006 (SD).

| probe policy | B30 MAE | NLL | B55 MAE | NLL | B110 MAE | NLL |
|---|---|---|---|---|---|---|
| naive (no IRT) | .1290 | .6445 | .1189 | .6348 | .1146 | .6302 |
| Random | .1039 | .6071 | .0941 | .6008 | .0905 | .5986 |
| theta-EIG (canonical) | .1017 | .6071 | .0946 | .6021 | .0884 | .5968 |
| 2PL Fisher (abl.) | .0983 | .6049 | .0914 | .6018 | .0881 | .5966 |

Reading. Any IRT transport beats the naive planner-mean by .024-.031 MAE;
between probe policies the paired deltas are all inside the intervals
(Random minus theta-EIG +.0022 ns at B30; 2PL Fisher minus theta-EIG -.0034 ns). The
bottleneck is the US difficulty prior (rho about .53), not probe placement
— UPS is reported as a transport result, not an acquisition result.

## Table 4 — the two-stage panel (`run_navhard.py`)

The same UP protocol on a panel with a different simulator, metric and
evaluation stage: the NAVSIM navhard leaderboard (two-stage
pseudo-closed-loop EPDMS), 87 unique submissions x 225 scored units,
pass = EPDMS >= 0.5. Per draw, 6 evaluation planners; K_cal subsampled
from the remaining 81 ({7, 10, 16} as on Bench2Drive, plus the full 81);
budgets are unit counts (no scenario-type structure, so no stratified
Random). SR-MAE, `*` vs SC-IRT as above; the strongest competitor per
K_cal in parentheses:

| K_cal | B=30 | B=55 | B=110 |
|---|---|---|---|
| 7 | **.0419** (metabench .0479) | **.0312** (Random+IRT .0344) | .0250 (Random+IRT .0242) |
| 10 | .0486 (tinyB .0469) | .0383 (tinyB .0334) | **.0203** (Random+IRT .0225) |
| 16 | .0487 (tinyB .0465) | .0345 (Random+IRT .0330) | **.0197** (Random+IRT .0229) |
| 81 | .0468 (DISCO .0396) | .0360 (DISCO .0299) | **.0166** (Fluid .0205*) |

Reading. The Bench2Drive pattern transfers to a two-stage panel without
retuning anything: at K_cal = 7 SC-IRT is best at B <= 55 with most
baselines significantly worse (Random .0674*, Fluid .0516*, Total-Fisher
.0531* at B30), and it is the best B = 110 method at every K_cal >= 10.
The saturation boundary moves the same way too — with the full 81-planner
calibration panel, response-clustering (DISCO) wins the low budgets, as
the rich-panel regime did on Bench2Drive. Caveats (REPRODUCIBILITY.md):
the leaderboard is dominated by a few teams' submission sweeps and
near-duplicates can straddle the planner split; this affects all methods
identically but limits reading the panel as 87 independent planners.

## Readout drop-in (`run_readout_dropin.py`)

Analysis, not a headline: every baseline's selected subset re-scored with
the SC-IRT readout (marginalised curves + theta-MAP fill), native readout
in parentheses. SR-MAE at K_cal = 7:

| selector | B=30 | B=55 | B=110 |
|---|---|---|---|
| Fluid | .0477 (.0486) | .0420 (.0421) | .0241 (.0258) |
| Total-Fisher | .0525 (.0581) | .0367 (.0406) | .0208 (.0268) |
| metabench | .0526 (.0554) | .0409 (.0424) | .0240 (.0236) |
| tinyBenchmarks | .0632 (.0626) | .0483 (.0475) | .0419 (.0411) |
| AnchorPoints | .0589 (.0805) | .0450 (.0669) | .0433 (.0649) |
| Random | .0617 (.0602) | .0408 (.0401) | .0215 (.0216) |

Reading. The readout is a drop-in improvement wherever the native readout
is a 2PL plug-in on a scarce panel — AnchorPoints about -.022 and
Total-Fisher -.004 to -.006 at every K_cal = 7 budget (same direction at
K_cal = 16, e.g. AnchorPoints .0243 -> .0194 at B110) — and neutral for
selectors whose native readout is already an average or a well-behaved
plug-in (Random, metabench, tinyBenchmarks, all within .003). So the
Table 1 gaps are not a readout artifact: with the readout equalised, the
remaining differences are attributable to scene selection.
