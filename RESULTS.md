# Results of record

Every number here is printed (and asserted, `anchors OK`) by the script named
in each section. Conventions: K = planners, S = scenes, K_cal = calibration
panel size, B = number of routes rolled out, SR-MAE = |SR_hat - SR| averaged
over 96 planner evaluations per cell (16 draws x 6 evaluation planners).
`*` marks a paired-bootstrap 95% CI vs SC-IRT that excludes zero; the
bootstrap resamples the 22 unique planners (the same planners recur across
draws), so differences below about .005 are read as ties. Acquisition ties
(routes of one type with identical posteriors are exchangeable) are broken
by bank order — a documented, deterministic choice that moves individual
cells by up to .004 (PROTOCOL section 4).

## Table 1 — UP at fixed budgets (`run_up_frontier.py`)

Unseen-planner SR reconstruction on the 22 x 220 Bench2Drive panel, random
16:6 planner split, B routes rolled out (30/55/110 = 5 x {6, 11, 22}).

| method | K7 B30 | K7 B55 | K7 B110 | K10 B30 | K10 B55 | K10 B110 | K16 B30 | K16 B55 | K16 B110 | macro |
|---|---|---|---|---|---|---|---|---|---|---|
| Random (IRT-free) | .0625* | .0417* | .0206 | .0625* | .0417* | .0206* | .0625* | .0417* | .0206* | .0416 |
| Random + IRT | .0540 | .0368* | .0194 | .0530 | .0362 | .0192* | .0534 | .0360 | .0191* | .0363 |
| Random-strat + IRT | .0509 | .0340 | .0164 | .0509 | .0333 | .0163 | .0510 | .0335 | .0163 | .0336 |
| DISCO-sel + IRT | .0581 | .0384* | .0217* | .0499 | .0340 | .0202* | .0438 | .0355 | .0225* | .0360 |
| AnchorPoints | .0786* | .0668* | .0649* | .0524 | .0377 | .0262* | .0545 | .0377 | .0255* | .0493 |
| Total-Fisher | .0545 | .0386* | .0223* | .0517 | .0381 | .0209* | .0522 | .0404 | .0233* | .0380 |
| Marginal-Fisher | .0566 | .0410* | .0220* | .0529 | .0401* | .0220* | .0568 | .0402 | .0246* | .0396 |
| tinyBenchmarks-lite | .0533 | .0396* | .0215* | .0465 | .0333 | .0195* | .0514 | .0377 | .0186 | .0357 |
| metabench-lite | .0606 | .0408* | .0215 | .0457 | .0386* | .0234* | .0477 | .0384 | .0226* | .0377 |
| Fluid-style | .0515 | .0428* | .0256* | .0496 | .0367 | .0206* | .0443 | .0355 | .0217 | .0365 |
| **SC-IRT** | **.0492** | **.0296** | .0165 | .0460 | **.0307** | **.0137** | .0447 | **.0317** | **.0139** | **.0307** |

The three random-policy rows are the expected error over five independent
orders per evaluation (the single-order value moves by up to .004 per cell
between seeds); every other row is deterministic.

Reading. SC-IRT has the lowest error in 6 of 9 cells and ties in the other
three (K7 B110: Random-strat .0164; K10 B30: metabench .0457; K16 B30:
DISCO-sel .0438, Fluid .0443 — inside the intervals), with the best macro
average (.0307 vs .0336 for the best baseline, type-stratified Random,
which uses the same scenario grouping). The gap is at the medium budget
(B = 55: .0296 / .0307 / .0317 vs .0340 / .0333 / .0335) and at B = 110 for
K10 / K16 (.0137 / .0139 vs .0163); at B = 30 the type-stratified random
order is within .002-.006, and at K7 B110 it is a tie. None of the
random-policy rows differs from SC-IRT significantly under the
planner-cluster bootstrap; the information-based orderings (Fisher,
Fluid, metabench, DISCO-sel) are significantly worse in most cells at
B >= 55.

## Table 2 — risk-target stopping (`run_tau_calibration.py`, `run_adaptive.py`)

Each order stops at the first t with c * R1_t <= eps, where R1 is the
posterior L1 risk of the reported SR under the common readout and c is
that order's calibration-fixed risk scale (leave-one-planner-out on the
calibration panel, 90th percentile of realised / predicted error; never
selected on evaluation planners; SC-IRT: c = 1.89 / 1.88 / 2.04 at
K_cal = 7 / 10 / 16). eps is an *error* target. Columns: mean rollouts spent
(110 = the bank cap), SR-MAE at the stop, coverage P(|SR_hat - SR| <= eps),
and the calibration gap (mean realised error minus mean c * R1 at the stop;
negative = conservative). d = paired delta of the SR-MAE vs SC-IRT.

| K_cal | eps | method | rollouts | SR-MAE | coverage | gap | d vs SC-IRT |
|---|---|---|---|---|---|---|---|
| 7 | .05 | **SC-IRT** | **65.9** | .0284 | .85 | -.021 | — |
| | | Fluid | 73.2 | .0323 | .80 | -.017 | +.0039 |
| | | metabench | 94.5 | .0278 | .85 | -.024 | -.0006 |
| | | Random | 86.7 | .0271 | .85 | -.023 | -.0013 |
| 7 | .03 | **SC-IRT** | **99.0** | **.0186** | **.83** | -.013 | — |
| | | Fluid | 102.0 | .0245 | .66 | -.009 | +.0059* |
| | | metabench | 109.5 | .0210 | .73 | -.024 | +.0025 |
| | | Random | 109.6 | .0182 | .80 | -.021 | -.0004 |
| 10 | .05 | **SC-IRT** | **66.3** | .0269 | .84 | -.023 | — |
| | | Fluid | 80.6 | .0268 | .85 | -.023 | -.0001 |
| | | metabench | 91.9 | .0279 | .84 | -.024 | +.0010 |
| | | Random | 90.1 | .0248 | .93 | -.025 | -.0020 |
| 10 | .03 | **SC-IRT** | **99.3** | **.0166** | **.84** | -.015 | — |
| | | Fluid | 106.4 | .0204 | .76 | -.016 | +.0037 |
| | | metabench | 109.4 | .0235 | .68 | -.020 | +.0069 |
| | | Random | 109.4 | .0172 | .81 | -.023 | +.0006 |
| 16 | .05 | **SC-IRT** | **69.4** | **.0252** | .85 | -.024 | — |
| | | Fluid | 82.2 | .0290 | .78 | -.021 | +.0038 |
| | | metabench | 96.7 | .0282 | .86 | -.023 | +.0030 |
| | | Random | 86.8 | .0246 | .89 | -.025 | -.0006 |
| 16 | .03 | **SC-IRT** | **103.2** | **.0162** | **.82** | -.015 | — |
| | | Fluid | 106.7 | .0212 | .74 | -.015 | +.0050 |
| | | metabench | 110.0 | .0254 | .67 | -.020 | +.0092* |
| | | Random | 109.4 | .0173 | .81 | -.021 | +.0011 |

Reading. For an error target of .05, SC-IRT stops after 66-69 rollouts
(37-39% of the per-draw bank, 30% of the 220-route benchmark) with 84-85%
coverage; the same target costs Fluid 73-82, Random 87-90 and metabench
92-97 rollouts. For .03 every order needs most of the bank; SC-IRT reaches
it earliest (99-103 rollouts) with the lowest error (.016-.019) and the
highest coverage (.82-.84 vs .66-.81). The negative gaps say the scaled
risk is conservative in the mean; the 90th-percentile scale, fixed on the
calibration panel, transfers to the evaluation planners as 82-85% coverage
rather than the nominal 90%, because their error / risk ratio has a
heavier tail. Random with its own calibrated scale is a fair competitor at
eps = .05 — a similar error at 17-24 more rollouts (ns). The raw R1 itself
tracks the realised error: pooled over the LOO tracks, the decile means of
raw R1 and of |SR_hat - SR| agree within .006 at every K_cal (e.g.
K_cal = 7: .0137 / .0153 in the lowest decile, .0266 / .0243 in the middle,
.0680 / .0735 in the highest).

**Matched cost (appendix).** The earlier rule — tau_hat chosen so that the
LOO mean rollouts hit 30 / 55 — is kept as the cost-matched comparison;
IES = (SR-MAE / Random-at-fixed-55) x (rollouts / 55):

| cell | SC-IRT | Fluid | metabench | Random |
|---|---|---|---|---|
| K7 target 30 | .0472 (29.5, IES .63) | .0486 (+.0014) | .0575 (+.0103) | .0609 (+.0137*) |
| K7 target 55 | .0314 (54.1, .77) | .0351 (+.0036) | .0423 (+.0109*) | .0383 (+.0069) |
| K10 target 30 | .0481 (30.0, .65) | .0449 (-.0032) | .0451 (-.0030) | .0596 (+.0115) |
| K10 target 55 | .0314 (55.2, .78) | .0330 (+.0016) | .0338 (+.0024) | .0379 (+.0065) |
| K16 target 30 | .0459 (30.2, .63) | .0455 (-.0003) | .0462 (+.0003) | .0607 (+.0149*) |
| K16 target 55 | .0319 (55.2, .80) | .0333 (+.0014) | .0344 (+.0025) | .0379 (+.0060) |

Realised budgets land on target without post-hoc adjustment (29.2-30.6,
54.1-55.8). SC-IRT is best or tied-best in every cell (K10 target 30 is a
tie against Fluid / metabench, -.003, ns).

## Component ablation (`run_ablation.py`)

Three components sit on top of a plug-in Rasch: the exact difficulty
posterior (off = point curves at b_hat), the planner x type testlet (off =
sigma_g = 0) and the risk acquisition (off = a random order under the same
posterior). Each is switched off alone; paired deltas vs full:

| cell | full | w/o b-uncertainty | w/o testlet | w/o risk acquisition |
|---|---|---|---|---|
| K7 B30 | .0492 | .0452 (-.0041) | .0529 (+.0036) | .0591 (+.0099) |
| K7 B55 | .0296 | .0261 (-.0035*) | .0362 (+.0066*) | .0384 (+.0088*) |
| K7 B110 | .0165 | .0168 (+.0003) | .0227 (+.0062*) | .0209 (+.0043) |
| K10 B30 | .0460 | .0477 (+.0017) | .0523 (+.0064) | .0581 (+.0121) |
| K10 B55 | .0307 | .0295 (-.0012) | .0381 (+.0074) | .0363 (+.0056) |
| K10 B110 | .0137 | .0134 (-.0003) | .0190 (+.0053*) | .0195 (+.0058*) |
| K16 B30 | .0447 | .0460 (+.0013) | .0551 (+.0104) | .0597 (+.0150) |
| K16 B55 | .0317 | .0334 (+.0016) | .0401 (+.0084) | .0364 (+.0047) |
| K16 B110 | .0139 | .0142 (+.0003) | .0187 (+.0048*) | .0193 (+.0054*) |

Reading. The risk acquisition carries the fixed-budget error (+.004 to
+.015 when removed, significant at K7 B55 and at B = 110 for K10 / K16). The
testlet is the second load-bearing piece: removing it costs +.004 to +.010
in every cell, significantly at B = 110 for all K_cal and at K7 B55 — the
independent-item model re-samples scenario types it has already seen and
mis-prices the remaining ones. The difficulty posterior is neutral for the
point estimate at fixed budgets (all cells inside +-.002 except K7, where
the point curves are slightly *better*, -.0041 / -.0035*): its role is the
risk — it is what makes R1 track the realised error (Table 2) and what gives
all-pass / all-fail routes a one-sided difficulty instead of a symmetric one.

## What the testlet does (`run_ablation.py`, `run_tau_calibration.py` / `run_adaptive.py` with `SCIRT_NO_TESTLET=1`)

The same SC-IRT with sigma_g fixed to 0 (routes of one type conditionally
independent), everything else unchanged:

| | with testlet (v5) | without (sigma_g = 0) |
|---|---|---|
| Table 1, SC-IRT B30 / B55 / B110 at K7 | .0492 / .0296 / .0165 | .0529 / .0362* / .0227* |
| at K10 | .0460 / .0307 / .0137 | .0523 / .0381 / .0190* |
| at K16 | .0447 / .0317 / .0139 | .0551 / .0401 / .0187* |
| raw R1 vs realised error, LOO deciles 1 / 5 / 10 at K7 | .0137/.0153, .0266/.0243, .0680/.0735 | .0129/.0180, .0267/.0313, .0671/.0759 |
| at K10 | .0135/.0173, .0260/.0247, .0671/.0697 | .0126/.0233, .0256/.0294, .0664/.0737 |
| at K16 | .0129/.0164, .0253/.0253, .0654/.0651 | .0118/.0208, .0243/.0330, .0643/.0709 |
| risk scale c (SC-IRT) at K7 / 10 / 16 | 1.89 / 1.88 / 2.04 | 2.34 / 2.47 / 2.54 |
| eps = .05: rollouts, SR-MAE, coverage at K7 | 65.9, .0284, .85 | 79.1, .0306, .81 |
| at K10 | 66.3, .0269, .84 | 82.3, .0275, .85 |
| at K16 | 69.4, .0252, .85 | 82.1, .0297, .80 |
| eps = .03: coverage at K7 / 10 / 16 | .83 / .84 / .82 | .72 / .79 / .78 |

Reading. Without the dependence structure the posterior L1 risk
under-states the realised error in every decile by 30-70% (K16 lowest
decile .0118 vs .0208), so the calibration has to inflate it (c 2.3-2.5
instead of 1.9-2.0), the error target of .05 costs 79-82 rollouts instead
of 66-69, and Random with its own scale reaches the same target with a
lower error. The testlet is therefore not a K16 patch: it is what makes R1
a risk. The grouping it uses is the benchmark's own scenario-type
annotation, entered only as "these routes share an offset"; no difficulty
or feature is read from it (PROTOCOL section 1).

## Table 3A — US: unseen scenes (`run_us.py`)

Predict scene difficulty (and per-cell outcomes) for the 8 evaluation
scenario types from the scene alone; pooled over 16 draws (640 route
evaluations). Descriptor rows are scored through a two-stage Ridge plug-in
fitted on the calibration types; the encoder row is the RelGraph R2
out-of-fold prediction (retrained on the v5 calibration). Planner-only
null: AUROC .694 / scene-MAE .199.

| difficulty source | AUROC | scene-MAE | rho(b_tilde, fail rate) |
|---|---|---|---|
| Min-TTC | .691 | .202 (-1.6%) | -.152 |
| Risk field | .713 | .195 (+2.1%) | +.133 |
| Route geometry | .723 | .193 (+3.1%) | +.361 |
| Agent density + kin. | .725 | .191 (+4.0%) | +.301 |
| Traffic entropy | .706 | .197 (+1.3%) | +.111 |
| Agent-JEPA | .699 | .199 (+0.2%) | +.051 |
| Kinematics (cmdkin, 25d) | .752 | .172 (+13.5%) | +.526 |
| Hand-crafted risk (cmdkin+gtrisk, 73d) | .753 | .171 (+14.3%) | +.558 |
| **SC-IRT: RelGraph R2 scene encoder (3 runs)** | **.747 +- .006** | **.184 +- .005** | **+.506 +- .024** |
| Oracle (response-calibrated) | .862 | .027 | +.997 |

Reading. The learned relational encoder and the two hand-crafted stacks
clear every single-descriptor baseline by +.02-.06 AUROC (13-14 points of
scene-MAE for the hand-crafted stacks, 5-9 for the encoder); between them
the encoder is tied on AUROC / MAE and behind on rank correlation (RelGraph
minus hand-crafted risk: Delta rho -.052 +- .024 across runs, i.e. about
two run-SDs). The encoder buys no difficulty signal beyond well-chosen
rollout descriptors on this bank; what it offers is the same signal from
the raw scene graph without feature engineering. The oracle (.862) is the
ceiling of any scene-only predictor: roughly half the difficulty variance
is not visible from the scene. (Earlier versions reported a descriptor
stack that included the scenario-definition parameters; it was removed
because those values are the benchmark's own construction parameters, not
observable scene content.)

## Table 3B — UPS: unseen planner x unseen scenes (`run_ups.py`)

Predict an unseen planner's behaviour on unseen scenario types with zero
rollouts on the target block: probe the planner on B calibration-type
routes, transport the ability posterior through the RelGraph difficulty
prior N(b_tilde_s, sigma^2) with the testlet prior on the (unobserved)
evaluation types. The MAE scores the posterior median of the block-D
success rate, the NLL scores the per-cell posterior predictive. 96
evaluations, RelGraph run s0. The canonical probe rule is Delta-R1 on the
block-D success rate — the UP acquisition with its risk evaluated on the
target block; theta-EIG and the 2PL Fisher rule are ablations.

| probe policy | B30 MAE | NLL | B55 MAE | NLL | B110 MAE | NLL |
|---|---|---|---|---|---|---|
| naive (no IRT) | .1233 | .6402 | .1215 | .6361 | .1151 | .6311 |
| Random | .1047 | .5959 | .0990 | .5931 | .0914 | .5893 |
| theta-EIG (abl.) | .1041 | .6001 | .0978 | .5949 | .0926 | .5899 |
| 2PL Fisher (abl.) | .0999 | .5946 | .0928 | .5907 | .0873 | .5875 |
| **SC-IRT (Delta-R1 on D)** | .1030 | .6003 | .0971 | .5942 | .0927 | .5900 |

Reading. Any transport beats the naive planner-mean by .02-.03 MAE; that
transport, not the probe placement, is the UPS result — the bottleneck is
the scene prior (rho about .5). Among probe rules the target-aligned
Delta-R1 is at or ahead of theta-EIG (+.0008* at B55) and Random (+.0017 at
B30, ns); the 2PL Fisher rule is .003-.005 better (significant at B110,
-.0054*): with 16 calibration planners the item discriminations it uses
are real information for placing probes. It stays an ablation because it
needs a second model; the paper's claim is the transport, not the probe.

## Table 4 — the two-stage panel (`run_navhard.py`)

The same UP protocol on a panel with a different simulator, metric and
evaluation stage: the NAVSIM navhard leaderboard (two-stage
pseudo-closed-loop EPDMS), 87 unique submissions x 225 scored units,
pass = EPDMS >= 0.5. Per draw, 6 evaluation planners; K_cal subsampled
from the remaining 81 ({7, 10, 16} as on Bench2Drive, plus the full 81);
budgets are unit counts and there is no scenario-type structure (sigma_g =
0, i.e. the independent-item special case). SR-MAE; the strongest
competitor per cell in parentheses:

| K_cal | B=30 | B=55 | B=110 |
|---|---|---|---|
| 7 | **.0429** (metabench .0466) | **.0347** (Random+IRT .0361) | .0251 (Random+IRT .0232) |
| 10 | **.0482** (Random+IRT .0493) | **.0348** (Random+IRT .0361) | **.0210** (tinyB .0221) |
| 16 | .0488 (Random+IRT .0488) | .0356 (Random+IRT .0356) | **.0191** (tinyB .0218) |
| 81 | .0464 (Fluid .0304*) | .0360 (DISCO-sel .0293*) | .0169 (Fluid .0150) |

Reading. With the calibration panels the Bench2Drive protocol uses
(K_cal <= 16) SC-IRT is best or tied-best in 8 of 9 cells of a panel it
was never tuned on and has no grouping to exploit; several published
orderings are significantly worse (K7: tinyBenchmarks .0535*, DISCO-sel
.0557*, Fluid .0584*). Random + IRT (expected error over five orders) is
the strongest competitor here — a plug-in IRT readout on a random subset
is hard to beat when the units are exchangeable — and it edges SC-IRT at
K7 B110 (ns). The picture inverts only with the full 81-planner
calibration panel at the low budgets, where 2PL discrimination is
precisely estimated and Fluid / Fisher orderings win by a margin (.0304*
vs .0464 at B30) — a regime that does not exist on Bench2Drive (22 planners
in total) and that the 1PL evaluation model does not try to exploit.
Caveats (REPRODUCIBILITY.md): the leaderboard is dominated by a few teams'
submission sweeps and near-duplicates can straddle the planner split; this
affects all methods identically but limits reading the panel as 87
independent planners.

## Readout drop-in (`run_readout_dropin.py`)

Analysis, not a headline: every baseline's selected subset re-scored with
the SC-IRT readout (exact posteriors, testlet, posterior median). SR-MAE:

| selector (subset) | K7 B30 | K7 B55 | K7 B110 | K16 B30 | K16 B55 | K16 B110 |
|---|---|---|---|---|---|---|
| Fluid | .0493 | .0387 | .0212 | .0445 | .0345 | .0197 |
| Total-Fisher | .0553 | .0428 | .0235 | .0485 | .0430 | .0286 |
| metabench | .0591 | .0395 | .0209 | .0449 | .0366 | .0254 |
| tinyBenchmarks | .0565 | .0398 | .0197 | .0503 | .0389 | .0185 |
| AnchorPoints | .0549 | .0349 | .0211 | .0515 | .0345 | .0207 |
| Random | .0576 | .0403 | .0182 | .0588 | .0399 | .0173 |
| SC-IRT (own order, Table 1) | .0492 | .0296 | .0165 | .0447 | .0317 | .0139 |

Reading. Under one readout the remaining differences are selection: the
SC-IRT order beats every re-scored subset at K7 B55 / B110 by .005-.013 and
at K16 B55 / B110 by .003-.015, and ties the Fluid / metabench subsets at
B = 30. The readout itself is a drop-in improvement for selectors whose
native readout is a plug-in on a scarce panel (compare the native Table 1
rows: AnchorPoints .0786 -> .0549 at K7 B30).

## Inside a real evaluation (`tools/b2d_adaptive_eval.py`)

The same posterior drives an actual Bench2Drive run: the bank is the full
22 x 220 panel (excluding the planner under test if it is in it), the risk
scale c is fixed by leave-one-planner-out on that bank (c = 2.36 for the
22-planner bank; 2.55 / 2.32 / 2.16 for the three held-out banks below),
and the driver picks one route at a time, runs it through the leaderboard
evaluator, reads the outcome and stops at c * R1 <= eps. Simulated from the
matrix (`--dry-run`, planner held out of the bank, eps = .03, cap 110
routes; the bank is the planner's recorded routes, 211-220 of 220):

| planner | true SR | routes run | stopped by | SR_hat | abs err | types covered |
|---|---|---|---|---|---|---|
| VAD | .155 | 106 | risk (c * R1 = .0296) | .170 | .015 | 37 / 44 |
| HiP-AD | .664 | 110 | cap (c * R1 = .039) | .645 | .019 | 42 / 44 |
| LEAD-tfv6 | .777 | 99 | risk (c * R1 = .0300) | .814 | .036 | 38 / 44 |

Half the benchmark and errors of .015-.019 SR for VAD and HiP-AD; the
LEAD-tfv6 stop (99 routes) missed its .03 target with a realised error of
.036 — the kind of miss the 82-85% coverage of Table 2 predicts for
roughly one run in six. The weakest
planner remains the hardest case: an all-fail record only bounds theta
from above, so its estimate leans on the prior early (SR_hat .22 after 40
routes) and settles as the acquisition finds the routes it can pass.
