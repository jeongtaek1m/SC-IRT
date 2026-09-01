# Results of record

Every number here is printed (and asserted, `anchors OK`) by the script named
in each section. Conventions: K = planners, S = scenes, K_cal = calibration
panel size, B = number of routes rolled out, SR-MAE = |SR_hat - SR| averaged
over 96 planner evaluations per cell (16 draws x 6 evaluation planners).
`*` marks a paired-bootstrap 95% CI vs SC-IRT that excludes zero; the
bootstrap resamples the 22 unique planners (the same planners recur across
draws), so differences below about .005 are read as ties.

## Table 1 — UP at fixed budgets (`run_up_frontier.py`)

Unseen-planner SR reconstruction on the 22 x 220 Bench2Drive panel, random
16:6 planner split, B routes rolled out (30/55/110 = 5 x {6, 11, 22}, so the
type-stratified baseline can execute whole scenario types).

| method | K7 B30 | K7 B55 | K7 B110 | K10 B30 | K10 B55 | K10 B110 | K16 B30 | K16 B55 | K16 B110 | macro |
|---|---|---|---|---|---|---|---|---|---|---|
| Random (IRT-free) | .0670* | .0405* | .0224* | .0670* | .0405* | .0224* | .0670* | .0405 | .0224* | .0433 |
| Random + IRT | .0587 | .0390* | .0208* | .0568* | .0367* | .0191* | .0584* | .0374 | .0191* | .0385 |
| Random-strat + IRT | .0560 | .0352 | .0170 | .0542* | .0354* | .0171 | .0549 | .0352 | .0172 | .0358 |
| DISCO | .0581* | .0384* | .0217* | .0499 | .0340* | .0202* | .0438 | .0355 | .0225* | .0360 |
| AnchorPoints | .0805* | .0670* | .0649* | .0559 | .0387* | .0264* | .0554* | .0470* | .0243* | .0511 |
| Total-Fisher | .0545 | .0386* | .0223* | .0517 | .0381* | .0209* | .0522 | .0404 | .0233* | .0380 |
| Marginal-Fisher | .0566* | .0410* | .0220* | .0529* | .0401* | .0220* | .0568 | .0402 | .0246* | .0396 |
| tinyBenchmarks | .0533 | .0402* | .0333* | .0465 | .0333 | .0214* | .0514 | .0377 | .0186 | .0373 |
| metabench | .0606* | .0408* | .0215* | .0457 | .0386* | .0234* | .0477 | .0384 | .0226* | .0377 |
| Fluid | .0515 | .0428* | .0256* | .0496 | .0367* | .0206* | .0443 | .0355 | .0217* | .0365 |
| **SC-IRT** | **.0464** | **.0287** | **.0156** | **.0417** | **.0261** | **.0135** | .0449 | .0357 | **.0142** | **.0296** |

Reading. SC-IRT has the lowest error in 7 of 9 cells and ties in the other
two (K16 B30: DISCO .0438, Fluid .0443; K16 B55: Random-strat .0352 —
all inside the intervals), with the best macro average by a margin
(.0296 vs .0358 for the best baseline, type-stratified Random). The gap is
largest where evaluation is hardest — a small calibration panel at a
medium budget (K7 B55: .0287 vs .0352; K10 B55: .0261 vs .0333) — and it
survives the rich-panel regime that the independent-item model of earlier
versions lost: with the planner x type testlet the acquisition spreads
over scenario types by itself. At B = 110 (61% of the per-draw bank) SC-IRT
is still best at every K_cal (.0135-.0156 vs .0170-.0186).

## Table 2 — risk-target stopping (`run_tau_calibration.py`, `run_adaptive.py`)

Each order stops at the first t with c * R1_t <= eps, where R1 is the
posterior L1 risk of the reported SR under the common readout and c is
that order's calibration-fixed risk scale (leave-one-planner-out on the
calibration panel, 90th percentile of realised / predicted error; never
selected on evaluation planners). eps is an *error* target. Columns: mean
rollouts spent (110 = the bank cap), SR-MAE at the stop, coverage
P(|SR_hat - SR| <= eps), and the calibration gap (mean realised error minus
mean c * R1 at the stop; negative = conservative). d = paired delta of the
SR-MAE vs SC-IRT.

| K_cal | eps | method | rollouts | SR-MAE | coverage | gap | d vs SC-IRT |
|---|---|---|---|---|---|---|---|
| 7 | .05 | **SC-IRT** | **65.4** | .0258 | .84 | -.024 | — |
| | | Fluid | 73.2 | .0323 | .80 | -.017 | +.0064* |
| | | metabench | 94.5 | .0278 | .85 | -.024 | +.0020 |
| | | Random | 82.5 | .0253 | .85 | -.025 | -.0005 |
| 7 | .03 | **SC-IRT** | **99.3** | **.0182** | **.86** | -.013 | — |
| | | Fluid | 102.0 | .0245 | .66 | -.009 | +.0063* |
| | | metabench | 109.5 | .0210 | .73 | -.024 | +.0028 |
| | | Random | 108.9 | .0209 | .74 | -.017 | +.0027 |
| 10 | .05 | **SC-IRT** | **66.9** | .0253 | .88 | -.024 | — |
| | | Fluid | 80.6 | .0268 | .85 | -.023 | +.0014 |
| | | metabench | 91.9 | .0279 | .84 | -.024 | +.0026 |
| | | Random | 85.3 | .0230 | .92 | -.027 | -.0023 |
| 10 | .03 | **SC-IRT** | **101.0** | **.0164** | **.86** | -.015 | — |
| | | Fluid | 106.4 | .0204 | .76 | -.016 | +.0039 |
| | | metabench | 109.4 | .0235 | .68 | -.020 | +.0071 |
| | | Random | 109.5 | .0197 | .78 | -.019 | +.0033 |
| 16 | .05 | **SC-IRT** | **67.6** | .0297 | .81 | -.020 | — |
| | | Fluid | 82.2 | .0290 | .78 | -.021 | -.0007 |
| | | metabench | 96.7 | .0282 | .86 | -.023 | -.0015 |
| | | Random | 82.5 | .0245 | .91 | -.025 | -.0052 |
| 16 | .03 | **SC-IRT** | **101.4** | **.0164** | **.88** | -.015 | — |
| | | Fluid | 106.7 | .0212 | .74 | -.015 | +.0048 |
| | | metabench | 110.0 | .0254 | .67 | -.020 | +.0090* |
| | | Random | 109.6 | .0194 | .77 | -.018 | +.0030 |

Reading. For an error target of .05, SC-IRT stops after 65-68 rollouts
(36-38% of the per-draw bank, 30% of the 220-route benchmark) with 81-88%
coverage; the same target costs Fluid 73-82, Random 83-85 and metabench
92-97 rollouts. For .03 every order needs most of the bank; SC-IRT reaches
it earliest (99-101 rollouts) with the lowest error (.016-.018) and the
highest coverage (.86-.88 vs .66-.78). The negative gaps say the scaled
risk is conservative in the mean; the 90th-percentile scale, fixed on the
calibration panel, transfers to the evaluation planners as 81-88% coverage
rather than the nominal 90%, because their error / risk ratio has a
heavier tail. The raw R1 itself tracks the realised error: pooled over the
LOO tracks, the decile means of raw R1 and of |SR_hat - SR| agree within
.0045 at K_cal = 7 and 16 and within .0075 at K_cal = 10 (mean absolute
decile gap .0018 / .0027 / .0015; e.g. K_cal = 7: .0137 / .0144 in the
lowest decile, .0681 / .0711 in the highest). Random with its own calibrated scale is a fair competitor at
eps = .05 and slightly ahead at K16 (-.0052, ns) at 15 more rollouts.

**Matched cost (appendix).** The earlier rule — tau_hat chosen so that the
LOO mean rollouts hit 30 / 55 — is kept as the cost-matched comparison;
IES = (SR-MAE / Random-at-fixed-55) x (rollouts / 55):

| cell | SC-IRT | Fluid | metabench | Random |
|---|---|---|---|---|
| K7 target 30 | .0458 (29.2, IES .63) | .0486 (+.0028) | .0575 (+.0117*) | .0591 (+.0133*) |
| K7 target 55 | .0297 (54.1, .76) | .0351 (+.0053) | .0423 (+.0126*) | .0373 (+.0076*) |
| K10 target 30 | .0435 (29.9, .65) | .0449 (+.0014) | .0451 (+.0016) | .0567 (+.0132) |
| K10 target 55 | .0293 (54.8, .80) | .0330 (+.0037) | .0338 (+.0045) | .0371 (+.0079*) |
| K16 target 30 | .0440 (29.9, .66) | .0455 (+.0015) | .0462 (+.0022) | .0589 (+.0148*) |
| K16 target 55 | .0340 (55.1, .94) | .0333 (-.0007) | .0344 (+.0004) | .0362 (+.0023) |

Realised budgets land on target without post-hoc adjustment (29.2-30.7,
54.1-55.9). SC-IRT is best or tied-best in every cell; the K16 inversion of
earlier versions is gone.

## Component ablation (`run_ablation.py`)

Three components sit on top of a plug-in Rasch: the exact difficulty
posterior (off = point curves at b_hat), the planner x type testlet (off =
sigma_g = 0) and the risk acquisition (off = a random order under the same
posterior). Each is switched off alone; paired deltas vs full:

| cell | full | w/o b-uncertainty | w/o testlet | w/o risk acquisition |
|---|---|---|---|---|
| K7 B30 | .0464 | .0454 (-.0010) | .0449 (-.0015) | .0591 (+.0128) |
| K7 B55 | .0287 | .0273 (-.0014) | .0344 (+.0057) | .0384 (+.0097*) |
| K10 B30 | .0417 | .0423 (+.0007) | .0440 (+.0023) | .0581 (+.0164*) |
| K10 B55 | .0261 | .0286 (+.0025) | .0357 (+.0096*) | .0363 (+.0102*) |
| K16 B30 | .0449 | .0457 (+.0008) | .0511 (+.0062) | .0597 (+.0148*) |
| K16 B55 | .0357 | .0312 (-.0045) | .0396 (+.0039) | .0364 (+.0007) |

Reading. The risk acquisition carries the fixed-budget error (+.010 to
+.016 when removed in five of six cells, significant in four; K16 B55 is the
saturated exception at +.0007). The testlet is what
makes the acquisition pay at the medium budget: without it the K10 B55
error rises from .0261 to .0357 (+.0096*) and every B = 55 / K16 cell
degrades — the independent-item model re-samples scenario types it has
already seen. The difficulty posterior is neutral for the point estimate
at fixed budgets (all cells inside the intervals; K16 B55 even favours the
point curves by -.0045, ns): its role is the risk — it is what makes R1
track the realised error in Table 2 and what gives all-pass / all-fail
routes a one-sided difficulty instead of a symmetric one.

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
scene-MAE for the hand-crafted stacks, 5-9 for the encoder); between them the encoder is tied on AUROC / MAE and behind
on rank correlation (RelGraph minus hand-crafted risk: Delta rho
-.052 +- .024 across runs, i.e. about two run-SDs). The encoder buys no
difficulty signal beyond well-chosen rollout descriptors on this bank; what
it offers is the same signal from the raw scene graph without feature
engineering. The oracle (.862) is the ceiling of any scene-only predictor:
roughly half the difficulty variance is not visible from the scene.
(Earlier versions reported a descriptor stack that included the
scenario-definition parameters; it was removed because those values are
the benchmark's own construction parameters, not observable scene content.)

## Table 3B — UPS: unseen planner x unseen scenes (`run_ups.py`)

Predict an unseen planner's behaviour on unseen scenario types with zero
rollouts on the target block: probe the planner on B calibration-type
routes, transport the ability posterior through the RelGraph difficulty
prior N(b_tilde_s, sigma^2) with the testlet prior on the (unobserved)
evaluation types. The MAE scores the posterior median of the block-D
success rate, the NLL scores the per-cell posterior predictive. 96
evaluations, RelGraph run s0 (across the three encoder runs the MAE cells
move by .005-.008, SD). The canonical probe rule is Delta-R1 on the
block-D success rate — the UP acquisition with its risk evaluated on the
target block; theta-EIG and the 2PL Fisher rule are ablations.

| probe policy | B30 MAE | NLL | B55 MAE | NLL | B110 MAE | NLL |
|---|---|---|---|---|---|---|
| naive (no IRT) | .1290 | .6445 | .1189 | .6348 | .1146 | .6302 |
| Random | .1085 | .6005 | .0983 | .5934 | .0934 | .5909 |
| theta-EIG (abl.) | .1111 | .6020 | .0991 | .5942 | .0927 | .5893 |
| 2PL Fisher (abl.) | .0999 | .5946 | .0928 | .5907 | .0873 | .5875 |
| **SC-IRT (Delta-R1 on D)** | .1044 | .5981 | .0996 | .5947 | .0931 | .5901 |

Reading. Any transport beats the naive planner-mean by .02-.03 MAE; that
transport, not the probe placement, is the UPS result — the bottleneck is
the scene prior (rho about .5). Among probe rules the target-aligned
Delta-R1 beats theta-EIG at the smallest budget (+.0066*) and ties Random
everywhere; the 2PL Fisher rule is .005-.007 better (significant at B110,
-.0058*): with 16 calibration planners the item discriminations it uses are
real information for placing probes. It stays an ablation because it needs
a second model; the paper's claim is the transport, not the probe.

## Table 4 — the two-stage panel (`run_navhard.py`)

The same UP protocol on a panel with a different simulator, metric and
evaluation stage: the NAVSIM navhard leaderboard (two-stage
pseudo-closed-loop EPDMS), 87 unique submissions x 225 scored units,
pass = EPDMS >= 0.5. Per draw, 6 evaluation planners; K_cal subsampled
from the remaining 81 ({7, 10, 16} as on Bench2Drive, plus the full 81);
budgets are unit counts and there is no scenario-type structure (sigma_g =
0). SR-MAE; the strongest competitor per cell in parentheses:

| K_cal | B=30 | B=55 | B=110 |
|---|---|---|---|
| 7 | .0478 (metabench .0466) | .0342 (Random+IRT .0351) | .0239 (Random+IRT .0236) |
| 10 | .0499 (Random+IRT .0473) | .0354 (Random+IRT .0342) | **.0211** (Random+IRT .0225) |
| 16 | **.0450** (Random+IRT .0495) | **.0332** (Random+IRT .0334) | **.0185** (tinyB .0218) |
| 81 | .0451 (Fluid .0304*) | .0374 (DISCO .0293*) | .0172 (Fluid .0150) |

Reading. With the calibration panels the Bench2Drive protocol uses
(K_cal <= 16) SC-IRT is best or tied-best in every cell of a panel it was
never tuned on, and most published orderings are significantly worse
(K7 B55: tinyBenchmarks .0423*, Marginal-Fisher .0486*, Total-Fisher
.0521*; Fluid .0402 and metabench .0369 are ties). The
picture inverts only with the full 81-planner calibration panel at the low
budgets, where 2PL discrimination is precisely estimated and Fluid /
Fisher orderings win by a margin (.0304* vs .0451 at B30) — a regime that
does not exist on Bench2Drive (22 planners in total) and that the 1PL
evaluation model does not try to exploit. Caveats (REPRODUCIBILITY.md):
the leaderboard is dominated by a few teams' submission sweeps and
near-duplicates can straddle the planner split; this affects all methods
identically but limits reading the panel as 87 independent planners.

## Readout drop-in (`run_readout_dropin.py`)

Analysis, not a headline: every baseline's selected subset re-scored with
the SC-IRT readout (exact posteriors, testlet, posterior median). SR-MAE:

| selector (subset) | K7 B30 | K7 B55 | K7 B110 | K16 B30 | K16 B55 | K16 B110 |
|---|---|---|---|---|---|---|
| Fluid | .0493 | .0387 | .0212 | .0445 | .0345 | .0197 |
| Total-Fisher | .0553 | .0428 | .0235 | .0485 | .0430 | .0286 |
| metabench | .0591 | .0395 | .0209 | .0449 | .0366 | .0254 |
| tinyBenchmarks | .0565 | .0397 | .0335 | .0503 | .0389 | .0185 |
| AnchorPoints | .0536 | .0369 | .0343 | .0462 | .0387 | .0189 |
| Random | .0591 | .0384 | .0209 | .0597 | .0364 | .0193 |
| SC-IRT (own order, Table 1) | .0464 | .0287 | .0156 | .0449 | .0357 | .0142 |

Reading. Under one readout the remaining differences are selection: the
SC-IRT order still beats every re-scored subset at K7 by .003-.019 and at
K16 B110 by .004-.014, while at K16 B30 / B55 the Fluid and metabench
subsets tie it. The readout itself is a drop-in improvement for
information-based selectors whose native readout is a 2PL plug-in on a
scarce panel (compare the native Table 1 rows: AnchorPoints .0805 -> .0536,
Total-Fisher .0545 -> .0553 at K7 B30).

## Inside a real evaluation (`tools/b2d_adaptive_eval.py`)

The same posterior drives an actual Bench2Drive run: the bank is the full
22 x 220 panel (excluding the planner under test if it is in it), the
risk scale c is fixed by leave-one-planner-out on that bank (c = 2.08 for
the 22-planner bank; 2.35 / 2.15 / 2.03 for the 21-planner banks of the
three dry-runs below), and the driver picks one route at a time, runs it through the leaderboard evaluator, reads the outcome and
stops at c * R1 <= eps. Simulated from the matrix (`--dry-run`, planner held out of the bank,
eps = .03, cap 110 routes):

| planner | true SR | routes run | stopped by | SR_hat | abs err | types covered |
|---|---|---|---|---|---|---|
| VAD | .155 | 102 | risk (c * R1 = .0298) | .168 | .013 | 38 / 44 |
| HiP-AD | .664 | 110 | cap (c * R1 = .036) | .663 | .0005 | 42 / 44 |
| LEAD-tfv6 | .777 | 110 | cap (c * R1 = .030) | .773 | .004 | 41 / 44 |

Half the benchmark, errors of .0005-.013 SR, and a risk that says when it
is safe to stop (the true SR is taken over the planner's recorded routes,
213-220 of 220; SR_hat is over all 220). The weakest planner is the hardest case: an all-fail
record only bounds theta from above, so its estimate leans on the prior
early (SR_hat .22 after 40 routes) and settles as the acquisition finds
the routes it can pass.
