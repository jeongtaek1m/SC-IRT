# Results — the numbers of record

Produced by the scripts in `experiments/` on the unified split
(13/3 planners x 36/8 types, R = 16; see [PROTOCOL.md](PROTOCOL.md)).
Best value per column in **bold**, including baseline wins. Coverage always
carries its count. `+-` on rollouts is the SE of the mean stopping length
over the 48 evaluation runs.

## Table 1 — Scene difficulty prediction (US; pooled 640) — `run_us.py`

| arm | AUROC | Scene-MAE | rel. | rho_scene |
|---|---|---|---|---|
| Planner-only null | 0.710 | 0.207 | — | 0 |
| Min-TTC | 0.700 | 0.216 | -4.2% | -0.092 |
| Risk field | 0.710 | 0.214 | -3.2% | +0.069 |
| Route geometry | 0.714 | 0.209 | -0.8% | +0.121 |
| Agent density + kin. | 0.722 | 0.214 | -3.4% | +0.182 |
| Traffic entropy | 0.716 | 0.210 | -1.3% | +0.090 |
| Agent-JEPA | 0.702 | 0.214 | -3.5% | -0.057 |
| SC-IRT stack, LLTM+e (canonical) | **0.764** | **0.177** | **+14.5%** | **+0.510** |
| Encoder (single run d64, 3-seed) | 0.753 +-.002 | 0.189 +-.001 | +8.9% | +0.469 +-.011 |
| Oracle (in-sample ceiling) | 0.876 | ~0 | — | +0.994 |

Relative reductions use unrounded MAEs. LLTM+e vs two-stage Ridge:
delta rho +0.023 CI[-0.002, +0.049] (adoption rationale is canonicity, not
significance). sigma-hat = 0.593 +- 0.118; plausible-values share of the
13-rater calibration noise in US rho uncertainty: 16.4%.

## Table 2 — US feature ablation — `run_us.py`

Kinematics only +0.428 / **hand-crafted (ck+gtr) +0.486** /
encoder d64 +0.469 +- 0.011 / **d96 +0.490 +- 0.011** (per-run delta vs
hand-crafted: all CIs cover zero — statistical tie; both column bests bolded).

## Table 3 — UP main (all-1PL + SR-variance acquisition) — `run_up_main.py`

| eps | Rollouts | SR-MAE | Coverage |
|---|---|---|---|
| +-10% | 29.0 +-1.0 | .0463 | **1.00 (48/48)** |
| +-5% | 69.1 +-2.2 | .0294 | 0.83 (40/48) |

Acquisition ablation (theta-EIG, same machine): 28.7/.0533/0.90(43) and
69.0/.0289/0.85(41) — same cost, the SRVar alignment buys the +-10% MAE and
fixes coverage. Posterior-a variant (2PL bank, EIG era): 24.7/.0451/44 and
67.6/.0315/44. Selected subsets are planner-specific: cross-planner
Jaccard(S30) 0.14 (chance 0.09), overlap governed by ability proximity
(Spearman -0.77) — `run_sel_diversity.py`.

### Table 3(b) — the 2x2 factorial: align selection x stopping — `run_factorial_2x2.py`

Same 1PL + marginalisation + posterior in every cell; {theta-EIG, SRVar} x
{SE(theta) <= tau, SR-CI}; tau swept over {0.5..0.2}, matched point
tau = 0.4 ~ 29 rollouts (cells B and D reproduce Table 3 — anchors).

| | theta-SE stop (tau=0.4) | SR-CI stop (+-10%) |
|---|---|---|
| theta-EIG | A: 28.7 / .0557 / 0.88 / hw .100+-.019 | B: 28.7 / .0533 / 0.90 / .098+-.002 |
| SRVar | C: 28.9 / .0432 / 0.94 / .099+-.020 | D: 29.0 / .0463 / **1.00** / **.098+-.002** |

Paired effects (seed clusters): the acquisition effect holds in both
stopping worlds (C-A dMAE -0.0125 [-0.0212,-0.0049]; D-B -0.0070 with
dcov +0.104 [+0.042,+0.188]). The stopping effect is MAE-neutral but decides
the certificate: under one tau the delivered SR half-width ranges .046-.125
across planners (+-.019-.020) — the tau -> SR-precision map is
planner-dependent — while the SR-CI stop delivers the contract uniformly
(.092-.100, +-.002, a 10x tightening). The safe form of the stopping claim:
a fixed theta-SE threshold does not map uniformly to a fixed SR precision
across planners — the dispersion above proves it directly. Uncapped check
(--max-steps 999): the low coverage of the capped tau = 0.2 row was a
truncation artifact — without the cap it recovers to 0.96, but at 159.5
rollouts (~89% of the bank, delivered half-width +-0.9%: near-enumeration,
unrelated to any +-eps contract; this reproduces the native-3PL bridge's
"tau = 0.2 consumes 91% of the bank"). The tau = 0.3 dip (cov 0.79 at 56
rollouts) is truncation-free and intrinsic (tau = 0.25 is mildly cap-affected:
90 -> 93 rollouts uncapped, coverage unchanged). So the strong claim, correctly
stated: theta-SE has no useful operating point on this bank — dispersed
precision and sub-nominal coverage at comparable budgets, with coverage
recoverable only by near-enumeration. (And 48/48 with n = 48 may simply be
conservative relative to nominal 95% — say "more reliable and uniform", not
"valid".) Full alignment (D-A): dcov +0.125
[+0.042,+0.229]. Honest nuance: C has the lowest matched-budget MAE (.0432,
ns vs D) — the claim is not that alignment minimises MAE but that selection
alignment buys accuracy, stopping alignment buys a more reliable and
uniform SR-precision certificate, and delivering '+-eps as contracted'
needs both.

## Table 4 — Published baselines (UP) — `run_up_baselines.py`, `run_atlas_bridge.py`

**(main) one table, one Mode column.** Static methods run at declared
budgets B = 50 / 100 (not matched to anyone's realised rollouts); dynamic
methods run their native stopping criteria. "—" in Coverage is a statement
of fact — the method offers no precision contract.

| Method | Mode | Rollouts | SR-MAE | Coverage |
|---|---|---|---|---|
| Random (no model) | static, B=50/100 | 50 / 100 | .0439 / .0249 | — |
| Random + IRT | static, B=50/100 | 50 / 100 | .0391 / .0217 | — |
| tinyBenchmarks | static, B=50/100 | 50 / 100 | .0395 / .0197 | — |
| metabench-lite | static, B=50/100 | 50 / 100 | **.0347** / **.0152** | — |
| Fluid-style | static, B=50/100 | 50 / 100 | .0395 / .0218 | — |
| AnchorPoints-adapted | static, B=50/100 | 50 / 100 | .0372 / .0212 | — |
| DISCO-adapted | static, B=50/100 | 50 / 100 | .0408 / .0282 | — |
| Total-Fisher static | static, B=50/100 | 50 / 100 | .0350 / .0258 | — |
| Random, no IRT | dynamic, Wilson+FPC, SR±10/5% | 51.9 / 111.2 | .0444 / .0165 | 0.98 / 0.96 |
| ATLAS-style (3PL) | dynamic, SE(theta)<=.3/.2/.1 | 63.7 / 163.5 / 177.6 | .0355 / .0045 / — | 0.77 / 0.96 / 1.00 |
| **SC-IRT (ours)** | dynamic, SR±10/5% | **29.0** / **69.1** | .0463 / .0294 | **1.00** / 0.83 |

ATLAS tau = 0.2 consumes 91% of the bank and tau = 0.1 is reachable only by
exhausting it (its MAE cell is enumeration, not estimation). Reverse mapping
between the stopping scales: our ±10% stop ~ SE(theta) 0.405, ±5% ~ 0.278.
Details: static frontier in 4-S, certification (half-widths, ladder) in 4-D.

Selection-isolation panel (common stopping machine): informative rules tie
on rollouts (Fluid-Fisher 27.3 / ATLAS-sel 28.5 / ours 29.0 at +-10%);
the ~30% saving over Random (40.7 / 98.9) and static orders
(metabench-greedy 38.9 / 95.2) comes from the calibrated bank plus the
Bayesian SR-stop, which no published method provides. IES panel
(ref Random-100 = .0217): tier-29 best Fluid 0.50, ours 0.59 (fixed) /
0.62 (adaptive); tier-69 best metabench 0.91, ours 0.94.

### Table 4-D — Dynamic: certification (stopping-rule methods ONLY) — `run_random_fpc.py`, `run_up_main.py`, `run_atlas_bridge.py`

Columns: rollouts /
SR-MAE / **achieved SR half-width mean +- SD** / coverage. The IRT-free row
(uniform random + Wilson interval with finite-population correction) is the
reference for "is IRT needed at all".

| method | stop | +-10%: roll / MAE / half-width / cov | +-5%: roll / MAE / half-width / cov |
|---|---|---|---|
| Random, no IRT | Wilson+FPC width <= 2 eps | 51.9 +-1.3 / .0444 / .099 +-.001 / 0.98 (47) | 111.2 +-1.6 / .0165 / .050 +-.000 / 0.96 (46) |
| Random + IRT posterior | SR-CI | 40.7 +-0.9 / .0459 / .099 +-.002 / 0.96 (46) | 98.9 +-1.4 / .0213 / .049 +-.001 / 0.94 (45) |
| theta-EIG + IRT | SR-CI | 28.7 +-1.0 / .0533 / .098 +-.002 / 0.90 (43) | 69.0 +-2.1 / .0289 / .049 +-.001 / 0.85 (41) |
| ATLAS-style (3PL) | SE(theta)<=0.3, min30 | 63.7 +-5.3 / .0355 / .056 +-.023 [0, .081] / 0.77 (37) | (tau=0.2: 163.5, 91% of bank / .0045 / .000 / 0.96 (46)) |
| **SC-IRT (SRVar)** | SR-CI | **29.0 +-1.0** / .0463 / .098 +-.002 / **1.00 (48)** | **69.1 +-2.2** / .0294 / .049 +-.001 / 0.83 (40) |

The classical interval keeps the contract well (coverage 0.98 / 0.96): the
claim is not that IRT is required, but that IRT plus aligned selection
delivers the same +-eps certificate at 44% / 38% lower cost (51.9 -> 29.0,
111.2 -> 69.1) — the IRT posterior buys 21% / 11% (+-10% / +-5%), aligned
selection a further 29% / 30%. ATLAS's theta-scale stop has no precision contract (half-width
dispersion +-.023, range 0-.081). Fluid-style defines no stopping rule and
appears only on the frontier. Per-planner required-rollout distributions:
`figs/fig_rollout_distribution.pdf`.

### Table 4-S — Static: fixed-budget accuracy on one budget grid — `run_budget_frontier.py`

B in {10, 20, 29, 30, 40, 60, 69, 80, 100, 120}; native scoring per
method. Added baselines: DISCO-adapted (inter-planner disagreement
p(1-p) top-B + 2PL p-IRT; binary adaptation of arXiv:2510.07959),
AnchorPoints-adapted (K-means on item response vectors, cluster-weighted
anchor mean), Total/Marginal-Fisher static, Random-strat, Random (IRT-free
mean). Figure: `figs/fig_budget_frontier.pdf`. SR-MAE:

| B | 10 | 20 | 29 | 40 | 60 | 69 | 80 | 100 | 120 |
|---|---|---|---|---|---|---|---|---|---|
| Random (IRT-free) | .0916 | .0752 | .0634 | .0500 | .0385 | .0359 | .0266 | .0249 | .0184 |
| Random + IRT | .0902 | .0676 | .0584 | .0466 | .0345 | .0324 | .0264 | .0217 | .0154 |
| DISCO-adapted | .0949 | .0633 | .0453 | .0463 | .0372 | .0379 | .0332 | .0282 | .0225 |
| AnchorPoints-adapted | .0778 | .0721 | .0620 | .0508 | **.0296** | **.0248** | .0231 | .0212 | .0159 |
| Total-Fisher static | .0831 | .0627 | .0490 | .0411 | .0317 | .0292 | .0306 | .0258 | .0201 |
| tinyBenchmarks | .1047 | .0683 | .0477 | .0432 | .0373 | .0297 | .0275 | .0197 | .0155 |
| metabench-lite | .0975 | .0602 | .0507 | **.0387** | .0321 | .0285 | **.0217** | **.0152** | **.0136** |
| Fluid-style | **.0601** | **.0480** | **.0375** | .0427 | .0390 | .0357 | .0288 | .0218 | .0182 |
| ours-EIG | .0827 | .0678 | .0519 | .0458 | .0323 | .0291 | .0260 | .0203 | .0187 |
| ours-SRVar (fixed-B, no stopping) | .0780 | .0574 | .0443 | .0448 | .0360 | .0297 | .0259 | .0198 | .0174 |

(Marginal-Fisher and Random-strat rows in the json.) Honest reading: at
small budgets (<= 30) Fluid's discrimination-aware selection has the best
MAE, at large budgets (>= 40) representative static subsets (metabench,
AnchorPoints) do; SRVar is second for 20 <= B <= 30 (third at B = 10) and mid-pack later. Fixed-
budget MAE is not the design target: *SC-IRT is not designed to minimise
MAE at a fixed budget; it minimises evaluation cost subject to a requested
precision on the reported quantity.* Rank agreement (pooled Spearman over
the 48 evaluations): SRVar is best at B = 29 (0.960; Fluid .957, metabench
.899), and every method saturates >= 0.95 beyond B ~ 60 — the true-SR spread of
the 48 evaluations (.114-.786) makes rank metrics weakly discriminative
(appendix only).

### Appendix — psychometric adequacy — `run_model_adequacy.py`

Held-out cell NLL on the calibration block (10% of cells, 16 draws):
**1PL .5303 +-.0119 / 2PL .5292 +-.0120 / 3PL .5340 +-.0117**; 2PL - 1PL =
-0.0012 +-0.0008 (ns), 3PL - 1PL = +0.0037. Split-half reliability of
log a-hat across 6/7-planner halves: **+0.095 +-0.021**. Discrimination
neither predicts held-out responses better nor is reliably estimable on a
13-planner panel: the Rasch model is the adequate model here, not a
simplification (with the a-marginalisation identity, max cell-probability
change 0.0009, this is the PROTOCOL section 2 justification).

### Appendix — calibration stability: full 16 x 220 fit vs the per-draw A block — `run_calibration_stability.py`

On the shared routes, b-hat from the 13 x ~180 calibration block vs the
full-panel fit: Pearson 0.978 (min 0.955 over 16 draws), Spearman 0.976;
mean-aligned RMSE 0.355 logit against sd(b-hat) = 1.63 and a per-route
posterior SD s_i = 0.81 — only 1.5% of routes move by more than their own
s_i (max 3.9%). theta-hat on the 13 shared planners: Pearson 0.997,
aligned RMSE 0.095 against sd(theta-hat) = 1.17 (max single shift 0.32).
Raw offsets (b -0.02, theta -0.01 on average, up to +-0.3 in a draw) are the
theta-mean-centring convention applied to 16 vs 13 planners, not
estimation change. Dropping three planners and the held-out types
therefore moves the bank by well under its own calibration uncertainty —
the regime is stable, and the residual s_i is exactly what the marginalised
curves carry.

## Table 5 — CAT under calibration scarcity — `run_scarcity.py`

| J_cal | SC-IRT (1PL+SRVar) | Fluid-style (2PL) | ATLAS-style (3PL) | Random |
|---|---|---|---|---|
| 4 | 25.5 / .0630 / **0.79** | 16.2 / .0956 / 0.56 | 15.4 / .1012 / 0.58 | 38.8 / **.0624** / 0.75 |
| 7 | 27.0 / **.0510** / **0.96** | 18.3 / .0571 / 0.88 | 17.9 / .0661 / 0.75 | 39.3 / .0560 / 0.83 |
| 10 | 28.4 / **.0450** / **0.94** | 20.7 / .0553 / 0.88 | 20.3 / .0638 / 0.85 | 40.5 / .0462 / 0.92 |
| 13 | 29.0 / .0463 / **1.00** | 21.9 / **.0442** / 0.92 | 22.4 / .0540 / 0.85 | 40.7 / .0459 / 0.96 |

(rollouts / SR-MAE / coverage; +-10% target; native end-to-end, so the J=13
CAT rows differ from the common-machine panel above by design.) The shrinking
Fluid/ATLAS rollouts are premature stops — their certificates break
(J4: error ~ the +-10% target itself at ~0.56 coverage; paired deltas
significant). Mechanism: corr(log a-hat) across calibration replicates
0.36 / 0.56 / 0.80 at J = 4/7/10 vs corr(b-hat) 0.67 / 0.83 / 0.94;
Jaccard(S30) drops -50%/-61%/-34% (Fluid/ATLAS/ours). Ours matches Random's
accuracy at ~35% fewer rollouts at every panel size — and degrades too
(J4 coverage 0.79): the claim is graceful degradation, not immunity.

### Table 5(b) — the causal ablation: toggle only uncertainty propagation — `run_plugin_ablation.py`

Same 1PL, same SR-variance acquisition, same stopping; the only change is
plug-in curves sigmoid(theta - b_hat) vs marginalised curves
int sigmoid(theta - b) N(b; b_hat, s^2) db. Subsample streams shared with
Table 5 (paired); +-10% target.

| J_cal | plug-in (roll / MAE / cov) | marginalised | dcov [CI] | dMAE [CI] |
|---|---|---|---|---|
| 4 | 18.8 / .0869 / 0.62 | 25.5 / .0630 / 0.79 | +0.167 [+0.06,+0.29] | -0.0239 [-0.036,-0.010] |
| 7 | 22.8 / .0694 / 0.75 | 27.0 / .0510 / 0.96 | +0.208 [+0.10,+0.31] | -0.0184 [-0.025,-0.012] |
| 10 | 26.2 / .0589 / 0.79 | 28.4 / .0450 / 0.94 | +0.146 [+0.04,+0.27] | -0.0139 [-0.025,-0.005] |
| 13 | 27.2 / .0611 / 0.85 | 29.0 / .0463 / 1.00 | +0.146 [+0.06,+0.25] | -0.0148 [-0.026,-0.002] |

Propagating calibration uncertainty is significant at every panel size, and
the failure it prevents — premature, overconfident stopping — amplifies as
panels shrink (the plug-in stops 1.7 -> 6.8 rollouts earlier as J_cal drops).
It also decomposes Table 5: ~73% of the Fluid J=4 collapse is explained by
non-propagation (shared by any point-parameter model), ~27% by 2PL a-hat
noise amplification. The extra rollouts of the marginalised arm are the
honest price: it does not stop less — it stops when it should.

### Appendix — rejected: pairwise-CML (item-vs-item) bank estimator — `run_cml_calibration.py`

The Rasch model contains a pure difficulty-vs-difficulty sub-model
(P(passed k, failed i | exactly one) = sigmoid(b_i - b_k), theta cancels);
estimating b from those pairwise votes is the conditional estimator,
consistent in the number of items however few planners there are. Swapping
only the bank estimator (then theta by ML given b, same s_i, curves, SRVar
and stop) across J_cal in {4, 7, 10, 13}: worse at every panel size —
J7 dMAE +0.0141 [+0.0006, +0.0281], dcov -0.250 [-0.354, -0.146]; J13 dcov
-0.083 [-0.146, -0.021]; others ns; CML stops 1-2 rollouts earlier
(overconfident). b-hat orderings agree (corr >= 0.993) but CML disperses b
6-10% more (sd 2.84 vs 2.67 at J4): the joint MAP's priors are useful
regularisation with few raters, and CML's consistency needs denser pairwise
votes than 4-13 planners provide. Joint MAP remains canonical; the
within-group relativity survives as the conditional decomposition of Rasch
(specific objectivity), not as an estimator.

### Appendix — multidimensional (residual-interaction) IRT, dimension sweep — `run_mirt_dsweep.py`

logit = theta_j - mu_i + u_j^T v_i, d in {1, 2, 4, 8} x prior lam in
{.05, .1, .2, .4}, unified split, cell level (scalar reference: oracle
AUROC .8761, amortised .7604). Columns: in-bank sd(u.v) / Procrustes
split-half excess reliability of U (minus a row-shuffle null) / oracle
dAUROC / **amortised dAUROC** / Spearman(v~, v_oracle).

| d | lam | sd(u.v) | excess rel. (null) | oracle d | **amort d** | G2 |
|---|---|---|---|---|---|---|
| 1 | .05 | .252 | +.32 (.20) | +.023 | +.000 | +.04 |
| 1 | >= .1 | .000 (collapse) | ~0 | 0 | 0 | — |
| 2 | .05 | .850 | +.28 (.31) | +.055 | -.001 | +.08 |
| **2** | **.1 (canonical)** | .323 | +.29 (.31) | +.033 | +.001 | +.08 |
| 2 | >= .2 | .000 (collapse) | ~0 | 0 | 0 | — |
| 4 | .05 / .1 / .2 | 1.79 / 1.07 / .36 | +.14 / +.18 / +.23 (.44-.47) | +.098 / +.081 / +.033 | -.002 / +.002 / +.002 | +.08-.10 |
| 4 | .4 | .000 (collapse) | ~0 | 0 | 0 | — |
| 8 | .05 / .1 / .2 / .4 | 2.64 / 1.85 / 1.11 / .36 | +.12 / +.09 / +.08 / +.21 (.49-.65) | +.087 / +.119 / +.088 / +.021 | -.001 / +.002 / +.004 / +.003 | +.08-.09 |

Deployable gain is zero at every dimension (amortised dAUROC within
+-.004; x -> v predictability <= .10). The oracle column is an in-sample
fit of d+1 parameters per scene on <= 13 responses, so its growth with d
(to +.119 at d = 8) is a degrees-of-freedom artifact, not a ceiling — and
the split-half excess reliability of U falls with d (+.29 -> +.09) while
the Procrustes null rises (.31 -> .65). The prior cliff (collapse to zero
vs overfit within a factor-2 window) reproduces at every d. The scalar
backbone stays; the d = 2, lam = .1 anchors reproduce (.9094 / .7610).

## Table 6 — UPS — `run_ups.py`

- **(a) decomposition** (B=30, zero target rollouts, common Rasch scale):
  Err(thetaB, b~) .1034 / Err(thetaB, b^C) .1005 / Err(thetaD, b^C) .0350 —
  amortisation gap +0.003 vs **transport gap +0.066 (22x)**;
  |thetaB - thetaD| 0.62, SE(thetaD) 0.41.
- **(b) composition baselines**: naive SR transfer .1259 / random-B .1206 /
  **ours .1035** (+-.0124).
- **(c) hybrid** (B=30 warm + D probes): D = 0/5/10/20 ->
  .1036 / .0832 / .0654 / **.0470**, coverage 41/42/44/**47** of 48.
- **(d) standalone** (bank = ~40 unseen-type routes, SRVar):
  +-10% 21.0 +-0.4 (54% pool) / .0495 / 0.96 (46);
  +-5% 30.5 +-0.4 (78%) / .0254 / 0.94 (45).

## Table 7 — NavSim scale-up (17 x 12,146 full OOF)

| seed | stack rho | enc rho | Delta [95% CI] | P(Delta>0) |
|---|---|---|---|---|
| 0 | +0.513 | +0.542 | **+0.029 [+0.003, +0.055]** | 0.986 |
| 1 | +0.513 | +0.533 | +0.020 [-0.005, +0.046] | 0.943 |
| 2 | +0.513 | +0.531 | +0.017 [-0.009, +0.043] | 0.903 |
| mean | +0.513 | +0.535 | +0.022 [-0.002, +0.047] | 0.962 |

Consistently positive across three independent runs, significant for one of
three seeds — no stronger claim. (The NavSim artifacts and runner are staged
for a follow-up commit; see REPRODUCIBILITY.md.)
