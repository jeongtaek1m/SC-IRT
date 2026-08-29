# Results — the numbers of record

One section per paper table; every number is produced by the named script,
which asserts it at the end of its run (`anchors OK`). Planner-side
results are means over 48 planner evaluations (16 draws x 3 held-out
planners); paired 95% intervals are cluster bootstraps over the 16 draws.
Differences below about .005 SR-MAE are inside those intervals.

## Table 1 — UP fixed-budget accuracy — `run_up_frontier.py`

Primary protocol: J_cal in {7, 10, 13} calibration planners x B in
{30, 60} rollouts, SR-MAE (lower is better), macro = mean of the six
cells. Baselines use their native readouts; SC-IRT the Rasch MAP-fill
readout. `*` = the paired 95% interval against SC-IRT excludes zero.

| method | J7·B30 | J7·B60 | J10·B30 | J10·B60 | J13·B30 | J13·B60 | macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random (IRT-free) | .0624* | .0385 | .0624 | .0385* | .0624* | .0385* | .0505 |
| Random + IRT | .0540 | .0331 | .0599 | .0352 | .0575* | .0345* | .0457 |
| Random-strat + IRT | .0574* | .0344 | .0576 | .0341* | .0574* | .0348* | .0460 |
| DISCO-adapted | .0744* | .0521* | .0596 | .0445* | .0472 | .0372* | .0525 |
| AnchorPoints-adapted | .0785* | .0695* | .0527 | .0403* | .0603* | .0296* | .0552 |
| Total-Fisher static | .0717* | .0522* | .0598* | .0400* | .0470* | .0317* | .0504 |
| Marginal-Fisher static | .0605* | .0580* | .0567 | .0379* | .0495* | .0346* | .0495 |
| tinyBenchmarks | .0724* | .0493* | .0485 | .0331* | .0516 | .0373* | .0487 |
| metabench-lite | .0480 | .0377 | .0490 | .0318 | .0464 | .0321* | .0408 |
| Fluid-style | .0576 | .0359 | .0514 | .0374* | **.0379** | .0390* | .0432 |
| **SC-IRT** | **.0423** | .0345 | .0494 | **.0260** | .0388 | **.0213** | **.0354** |

Reading. Macro .0354 against metabench .0408 (-13%) and Fluid .0432. Three
cells are the lowest point estimate (J7·B30, J10·B60, J13·B60) and three
are ties within +5% of the best baseline (J7·B60 vs Random+IRT .0331,
J10·B30 vs tinyBenchmarks .0485, J13·B30 vs Fluid .0379). At J13·B60
SC-IRT improves significantly over every reported baseline; at J10·B60 it
has the lowest point estimate and improves significantly over several, but
not all (metabench .0318 and Random+IRT .0352 are inside the interval).
The first K = 20 picks of SC-IRT are the Fluid selection rule; at B <= 20
the two rows differ only through their readouts.

### Full budget grids (SR-MAE, B in {10, ..., 120})

| J_cal = 13 | 10 | 20 | 30 | 40 | 60 | 80 | 100 | 120 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random (IRT-free) | .0916 | .0752 | .0624 | .0500 | .0385 | .0266 | .0249 | .0184 |
| Random + IRT | .0902 | .0676 | .0575 | .0466 | .0345 | .0264 | .0217 | .0154 |
| AnchorPoints-adapted | .0778 | .0721 | .0603 | .0508 | .0296 | .0231 | .0212 | .0159 |
| Total-Fisher static | .0831 | .0627 | .0470 | .0411 | .0317 | .0306 | .0258 | .0201 |
| tinyBenchmarks | .1047 | .0683 | .0516 | .0432 | .0373 | .0275 | .0197 | .0155 |
| metabench-lite | .0975 | .0602 | .0464 | .0387 | .0321 | **.0217** | **.0152** | **.0136** |
| Fluid-style | **.0601** | .0480 | **.0379** | .0427 | .0390 | .0288 | .0218 | .0182 |
| **SC-IRT** | .0617 | **.0479** | .0388 | **.0342** | **.0213** | .0222 | .0205 | .0168 |

| J_cal = 7 | 10 | 20 | 30 | 40 | 60 | 80 | 100 | 120 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random (IRT-free) | .0916 | .0752 | .0624 | .0500 | .0385 | .0266 | .0249 | .0184 |
| Random + IRT | .0896 | .0629 | .0540 | .0435 | **.0331** | .0264 | .0210 | **.0151** |
| AnchorPoints-adapted | .1263 | .0922 | .0785 | .0722 | .0695 | .0697 | .0697 | .0697 |
| tinyBenchmarks | .0905 | .0779 | .0724 | .0595 | .0493 | .0504 | .0504 | .0504 |
| metabench-lite | .1238 | .0605 | .0480 | .0414 | .0377 | .0264 | **.0178** | .0179 |
| Fluid-style | .0800 | .0570 | .0576 | .0666 | .0359 | .0298 | .0268 | .0213 |
| **SC-IRT** | **.0795** | **.0536** | **.0423** | **.0383** | .0345 | **.0255** | .0213 | .0209 |

(J_cal = 10 and the DISCO / Marginal-Fisher / Random-strat rows are printed
by `run_up_frontier.py`; the J_cal = 4 grid is shown as the map below.)
At B >= 100 representative static subsets (metabench, tinyBenchmarks) win at
every J_cal, and random sampling does too at J_cal <= 10 — adaptive
selection buys its accuracy at 20-80 rollouts.

### The J_cal x B map — `make_figures.py` (`figs/fig_jb_map`)

SC-IRT minus the best baseline, SR-MAE. With J_cal = 4 the calibrated bank
is too wrong for model-based extrapolation: random sampling (design-
unbiased) is the floor from B ~ 40 on (J4: SC-IRT .0699/.0522 at B = 30/60
vs Random+IRT .0580/.0367). Under a J_cal = 4 bank *every* adaptive
selector loses to random: the executed set of a greedy rule is
non-representative by design and its fill is model-dependent, so a bank
that is too wrong cannot repay that non-representativeness. That is when
not to use adaptive evaluation.

## Table 2 — Adaptive evaluation under a common stopping machine — `run_tau_calibration.py`, `run_adaptive.py`

Every bank order runs with SC-IRT's readout and the posterior-risk stop
R1(D_t) <= tau. Thresholds are fixed on the calibration panel by
leave-one-planner-out simulation to a *target* mean budget (30 / 60) and
never selected on held-out planners; the table reports the achieved mean
rollouts. IES = (SR-MAE / SR-MAE of Random at fixed B = 60) x (rollouts /
60). `*` = paired 95% interval vs SC-IRT excludes zero.

| J_cal | target | method | tau_hat (median) | rollouts | SR-MAE | IES |
|---|---|---|---|---:|---:|---:|
| 7 | 30 | Random | .048 | 29.5 | .0586* | 0.87 |
| 7 | 30 | Fluid | .038 | 29.7 | .0410 | 0.61 |
| 7 | 30 | metabench | .048 | 30.1 | .0445 | 0.67 |
| 7 | 30 | **SC-IRT** | .040 | 29.2 | .0433 | 0.63 |
| 7 | 60 | Random | .032 | 59.3 | .0321 | 0.95 |
| 7 | 60 | Fluid | .023 | 58.1 | .0367 | 1.07 |
| 7 | 60 | metabench | .031 | 58.7 | .0338 | 0.99 |
| 7 | 60 | **SC-IRT** | .027 | 60.2 | **.0306** | **0.92** |
| 10 | 30 | Random | .049 | 29.8 | .0610* | 0.86 |
| 10 | 30 | Fluid | .040 | 29.0 | .0455 | 0.62 |
| 10 | 30 | metabench | .048 | 30.1 | .0437 | 0.62 |
| 10 | 30 | **SC-IRT** | .042 | 29.0 | .0448 | **0.61** |
| 10 | 60 | Random | .032 | 59.8 | .0316 | 0.89 |
| 10 | 60 | Fluid | .024 | 58.9 | .0347 | 0.96 |
| 10 | 60 | metabench | .031 | 59.8 | .0282 | 0.79 |
| 10 | 60 | **SC-IRT** | .028 | 57.9 | **.0265** | **0.72** |
| 13 | 30 | Random | .049 | 29.9 | .0589* | 0.85 |
| 13 | 30 | Fluid | .041 | 28.7 | .0399 | 0.55 |
| 13 | 30 | metabench | .050 | 29.4 | .0461 | 0.65 |
| 13 | 30 | **SC-IRT** | .042 | 29.1 | **.0384** | **0.54** |
| 13 | 60 | Random | .032 | 58.9 | .0316* | 0.89 |
| 13 | 60 | Fluid | .024 | 58.6 | .0358* | 1.01 |
| 13 | 60 | metabench | .032 | 59.2 | .0307* | 0.87 |
| 13 | 60 | **SC-IRT** | .027 | 59.1 | **.0228** | **0.65** |

At target 60 SC-IRT has the lowest error and IES at every J_cal, and at
J_cal = 13 significantly so against all three orders; at target 30 SC-IRT
and Fluid are within .002 of each other (they share the first 20 picks),
metabench is within .001-.008, and only Random is separated. The
adaptive stop also beats SC-IRT's own fixed-budget curve at matched mean
rollouts in the 30-50 range (J13: -.0023 to -.0037 for tau in [.03, .04],
printed by `run_adaptive.py --merge`)
and the risk is honest there (MAE / tau 0.8-1.0; under a J_cal = 4 bank
it is over-confident, 1.7-2.2, as the bank error is not in the posterior).
The semantic tau sweep and design B (SC-IRT's tau applied to every order)
are printed by `run_adaptive.py --merge`; the cost-error curves are
`figs/fig_cost_error`.

## Table 3 — Generalisation

### Table 3A — Scene difficulty prediction (US; pooled 640) — `run_us.py`

| arm | AUROC | Scene-MAE | rel. | rho_scene |
|---|---|---|---|---|
| Planner-only null | 0.710 | 0.207 | — | 0 |
| Min-TTC | 0.700 | 0.216 | -4.2% | -0.092 |
| Risk field | 0.710 | 0.214 | -3.2% | +0.069 |
| Route geometry | 0.714 | 0.209 | -0.8% | +0.121 |
| Agent density + kin. | 0.722 | 0.214 | -3.4% | +0.182 |
| Traffic entropy | 0.716 | 0.210 | -1.3% | +0.090 |
| Agent-JEPA | 0.702 | 0.214 | -3.5% | -0.057 |
| SC-IRT stack, LLTM+e (canonical) | **0.764** | **0.177** | **+14.4%** | **+0.510** |
| Encoder (single run d64, 3-seed) | 0.753 +-.002 | 0.189 +-.001 | +8.8% | +0.469 +-.011 |
| Oracle (in-sample ceiling) | 0.876 | ~0 | — | +0.994 |

Relative reductions use unrounded MAEs. LLTM+e vs two-stage Ridge:
delta rho +0.023 CI[-0.002, +0.049] (adoption rationale is canonicity, not
significance). sigma-hat = 0.593 +- 0.118; plausible-values share of the
13-rater calibration noise in US rho uncertainty: 16.4%.

### Table 3A(b) — US descriptor ablation — `run_us.py`

Kinematics only +0.428 / **hand-crafted (ck+gtr) +0.486** /
encoder d64 +0.469 +- 0.011 / **d96 +0.490 +- 0.011** (per-run delta vs
hand-crafted: all CIs cover zero — statistical tie; both column bests bolded).

### Table 3B — UPS: unseen planner x unseen scenes — `run_ups.py`

30 probes on the calibrated bank, zero rollouts on the unseen-type routes
D; ability = MAP of the Rasch posterior on the probes; D success predicted
through the feature path with the difficulty prior marginalised
(ridge b_tilde + residual tau). 48 evaluations.

| probe policy (30 rollouts on B) | D-SR MAE | D NLL |
|---|---:|---:|
| naive SR transfer (probe mean) | .1282 | .7235 |
| Random probes | .1194 | .6251 |
| **theta-EIG under the evaluation model (canonical)** | **.1034** | **.6095** |
| Localize (2PL Fisher, the UP rule) | .1083 | .6126 |

Paired vs theta-EIG: Random +.0160 [-.0010, +.0332]; Localize +.0049
[-.0151, +.0241]. The UP localize rule brings no additional gain for
transport — the quantity that must generalise here is the evaluation-scale
ability, and the probe rule is chosen by alignment and simplicity, not by
a statistical win.

## Analysis — the Rasch readout as a drop-in for any selector — `run_readout_dropin.py`

Selector subsets re-scored with SC-IRT's readout; native readout in
parentheses. SR-MAE at B = 30 / 60.

| selector | J7 B30 | J7 B60 | J13 B30 | J13 B60 |
|---|---:|---:|---:|---:|
| Fluid-style | .0421 (.0576) | .0333 (.0359) | .0387 (.0379) | .0394 (.0390) |
| Total-Fisher static | .0623 (.0717) | .0420 (.0522) | .0418 (.0470) | **.0266** (.0317) |
| metabench-lite | .0459 (.0480) | .0338 (.0377) | .0452 (.0464) | .0307 (.0321) |
| AnchorPoints-adapted | .0699 (.0785) | .0551 (.0695) | .0512 (.0603) | .0294 (.0296) |
| tinyBenchmarks | .0763 (.0724) | .0534 (.0493) | .0560 (.0516) | .0385 (.0373) |
| Random | .0584 (.0540) | .0333 (.0331) | .0575 (.0575) | .0347 (.0345) |

The inference layer helps every information-based selector, most under
scarcity (Fluid at J7·B30: .0576 -> .0421), and it is neutral-to-worse for
the two K-means designs and for random subsets, whose native estimators
are already representative. SC-IRT's cover phase is the Total-Fisher order:
that row at J13·B60 (.0266 with the readout) is what the cover phase alone
delivers before the localize phase is added (.0213).

## Analysis — the localize budget K is a bank constant — `run_k_calibration.py`

Leave-one-planner-out simulation on the calibration panel, SR-MAE averaged
over B in {30, 40, 60, 80}:

| J_cal | K = 0 | 5 | 10 | 15 | 20 | 25 | 30 | 40 | 60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | .0467 | .0467 | .0452 | .0424 | .0410 | .0402 | **.0393** | .0399 | .0402 |
| 10 | .0381 | .0378 | .0378 | .0375 | .0369 | .0351 | **.0335** | .0351 | .0373 |
| 13 | .0325 | .0319 | .0318 | **.0312** | .0313 | .0318 | .0316 | .0352 | .0372 |

Flat over K in [15, 30] (range .0006 at J13) with a cliff at K >= 40
(never switching to cover); K = 20 is used at every J_cal. A per-bank
argmin is not used: with <= 13 planners the LOO curve is too flat for its
argmin to be a reliable estimate, so only the plateau is read from it.

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

## Table 4 — NavSim scale-up (US) (17 x 12,146 full OOF)

| seed | stack rho | enc rho | Delta [95% CI] | P(Delta>0) |
|---|---|---|---|---|
| 0 | +0.513 | +0.542 | **+0.029 [+0.003, +0.055]** | 0.986 |
| 1 | +0.513 | +0.533 | +0.020 [-0.005, +0.046] | 0.943 |
| 2 | +0.513 | +0.531 | +0.017 [-0.009, +0.043] | 0.903 |
| mean | +0.513 | +0.535 | +0.022 [-0.002, +0.047] | 0.962 |

Consistently positive across three independent runs, significant for one of
three seeds — no stronger claim. (The NavSim artifacts and runner are staged
for a follow-up commit; see REPRODUCIBILITY.md.)
