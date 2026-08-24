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

## Table 4 — Published baselines (UP) — `run_up_baselines.py`, `run_atlas_bridge.py`

**(main) native-vs-native, each method in its published operating mode:**

| method | operating point | Rollouts | SR-MAE | Coverage |
|---|---|---|---|---|
| tinyBenchmarks | fixed K=29 / 69 | 29 / 69 | .0477 / .0297 | — |
| metabench-lite | fixed K=29 / 69 | 29 / 69 | .0507 / **.0285** | — |
| Fluid-style | fixed B=29 / 69 | 29 / 69 | **.0375** / .0357 | — |
| ATLAS-style | SE(theta)<=0.3, min30 | 63.7 +-5.3 | .0355 | 0.77 (37/48) |
| ATLAS-style | SE(theta)<=0.2 | 163.5 (91% of bank) | .0045 | 0.96 (46/48) |
| ATLAS-style | SE(theta)<=0.1 | 177.6 (pool exhausted) | — | 1.00 |
| **ours** | SR +-10% | **29.0 +-1.0** | .0463 | **1.00 (48/48)** |
| **ours** | SR +-5% | **69.1 +-2.2** | .0294 | 0.83 (40/48) |

Fixed-budget methods carry no precision certificate (—); ATLAS's
theta-scale certificate degenerates on a 180-item bank (tau=0.1 reachable
only by full enumeration); ours certifies the evaluand itself. Reverse
mapping: our +-10% stop corresponds to SE(theta) 0.405, +-5% to 0.278.

Selection-isolation panel (common stopping machine): informative rules tie
on rollouts (Fluid-Fisher 27.3 / ATLAS-sel 28.5 / ours 29.0 at +-10%);
the ~30% saving over Random (40.7 / 98.9) and static orders
(metabench-greedy 38.9 / 95.2) comes from the calibrated bank plus the
Bayesian SR-stop, which no published method provides. IES panel
(ref Random-100 = .0217): tier-29 best Fluid 0.50, ours 0.59 (fixed) /
0.62 (adaptive); tier-69 best metabench 0.91, ours 0.94.

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
