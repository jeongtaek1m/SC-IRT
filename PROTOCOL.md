# Protocol — the specification

Normative for the release: every experiment in `experiments/` implements
exactly this. Numbers of record: [RESULTS.md](RESULTS.md); anchors and RNG
registry: [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## 1. Data

16 open-source end-to-end planners x 220 Bench2Drive closed-loop routes;
y_sk = 1 iff planner k completes scene (route) s (`data/matrices/`, 3,482
of 3,520 cells observed). The 16 are chosen from the 22 planners with a
complete closed-loop record (`b2d_e2e22_response_matrix.csv`, kept for
provenance and for the live tool) so that the panel covers the ability
range evenly with one planner per model family: of the six dropped, four
are the second member of a family within .04 SR of the kept one
(SimLingo-IVL35-1B, MindDrive, Orion-Lite, UniAD-Base), one a same-SR
duplicate (DriveMoE-Base, .486 = PGS) and one a crowded-band newcomer
(R2SE). Success rates of the kept 16 run from .135 to .777. Each route carries one of 44 scenario types (5 routes per
type). Scene descriptors (`data/features/`) serve only as US baselines;
the scene encoder consumes the raw scene graph (Section 3.1). Route order
is the response-matrix CSV column order — part of the reproduction contract.

**What the UP model is given.** Exactly two things: the pass/fail response
matrix, and the benchmark's item grouping g(s) (the scenario type of each
route). The grouping enters only as a dependence structure (Section 3):
no difficulty, feature or parameter is read from the type label, the
prior on the type effect is the same zero-mean N(0, sigma_g^2) for every
type, and sigma_g is estimated on the calibration panel — when the grouping
carries no information it is fitted to 0 and the model reduces to the
independent-item model when no grouping is available. The grouping
is public metadata available to every method: the type-stratified Random
baseline uses it, and the component ablation reports DriveAT without it.
Scenario-definition parameters (trigger distances, flow speeds, ...) are
not used anywhere.

## 2. The split

12 calibration : 4 evaluation planners and 36 calibration : 8 evaluation
scenario types, drawn per Monte-Carlo draw (R = 16 draws,
RandomState(1000 + draw); `driveat/splits.py`). One draw partitions both axes
at once, so the three regimes share the planner split:

|                          | calibration planners (12) | evaluation planners (4) |
|--------------------------|---------------------------|-------------------------|
| calibration types (36)   | A: calibration block      | **UP** evaluation       |
| evaluation types (8)     | **US** evaluation         | **UPS** target          |

**UP holds out planners only** (`splits.up_split`): the bank of the new
planner is the whole benchmark — all 220 routes of all 44 scenario types —
and it is calibrated from the 12 calibration planners on those same 220
routes. That is the configuration a user of the tool is in: a panel of
published planners, a new planner to place on the full benchmark. The
36 : 8 type hold-out is kept for US and UPS, whose target block is the 40
routes of the 8 evaluation types.

Sample sizes: UP and UPS 16 x 4 = 64 planner evaluations per condition;
US pooled 16 x 40 = 640 route evaluations. Types are held out as whole
types.

**Calibration scarcity.** K_cal in {4, 8, 12}: at K_cal < 12 the
calibration planners are subsampled once per draw
(RandomState(9000 + 100 draw + 10 K_cal)) and *every* model — ours and each
baseline's own — is re-fitted from those planners only; the evaluation
planners are unchanged. K_cal = 12 is the full calibration pool.

**Budgets.** B = number of routes rolled out (the acquisition of Section 4
selects one route at a time); B = 5 x {6, 11, 22, 33} = {30, 55, 110, 165},
i.e. 14% / 25% / 50% / 75% of the 220-route benchmark. The stopping
experiments (Section 4) let every order run through the whole bank (all 220
routes), so a reported stop is the rule's own stop; the fraction of
evaluations that exhaust the bank is reported. (The grid was chosen as multiples of the 5 routes per scenario
type; the stratified baseline visits types round-robin, so no budget
corresponds to whole types for any method.)

## 3. The model — one uncertainty-aware Rasch posterior

```
y_sk ~ Bernoulli( sigmoid(theta_k - b_s + u_{k,g(s)}) ),   u_kg ~ N(0, sigma_g^2)
```

g(s) is the scenario type of route s and u_kg a planner x type testlet
effect: on this panel the Rasch residuals of two routes of the same type
correlate +.21 (different types: .00), so routes are not conditionally
independent given theta and a rule that assumes they are under-states its
own risk.

**Calibration** on the block A (`driveat/calibration.py`) maximises the
posterior of the Rasch model under explicit priors — the same prior forms
the evaluation posterior uses. Because both priors are proper and centred at
zero, they already identify the location of the latent scale: the curvature
along the joint shift (theta, b) -> (theta + t, b + t) is
K_cal / sigma_theta^2 + n / sigma_b^2 > 0, so no identification constraint is
needed or imposed.

`calibrate` nevertheless subtracts c = mean(theta_hat) from theta_hat and
b_hat before returning. That is a **change of origin, not identification**,
and what it returns is therefore a translation of the MAP rather than the
MAP itself: the objective is higher by c^2 (K_cal / sigma_theta^2 +
n / sigma_b^2) / 2, a median of 3.3 / 1.5 / 1.1 nats at K_cal = 4 / 8 / 12.
|c| reaches .95 (median .22) at K_cal = 4 and .33 (median .14) at
K_cal = 12. Everything downstream is then read in that centred frame with
its priors left at zero, so the origin of the reported scale is the
calibration panel's mean ability and the same convention applies to every
method scored against the bank. Two consequences are worth stating: the
posterior mean E[b | A] carries a bias of +lambda c with lambda in (0, 1)
(up to .47 logits at K_cal = 4, largest on the all-pass and all-fail routes),
while sigma_g and every predicted probability sigmoid(theta - b) are exactly
invariant to the shift. Removing the centring, or re-anchoring both priors at
-c, moves no cell of Table 1 by more than .003 SR-MAE and the macro average
by .0003 (below the .005 tie threshold of RESULTS.md).

The difficulty posterior below is taken in that centred frame:

```
theta_k ~ N(0, 1)          b_s ~ N(0, sigma_b^2)          (2PL/3PL baselines: log a_s ~ N(0, .5^2), logit c_s ~ N(-2.2, 1))
```

Adam lr .05, 800 iterations, zero init, theta-mean centring. sigma_b is
chosen by empirical Bayes on the grid {.5, .75, 1, 1.5, 2, 3} — the exact
marginal likelihood sum_s log int prod_k Bern(y_sk | theta_hat_k, b) N(b; 0, sigma_b^2) db
on the difficulty grid below (1.5 in 36 of the 48 (draw, K_cal) fits on
this panel and in all 16 at K_cal = 12; 1.0 in 9; 2.0 in 3, all three at
K_cal = 4). The testlet SD sigma_g is the profile
marginal likelihood at (theta_hat, b_hat) over {0, .25, .5, .75, 1, 1.25,
1.5, 2} with u integrated on the u grid (0-1.5 at K_cal = 4, .75-1.25 at
K_cal = 12; u itself is not a MAP parameter — the calibration block has one
u_kg per cell block and only its variance is identified).

**Difficulty posterior — exact, conditional on the fitted abilities.**
Given theta_hat the items are independent and one-dimensional, so every
difficulty keeps its full posterior on an 801-point grid over [-10, 10]:

```
p(b_s | A) ∝ N(b_s; 0, sigma_b^2) prod_{k:(s,k) in A} Bern(y_sk | sigmoid(theta_hat_k - b_s))
```

No Gaussian approximation: the all-pass / all-fail routes of a K_cal = 4
panel (32-79 of 220 in the first four draws; 10-20 at K_cal = 12) get their
one-sided posterior. Every downstream probability
marginalises it:

```
m_s(x) = E_{b_s | A}[ sigmoid(x - b_s) ]      tabulated on XG = [-9, 9], step .05 (`driveat/curves.py`)
```

so that m_s(theta + u) is an index lookup.

Conditioning on theta_hat rather than marginalising it is deliberate, and the
asymmetry with b is the data's, not a modelling choice: each calibration
planner is scored on ~217 of the 220 routes, so SE(theta_hat) = .17 (.19 with
b integrated out, .22 once the testlet is accounted for), while each route is
scored by only 4-12 planners, so SD(b | A) = .67-1.15. Marginalising the
abilities adds at most sigma_b^2 / (routes per calibration planner) = .010 to
each difficulty variance (measured +.008, i.e. +0.4-0.9% in SD), moves m_s(x)
by at most .0005 in probability and the reported success rate by 6e-5 per
evaluation (macro SR-MAE .0000, 95% CI +-.0009). About 87% of that widening is
a common level shift of the whole bank, which the evaluation planner's own
ability absorbs because it is estimated on those same routes. The bound
saturates in planners per route, so it would only reach .005 in probability on
a bank giving each calibration planner fewer than about 20 routes — a regime
probed by shrinking the bank to 14 routes, where the readout still does not
move.

The evaluation planner's ability is kept as a full posterior for a different
reason. Collapsing it to a point at readout costs .0002 macro SR-MAE, which is
nothing, and at a matched number of rollouts the two readouts stop within 2.4
routes of each other with the same error; but the point readout's risk is a
different quantity. Its raw R1 saturates at .031, so it cannot certify any
error target above about .075, and its realised-to-predicted error ratio moves
from 1.0 to 1.5, which the calibration scale c then has to absorb (2.0 -> 2.7
at K_cal = 12). Keeping the posterior is what makes R1 readable as an absolute
expected error before c is applied, and it is already computed.

**Evaluation posterior.** For a new planner with administered set 𝒜_t and
outcomes y, theta lives on a 241-point grid over [-6, 6] with the N(0, 1)
prior and each u_g on a 61-point grid over [-3, 3] with N(0, sigma_g^2).
Given theta the types factorise, so the posterior is one theta vector plus
one (theta x u) table per observed type (`driveat/bayes.State`):

```
q_t(theta) ∝ N(theta; 0, 1) prod_g l_g(theta),
l_g(theta) = sum_u p(u) prod_{s in 𝒜_t ∩ g} m_s(theta + u)^{y_s} (1 - m_s(theta + u))^{1 - y_s}
```

Types without observations keep the prior on u. sigma_g = 0 recovers the
independent-item model exactly (one grid point for u).

### 3.1 Unseen scenes — the difficulty prior from the scene

```
b_s | scene_s ~ N( b_tilde_s , sigma^2 )        (RelGraph R2 scene encoder)
```

The scene encoder is the RelGraph R2: ego, agent and lane tokens with R-GCN
lane-lane message passing, agent-lane cross-attention over relative-geometry
relations and an ego-route relation (d = 64). Per draw it is trained
end-to-end on the calibration types only, with the cell likelihood
marginalised over the residual b_s - b_tilde_s ~ N(0, sigma^2) by
Gauss-Hermite and sigma learned jointly (the encoder's shared residual SD,
~.65 on this panel); the release
ships its per-run out-of-fold predictions (`data/encoder/relgraph_r2_s*.npz`,
three independent runs, no ensembling; the structural controls of
Table 3A(b) — route relation removed, route or agent-lane correspondence
shuffled — are the same architecture and recipe with different graph
tensors, `data/encoder/relgraph_r2_{noroute,sroute,sa2l}_s*.npz`). Table 3A
scores these point predictions; on this panel the encoder is
tied with the hand-crafted descriptor baselines on AUROC / scene-MAE and
behind them on rank correlation (RelGraph minus cmdkin+gtrisk: Delta rho
-.052 +- .024 across runs, about two run-SDs).

## 4. UP — one posterior for inference, acquisition and stopping

The reported quantity is the full-bank success rate

```
SR = (1/S) [ sum_{s in 𝒜_t} y_s + T ],   T = sum_{s not in 𝒜_t} Y_s
```

Given theta the unobserved total T is approximated by a normal whose mean
is sum_s p_s(theta), p_s(theta) = E_{u_g | D_t}[m_s(theta + u_g)], and whose
variance sums per type E_u[Var(T_g | u)] + Var_u[E(T_g | u)] — routes of
one type share their u_g, so the variance is not the independent
sum_s p_s(1 - p_s). Mixing over q_t gives the posterior of T
(`driveat/bayes.State.stats`). The evaluation loss is L1, so one action
serves everywhere:

```
readout      SR_hat_t = (1/S) [ sum_{s in 𝒜_t} y_s + median(T | D_t) ]                (the L1 Bayes action)
risk         R1(D_t)  = E[ |SR - SR_hat_t| | D_t ]                                      (closed form on the mixture)
acquisition  s* = argmax_s  R1(D_t) - E_{Y_s | D_t}[ R1(D_t + (s, Y_s)) ]               (`driveat/acquisition.r1_pick`)
stopping     stop when  c * R1(D_t) <= eps
```

Each acquisition branch is scored at its own posterior median, i.e. with
the same action the readout would report; candidates of one type share the
type's (theta x u) table, so branches are vectorised per type. Ties — exact
ones are common, because all-pass / all-fail routes of one type have
identical posteriors and are exchangeable — are broken by rounding the
score to 1e-10 and taking the lowest bank index (`acquisition.argmin_stable`),
so the choice does not depend on floating-point rescaling. No
auxiliary model, no phase switch, no localisation budget. The component
ablation (`run_ablation.py`) switches off, one at a time, the difficulty
posterior (point curves at b_hat), the testlet (sigma_g = 0) and the risk
acquisition (a random order); 2PL selection survives only inside the
published baselines.

**The risk scale c is fixed on the calibration panel, never on evaluation
planners** (`run_tau_calibration.py`, results/risk_cal.json):
leave-one-planner-out Delta-R1 trajectories on the calibration panel give
realised errors |SR_hat_t - SR| and predicted risks R1(D_t); c is the 90th
percentile of their ratio over the left-out planners and t in [10, n], n the
bank size (the LOO trajectories run through the whole bank).
The stopping rule c * R1 <= eps is therefore an *error* target: eps in
{.03, .05}, and Table 2 reports the mean rollouts it spends, the SR-MAE at
the stop and the calibration gap
(mean realised error minus mean c * R1 at the stop). The reliability of raw
R1 against realised error, by deciles, is printed with the same merge. The
earlier matched-cost rule (tau_hat chosen so that the LOO mean rollouts hit
30 / 55) is kept as the appendix table.

## 5. UPS — theta transport to unseen scenes

The probe bank exists only to learn the evaluation-scale ability that is
transported to the evaluation-type routes D:

```
probes: Delta-R1 on the block-D success rate (driveat/acquisition.r1_pick_transfer) — the probe-bank
        route whose outcome most reduces E|SR_D - SR_hat_D|; theta-EIG and 2PL Fisher are ablations
D:      b_s | scene_s ~ N(b_tilde_s, sigma^2),  u_g ~ N(0, sigma_g^2) for the (unobserved) evaluation types
```

with b_tilde_s the RelGraph R2 out-of-fold prediction of that draw and
sigma the residual SD the encoder learned on the calibration block — the
same prior as Section 3.1, so US and UPS share one difficulty model (run
s0 is canonical; the across-run SD is reported). The probe posterior q_B
is transported as is (`driveat/bayes.transfer`): the D-block success rate is
its posterior median (the quantity the MAE scores) and each D cell is
scored by its posterior predictive probability (the quantity the NLL
scores) — two Bayes actions for two losses, one posterior. The probe rule
is the same Delta-R1 machinery with the risk evaluated on D instead of on
the probe bank: the principle is *acquire for the quantity that must
generalise* — full-bank SR in UP, the block-D SR in UPS, nothing in US;
theta-EIG and the 2PL Fisher rule are reported as ablations (Table 3B).

## 6. Published baselines (`driveat/baselines.py`)

Native-vs-native (Table 1): each method's *selection rule* runs on the same
bank and the same calibration panel, with the readout its published
description implies where that is possible on 4-12 calibration planners.
All 2PL item parameters come from one calibration (`calibrate(mode='2pl')`,
explicit priors log a ~ N(0, .5^2), b ~ N(0, sigma_b^2) with sigma_b from
the 1PL empirical-Bayes fit) — not from the methods' own fitting code. The
adaptations, per row:

- **Random (IRT-free)** — mean of the rolled-out outcomes.
- **Random (IRT-free) / Random + IRT / Random-strat + IRT** — random order
  (type round-robin for the stratified one) read with the sample mean or
  the Rasch plug-in (1PL b_hat, a = 1, Newton-MAP theta with the N(0, 1)
  prior). The stratified order visits scenario types round-robin (one route
  per type per pass), not whole types. Table 1 / Table 4 report these rows
  as the expected error over five independent orders per evaluation
  (`NREP = 5`); the adaptive tables use one order (the stopping rule is
  applied to a single trajectory).
- **tinyBenchmarks-lite** (Polo et al., 2024) — K-means with K = B on the
  (a_hat, b_hat) embedding, one medoid per cluster, read with the p-IRT
  plug-in; duplicate (a, b) points (routes with identical calibration
  responses) leave clusters empty, so the budget is filled with the routes
  closest to their centroid. Their gp-IRT blend and the anchor-weighted
  correctness estimate are not used.
- **metabench-lite** (Kipnis et al., 2024) — greedy maximum 2PL information
  over a fixed 25-point quantile grid of b_hat (prefix order), read with the
  p-IRT plug-in; the published subset-size tuning and GAM readout are
  replaced by the budget grid and the plug-in.
- **Fluid-style** (Hofmann et al., 2025) — adaptive Fisher selection at the
  Newton-MAP 2PL ability, read with the p-IRT plug-in on the SR scale
  (Fluid reports the ability itself).
- **AnchorPoints** (Vivek et al., 2023) — K-medoids (PAM) with K = B on
  1 - correlation of the calibration response vectors (identical rows at
  distance 0; a constant row at distance 1 from every non-identical row),
  cluster-size-weighted mean of the anchors' outcomes — their own estimator,
  no IRT.
- **DISCO-sel + IRT** (Rubinstein et al., 2025, arXiv:2510.07959) — DISCO's
  selection stage only: routes ordered by the calibration planners'
  disagreement pbar (1 - pbar) (order-equivalent to its Jensen-Shannon
  criterion on 0/1 outcomes), read with the p-IRT plug-in. DISCO's own
  prediction stage is a signature metamodel (kNN / random forest over
  source models), which cannot be trained on 7-16 source planners and is
  far worse here (macro SR-MAE about .08 on two draws vs .03 with the
  plug-in); the substitution helps the baseline.
- **Total-/Marginal-Fisher static** — classical IRT information orders
  (population-integrated / marginal), read with the p-IRT plug-in.

The adaptive comparison (Table 2) runs the bank orders of Random, Fluid and
metabench under DriveAT's readout, risk and stopping rule, each with its own
calibration-fixed risk scale. metabench and Random define no stopping rule;
Fluid's published procedure is fixed-budget, and its dynamic-stopping
demonstration (ability standard error below a leaderboard rank-gap
threshold) targets ability precision with a threshold that has no
counterpart on this panel, so all orders are compared under the common
SR-error rule — the Table 2 "Fluid" row is Fluid's item order only.

## 7. Metrics

- **UP (primary)**: SR-MAE at the fixed budgets; the primary protocol is
  K_cal in {4, 8, 12} x B in {30, 55, 110, 165} with the macro average. Ties
  are reported when the paired 95% interval (cluster bootstrap over
  planners — the unique planner ids resampled with replacement, since the
  same planners recur across draws) includes zero.
- **Adaptive (risk target)**: at eps in {.03, .05}, mean rollouts spent,
  SR-MAE at the stop and the calibration
  gap; every order (DriveAT, Fluid, metabench, Random) stops on its own
  calibration-fixed c under the common readout. Appendix (matched cost):
  SR-MAE at the tau_hat that spends 30 / 55 rollouts on average and
  IES = (MAE / MAE_ref) x (rollouts / 55), reference = the random order at
  fixed B = 55.
- **US**: rho_scene = Spearman of predicted difficulty vs observed fail
  rate (primary); pooled cell AUROC and Scene-MAE vs the planner-only null.
- **UPS**: D-SR MAE at zero rollouts on the evaluation-type routes
  (primary), at probe budgets {30, 55, 110}; D-cell NLL.
- Resampling: paired cluster bootstrap — (planner) the unique evaluation-planner ids (16 on Bench2Drive) resampled with replacement for planner-side deltas, every evaluation of a resampled planner weighted equally (the same planners recur across the 16 draws, so draws are not independent clusters); US encoder-vs-baseline deltas are reported as mean +- SD across the three encoder runs.
