# Protocol — the specification

Normative for the release: every experiment in `experiments/` implements
exactly this. Numbers of record: [RESULTS.md](RESULTS.md); anchors and RNG
registry: [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## 1. Data

22 open-source end-to-end planners x 220 Bench2Drive closed-loop routes;
y_sk = 1 iff planner k completes scene (route) s (`data/matrices/`,
4,796 of 4,840 cells observed; the build report records derived vs published
success rates). Each route carries one of 44 scenario types (5 routes per
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
independent-item model (as on navhard, which has no grouping). The grouping
is public metadata available to every method: the type-stratified Random
baseline uses it, and the component ablation reports SC-IRT without it.
Scenario-definition parameters (trigger distances, flow speeds, ...) are
not used anywhere.

## 2. The split

16 calibration : 6 evaluation planners and 36 calibration : 8 evaluation
scenario types, drawn per Monte-Carlo draw (R = 16 draws,
RandomState(1000 + draw); `scirt/splits.py`). One draw partitions both axes
at once, so the three regimes share it:

|                          | calibration planners (16) | evaluation planners (6) |
|--------------------------|---------------------------|-------------------------|
| calibration types (36)   | A: calibration block      | **UP** evaluation       |
| evaluation types (8)     | **US** evaluation         | **UPS** target          |

Sample sizes: UP and UPS 16 x 6 = 96 planner evaluations per condition;
US pooled 16 x ~40 = 640 route evaluations. Types are held out as whole
types.

**Calibration scarcity.** K_cal in {7, 10, 16}: at K_cal < 16 the
calibration planners are subsampled once per draw
(RandomState(9000 + 100 draw + 10 K_cal)) and *every* model — ours and each
baseline's own — is re-fitted from those planners only; the evaluation
planners are unchanged.

**Budgets.** B = number of routes rolled out (the acquisition of Section 4
selects one route at a time); B = 5 x {6, 11, 22} = {30, 55, 110}, i.e.
14% / 25% / 50% of the benchmark; 110 executes 61% of the per-draw bank,
the reference budget at which representative samplers have largely
converged. (The grid was chosen as multiples of the 5 routes per scenario
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

**Calibration** on the block A (`scirt/calibration.py`) is a MAP fit with
explicit priors — the same prior forms the evaluation posterior uses. After
the theta-mean centring the MAP's b prior sits c = mean theta_hat away from
the centred frame (|c| up to .65 at K_cal = 7, median .2; .05 at K_cal =
16); the difficulty posterior below is taken in the centred frame, i.e. the
frame of the evaluation prior, and only the MAP point b_hat (baselines,
point-curve ablation, sigma_g estimate) carries the offset:

```
theta_k ~ N(0, 1)          b_s ~ N(0, sigma_b^2)          (2PL/3PL baselines: log a_s ~ N(0, .5^2), logit c_s ~ N(-2.2, 1))
```

Adam lr .05, 800 iterations, zero init, theta-mean centring. sigma_b is
chosen by empirical Bayes on the grid {.5, .75, 1, 1.5, 2, 3} — the exact
marginal likelihood sum_s log int prod_k Bern(y_sk | theta_hat_k, b) N(b; 0, sigma_b^2) db
on the difficulty grid below (1.5 in 45 of the 48 (draw, K_cal) fits on
this panel, 1.0 or 2.0 in the rest). The testlet SD sigma_g is the profile
marginal likelihood at (theta_hat, b_hat) over {0, .25, .5, .75, 1, 1.25,
1.5, 2} with u integrated on the u grid (.5-1.25 at K_cal = 7, 1.0-1.25 at
K_cal = 16; u itself is not a MAP parameter — the calibration block has one
u_kg per cell block and only its variance is identified).

**Difficulty posterior — exact, conditional on the fitted abilities.**
Given theta_hat the items are independent and one-dimensional, so every
difficulty keeps its full posterior on an 801-point grid over [-10, 10]:

```
p(b_s | A) ∝ N(b_s; 0, sigma_b^2) prod_{k:(s,k) in A} Bern(y_sk | sigmoid(theta_hat_k - b_s))
```

No Gaussian approximation: the all-pass / all-fail routes of a K_cal = 7
panel (19-28 of 180 in the first four draws) get their one-sided posterior. Every downstream probability
marginalises it:

```
m_s(x) = E_{b_s | A}[ sigmoid(x - b_s) ]      tabulated on XG = [-9, 9], step .05 (`scirt/curves.py`)
```

so that m_s(theta + u) is an index lookup.

**Evaluation posterior.** For a new planner with administered set 𝒜_t and
outcomes y, theta lives on a 241-point grid over [-6, 6] with the N(0, 1)
prior and each u_g on a 61-point grid over [-3, 3] with N(0, sigma_g^2).
Given theta the types factorise, so the posterior is one theta vector plus
one (theta x u) table per observed type (`scirt/bayes.State`):

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
(`scirt/bayes.State.stats`). The evaluation loss is L1, so one action
serves everywhere:

```
readout      SR_hat_t = (1/S) [ sum_{s in 𝒜_t} y_s + median(T | D_t) ]                (the L1 Bayes action)
risk         R1(D_t)  = E[ |SR - SR_hat_t| | D_t ]                                      (closed form on the mixture)
acquisition  s* = argmax_s  R1(D_t) - E_{Y_s | D_t}[ R1(D_t + (s, Y_s)) ]               (`scirt/acquisition.r1_pick`)
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
percentile of their ratio over the left-out planners and t in [10, 110].
The stopping rule c * R1 <= eps is therefore an *error* target: eps in
{.03, .05}, and Table 2 reports the mean rollouts it spends, the SR-MAE at
the stop, the coverage P(|SR_hat - SR| <= eps) and the calibration gap
(mean realised error minus mean c * R1 at the stop). The reliability of raw
R1 against realised error, by deciles, is printed with the same merge. The
earlier matched-cost rule (tau_hat chosen so that the LOO mean rollouts hit
30 / 55) is kept as the appendix table.

## 5. UPS — theta transport to unseen scenes

The probe bank exists only to learn the evaluation-scale ability that is
transported to the evaluation-type routes D:

```
probes: Delta-R1 on the block-D success rate (scirt/acquisition.r1_pick_transfer) — the probe-bank
        route whose outcome most reduces E|SR_D - SR_hat_D|; theta-EIG and 2PL Fisher are ablations
D:      b_s | scene_s ~ N(b_tilde_s, sigma^2),  u_g ~ N(0, sigma_g^2) for the (unobserved) evaluation types
```

with b_tilde_s the RelGraph R2 out-of-fold prediction of that draw and
sigma the residual SD the encoder learned on the calibration block — the
same prior as Section 3.1, so US and UPS share one difficulty model (run
s0 is canonical; the across-run SD is reported). The probe posterior q_B
is transported as is (`scirt/bayes.transfer`): the D-block success rate is
its posterior median (the quantity the MAE scores) and each D cell is
scored by its posterior predictive probability (the quantity the NLL
scores) — two Bayes actions for two losses, one posterior. The probe rule
is the same Delta-R1 machinery with the risk evaluated on D instead of on
the probe bank: the principle is *acquire for the quantity that must
generalise* — full-bank SR in UP, the block-D SR in UPS, nothing in US;
theta-EIG and the 2PL Fisher rule are reported as ablations (Table 3B).

## 6. Published baselines (`scirt/baselines.py`)

Native-vs-native (Table 1): each method's *selection rule* runs on the same
bank and the same calibration panel, with the readout its published
description implies where that is possible on 7-16 calibration planners.
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
metabench under SC-IRT's readout, risk and stopping rule, each with its own
calibration-fixed risk scale. metabench and Random define no stopping rule;
Fluid's published procedure is fixed-budget, and its dynamic-stopping
demonstration (ability standard error below a leaderboard rank-gap
threshold) targets ability precision with a threshold that has no
counterpart on this panel, so all orders are compared under the common
SR-error rule — the Table 2 "Fluid" row is Fluid's item order only.

## 7. Metrics

- **UP (primary)**: SR-MAE at the fixed budgets; the primary protocol is
  K_cal in {7, 10, 16} x B in {30, 55, 110} with the macro average. Ties
  are reported when the paired 95% interval (cluster bootstrap over
  planners — the unique planner ids resampled with replacement, since the
  same planners recur across draws) includes zero.
- **Adaptive (risk target)**: at eps in {.03, .05}, mean rollouts spent,
  SR-MAE at the stop, coverage P(|SR_hat - SR| <= eps) and the calibration
  gap; every order (SC-IRT, Fluid, metabench, Random) stops on its own
  calibration-fixed c under the common readout. Appendix (matched cost):
  SR-MAE at the tau_hat that spends 30 / 55 rollouts on average and
  IES = (MAE / MAE_ref) x (rollouts / 55), reference = the random order at
  fixed B = 55.
- **US**: rho_scene = Spearman of predicted difficulty vs observed fail
  rate (primary); pooled cell AUROC and Scene-MAE vs the planner-only null.
- **UPS**: D-SR MAE at zero rollouts on the evaluation-type routes
  (primary), at probe budgets {30, 55, 110}; D-cell NLL.
- Resampling: paired cluster bootstrap — (planner) the unique evaluation-planner ids (22 on Bench2Drive) resampled with replacement for planner-side deltas, every evaluation of a resampled planner weighted equally (the same planners recur across the 16 draws, so draws are not independent clusters); US encoder-vs-baseline deltas are reported as mean +- SD across the three encoder runs.
