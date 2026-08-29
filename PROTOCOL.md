# Protocol — the specification

This document is normative for the release: every experiment in
`experiments/` implements exactly this. The numbers of record are in
[RESULTS.md](RESULTS.md); the anchors ledger and RNG registry are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## 1. Data

16 open-source end-to-end planners x 220 Bench2Drive closed-loop routes;
y_ij = 1 iff planner j completes route i (`data/matrices/`). 3,476 of 3,520
cells observed. Each route carries one of 44 scenario types. Scene
descriptors x_i (`data/features/`): the SC-IRT stack is cmdkin (25d) +
scenparamz (31d); the baseline descriptors of Table 3A ship alongside.
Route order is the response-matrix CSV column order — part of the
reproduction contract.

## 2. Evaluation model — uncertainty-aware Rasch

```
y_ij ~ Bernoulli( sigmoid(theta_j - b_i) )
```

Calibration on the response block A (planners j in the calibration panel,
routes i in the bank) is a regularised MAP (`scirt/calibration.py`, Adam
lr 0.05, 800 iterations, zero init, theta-mean centring):

```
(theta_hat, b_hat) = argmax  sum_{(i,j) in A} log Bern(y_ij | sigmoid(theta_j - b_i))
                              - 1e-2 mean_j theta_j^2 - 1e-3 mean_i b_i^2
```

The difficulty is kept as a distribution, not a point — the Laplace
approximation at the fitted calibration abilities:

```
p(b_i | A) ~ N( b_hat_i, s_i^2 ),   s_i^2 = 1 / sum_{j:(i,j) in A} p_hat_ij (1 - p_hat_ij)
```

and every downstream probability marginalises it (21-node Gauss-Hermite,
`scirt/curves.py`):

```
m_i(theta) = E_{b_i | A}[ sigmoid(theta - b_i) ]
```

**Readout for a new planner.** With administered set S_t and responses y,
the ability posterior is computed on a 241-point grid over [-6, 6] with a
N(0,1) prior,

```
q_t(theta) ∝ N(theta; 0, 1) prod_{i in S_t} m_i(theta)^{y_i} (1 - m_i(theta))^{1 - y_i}
theta_hat_t = argmax q_t
S_hat_t = (1/N) [ sum_{i in S_t} y_i + sum_{i not in S_t} m_i(theta_hat_t) ]
```

(`scirt/bayes.py`). Integrating the readout over q_t instead of plugging in
the MAP changes SR-MAE by less than .0005 at every budget; the MAP fill is
the reported estimator.

### 2.1 Unseen scenes — the difficulty prior from the scene

```
b_i | x_i ~ N( w^T x_i , sigma^2 )        (LLTM+e, canonical descriptor path)
```

(w, log sigma, theta) fitted jointly by MAP on the calibration block with
eps marginalised by Gauss-Hermite (`scirt/lltm.py`; lam_w = 0.5, 1500
iterations). The trajectory encoder (`scirt/encoder.py`,
`train/train_encoder_unified.py`) predicts b_tilde(x) end-to-end from the
cell likelihood; evaluation treats both paths identically (Table 3A scores
the point prediction b_tilde; single runs, seeds as mean +- SD).

## 3. Acquisition model — a joint 2PL used only to choose scenes

```
y_ij ~ Bernoulli( sigmoid( a_i^sel (theta_j^sel - b_i^sel) ) )
```

fitted on the same block A (`calibrate(..., mode='2pl')`: the 1PL
regulariser plus 0.5 mean_i (log a_i)^2). Held-out cell NLL does not
favour it over the Rasch model (Appendix: adequacy), which is why it never
scores anything: a_i^sel is a *ranking statistic* — how well scene i
separates planners — not a predictive parameter. It is only ever fitted
on a calibrated bank; unseen scenes never need it.

## 4. Target-aligned acquisition — *acquire for the quantity that must generalise*

| regime | given                                   | inferred            | acquisition                          |
|--------|-----------------------------------------|---------------------|--------------------------------------|
| UP     | calibrated bank p(b|A), (a,b,theta)^sel | theta_new, S_full   | localize (2PL Fisher, K) -> cover    |
| US     | theta_cal, x_new                        | b_new               | none                                 |
| UPS    | calibrated probe bank B, x_D            | theta_new (from B), b_D (from x_D) | theta-EIG on B under the evaluation model |

For the UPS target block the difficulty prior is the two-stage path fitted
on the calibration difficulties — b_tilde = ridge(alpha = 100) on the
descriptor stack, tau = its residual SD — marginalised in the response
probability (`run_ups.py`); the LLTM+e path is the canonical *US* estimator
(Table 3A).

**UP — localize, then cover** (`scirt/acquisition.localize_cover`):

```
t <= K :  theta_hat^sel_t = Newton MAP of the 2PL ability on S_t (N(0,1) prior)
          i_t = argmax_{i in U_t}  (a_i^sel)^2 p_i(theta_hat^sel_t) (1 - p_i(theta_hat^sel_t))
t >  K :  I_i^pop = (1 / J_cal) sum_{j in cal} (a_i^sel)^2 p_ij (1 - p_ij),  p_ij = sigmoid(a_i^sel (theta_j^sel - b_i^sel))
          i_t = argmax_{i in U_t} I_i^pop                       (planner-independent static order)
```

The first phase is the Fluid selection rule; the second is the
Total-Fisher static order. Their combination is what the failure analysis
asks for: adaptive exploitation alone leaves a non-representative executed
set whose fill is model-dependent, so once the planner is localised the
remaining error is reduced by covering the difficulty spectrum with the
scenes most informative for the calibration population, not by more
focusing.

**K is a bank constant, not a budget fraction.** It is estimated at
calibration time by leave-one-planner-out simulation on the calibration
panel (`run_k_calibration.py`): every calibration planner is evaluated with
the bank re-fitted from the others, running localize(K) -> cover for K in
{0, 5, 10, 15, 20, 25, 30, 40, 60}. On Bench2Drive the simulated SR-MAE is flat for K in
[15, 30] (range .0006 at J_cal = 13) and degrades for K >= 40 (never
switching, +.004); K = 20 is used everywhere. Closed-form triggers that
compare ability variance with item variance fire far past the plateau —
they measure the end of ability *learning*, whereas the switch is the end of
ability *over-focusing* — and are not used.

**UPS — theta-EIG under the evaluation model** (`scirt/acquisition.eig_pick`):
the probe bank exists only to learn the evaluation-scale ability that will
be transported to the unseen scenes, so the probe rule maximises expected
information about theta under the Rasch curves m_i. Choosing probes by the
2PL localize rule brings no additional gain there (Table 3B: +.0049,
95% CI [-.0151, +.0241]); the reason is alignment, not a statistical
defeat — argmax_i I_i^2PL(theta^sel) need not be argmax_i I_i^1PL(theta^eval).

## 5. Risk-based adaptive stopping

The reported quantity is the full-bank success rate S; the stopping
quantity is its posterior expected absolute error under the evaluation
model:

```
R1(D_t) = E[ |S - S_hat_t| | D_t ] <= tau   ->  stop
```

Closed form on the grid (`scirt/bayes.r1_risk`): given theta, the
unobserved successes are approximated by N(sum_U m_i, sum_U m_i(1 - m_i));
the mixture over q_t gives E|X - c| = sigma [2 phi(z) + z (2 Phi(z) - 1)].
The rule is aligned with the primary metric (SR-MAE) and, unlike an
interval-width contract, has no arbitrary precision level: tau is a
tolerance on the expected error.

**Thresholds are fixed on the calibration panel, never on held-out
planners** (`run_tau_calibration.py`): for a target mean budget B* in
{30, 60},

```
tau_hat(draw, J_cal, method, B*) = argmin_tau | E_LOO[ rollouts(tau) ] - B* |
```

a cost target, so held-out SR-MAE and efficiency are measured, not
selected. Table 2 reports the *achieved* mean rollouts. The same machine
(readout + R1 stop) is applied to every bank order it compares (design A:
each method at its own calibration-fixed tau for a common target budget;
design B, appendix: SC-IRT's tau applied to all).

## 6. The split

13/3 planners x 36/8 scenario types, R = 16 Monte-Carlo draws
(`scirt/splits.py`, RandomState(1000 + draw)). One draw partitions both
axes at once, so US / UP / UPS share it. Sample sizes: UP and UPS
16 x 3 = 48 planner evaluations; US pooled 16 x ~40 = 640 route
evaluations. Types are held out as whole types.

**Calibration scarcity.** J_cal in {4, 7, 10, 13}: at J_cal < 13 the
calibration planners are subsampled once per draw
(RandomState(9000 + 100 draw + 10 J_cal)) and *every* model — evaluation,
acquisition, and each baseline's own — is re-fitted from those planners
only; the held-out planners are the same across J_cal.

## 7. Published baselines (`scirt/baselines.py`)

Native-vs-native: each method runs in its published operating mode on the
same bank and the same calibration panel, with its own readout —
tinyBenchmarks (K-means anchors on (a, b) + plug-in IRT), metabench-lite
(information-grid greedy subset), Fluid-style (2PL Fisher argmax at the
Newton-MAP ability), AnchorPoints-adapted (K-means on calibration response
vectors, cluster-weighted medoid mean), DISCO-adapted (inter-planner
disagreement), Total-/Marginal-Fisher static orders, Random and
type-stratified Random with plug-in IRT, and the IRT-free random mean.
"-style / -lite / -adapted" marks re-implementations from the method
descriptions. The adaptive comparison (Table 2) runs the bank orders of
Random, Fluid and metabench under SC-IRT's readout and stopping rule —
none of them defines a stopping rule of its own.

## 8. Metrics

- **UP (primary)**: SR-MAE at fixed rollout budgets; the primary protocol is
  J_cal in {7, 10, 13} x B in {30, 60} with the macro average of the six
  cells. Ties are reported when the paired 95% interval (cluster bootstrap
  over the 16 draws) includes zero.
- **Adaptive**: mean rollouts and SR-MAE at the calibration-fixed tau;
  IES = (MAE / MAE_ref) x (rollouts / 60) with the reference declared as the
  random order at a fixed budget of 60 under the common readout (secondary).
  The ratio MAE / tau is the honesty diagnostic of the risk.
- **US**: rho_scene = Spearman of predicted difficulty vs observed fail rate
  (primary); pooled cell AUROC and Scene-MAE vs the planner-only null.
- **UPS**: D-SR MAE at zero rollouts on the unseen-type routes with 30 bank
  probes (primary); D-cell NLL.
- Resampling: paired cluster bootstrap — (draw) 16 clusters for planner-side
  deltas, (draw, type) 128 clusters for US pooled deltas.
