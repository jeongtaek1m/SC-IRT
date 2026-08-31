# Protocol — the specification

Normative for the release: every experiment in `experiments/` implements
exactly this. Numbers of record: [RESULTS.md](RESULTS.md); anchors and RNG
registry: [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## 1. Data

22 open-source end-to-end planners x 220 Bench2Drive closed-loop routes;
y_sk = 1 iff planner k completes scene (route) s (`data/matrices/`,
4,796 of 4,840 cells observed; the build report records derived vs published
success rates). Each route carries one of 44 scenario types (5 routes per
type). Scene descriptors x_s (`data/features/`): the SC-IRT stack is
cmdkin (25d) + scenparamz (31d). Route order is the response-matrix CSV
column order — part of the reproduction contract.

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

**Budgets.** Rollout budgets are whole scenario types:
B = 5 x {6, 11, 22} = {30, 55, 110} (14% / 25% / 50% of the benchmark;
110 executes 61% of the per-draw bank and is the saturation reference where
any representative sampler converges).

## 3. The model — one uncertainty-aware Rasch posterior

```
y_sk ~ Bernoulli( sigmoid(theta_k - b_s) )
```

Calibration on the block A is a regularised MAP (`scirt/calibration.py`,
Adam lr 0.05, 800 iterations, zero init, theta-mean centring):

```
(theta_hat, b_hat) = argmax  sum_A log Bern(y_sk | sigmoid(theta_k - b_s))
                              - 1e-2 mean_k theta_k^2 - 1e-3 mean_s b_s^2
```

The scene difficulty is kept as a distribution — the conditional Laplace
curvature at the fitted calibration abilities:

```
p(b_s | A) ~ N( b_hat_s, s_s^2 ),   s_s^2 = 1 / sum_{k:(s,k) in A} p_hat_sk (1 - p_hat_sk)
```

(A joint (theta, b) Laplace with the prior curvature and a centring
constraint changes s_s by ~1% on this panel and no downstream number —
each calibration planner answers ~180 scenes, so theta_cal is precise; the
conditional form is used.) Every downstream probability marginalises it
(21-node Gauss-Hermite, `scirt/curves.py`):

```
m_s(theta) = E_{b_s | A}[ sigmoid(theta - b_s) ]
```

For a new planner with administered set 𝒜_t and outcomes y, the ability
posterior lives on a 241-point grid over [-6, 6] with a N(0,1) prior:

```
q_t(theta) ∝ N(theta; 0, 1) prod_{s in 𝒜_t} m_s(theta)^{y_s} (1 - m_s(theta))^{1 - y_s}
```

### 3.1 Unseen scenes — the difficulty prior from the scene

```
b_s | x_s ~ N( w^T x_s , sigma^2 )        (LLTM+e, canonical descriptor path)
```

(w, log sigma, theta) fitted jointly by MAP on the calibration block with
the residual marginalised by Gauss-Hermite (`scirt/lltm.py`). The learned
scene encoder is the RelGraph R2: ego, agent and lane tokens with R-GCN
lane-lane message passing, agent-lane cross-attention over relative-geometry
relations and an ego-route relation (d = 64), trained end-to-end with the
same residual-marginalised cell likelihood; the release
ships its per-run out-of-fold predictions (`data/encoder/relgraph_r2_s*.npz`,
three independent runs, no ensembling). Table 3A scores point predictions;
statistically the two paths are tied on this panel.

## 4. UP — one posterior for inference, acquisition and stopping

The reported quantity is the full-bank success rate SR and its point
estimate is the MAP fill (`scirt/bayes.py`):

```
theta_hat_t = argmax q_t ,   SR_hat_t = (1/S) [ sum_{s in 𝒜_t} y_s + sum_{s not in 𝒜_t} m_s(theta_hat_t) ]
```

The stopping quantity is the posterior L1 risk of that estimate, in closed
form on the grid (unobserved successes given theta approximated by a
normal; mixture over q_t):

```
R1(D_t) = E[ |SR - SR_hat_t| | D_t ]          stop when  R1(D_t) <= tau_hat
```

The acquisition selects the scene whose outcome most reduces the same risk
(`scirt/acquisition.r1_pick`):

```
s* = argmax_s  R1(D_t) - E_{Y_s | D_t}[ R1(D_t + (s, Y_s)) ]
```

(Inside the acquisition the branch risk is evaluated at the branch
posterior median — the L1-optimal point — which differs from the MAP fill
by < .0005 SR on this panel.) No auxiliary model, no phase switch, no
localisation budget: an acquisition-criterion comparison run at protocol
freeze (theta-EIG, uncertainty-aware marginal Fisher, the former two-phase
2PL selector; research-side, not shipped as an entry point) showed every
posterior-based criterion within the paired intervals of every cell, so the
target-aligned one is used. The component ablation
(`run_ablation.py`) switches the two model components off independently;
2PL selection survives only inside the published baselines.

**tau_hat is fixed on the calibration panel, never on evaluation
planners** (`run_tau_calibration.py`): leave-one-planner-out simulation,
tau_hat(draw, K_cal, method, B*) = argmin_tau |E_LOO[rollouts(tau)] - B*|
for target budgets B* in {30, 55} — a cost target, so evaluation SR-MAE and
efficiency are measured, not selected. Table 2 reports achieved mean
rollouts; MAE / tau is the honesty diagnostic.

## 5. UPS — theta transport to unseen scenes

The probe bank exists only to learn the evaluation-scale ability that is
transported to the evaluation-type routes D:

```
probes: theta-EIG under the evaluation model (scirt/acquisition.eig_pick)
P(y = 1 | x, B) = int sigmoid(theta_hat - b) N(b; ridge(x), tau_D^2) db,   zero rollouts on D
```

with b_tilde = ridge(alpha = 100) on the descriptor stack fitted to the
calibration difficulties and tau_D its residual SD. Choosing probes by the
2PL Fisher rule brings no additional gain (ablation in Table 3B): the
principle is *acquire for the quantity that must generalise* — full-bank SR
in UP, the transportable ability in UPS, nothing in US.

## 6. Published baselines (`scirt/baselines.py`)

Native-vs-native (Table 1): each method runs in its published operating
mode on the same bank and the same calibration panel with its own readout —
tinyBenchmarks, metabench-lite, Fluid-style, AnchorPoints-adapted,
DISCO-adapted, Total-/Marginal-Fisher static, Random and type-stratified
Random with plug-in IRT, and the IRT-free random mean. "-style / -lite /
-adapted" marks re-implementations from the method descriptions. The
adaptive comparison (Table 2) runs the bank orders of Random, Fluid and
metabench under SC-IRT's readout and stopping rule — none defines a
stopping rule of its own.

## 7. Metrics

- **UP (primary)**: SR-MAE at the fixed budgets; the primary protocol is
  K_cal in {7, 10, 16} x B in {30, 55, 110} with the macro average. Ties
  are reported when the paired 95% interval (cluster bootstrap over the 16
  draws, 6 planners per cluster) includes zero.
- **Adaptive**: achieved mean rollouts and SR-MAE at the calibration-fixed
  tau; IES = (MAE / MAE_ref) x (rollouts / 55), reference = the random
  order at fixed B = 55 under the common readout (secondary).
- **US**: rho_scene = Spearman of predicted difficulty vs observed fail
  rate (primary); pooled cell AUROC and Scene-MAE vs the planner-only null.
- **UPS**: D-SR MAE at zero rollouts on the evaluation-type routes
  (primary), at probe budgets {30, 55, 110}; D-cell NLL.
- Resampling: paired cluster bootstrap — (draw) 16 clusters of 6 for
  planner-side deltas, (draw, type) clusters for US pooled deltas.
