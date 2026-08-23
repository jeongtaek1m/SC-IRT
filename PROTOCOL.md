# Protocol — the unified specification

This document is normative for the release: every experiment in
`experiments/` implements exactly this. The numbers of record are in
[RESULTS.md](RESULTS.md); the anchors ledger and RNG registry are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## 1. Data

16 open-source end-to-end planners x 220 Bench2Drive closed-loop routes;
y_ij = 1 iff planner j completes route i (`data/matrices/`). 3,476 of 3,520
cells observed. Each route carries one of 44 scenario types. Scene
descriptors x_i (`data/features/`): the SC-IRT stack is cmdkin (25d) +
scenparamz (31d); the baseline descriptors of Table 1 ship alongside.
Route order is the response-matrix CSV column order — it is part of the
reproduction contract.

## 2. Generative model (one equation for every regime)

```
theta_j ~ N(0, 1)
b_i | x_i ~ N(b_tilde_phi(x_i), sigma^2)
y_ij ~ Bernoulli( sigmoid(theta_j - b_i) )          all-1PL
```

The three regimes differ only in the *state* of the item parameters:

| regime | item difficulty state                       | inferred |
|--------|---------------------------------------------|----------|
| US     | the prior itself is learned (x -> b~, sigma)| scene    |
| UP     | posterior from responses: N(b^_i, s_i^2)    | planner  |
| UPS    | no responses -> posterior = prior           | planner  |

The inference engine is shared: item curves m_i(theta) = E_b[sigmoid(theta-b)]
(21-node Gauss-Hermite), a 241-point theta grid on [-6, 6] with N(0,1) prior,
the SR-variance acquisition (section 4.0), and the realised-SR
posterior-predictive quantile stop. **UPS = the UP engine + the prior US
supplies.**

Discrimination: with the panel's measured sigma_log a = 0.089, marginalising
a changes cell probabilities by at most 0.0009 — the 1PL is not a separate
model but the same model with a marginalised. A fitted-discrimination
variant (2PL bank + marginalisation) is retained as the appendix row of
Table 3; log-discrimination has split-half reliability 0.08 on this panel,
which is why no unseen-scene path ever predicts a.

### 2.1 The canonical descriptor estimator — one-stage LLTM+e

```
z_ij = theta_j - (w^T x~_i + eps_i),   eps_i ~ N(0, sigma^2)
```

(theta, w, log sigma) fitted jointly by MAP on the calibration block, eps
marginalised by Gauss-Hermite (`scirt/lltm.py`; Fischer 1973; Janssen et al.
2004; De Boeck 2008). This instantiates the generative model's difficulty
prior directly; sigma-hat (~0.59) is the model's own estimate of the
difficulty share features cannot explain. The two-stage alternative
(Rasch b-hat then Ridge) is reported as a comparison row; the paired delta
and the plausible-values decomposition (Mislevy 1991; Rubin 1987 — the
13-rater calibration-noise share of US rho uncertainty, 16.4%) are in
`experiments/run_us.py`.

The encoder path predicts b_tilde with a trajectory encoder instead of
w^T x~; evaluation treats both identically. Single runs only; seeds are
summarised as metric mean +- SD.

## 3. The split

13/3 planners x 36/8 scenario types, R = 16 Monte-Carlo draws
(`scirt/splits.py`, RandomState(1000 + draw)). One draw partitions both axes
at once, so US / UP / UPS share it (blocks A, C, UP-eval, D as in README).
Sample sizes: UP and UPS 16 x 3 = 48 planner evaluations; US pooled
16 x ~40 = 640 route evaluations. Types are held out as whole types — route-
level splits leak near-duplicates of the same scenario into training.

## 4. Adaptive testing (UP)

### 4.0 Acquisition — SR-variance (canonical, 2026-08-22)

The evaluand is the realised full-bank success rate
S = (1/N)(sum_observed y + sum_unobserved Y). Select the item whose
observation most reduces its posterior variance:

```
A_i = Var(S | D) - E_{Y_i}[ Var(S | D, Y_i) ]

Var(N S | D) = E_q[ sum_{j in U} m_j (1 - m_j) ]   (per-response uncertainty)
             + Var_q( sum_{j in U} m_j )           (ability uncertainty)
```

Closed form on the theta grid — for each candidate update q_y ~ q m_i^y
(1-m_i)^(1-y), drop i from U, re-evaluate; argmin expected posterior
variance (`scirt/acquisition.srvar_pick`). Early on the ability term
dominates (Fisher-like behaviour); as q narrows the direct-removal term
takes over and targets the residual-uncertainty items. Selection and
stopping now optimise the same quantity; the paired price/benefit against
the theta-EIG rule it replaced is in RESULTS Table 3.

**Scope.** UP and UPS-standalone certify S, so they use SR-variance.
The UPS-extend bank probes exist to learn theta for transport — there the
theta-EIG rule remains correct. The principle is: *the acquisition target is
the quantity being certified.*

### 4.1 Stopping and the certificate

Stop when the 95% posterior-predictive interval of S has width <= 2 eps
(eps in {0.10, 0.05}). The interval is a quantile interval from theta draws
+ Bernoulli fills (`scirt/bayes.sr_ci`) — a normal approximation changes
measured coverage and is not used. Coverage is always reported as count/48.
The second variance term above is *unobserved-response predictive
uncertainty* (not "binomial noise" — the simulator is deterministic; the
uncertainty is that unrun responses are unknown).

### 4.2 Published-baseline fidelity (Table 4)

Native-vs-native is the main comparison: every method runs in its published
operating mode on the same bank and the same 13-planner calibration —
tinyBenchmarks (K-means anchors + p-IRT, fixed K), metabench-lite
(information-grid subset, fixed K), Fluid-style (2PL Fisher argmax, fixed
budget; the paper defines no stopping rule), ATLAS-style (3PL + top-5
randomesque + EAP, SE(theta) <= tau in {0.1, 0.2, 0.3}, min 30 items), ours
(SR +- eps). Panel (a) is the selection-isolation ablation (common stopping
machine, rules swapped); panel (b) the equal-cost fixed-budget snapshot;
panel (c) the ATLAS-format IES with a declared Random-100 reference (valid
only within a matched-cost tier); panel (e) the tau <-> eps bridge.
"-style/-lite" marks reimplementations from the method descriptions.

### 4.3 CAT under calibration scarcity (Table 5)

Subsample J_cal in {4, 7, 10, 13} of the calibration planners and
*re-calibrate the bank from scratch* with only those responses (1PL / 2PL /
3PL each); held-out planners fixed across J_cal. Each arm runs its native
item model end to end. Two subsample replicates at J_cal < 13 give the
selection-stability (Jaccard of first-30 selections) and the parameter
reliability chain (corr of log a-hat vs b-hat across replicates). Read the
rollout column together with coverage: a point-parameter posterior that
mistakes small-panel noise for certainty stops *earlier* while its error
grows past the target — premature stops are a symptom, not efficiency.

## 5. UPS — composition

```
P(y | B, x) = int int sigmoid(theta - b) dp(theta | B) dp(b | x)
```

theta posterior from B = 30 bank probes (theta-EIG), difficulty prior from
the feature path. Main = extend (zero rollouts on the target routes), with
the common-scale decomposition separating the amortisation gap (swap
b_tilde for response-calibrated b-hat, same theta) from the theta-transport
gap (swap the bank theta for the target-domain theta) — transport dominates
by ~20x. Hybrid adds D probes on top of the warm-start posterior;
standalone treats the ~40 target routes as their own bank (SR-variance
acquisition, cold-start stress test).

## 6. Metrics

- **US**: pooled cell AUROC vs the planner-only null (b = 0 — a reference,
  not a floor), Scene-MAE with relative reduction, rho_scene = Spearman of
  predicted difficulty vs observed fail rate. Oracle = response-calibrated
  b on the evaluation block (in-sample ceiling).
- **UP**: rollouts to certification, SR-MAE at stop, coverage (count/48).
- **UPS**: |SR error| at zero target rollouts; pool-exhaustion fraction for
  the standalone.
- **IES** (ATLAS-style) = (MAE / MAE_Random100) x (Items / 100), reference
  declared, matched-cost tiers only.
- Resampling: paired cluster bootstrap — (seed) 16 clusters for planner-side
  deltas, (draw, type) 128 clusters for US pooled deltas.
