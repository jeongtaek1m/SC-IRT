# SC-IRT protocol

Metric definitions, split construction and the estimator settings behind every
number in [RESULTS.md](RESULTS.md). Where this document and the code disagree,
the code is authoritative and the discrepancy is listed in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

Scope: Bench2Drive only.

## 1. Data

**Bench2Drive (B2D).** 16 end-to-end planners x 219 routes. 220 routes were
collected; one (`11755`) failed collection and is excluded, which is what makes
the evaluation bank 219. The response is the success flag, `y_ij` in {0, 1}. Routes
carry one of 44 scenario types (`data/matrices/b2d_route_types.csv`); 43 types have
five routes and one has four.

The panel is incomplete — 3476 of 3520 cells are observed — and every estimator
below masks the missing cells rather than imputing them.

## 2. Models

### 2.1 IRT calibration (2PL throughout)

```
P(y_ij = 1) = sigmoid( a_i (theta_j - b_i) ),    a_i = exp(loga_i)

MAP, Adam lr 0.05:
  NLL  +  1e-2 * mean(theta^2)  +  1e-3 * mean(b^2)  +  0.5 * mean(loga^2)

Identification:  theta <- theta - mean(theta)
```

Adam step counts are **not** uniform, and the code names each one explicitly:

| Site | steps |
|---|---|
| Gold anchor, noise ceiling, `cat_up` bank | 800 |
| `cat_ups` calibration and full bank | 600 |
| Leave-one-type-out folds, descriptor table | 400 |

Identification is applied *after* the fit, by one of three named policies, because
the correct choice differs per experiment (`scirt/irt.py`). The encoder evaluation
in particular must **not** re-centre: its difficulty is frozen input and is itself
the scale anchor.

**Gold difficulty.** One calibration on the full panel and the full item bank,
then frozen. It is used only as an evaluation anchor and enters no training
procedure of any kind.

### 2.2 Ours — difficulty-supervised interaction encoder

Ego-relative ground-truth agent tracks (up to 48 agents x 12 steps x 8 channels)
plus the ego trajectory and the navigation command. A per-agent BiGRU feeds a
2-3 layer transformer to a route embedding; kinematic summary statistics are
concatenated *inside* the model through an MLP. The output is a scalar difficulty
`b_tilde`.

Training minimises the cell-level BCE
`-sum_ij log P(y_ij | sigmoid(theta_j - b_tilde(x_i)))`, with `theta` fitted by a
within-fold Rasch calibration and then frozen. Six runs are reported (two widths
x three seeds) plus their ensemble; Table I uses the 6-seed logit mean (ens6L).
The shipped artifact's original training selected checkpoints on an inner
validation NLL; the released reference trainer runs a fixed epoch budget and is
not claimed to regenerate the artifact bit-for-bit (GPU tier, REPRODUCIBILITY.md).

This repository ships both the encoder's frozen out-of-fold predictions
(`data/interact/interact_b2d_w2a_final.npz`, the reference artifact every
number in RESULTS.md is computed from) and a reference training pipeline
(`train/`).

### 2.3 Baselines and ablations (not part of this release)
Hand-crafted descriptor stacks, kin-fusion and window variants, rank fusion and
the adaptive-testing experiments are comparisons and ablations reported in the
paper. Their code and data ship on the `full-reproduction` branch; this branch
contains the method and its evaluation against the frozen anchor, nothing else.

## 3. Splits

Every reported number is out-of-fold. Standardisation, calibration and encoder
training all happen strictly within the training fold.

| Regime | Axis cut | Construction |
|---|---|---|
| **US** — unseen scenario | scene | 44-type leave-one-type-out; sibling routes of the held-out type are removed with it |
| **UP** — unseen planner | planner | leave-one-planner-out over 16; the held-out planner's responses are excluded from calibration entirely |
| **UPS** — both | scene and planner | 3 seeds x [random half of types held out] x leave-one-planner-out; held-out-type routes have no response-based calibration at all |

## 4. Metrics

Notation: `i` indexes scenes, `j` planners, `p_ij` the model probability, `y_ij` the
observation, `N` the bank size, `J` the panel size.

### 4.1 rho — difficulty recovery (Table I headline)

```
rho = Spearman( {b_tilde_i} over all out-of-fold scenes , {b_gold_i} )
```

Pooled across folds, **not** averaged per fold: a per-fold average would be a
within-type correlation, which is a different quantity.

Uncertainty is a **cluster bootstrap** over the 44 scenario types, 10,000
replicates, percentile interval. Comparisons are **paired** — both predictors are
scored on the same resampled index set within a replicate — giving an interval on
the difference and an exceedance probability. A comparison is called significant
only when the 95% interval excludes zero.

### 4.2 AUROC — response ranking

```
US : micro — one pooled AUC over all held-out cells
UP : macro — per-planner AUC, averaged over the 16 folds (± fold SE)
```

Threshold-free, and deliberately not a calibration claim: it says nothing about
absolute probability level, which is why MAE is reported alongside it.

### 4.3 MAE — performance reconstruction, aggregate level only

```
US (scene pass rate):
    MAE = (1/N_test) * sum_i | mean_j p_ij  -  mean_j y_ij |

UP / CAT (p-IRT):
    predSR = ( sum_{administered} y  +  sum_{not administered} p ) / N_bank
    MAE    = | predSR - SR_bank |,   averaged over planners (± SE)
```

The p-IRT form follows the tinyBenchmarks convention: observed responses enter as
themselves, unobserved ones as their predicted probability. Because the observed
terms cancel, this is numerically identical to the held-out-only definition scaled
by the unobserved fraction.

Per-cell `|p - y|` is an improper score — its reference floor is 0.293 and its
ceiling 0.499 — and is never reported. It is retained for diagnostics only.

### 4.4 Adaptive testing (Table II)

```
Ability:  Newton MAP,  g = sum_S a_i (y_i - p_i) - theta
                       h = -sum_S a_i^2 p_i (1 - p_i) - 1

Stop when SE(theta) = 1 / sqrt( sum_S a_i^2 p_i (1-p_i) + 1 ) < tau
          tau = 0.35 in the main text; 0.40 and 0.30 in the appendix

items n : routes administered before stopping (fold mean ± SE)
IES     : N / n, with N = 219 for UP and the new-type bank (~110) for UPS.
          Undefined for the oracle row, which is not deployable.
theta err : | theta_hat - theta_full |, against the full-response MAP estimate
```

Selection rules:

- **UP** — `argmax a^2 p (1-p)`, classical 2PL Fisher information. Available only
  because the bank is calibrated.
- **UPS (ours)** — shrunk information. Greedy Fisher selection on *predicted*
  difficulty fails by a winner's curse: it favours routes whose difficulty was
  over-predicted, buying prediction error rather than information. Marginalising
  the logistic over a Gaussian difficulty posterior gives

  ```
  info_i = p~ (1 - p~),   p~ = sigmoid( (theta_hat - b_tilde_i) / sqrt(1 + s^2 / c) )
  ```

  with `s^2` the Stage-2 in-fold residual variance and `c` the probit-logit
  matching constant. **The code uses `c = 2.9`**; the value implied by the usual
  1.7 scaling is 2.89. The code value is what produced the published numbers.

### 4.5 Noise ceiling (Appendix A3)

```
Split the 16-planner panel into halves, calibrate each independently,
correlate the two difficulty vectors  ->  r_half  (mean of 20 random splits)

Spearman-Brown:   reliability = 2 r_half / (1 + r_half)
Ceiling        =  sqrt(reliability)
```

B2D: r_half 0.691 -> reliability 0.817 -> **ceiling 0.904**. Attenuation-corrected
values `rho* = rho / ceiling` may be reported alongside raw rho.

This estimate runs on the full 220-route collection, not the 219-route evaluation
bank; its inclusion filter is "at least eight observed responses", which excludes
nothing.

## 5. Reproduction

`bash run_all.sh` regenerates every number from `data/`, in dependency order,
under a pinned runtime (CPU, single-threaded, pinned library versions). Compare
with `tools/compare_outputs.py`, not `diff`. Tolerances: rho and MAE ±0.002,
interval bounds ±0.005, counts exact — except for the three chaotic
adaptive-selection rows documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
