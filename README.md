# SC-IRT — Scene-Conditioned Item Response Theory

Treat a driving scenario as a test item and a planner as an examinee. Fitting an
item response model to the pass/fail panel recovers a **per-scenario difficulty**;
learning to *predict* that difficulty from the scene alone makes it available for
scenarios no planner has ever driven — which is what turns a fixed benchmark into
an adaptive one.

Two things follow. Scenario difficulty becomes a measurable quantity rather than
an intuition, so a benchmark can be described by what it actually tests. And a new
planner can be evaluated to a target precision in a fraction of the scenarios,
because the ones that carry no information about *this* planner need not be run.

**Headline results** (Bench2Drive, 16 end-to-end planners x 219 routes):

| | |
|---|---|
| Difficulty recovery | rho **+0.520** for the learned encoder, against **+0.191** for the best hand-crafted descriptor (ceiling +0.904) |
| Evaluation efficiency | success rate to ±4.4% in **25.6 routes** instead of 219 — **8.6x** fewer |
| Unseen scenarios | with no calibration available at all, adaptive selection on predicted difficulty matches the unattainable oracle (31.5 vs 32.9 routes) |

## Install

```bash
git clone <repo-url> && cd scirt
pip install -e .
```

Versions are pinned in `pyproject.toml`. They are pins rather than lower bounds
on purpose — see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Reproduce

```bash
bash run_all.sh                       # writes expected_local/, ~10 min on CPU
python tools/compare_outputs.py expected expected_local --tol 0.002
```

No GPU is needed or wanted: the runtime is pinned to single-threaded CPU, which
is both faster at this problem size and free of the cross-device float32
disagreement that would otherwise move every number.

## What is here

```
scirt/            the library
  runtime.py      pinned device, threads, dtype, seeds
  data.py         response panel, route types, evaluation bank
  features.py     scene descriptors and the reported registry
  irt.py          MAP calibration kernel + identification policies
  theta.py        ability estimation for adaptive testing
  stage2.py       feature -> item-parameter ridge regressions
  selection.py    CAT item-selection strategies
  resample.py     scenario-type cluster bootstrap
  stats.py        summary statistics

experiments/      one entry point per reported table
  gold_anchor.py       frozen gold difficulty        (run first)
  noise_ceiling.py     Appendix A3
  descriptor_table.py  Table I    hand-crafted descriptors
  encoder_us.py        Table I    "Ours"
  encoder_verify.py    Table IV   ablation and tie test
  hybrid_prereg.py     Table IV   rank fusion
  cat_up.py            Table II(a) calibrated-bank CAT
  cat_ups.py           Table II(b) amortised-calibration CAT

data/             1.4 MB — response matrix, route types, 9 descriptor files,
                  frozen encoder predictions
expected/         reference outputs from this code, pinned configuration
expected_legacy/  the authors' original outputs, unmodified
tests/            kernel-equivalence and invariant tests
tools/            baseline freezing, output comparison
```

`gold_anchor.py` must run first: it writes the frozen difficulty anchor that five
other experiments read. `run_all.sh` encodes the order.

## Method in one page

**Calibration.** A 2PL model, `P(y_ij = 1) = sigmoid(a_i (theta_j - b_i))`, fitted
by MAP with Adam. The panel is 16 planners x 219 routes with missing cells. The
model is invariant to a common shift of ability and difficulty, so one of three
identification policies is applied explicitly after each fit
(`scirt.irt.center_b` / `center_both` / `uncentred_theta`) — never inside the
fitter, because the right choice differs per experiment.

**The gold anchor** is one calibration on the complete panel, frozen and never
refitted. Nothing is ever *trained* against it, so a predictor correlating with it
is being scored rather than fitted.

**Predicting difficulty.** Two routes to a difficulty for an undriven scene:
a two-stage explanatory baseline (ridge from scene features to calibrated
difficulty), and a difficulty-supervised interaction encoder trained end to end
on the response panel. Table I compares them.

**Splits are the whole story.** Difficulty prediction is only useful if it
generalises to genuinely new scenarios, so every number is out-of-fold under a
split that cuts the axis being claimed: leave-one-scenario-type-out for unseen
scenarios (44 types, sibling routes excluded with their type), leave-one-planner-out
for unseen planners, and the crossed version for both. Standardisation,
calibration and encoder training all happen inside the training fold.

**Adaptive evaluation.** With a calibrated bank, classical Fisher information
picks the next route. Without one — the case the method exists for — only
predicted difficulty is available, and greedy Fisher selection on it *fails*: it
preferentially picks routes whose difficulty was over-predicted, buying prediction
error instead of information. Discounting information by the regression's own
residual variance removes that bias and closes most of the gap to the oracle.

**Uncertainty.** Routes of the same scenario type share a generator and a failure
mode, so intervals resample whole types, not routes: the effective sample size for
a confidence interval is 44, not 219. Comparisons are paired within a replicate,
and are called significant only when the interval excludes zero.

## Scope of this release

Bench2Drive experiments only. The paper's NavSim scale-up (Table III) is not
included here.

## Caveats worth reading before citing

- Difficulty is only identified up to the panel that produced it. The reliability
  ceiling is 0.904; a raw rho of 0.520 is 58% of what is attainable.
- The encoder and the best hand-crafted stack **tie** on this bank — the interval
  on the difference straddles zero. Table I's gap is against the best *single*
  descriptor family, not against the best stack.
- Aggregate MAE saturates around 0.05 for every descriptor, which is why
  difficulty recovery, not MAE, is the discriminating metric.
- Three adaptive-selection rows are chaotic across environments. See
  [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Documents

- [PROTOCOL.md](PROTOCOL.md) — metric definitions and split construction
- [RESULTS.md](RESULTS.md) — the reported tables
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — stability tiers, measured drift, pinned versions
