# SC-IRT — Scene-Conditioned Item Response Theory for Closed-Loop Driving Evaluation

Official code release. Treat a driving scenario as a test item and a planner
as an examinee. SC-IRT is built around a single probabilistic object — an
uncertainty-aware Rasch posterior over the scene bank and the planner's
ability — and uses it for everything: reconstructing the benchmark success
rate, choosing which scene to roll out next, and deciding when to stop.

```
evaluation model   y_sk ~ Bernoulli(sigmoid(theta_k - b_s + u_kg)),  b_s | A exact grid posterior,  u_kg ~ N(0, sigma_g^2)
readout            SR_hat_t = posterior median of SR                     (the L1 Bayes action)
acquisition        argmax_s  R1(D_t) - E_{Y_s}[ R1(D_t + (s, Y_s)) ]      (Delta-R1)
stopping           c * R1(D_t) <= eps,  R1 = E[ |SR - SR_hat_t| | D_t ],  c fixed on the calibration panel
```

*Same posterior, same loss, same action for inference, acquisition and
stopping.* No auxiliary discrimination model, no phase switch, no
localisation budget. The posterior knows two things a plug-in Rasch does
not: how uncertain each scene's difficulty is on a small calibration panel
(exact, not Gaussian), and that routes of one scenario type move together
(a planner x type testlet effect).

Everything in the paper runs from this repository: the data ships in
`data/` (under 1 MB, no external downloads) and every table-producing entry
point ends by asserting the published numbers (`anchors OK`).

- [[Protocol]](PROTOCOL.md) — models, split, acquisition, stopping, metrics.
- [[Results]](RESULTS.md) — the numbers of record, one section per table.
- [[Reproducibility]](REPRODUCIBILITY.md) — anchors ledger, RNG registry, environment.

## Contributions

- **C1 Uncertain item-bank inference.** A small calibration panel makes
  every scene difficulty uncertain; SC-IRT keeps the exact conditional
  posterior p(b_s | A) and the planner x scenario-type dependence (a
  testlet effect over the benchmark's own route grouping, fitted to zero
  when the grouping is uninformative), and marginalises both into every
  probability it computes. Inputs: the response matrix and the grouping,
  nothing else.
- **Baselines.** tinyBenchmarks-lite, metabench-lite, Fluid-style,
  AnchorPoints, DISCO-sel + IRT, Fisher-static, Random (+IRT, stratified):
  each adaptation is spelled out in PROTOCOL.md section 6.
- **C2 Target-aligned acquisition.** *Acquire for the quantity that must
  generalise*: the posterior L1 risk of the reported success rate in UP, of
  the transported block-D success rate in UPS (Delta-R1 on D), nothing in
  US.
- **C3 Risk-based adaptive stopping.** Stop when the calibrated risk
  c * R1 falls below the error target eps; c is fixed on the calibration
  panel (leave-one-planner-out) and never selected on evaluation planners,
  and the realised coverage P(|SR_hat - SR| <= eps) is reported.
- **C4 Scene-conditioned difficulty.** scene -> b via the RelGraph
  relational scene-graph encoder (learned per draw on the calibration
  types, shipped as per-run out-of-fold predictions), enabling unseen-scene
  (US) and joint (UPS) generalisation.

## The primary protocol

16-planner x 220-route Bench2Drive panel (one planner per model family,
chosen from 22 with complete records to cover the ability range evenly);
per draw (R = 16), 12 calibration : 4 evaluation planners and 36 : 8 scenario types. B = number of routes
rolled out (30/55/110 = 5 x {6, 11, 22}, so the type-stratified baseline
are multiples of the 5 routes per scenario type); calibration-panel sizes
K_cal in {4, 8, 12}. 64 evaluations per cell. A second, two-stage panel
(NAVSIM navhard leaderboard, 87 unique submissions x 225 units) reproduces
the UP comparison off Bench2Drive.

|                          | calibration planners (12) | evaluation planners (4) |
|--------------------------|---------------------------|-------------------------|
| calibration types (36)   | A: calibration            | **UP** evaluation       |
| evaluation types (8)     | **US** evaluation         | **UPS** target          |

## Getting started

```bash
pip install -e .[figs]      # numpy, scipy, scikit-learn, torch (+ matplotlib)
pytest tests/ -q            # fast invariants, CPU

# Reproduce the paper (GPU). Heavy scripts accept --seeds lo hi shards + --merge.
python experiments/run_up_frontier.py       # Table 1 (fixed budgets)
python experiments/run_tau_calibration.py   # calibration-fixed risk scale (+ matched-cost thresholds)
python experiments/run_adaptive.py          # Table 2 (risk-target stopping) + cost-error data
python experiments/run_ablation.py          # component ablation (one component off at a time)
python experiments/run_us.py                # Table 3A
python experiments/run_ups.py               # Table 3B
python experiments/run_navhard.py           # Table 4: the two-stage NAVSIM panel
python experiments/run_readout_dropin.py    # analysis: the readout under every selector
python experiments/run_model_adequacy.py    # appendix diagnostics
python experiments/run_calibration_stability.py
python experiments/make_figures.py          # figures from the results jsons

# Use it inside a real closed-loop evaluation (UP): pick routes, ingest outcomes, stop on risk
python tools/b2d_adaptive_eval.py --dry-run VAD --eps 0.03          # simulate from the matrix
python tools/b2d_adaptive_eval.py --name my_planner --agent ... --agent-config ... --eps 0.03
```

## What is in the box

```
scirt/          the library (PROTOCOL.md has the maths)
experiments/    one entry point per paper table + build_data.py (provenance)
tools/          b2d_adaptive_eval.py — SC-IRT inside a real Bench2Drive evaluation (scirt/live.py)
data/
  matrices/     16 x 220 pass/fail panel of record (+ the 22-planner matrix it was drawn from and the older 16-planner panel)
  features/     scene-descriptor sets used as US baselines (cmdkin, gtrisk, ...)
  b2d/          traffic-feature table and kin/density baselines
  encoder/      RelGraph R2 per-run out-of-fold difficulty predictions + the
                learned residual SD per draw (3 independent runs; no ensembling)
  navhard/      the two-stage NAVSIM leaderboard panel (provenance in REPRODUCIBILITY.md)
  live/         cached leave-one-planner-out risk scales for the live evaluator (keyed by bank)
results/        written by the scripts (gitignored); numbers of record are RESULTS.md
tests/          fast invariants
```

## Honest caveats

- Differences below about .005 SR-MAE are inside the paired 95% intervals
  at 64 evaluations per cell; the tables mark which cells are.
- SC-IRT has the lowest error in 7 of 9 cells of Table 1 (macro .0296 vs
  .0327 for Fluid and .0358 for type-stratified Random); at B = 30 for
  K_cal = 4 and 12 the Fluid order is lower by .001-.002, inside the
  intervals. Random-policy rows are expected errors over five orders.
  On the navhard panel SC-IRT wins only the B = 110 column for
  K_cal <= 12 (the other cells are ties or ns losses to Random + IRT,
  metabench and Fluid), and with an 81-planner calibration set the 2PL
  Fisher orderings win the low budgets outright. Those cells are part of
  the result.
- The stopping rule is conservative in the mean (the calibration gap is
  negative in every cell) but not always at the nominal level: the
  90th-percentile risk scale fixed on the calibration panel yields 84-94%
  realised coverage on evaluation planners.
- Exact ties in the acquisition score (routes of one type with identical
  posteriors) are broken by bank order; this arbitrary but documented
  choice moves individual cells by up to .004 SR-MAE, inside the intervals.
- UPS (Table 3B) is a per-cell result: the transport lowers the predictive
  NLL of the unseen cells at every budget, but its block-SR error sits on
  the scene-prior floor (.095) and does not beat the planner's own probed
  success rate at B >= 55 (.090 / .087) on this panel.
- The RelGraph encoder is tied with the hand-crafted descriptor stack on
  AUROC and behind it on scene-MAE and rank correlation (-.04 +- .02); its
  contribution is the input (raw tracks), not extra accuracy. Structural
  controls show the lane-graph relations are inert on this bank (removing
  the ego-route relation improves rho by +.03; shuffling correspondences
  changes nothing beyond seed noise).
- The RelGraph encoder ships as predictions; its training code depends on
  Bench2Drive raw rollouts (not redistributable) and is staged for a
  separate release; everything downstream of the predictions (US scoring,
  the UPS prior) is reproducible here.
- The navhard panel is dominated by a few teams' submission sweeps;
  near-duplicate submissions can appear on both sides of the planner split
  (this affects every method identically). See RESULTS.md for the caveats.

## Citation

```bibtex
(to appear)
```
