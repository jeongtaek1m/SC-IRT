# SC-IRT — Scene-Conditioned Item Response Theory for Closed-Loop Driving Evaluation

Official code release. Treat a driving scenario as a test item and a planner
as an examinee. SC-IRT is built around a single probabilistic object — an
uncertainty-aware Rasch posterior over the scene bank and the planner's
ability — and uses it for everything: reconstructing the benchmark success
rate, choosing which scene to roll out next, and deciding when to stop.

```
evaluation model   y_sk ~ Bernoulli(sigmoid(theta_k - b_s)),  b_s | A ~ N(b_hat_s, s_s^2)
readout            SR_hat = (1/S) [ sum observed y + sum m_s(theta_hat) ]
acquisition        argmax_s  R1(D_t) - E_{Y_s}[ R1(D_t + (s, Y_s)) ]      (Delta-R1)
stopping           R1(D_t) <= tau_hat,  R1 = E[ |SR - SR_hat_t| | D_t ]
```

*Same posterior, same target for inference, acquisition and stopping.* No
auxiliary discrimination model, no phase switch, no localisation budget: an
acquisition-criterion ablation shows every posterior-based criterion within
the paired intervals, so the target-aligned one is used.

Everything in the paper runs from this repository: the data ships in
`data/` (under 1 MB, no external downloads) and every table-producing entry
point ends by asserting the published numbers (`anchors OK`).

- [[Protocol]](PROTOCOL.md) — models, split, acquisition, stopping, metrics.
- [[Results]](RESULTS.md) — the numbers of record, one section per table.
- [[Reproducibility]](REPRODUCIBILITY.md) — anchors ledger, RNG registry, environment.

## Contributions

- **C1 Uncertain item-bank inference.** A small calibration panel makes
  every scene difficulty uncertain; SC-IRT keeps p(b_s | A) (conditional
  Laplace) and marginalises it into every probability it computes.
- **C2 Target-aligned acquisition.** *Acquire for the quantity that must
  generalise*: the posterior L1 risk of the reported success rate in UP,
  evaluation-model information for the transported ability in UPS, nothing
  in US.
- **C3 Risk-based adaptive stopping.** Stop when the same risk falls below
  tau_hat, which is fixed on the calibration panel (leave-one-planner-out,
  cost-matched) and never selected on evaluation planners.
- **C4 Scene-conditioned difficulty.** scene -> b via the RelGraph
  relational scene-graph encoder (learned per draw on the calibration
  types, shipped as per-run out-of-fold predictions), enabling unseen-scene
  (US) and joint (UPS) generalisation.

## The primary protocol

22-planner x 220-route Bench2Drive panel; per draw (R = 16), 16 calibration
: 6 evaluation planners and 36 : 8 scenario types. Rollout budgets are whole
scenario types: B = 5 x {6, 11, 22} = {30, 55, 110}; calibration-panel sizes
K_cal in {7, 10, 16}. 96 evaluations per cell. A second, two-stage panel
(NAVSIM navhard leaderboard, 87 unique submissions x 225 units) reproduces
the UP comparison off Bench2Drive.

|                          | calibration planners (16) | evaluation planners (6) |
|--------------------------|---------------------------|-------------------------|
| calibration types (36)   | A: calibration            | **UP** evaluation       |
| evaluation types (8)     | **US** evaluation         | **UPS** target          |

## Getting started

```bash
pip install -e .[figs]      # numpy, scipy, scikit-learn, torch (+ matplotlib)
pytest tests/ -q            # fast invariants, CPU

# Reproduce the paper (GPU). Heavy scripts accept --seeds lo hi shards + --merge.
python experiments/run_up_frontier.py       # Table 1 (fixed budgets)
python experiments/run_tau_calibration.py   # calibration-fixed stopping thresholds
python experiments/run_adaptive.py          # Table 2 (adaptive) + cost-error data
python experiments/run_ablation.py          # component ablation (2 x 2)
python experiments/run_us.py                # Table 3A
python experiments/run_ups.py               # Table 3B
python experiments/run_navhard.py           # Table 4: the two-stage NAVSIM panel
python experiments/run_readout_dropin.py    # analysis: the readout under every selector
python experiments/run_model_adequacy.py    # appendix diagnostics
python experiments/run_calibration_stability.py
python experiments/make_figures.py          # figures from the results jsons
```

## What is in the box

```
scirt/          the library (PROTOCOL.md has the maths)
experiments/    one entry point per paper table + build_data.py (provenance)
data/
  matrices/     22 x 220 pass/fail response panel (+ the 16-planner panel it extends)
  features/     scene-descriptor sets used as US baselines (cmdkin, gtrisk, ...)
  b2d/          traffic-feature table and kin/density baselines
  encoder/      RelGraph R2 per-run out-of-fold difficulty predictions + the
                learned residual SD per draw (3 independent runs; no ensembling)
  navhard/      the two-stage NAVSIM leaderboard panel (provenance in REPRODUCIBILITY.md)
results/        written by the scripts (gitignored); numbers of record are RESULTS.md
tests/          fast invariants
```

## Honest caveats

- Differences below about .005 SR-MAE are inside the paired 95% intervals
  at 96 evaluations per cell; the tables mark which cells are.
- SC-IRT's advantage concentrates where evaluation is hard: small
  calibration panels (K_cal <= 10) and low-to-medium budgets. With a rich
  panel at B = 110 (61% of the per-draw bank executed) representative random
  sampling wins — that saturation column is part of the result.
- The RelGraph encoder ships as predictions; its training code depends on
  Bench2Drive raw rollouts (not redistributable) and is staged for a
  separate release; everything downstream of the predictions (US scoring,
  the UPS prior) is reproducible here, and the hand-crafted descriptor
  baselines in Table 3A tie it.
- The navhard panel is dominated by a few teams' submission sweeps;
  near-duplicate submissions can appear on both sides of the planner split
  (this affects every method identically). See RESULTS.md for the caveats.

## Citation

```bibtex
(to appear)
```
