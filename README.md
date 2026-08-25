# SC-IRT — Scene-Conditioned Item Response Theory for Closed-Loop Driving Evaluation

Official code release. Treat a driving scenario as a test item and a planner
as an examinee: fitting an IRT model to the pass/fail panel recovers a
per-scenario difficulty, learning to predict that difficulty from the scene
makes it available for scenarios no planner has driven, and an
uncertainty-aware adaptive tester certifies a new planner's benchmark success
rate to a chosen precision at a fraction of the rollouts.

Everything in the paper runs from this repository: the data ships in
`data/` (~7 MB, no external downloads), and every experiment entry point ends
by asserting the published numbers (`anchors OK`).

- [[Protocol]](PROTOCOL.md) — the unified specification: generative model,
  split, acquisition, stopping, metrics.
- [[Results]](RESULTS.md) — the numbers of record, one section per table.
- [[Reproducibility]](REPRODUCIBILITY.md) — anchors ledger, RNG registry,
  environment, determinism notes.

## The one model and the one split

```
theta_j ~ N(0, 1)                         planner ability
b_i | x_i ~ N(b_tilde(x_i), sigma^2)      scenario difficulty given the scene
P(pass)   = sigmoid(theta_j - b_i)        all-1PL, single equation
```

Split: 13/3 planners x 36/8 scenario types, 16 Monte-Carlo draws
(`scirt/splits.py`). Within a draw the three regimes share the partition:

|                    | train planners (13) | held-out planners (3) |
|--------------------|---------------------|-----------------------|
| train types (36)   | A: calibration      | **UP** evaluation     |
| held-out types (8) | **US** evaluation   | **UPS** target        |

**US** predicts difficulty for unseen scenes (Tables 1-2). **UP** certifies a
new planner's full-bank success rate to +-eps on the calibrated bank
(Tables 3-4), selecting the next rollout by the SR-variance acquisition —
the item whose observation most shrinks the posterior variance of the very
quantity the stopping rule certifies. **UPS** composes the two: predict a new
planner on scenes with zero calibration responses (Table 6). Table 5 is the
paper's central claim — *CAT under calibration scarcity*: published
2PL/3PL adaptive testers silently break when the item bank is calibrated
from a handful of examinees, while the Rasch-plus-marginalisation design
degrades gracefully.

## Getting started

```bash
pip install -e .            # numpy, scipy, scikit-learn, torch (see pyproject)
pytest tests/ -q            # fast invariants, no GPU needed for the unified ones

# Reproduce the paper, one table per script (GPU, minutes each):
python experiments/run_up_main.py        # Table 3   (~8 min)
python experiments/run_up_baselines.py   # Table 4ab (~35 min)
python experiments/run_atlas_bridge.py   # Table 4e  (~10 min)
python experiments/run_scarcity.py       # Table 5   (~25 min)
python experiments/run_us.py             # Tables 1-2 (~40 min)
python experiments/run_ups.py            # Table 6   (~25 min)
python experiments/run_sel_diversity.py  # adaptivity diagnostic (~8 min)
python experiments/run_plugin_ablation.py # Table 5b: uncertainty-propagation toggle (~15 min)
python experiments/run_factorial_2x2.py   # Table 3b: selection x stopping factorial (~20 min)
python experiments/run_random_fpc.py      # IRT-free certification reference (seconds, CPU)
python experiments/run_budget_frontier.py # Table 4g: efficiency frontier, 12 methods (~45 min)
python experiments/run_model_adequacy.py  # psychometric adequacy diagnostic (~10 min)
```

Each script prints its table and finishes with `anchors OK` — an assertion
against the published numbers, so a silent environment drift fails loudly.

## What is in the box

```
scirt/          the library (see PROTOCOL.md for the maths)
experiments/    one entry point per paper table + build_data.py (provenance)
data/
  matrices/     16 x 220 pass/fail response panel; route -> scenario type
  features/     six scene-descriptor sets (cmdkin, scenparamz, gtrisk, ...)
  b2d/          traffic-feature table and kin/density baselines
  encoder/      per-run encoder predictions for the unified split (d64/d96,
                3 seeds each — single runs; prediction ensembling is banned)
                + the trajectory tensors to retrain from scratch
tests/          fast invariants (unified + legacy)
train/          encoder training (Bench2Drive annotations -> tensors -> runs)
```

The response panel: 16 open-source end-to-end planners x 220 Bench2Drive
routes (3,476 of 3,520 cells observed). Scenario types: 44. All derived data
was packaged by `experiments/build_data.py`, which re-verifies each artifact
against its source (the route->type map is checked against the raw CARLA
checkpoint JSONs it came from).

## Honest caveats

- The calibration panel is 13 planners per draw. That scarcity is the point
  of the paper, and Table 5 measures what it does to every method — including
  ours, which degrades too (gracefully; see the J_cal = 4 row before quoting
  headline numbers).
- Difficulty is calibrated against *this* planner population; a different
  population defines a correlated but not identical difficulty.
- Coverage is reported as count/48 next to every fraction; with 48 evaluation
  runs the resolution is ~2%.
- Adaptive selection wins at small budgets; at large fixed budgets a
  representative random sample estimates the mean better. The fixed-budget
  panel of Table 4 shows both regimes.
- The encoder rows use single runs; seeds are summarised as metric mean +- SD.
  Prediction ensembling is banned project-wide (including the ablations).

## Legacy layer

The pre-unified snapshot (220-route bank, leave-one-planner-out, 2PL bank,
target-EIG) is retained as `scirt/api.py` + friends with its own tests — it
produced the paper's robustness appendix and stays reproducible. New work
should use the unified layer. The even older 219-route pipeline lives on the
`full-reproduction` branch.

## Citation

```bibtex
(to appear)
```
