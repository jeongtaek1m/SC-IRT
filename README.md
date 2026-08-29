# SC-IRT — Scene-Conditioned Item Response Theory for Closed-Loop Driving Evaluation

Official code release. Treat a driving scenario as a test item and a planner
as an examinee. SC-IRT separates *robust performance inference* from
*acquisition*: it propagates the uncertainty of a scene bank calibrated from
a handful of planners, then chooses which scenes to run according to the
quantity each evaluation regime needs — and it learns scene difficulty from
the scene itself so that scenarios no planner has driven can be scored.

Everything in the paper runs from this repository: the data ships in
`data/` (~2.3 MB, no external downloads), and every table-producing entry
point ends by asserting the published numbers (`anchors OK`).

- [[Protocol]](PROTOCOL.md) — the specification: models, split, acquisition, stopping, metrics.
- [[Results]](RESULTS.md) — the numbers of record, one section per table.
- [[Reproducibility]](REPRODUCIBILITY.md) — anchors ledger, RNG registry, environment.

## The method in four lines

```
evaluation model    y_ij ~ Bernoulli(sigmoid(theta_j - b_i)),   b_i | A ~ N(b_hat_i, s_i^2)   (Rasch, item uncertainty kept)
acquisition model   a joint 2PL fit of the same panel — used only to choose scenes, never to score
unseen scenes       b_i | x_i ~ N(w^T x_i, sigma^2)  (LLTM+e)  or a trajectory encoder
stopping            R1(D_t) = E[ |S - S_hat_t| | D_t ] <= tau     (posterior L1 risk of the reported success rate)
```

Contributions, in the order the paper makes them:

- **C1 Uncertain item-bank inference.** Small planner panels make every
  scene difficulty uncertain; SC-IRT propagates p(b_i | A) into the item
  curves instead of fixing b_hat_i. The Rasch readout is a drop-in for any
  selector (`run_readout_dropin.py`).
- **C2 Target-aligned acquisition.** *Acquire for the quantity that must
  generalise.* UP (full-bank success rate): localize the new planner for
  K = 20 rollouts with 2PL Fisher, then cover the bank in population-Fisher
  order. UPS (an ability to transport to unseen scenes): theta-EIG under the
  evaluation model. US: no acquisition.
- **C3 Risk-based adaptive stopping.** Stop when the posterior expected
  absolute error of the reported success rate is below tau; thresholds are
  fixed on the calibration panel, never on held-out planners.
- **C4 Unseen-scene difficulty transfer.** x -> b enables US and UPS.

## The split

13/3 planners x 36/8 scenario types, 16 Monte-Carlo draws
(`scirt/splits.py`). Within a draw the three regimes share the partition:

|                    | train planners (13) | held-out planners (3) |
|--------------------|---------------------|-----------------------|
| train types (36)   | A: calibration      | **UP** evaluation     |
| held-out types (8) | **US** evaluation   | **UPS** target        |

## Getting started

```bash
pip install -e .[figs]      # numpy, scipy, scikit-learn, torch (+ matplotlib for the figures)
pytest tests/ -q            # fast invariants, CPU

# Reproduce the paper (GPU). Heavy scripts accept --seeds lo hi shards + --merge.
python experiments/run_up_frontier.py       # Table 1 + full budget grids (~1 h)
python experiments/run_tau_calibration.py   # calibration-fixed stopping thresholds (~1.5 h)
python experiments/run_adaptive.py          # Table 2 + cost-error figure data (~1.5 h)
python experiments/run_k_calibration.py     # the localize budget K on the calibration panel (~1.5 h)
python experiments/run_us.py                # Table 3A + descriptor ablation (~40 min)
python experiments/run_ups.py               # Table 3B (~10 min)
python experiments/run_readout_dropin.py    # analysis: the Rasch readout under every selector (~40 min)
python experiments/run_model_adequacy.py    # appendix: evaluation-model adequacy (~10 min)
python experiments/run_calibration_stability.py   # appendix: calibration stability
python experiments/make_figures.py          # figures from the results jsons
```

Each table-producing script prints its table and finishes with `anchors OK`;
the two appendix diagnostics, `eval_us_predictions.py` and `make_figures.py`
print their numbers without an assertion.

## What is in the box

```
scirt/          the library (PROTOCOL.md has the maths)
experiments/    one entry point per paper table + build_data.py (provenance)
data/
  matrices/     16 x 220 pass/fail response panel; route -> scenario type
  features/     six scene-descriptor sets (cmdkin, scenparamz, gtrisk, ...)
  b2d/          traffic-feature table and kin/density baselines
  encoder/      per-run encoder predictions for the unified split (d64/d96,
                3 seeds each — single runs; prediction ensembling is banned)
                + the trajectory tensors to retrain from scratch
results/        written by the scripts (gitignored); the numbers of record are RESULTS.md
tests/          fast invariants
train/          train_encoder_unified.py — the paper's encoder recipe
```

The response panel: 16 open-source end-to-end planners x 220 Bench2Drive
routes (3,476 of 3,520 cells observed), 44 scenario types. All derived data
was packaged by `experiments/build_data.py`, which re-verifies each artifact
against its source.

## Honest caveats

- Differences below about .005 SR-MAE are inside the paired 95% intervals
  at 48 evaluations per cell; Table 1 marks which cells are.
- At large fixed budgets (B >= 100) representative static subsets and
  random sampling estimate the mean better than any adaptive rule, and with
  a 4-planner calibration panel the bank is too wrong for model-based
  extrapolation at any budget above ~30 (`figs/fig_jb_map`). The J_cal x B
  map is part of the result, not a footnote.
- The first K = 20 picks of the UP rule are the Fluid selection rule; the
  cover phase is the Total-Fisher static order. What is new is the
  inference layer, the target-aligned two-phase design and its
  calibration-time constants, and the risk-based stop — not a new item
  selection criterion.
- Encoder rows use single runs; seeds are summarised as mean +- SD.

## Citation

```bibtex
(to appear)
```
