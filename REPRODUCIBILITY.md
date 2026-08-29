# Reproducibility

## Anchors ledger

Every table-producing entry point ends with hard assertions against the
published numbers and prints `anchors OK`; the appendix diagnostics,
`eval_us_predictions.py` and `make_figures.py` print without asserting. If an
environment change breaks the computation, the run fails loudly instead of
drifting silently. Tolerances are stated in each script (typically the last
published digit).

| script | asserted anchors |
|---|---|
| `run_up_frontier.py` | SC-IRT J13 B30/40/60/80 = .0388/.0342/.0213/.0222, J7 B30 .0423, J10 B60 .0260 (+-.002); macro .0354; Fluid J13 B30 .0379; metabench J13 B80 .0217; Random+IRT J13 B60 .0345 |
| `run_adaptive.py` | fixed-B SC-IRT J13 as above (+-.002); Table 2 design A SC-IRT J13 SR-MAE .0384 (target 30) / .0228 (target 60) (+-.003) |
| `run_tau_calibration.py` | writes `results/tau_hat.json`; all 16 draws present |
| `run_k_calibration.py` | J13 LOO loss range over K in [15,30] < .0015 and K = 40 penalty > .002; every J range < .005 |
| `run_ups.py` | naive .1282 / Random .1194 / theta-EIG .1034 / Localize .1083 (+-.003) |
| `run_readout_dropin.py` | Fluid + Rasch readout J13 B30 .0387, J7 B30 .0421; Total-Fisher + readout J13 B60 .0266; metabench + readout J13 B80 .0236 (+-.002) |
| `run_us.py` | null 0.710/.207; LLTM+e 0.764/+0.510; two-stage 0.760/+0.487; sigma-hat 0.593; PV share 16.4%; hc +0.486; kin +0.428; encoder d64 AUROC .753, rho +0.469 |
| `eval_us_predictions.py` | reproduces the shipped encoder summaries 0.753 / .189 / +0.469 |
| `run_model_adequacy.py`, `run_calibration_stability.py` | diagnostics (no anchor) |

`tests/test_unified_anchors.py` pins everything cheap and CPU-deterministic
(panel invariants, the draw-0 split, grid constants, readout/risk/acquisition
determinism, the prefix property of the static orders).

## RNG registry

Random streams are protocol constants. Fresh `np.random.RandomState`
instances are created at the documented points; nothing shares state across
arms, so run order between arms is irrelevant.

| stream | seed | used for |
|---|---|---|
| split | 1000 + draw | planner / type holdout (`scirt/splits.py`) |
| calibration-panel subsample | 9000 + 100*draw + 10*J_cal | J_cal < 13 |
| held-out random order | 100 + 20*draw + planner | Random / Random-strat rows, Random bank order, UPS random probes |
| LOO random order | 700 + 20*draw + calibration planner | `run_tau_calibration.py` |
| plausible values | 400 + draw | b-hat posterior draws in `run_us.py` (M = 20) |
| adequacy held-out cells | 2000 + draw | 10% held-out cells + planner halves |
| K-means | random_state = 0 | tinyBenchmarks / AnchorPoints |
| global | numpy 0 / torch 0 | set at the top of every entry point |

Calibration kernels are RNG-free: zeros initialisation, Adam lr 0.05, 800
iterations (LLTM+e: 1500), fixed regularisation, theta-mean centring. The
acquisition rules, the readout and the stopping rule are deterministic
given the fitted parameters; no Monte Carlo is used anywhere in the UP /
UPS pipeline (the posterior risk is closed-form).

## Protocol constants estimated on the calibration panel

- K = 20 (`scirt.acquisition.K_LOCALIZE`): leave-one-planner-out
  simulation, `run_k_calibration.py`, flat over [15, 30].
- tau_hat(draw, J_cal, method, target budget): `run_tau_calibration.py`,
  cost-matched on the calibration panel; medians on Bench2Drive: SC-IRT
  .042 (target 30) / .027 (target 60) at J_cal = 13.

## Environment and determinism

The anchors were produced with Python 3.9, torch on a single CUDA GPU,
numpy / scipy / scikit-learn as pinned in `pyproject.toml`. CUDA float32
Adam reductions are not bit-identical to CPU (up to ~2e-3 on a fitted
difficulty vector); the anchor tolerances absorb that. The kernels accept
`device='cpu'`; expect the same values within the stated tolerances, not
bit-equality. The heavy scripts accept `--seeds lo hi` shards and
`--merge`; shards are independent (fresh RNG streams per draw), so sharding
does not change any number.

## Data provenance

`experiments/build_data.py` packaged every artifact in `data/` from the
research tree, checking md5 equality where a source copy exists and
re-deriving the route->scenario-type map from the raw CARLA checkpoint JSONs.
The response panel is the raw 16 x 220 pass/fail matrix; nothing in `data/`
is a model output except `data/encoder/*` — per-run out-of-fold difficulty
predictions of the trajectory encoder on the unified split (d64/d96 x seeds
0-2, single runs — prediction ensembling is banned project-wide), shipped so
Table 3A reproduces without retraining. Full retraining:
`train/train_encoder_unified.py` with `data/encoder/b2d_tensors.npz`
(a full seed-0 / d64 retrain reproduced the shipped artifact bit-for-bit on
the development GPU; elsewhere expect rho within about +-0.01). Bench2Drive
rollout annotations themselves are not redistributed.

## Known gaps

- The NavSim panel (Table 4) is computed from a 17 x 12,146 out-of-fold
  artifact set not yet packaged here; the numbers of record are in
  RESULTS.md.
- Table 3A scores the point prediction b_tilde. `run_us.py` also prints the
  LLTM+e row with the residual sigma marginalised in the cell probability
  (AUROC 0.764 / Scene-MAE .177 / rho +0.510 — identical to the plug-in row
  at the reported precision); the encoder path has no sigma and is plug-in
  only.
