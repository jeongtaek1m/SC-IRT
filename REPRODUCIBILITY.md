# Reproducibility

## Environment

Python 3.9, numpy 1.23, scipy 1.13-era API (verified on 1.10/1.13), scikit-learn 1.2,
torch 2.0. A CUDA GPU is required as shipped — the 1PL/2PL calibration
runs on `device='cuda'` — though any single GPU suffices (the RelGraph
encoder itself ships as predictions). All experiment scripts are
deterministic given the RNG registry below; a full 16-draw run of the
heaviest script is a few hours on one GPU, and every heavy script accepts
`--seeds lo hi` sharding plus `--merge`.

## Anchors ledger

Every table-producing entry point ends by asserting its published numbers
against the freshly computed ones and printing `anchors OK`. If an assert
fires, the environment does not reproduce the paper — do not report numbers
past a failed anchor.

| script | asserts |
|---|---|
| `run_up_frontier.py --merge` | Table 1: 4 SC-IRT cells, Fluid / Random-strat / Random cells, SC-IRT macro .0307 |
| `run_tau_calibration.py --merge` | risk-scale medians c per K_cal (SC-IRT) + one matched-cost tau_hat median |
| `run_adaptive.py --merge` | fixed-t track errors of SC-IRT / Random / Fluid at representative (K_cal, t) |
| `run_ablation.py --merge` | full and each off-arm at representative cells |
| `run_us.py` | Table 3A: null, the two hand-crafted rows, RelGraph 3-run means |
| `run_ups.py` | Table 3B: representative MAE cells per policy (tol .003) |
| `run_navhard.py --merge` | Table 4 cells |
| `run_readout_dropin.py` | the drop-in cells |
| `tests/` | grids and index identities, exact-posterior and testlet invariants, split pinning (draw 0), panel shape, r1_pick determinism, IES definition |

## Protocol constants (`scirt/curves.py`, `scirt/calibration.py`)

| constant | value |
|---|---|
| theta grid THG | 241 points on [-6, 6]; prior N(0, 1) |
| extended axis XG | 361 points on [-9, 9] (theta + u lookups) |
| difficulty grid BG | 801 points on [-10, 10] (exact item posterior) |
| testlet grid UG | 61 points on [-3, 3] |
| calibration priors | theta ~ N(0, 1); b ~ N(0, sigma_b^2), sigma_b by empirical Bayes on {.5, .75, 1, 1.5, 2, 3}; log a ~ N(0, .5^2); logit c ~ N(-2.2, 1) |
| testlet SD grid | sigma_g on {0, .25, .5, .75, 1, 1.25, 1.5, 2} by profile marginal likelihood |
| optimiser | Adam lr .05, 800 iterations, zero init, theta-mean centring |
| risk scale | c = 90th percentile of realised / predicted error over LOO trajectories, t in [10, 110] |
| acquisition ties | score rounded to 1e-10, lowest bank index (`TIE_DECIMALS`) |
| random-policy rows (Tables 1, 4) | expected error over `NREP = 5` orders, seeds `100 + K*draw + planner_id + 100000*rep` |
| stopping targets | eps in {.03, .05}; matched-cost appendix targets 30 / 55 rollouts |

## RNG registry

All randomness is `np.random.RandomState` with fixed formulas (sklearn
k-means inside tinyBenchmarks passes `random_state=0` explicitly); the
top-of-script `np.random.seed(0); torch.manual_seed(0)` is belt-and-braces.

| purpose | seed formula |
|---|---|
| planner/type split (draw = 0..15) | `RandomState(1000 + draw)`: first 6 of 22 planners, then 8 of 44 types, from the same stream |
| K_cal subsample from the 16 calibration planners | `RandomState(9000 + 100*draw + 10*K_cal)` |
| Random / Random-strat rollout order (per evaluation) | `RandomState(100 + K*draw + planner_id)`, K = number of planners in the panel (22; 87 on navhard) — injective |
| leave-one-planner-out Random order (tau calibration) | `RandomState(700 + K*draw + j)` |
| navhard panel (Table 4) | same formulas on the 87 submissions: 6 evaluation planners `RandomState(1000 + draw)`, K_cal subsample `RandomState(9000 + 100*draw + 10*K_cal)`, Random order `RandomState(100 + 87*draw + planner_id)`; no scenario types |
| model-adequacy simulation | `RandomState(2000 + draw)` |
| paired bootstrap | `RandomState(0)`; 4,000 resamples of the unique planner ids (planner clusters, 22 on Bench2Drive), every evaluation of a resampled planner weighted equally |

Policy: no prediction or run ensembling anywhere — single-run numbers with
across-draw dispersion only. The RelGraph rows report 3 independent runs as
mean +- SD, never an averaged prediction.

## Data provenance

- `data/matrices/b2d_e2e22_response_matrix.csv` — 22 planners x 220
  Bench2Drive routes, pass/fail from official closed-loop evaluation; the
  per-planner sources and re-driven cells are recorded in the research-side
  build report; `experiments/build_data.py` md5-checks the 16-planner matrix
  against the research tree and `tests/` pins the 22-planner panel's shape
  (22 x 220, 4,796 cells). The 16-planner matrix it extends is kept
  alongside.
  `SCIRT_RESPONSE_CSV` overrides the panel for new matrices.
- `data/features/` — per-route descriptor sets (cmdkin, gtrisk, routegeom,
  ...) used as US baselines. They are computed from a fixed probe rollout
  per route, so they are probe-conditioned but contain nothing from the
  evaluated planners' rollouts. The scenario-definition parameters
  (scenparamz) that earlier versions used were removed: they are the
  benchmark's own construction values, not observable scene content.
- `data/encoder/relgraph_r2_s{0,1,2}.npz` — per-draw difficulty predictions
  from the RelGraph R2 scene encoder (ego, agent and lane tokens; R-GCN
  lane-lane message passing, agent-lane cross-attention over
  relative-geometry relations and an ego-route relation; d = 64), trained
  per draw on the 36 calibration types
  with the shared-sigma epsilon-marginalised objective and predicting the 8
  evaluation types out-of-fold; three seeds, one file per run. Each file
  also carries `draw{r}_sigma`, the shared residual SD the encoder learned
  on that draw's calibration block (the UPS prior width).
  `relgraph_r2_{noroute,sroute,sa2l}_s{0,1,2}.npz` are the structural
  controls of Table 3A(b): the same architecture, recipe and seeds with the
  ego-route relation removed / the route correspondence shuffled / the
  agent-lane correspondence shuffled (shuffle seed = model seed). Training code
  depends on Bench2Drive raw rollouts and is staged for a separate release.
- `data/live/risk_scale.json` — cached risk scales c of `scirt.live.LiveEvaluator`,
  keyed by a bank fingerprint (planner set, route list, iterations) and
  validated against a digest of the responses; the shipped entries are the
  22-planner bank (c = 2.36) and the three 21-planner banks of the dry-runs
  in RESULTS.md. `calibrate_risk()` recomputes an entry when the matrix
  changes (~22 GPU calibrations).
- `data/navhard/navhard_binary_panel.npz` — NAVSIM navhard leaderboard
  (two-stage pseudo-closed-loop EPDMS): 115 submissions scraped from the
  public leaderboard, near-duplicate resubmissions collapsed to 87 unique
  submissions from 25 teams, 225 scored units, pass = EPDMS >= 0.5.
  Caveats: the panel is dominated by a few teams' submission sweeps and
  near-duplicate variants can appear on both sides of the planner split;
  this affects every compared method identically but limits how far the
  panel can be read as 87 independent planners.

## How to re-run from scratch

```bash
pytest tests/ -q
for lo in 0 4 8 12; do python experiments/run_up_frontier.py --seeds $lo $((lo+4)) & done; wait
python experiments/run_up_frontier.py --merge          # Table 1, anchors OK
for lo in 0 2 4 6 8 10 12 14; do python experiments/run_tau_calibration.py --seeds $lo $((lo+2)) & done; wait
python experiments/run_tau_calibration.py --merge      # risk scale c (+ matched-cost tau_hat)
for lo in 0 4 8 12; do python experiments/run_adaptive.py --seeds $lo $((lo+4)) & done; wait
python experiments/run_adaptive.py --merge             # Table 2
for lo in 0 4 8 12; do python experiments/run_ablation.py --seeds $lo $((lo+4)) & done; wait
python experiments/run_ablation.py --merge             # component ablation
python experiments/run_us.py                           # Table 3A
python experiments/run_ups.py                          # Table 3B
for lo in 0 4 8 12; do python experiments/run_navhard.py --seeds $lo $((lo+4)) & done; wait
python experiments/run_navhard.py --merge              # Table 4
python experiments/run_readout_dropin.py
python experiments/run_model_adequacy.py
python experiments/run_calibration_stability.py
python experiments/make_figures.py
```
