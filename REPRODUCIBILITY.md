# Reproducibility

## Environment

Python 3.9, numpy 1.23, scipy 1.13-era API (verified on 1.10/1.13), scikit-learn 1.2,
torch 2.0. A CUDA GPU is required as shipped — the 1PL/2PL calibration and
the LLTM+e fit run on `device='cuda'` — though any single GPU suffices (the
RelGraph encoder itself ships as predictions). All experiment scripts are
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
| `run_up_frontier.py --merge` | Table 1: 4 SC-IRT cells, Fluid/metabench/Random cells, SC-IRT macro .0365 |
| `run_tau_calibration.py --merge` | tau_hat per (K_cal, target) + LOO expected-budget table |
| `run_adaptive.py --merge` | Table 2 adaptive SR-MAE cells |
| `run_ablation.py --merge` | component table at K7/K10 B30 and K16 B55 (full and off-arms) |
| `run_us.py` | Table 3A: null, canonical, two-stage, RelGraph mean, sigma_hat, PV share, descriptor ablation |
| `run_ups.py` | Table 3B: representative MAE cells per policy (tol .003) |
| `run_navhard.py --merge` | Table 4 cells |
| `run_readout_dropin.py` | the drop-in deltas |
| `tests/` | split pinning (draw 0), panel shape 22 x 220 / 4796 cells, r1_pick determinism, IES definition |

## RNG registry

All randomness is `np.random.RandomState` with fixed formulas; nothing reads
global seed state except the top-of-script `np.random.seed(0); torch.manual_seed(0)`
(which covers sklearn k-means inside tinyBenchmarks).

| purpose | seed formula |
|---|---|
| planner/type split (draw = 0..15) | `RandomState(1000 + draw)`: first 6 of 22 planners, then 8 of 44 types, from the same stream |
| K_cal subsample from the 16 calibration planners | `RandomState(9000 + 100*draw + 10*K_cal)` |
| Random / Random-strat rollout order (per evaluation) | `RandomState(100 + 20*draw + planner_id)` |
| leave-one-planner-out Random order (tau calibration) | `RandomState(700 + 20*draw + j)` |
| US plausible-values draw | `RandomState(400 + draw)` |
| model-adequacy simulation | `RandomState(2000 + draw)` |
| paired bootstrap | `RandomState(0)`; 4,000 resamples over 16 draw-clusters of 6 planner evaluations (10,000 for the US rho cluster bootstrap) |

Policy: no prediction or run ensembling anywhere — single-run numbers with
across-draw dispersion only. The RelGraph rows report 3 independent runs as
mean +- SD, never an averaged prediction.

## Data provenance

- `data/matrices/b2d_e2e22_response_matrix.csv` — 22 planners x 220
  Bench2Drive routes, pass/fail from official closed-loop evaluation; the
  per-planner sources and re-driven cells are recorded in the research-side
  build report, and `experiments/build_data.py` verifies the shipped copies
  against the research tree. The 16-planner matrix it extends is kept
  alongside.
  `SCIRT_RESPONSE_CSV` overrides the panel for new matrices.
- `data/features/` — per-route descriptor sets (cmdkin, scenparamz, gtrisk,
  ...) computed from Bench2Drive route definitions and expert logs only
  (nothing from the evaluated planners' rollouts).
- `data/encoder/relgraph_r2_s{0,1,2}.npz` — per-draw difficulty predictions
  from the RelGraph R2 scene encoder (ego, agent and lane tokens; R-GCN
  lane-lane message passing, agent-lane cross-attention over
  relative-geometry relations and an ego-route relation; d = 64), trained
  per draw on the 36 calibration types
  with the shared-sigma epsilon-marginalised objective and predicting the 8
  evaluation types out-of-fold; three seeds, one file per run. Training code
  depends on Bench2Drive raw rollouts and is staged for a separate release.
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
python experiments/run_tau_calibration.py --merge      # tau_hat ledger
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
