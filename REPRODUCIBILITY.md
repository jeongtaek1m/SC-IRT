# Reproducibility

## Environment

Python 3.9, numpy 1.23, scipy 1.13-era API (verified on 1.10/1.13), scikit-learn 1.2,
torch 2.0. The numbers of record were fitted on a CUDA GPU (any single GPU
suffices; the RelGraph encoder itself ships as predictions). The
calibration (`atdrive.calibration.DEVICE`) falls back to CPU when no GPU
is visible, which moves third decimals of the fitted parameters — enough
to flip a near-tied 2PL Fisher pick (see the anchors ledger), not enough
to move a table cell past its anchor tolerance. GPU entry points: the
compute mode (`--seeds`) of every sharded script, `run_us.py`,
`run_ups.py`, `run_readout_dropin.py`, `run_model_adequacy.py`,
`eval_us_predictions.py` and the live tool. CPU entry points: every
`--merge` / result-reading script, `run_policy_matrix.py`,
`run_ranking_quality.py`, `run_nuplan_zeroshot.py`, `run_ups_full.py`
(compute and merge), the figures and the tests. All experiment scripts are
deterministic given the RNG registry below; a full 16-draw run of the
heaviest script is a few hours on one GPU, and every heavy script accepts
`--seeds lo hi` sharding plus `--merge`. On a clone without the (gitignored)
shards, `--merge` reads the tracked merged json of record and re-asserts
its anchors in seconds; recomputing a shard reproduces the tracked shard
to <= 1e-12 except that the 2PL Fisher order can flip a near-tied pick
under GPU rounding (cell means unaffected at the anchor tolerances).

## Anchors ledger

Every table-producing entry point ends by asserting its published numbers
against the freshly computed ones and printing `anchors OK`. If an assert
fires, the environment does not reproduce the paper — do not report numbers
past a failed anchor.

| script | asserts |
|---|---|
| `run_up_frontier.py --merge` | Table 1: 5 ATDrive cells, Fluid / Random-strat / Random cells, ATDrive macro .0262 |
| `run_tau_calibration.py --merge` | risk-scale medians c per K_cal (ATDrive) + one matched-cost tau_hat median (from risk_cal.json / tau_hat.json on a clone) |
| `run_adaptive.py --merge` | fixed-t track errors of ATDrive / Random / Fluid at representative (K_cal, t); "what each budget buys" rows at K4 B30, K8 B55, K12 B30, K12 B165 |
| `run_ablation.py --merge` | full and each off-arm at representative cells |
| `run_system_ablation.py` | full-system ablation: fixed-budget and risk-target cells of the five arms |
| `run_system_comparison.py --merge` | complete-system comparison: ATDrive rows (= Table 2), ATLAS tau cells, Fluid B=100 / B=match / SE<=delta* cells |
| `run_cat_objective.py --merge` | the pooled-c CAT-objective cells (Delta-R1 / Fisher / Random at eps=.05, the matched ability-SD stop) at K4 / K8 / K12 |
| `run_policy_matrix.py` | policy matrix: ATDrive rows (= Table 2), the degenerate ATLAS rows, the matched rows, the IES reference .0393 at K12; factorial: C-B, F-E, G-B exclude zero, B-A, E-A, F-C include zero, fixed-budget Fisher - Delta-R1 at B=55 / 78 |
| `run_ranking_quality.py` | 13 insertion cells, the \|Delta rank\| identity < 1e-9, mae_only in {5, 6} with the boundary cell, mae_rank_exact .0203 (1417 of 2112), rank<=1 at K8, Fluid SE stop macro; co-estimated and within-draw rank-only cells, Table 1's pairwise column reproduced, K8 Fluid B=match +.0521 |
| `run_route_discrimination.py --merge` | 6 AUROC cells, macro AUROC / Brier of 3 orders, 36 drops, both rank-agreement rhos, pooled ATDrive - Random-strat spans 0 (n 751, 9 of 12), zero-rollout macro Brier of every order, Brier skill / AUROC gain of Random-strat > 0, common-set macro AUROC and spread |
| `run_us.py` | Table 3A: null, the two hand-crafted rows, RelGraph 3-run means; Table 3A(b): the rho of the four controls |
| `run_ups.py` | Table 3B: representative MAE cells per policy (tol .003), incl. the Delta-R1 cells under the speed-ablated prior |
| `run_ups_full.py --merge` | 12 full-SR cells (incl. the scene-free acquisition arm), 5 Table 3B cells, 2 AUROC cells; the null: scene-prior deltas include 0 and are < .0025 in the readout and in the acquisition |
| `run_nuplan_zeroshot.py` | panel constants, oracle / arm / single-seed null point estimates, the matched three-seed-mean T_null and p, the exact permutation counts, the paired-contrast CI signs, the whole-panel Spearman and its permutation p |
| `run_model_adequacy.py` | held-out NLL of 1PL / 2PL / 3PL and the split-half reliability of log a on the UP bank |
| `run_readout_dropin.py` | the drop-in cells (incl. AnchorPoints K12 B55 / B110) |
| `tests/` | grids and index identities, exact-posterior and testlet invariants, split pinning (draw 0), panel shape, r1_pick determinism, IES definition, the entry-point registry |

## Protocol constants (`atdrive/curves.py`, `atdrive/calibration.py`)

| constant | value |
|---|---|
| theta grid THG | 241 points on [-6, 6]; prior N(0, 1) |
| extended axis XG | 361 points on [-9, 9] (theta + u lookups) |
| difficulty grid BG | 801 points on [-10, 10] (exact item posterior) |
| testlet grid UG | 61 points on [-3, 3] |
| calibration priors | theta ~ N(0, 1); b ~ N(0, sigma_b^2), sigma_b by empirical Bayes on {.5, .75, 1, 1.5, 2, 3}; log a ~ N(0, .5^2); logit c ~ N(-2.2, 1) |
| testlet SD grid | sigma_g on {0, .25, .5, .75, 1, 1.25, 1.5, 2} by profile marginal likelihood |
| optimiser | Adam lr .05, 800 iterations, zero init, theta-mean centring |
| risk scale | c = 90th percentile of realised / predicted error over LOO trajectories, t in [10, bank size] (the LOO trajectories run through the whole bank; the live tool's cached c uses t in [10, 110]) |
| acquisition ties | score rounded to 1e-10, lowest bank index (`TIE_DECIMALS`) |
| random-policy rows (Table 1) | expected error over `NREP = 5` orders, seeds `100 + K*draw + planner_id + 100000*rep` |
| stopping targets | eps in {.03, .05}; matched-cost appendix targets 30 / 55 rollouts |

## RNG registry

All randomness is `np.random.RandomState` with fixed formulas (sklearn
k-means inside tinyBenchmarks passes `random_state=0` explicitly); the
top-of-script `np.random.seed(0); torch.manual_seed(0)` is belt-and-braces.

| purpose | seed formula |
|---|---|
| planner/type split (draw = 0..15) | `RandomState(1000 + draw)`: first 4 of 16 planners, then 8 of 44 types, from the same stream |
| K_cal subsample from the 12 calibration planners | `RandomState(9000 + 100*draw + 10*K_cal)` |
| Random / Random-strat rollout order (per evaluation) | `RandomState(100 + K*draw + planner_id)`, K = number of planners in the panel (16) — injective |
| leave-one-planner-out Random order (tau calibration) | `RandomState(700 + K*draw + j)` |
| policy-matrix trajectories (`run_cat_objective.py`) | evaluation Random order `RandomState(100 + K*draw + slot)`, slot = the 0-3 index of the held-out planner in its draw (a different permutation from Table 2's Random row); LOO Random order `RandomState(100 + K*(700 + K_cal) + planner_id)` |
| ATLAS-style top-5 randomesque draw (`run_system_comparison.py`) | `RandomState(5000 + 1000*draw + 100*K_cal + planner_id)` |
| UPS-full Random probe order | `RandomState(100 + K*draw + planner_id)` (as `run_ups.py`) |
| nuPlan zero-shot | random q% subsets `default_rng(0)`; per-arm-seed log bootstrap `default_rng(1)` (1,000); paired arm-minus-null log bootstrap `default_rng(1)` (2,000) |
| model-adequacy held-out cells and planner halves | `RandomState(2000 + draw)` |
| paired bootstrap | `RandomState(0)`; 4,000 resamples of the unique planner ids (planner clusters, 16 on Bench2Drive), every evaluation of a resampled planner weighted equally; `run_policy_matrix.py` and `run_ranking_quality.py` are deterministic on the saved trajectories apart from this bootstrap |

Policy: no prediction or run ensembling anywhere — single-run numbers with
across-draw dispersion only. The RelGraph rows report 3 independent runs as
mean +- SD, never an averaged prediction.

## Data provenance

- `data/matrices/b2d_e2e16sel_response_matrix.csv` — the panel of record:
  16 of the 22 planners below, one per model family, spread over the
  ability range (selection rule in PROTOCOL section 1).
- `data/matrices/b2d_e2e22_response_matrix.csv` — 22 planners x 220
  Bench2Drive routes, pass/fail from official closed-loop evaluation; the
  per-planner sources and re-driven cells are recorded in
  `data/matrices/b2d_e2e22_build_report.json`; `tests/` pins the 16-planner
  panel's shape (16 x 220, 3,482 cells). The 16-planner panel of record was
  selected from it (PROTOCOL section 1).
  `ATDRIVE_RESPONSE_CSV` overrides the panel for new matrices.
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
  per draw on the 36 calibration types of the 12 calibration planners
  with the shared-sigma epsilon-marginalised objective and predicting the 8
  evaluation types out-of-fold; three seeds, one file per run. Each file
  also carries `draw{r}_sigma`, the shared residual SD the encoder learned
  on that draw's calibration block (the UPS prior width).
  `relgraph_r2_{noroute,sroute,sa2l}_s{0,1,2}.npz` are the structural
  controls of Table 3A(b): the same architecture, recipe and seeds with the
  ego-route relation removed / the route correspondence shuffled / the
  agent-lane correspondence shuffled (shuffle seed = model seed);
  `relgraph_r2_nospeed_s{0,1,2}.npz` is the channel control with the
  ego-speed channel removed from both ego paths (the arm that clears the
  nuPlan shuffle null; it also drives the speed-ablated repeat of Table 3B,
  `results/ups_nospeed.json`). All 15 files are exported by
  `experiments/build_data.py` from the run outputs of the 16-planner RelGraph
  training harness (`relgraph_e16sel`: `r2_b2d_s{k}.npz` and the four control
  runs); the harness calibrates its training-fold abilities with
  `atdrive.calibration.calibrate_dense` from this package, so the encoder is
  trained against the release's own Rasch fit. The harness itself depends on
  Bench2Drive raw rollouts (not redistributable) and is staged for a separate
  release.
- `data/nuplan/val14_zeroshot.npz` — the nuPlan val14 zero-shot panel of
  `run_nuplan_zeroshot.py`: 584 scenarios (tokens, logs), the 11 x 584
  closed-loop score matrix with the planner names, its binarised failures,
  the per-scene failure rate and the response-calibrated difficulty b_ref,
  and the logged-ego predicted difficulty of every encoder arm and
  label-shuffle seed (C0e x 3, A2e x 3, C4r2n x 10, C4r2e x 10), exported by
  `experiments/build_data.py` from the encoder's stage-2 transfer outputs.
- `data/live/risk_scale.json` — cached risk scales c of `atdrive.live.LiveEvaluator`,
  keyed by a bank fingerprint (planner set, route list, iterations) and
  validated against a digest of the responses; the shipped entries are the
  22-planner bank (c = 2.36) and five 21-planner banks — the three of the
  dry-runs in RESULTS.md (c = 2.55 / 2.32 / 2.16) and two superseded VAD /
  LEAD-tfv6 banks (2.66 / 2.03) that no run reads. `calibrate_risk()`
  recomputes an entry when the matrix changes (~22 GPU calibrations; its LOO
  trajectories run 110 routes, PROTOCOL section 4).

## How to re-run from scratch

```bash
pytest tests/ -q
for lo in 0 4 8 12; do python experiments/run_up_frontier.py --seeds $lo $((lo+4)) & done; wait
python experiments/run_up_frontier.py --merge          # Table 1, anchors OK
for lo in 0 2 4 6 8 10 12 14; do python experiments/run_tau_calibration.py --seeds $lo $((lo+2)) & done; wait
python experiments/run_tau_calibration.py --merge      # risk scale c (+ matched-cost tau_hat)
for lo in 0 4 8 12; do python experiments/run_adaptive.py --seeds $lo $((lo+4)) & done; wait
python experiments/run_adaptive.py --merge             # Table 2 + "what each budget buys"
for lo in 0 4 8 12; do python experiments/run_ablation.py --seeds $lo $((lo+4)) & done; wait
python experiments/run_ablation.py --merge             # component ablation
# the switched arms of the full-system ablation and the testlet table (results/notestlet, results/pointcurves)
for sw in "ATDRIVE_NO_TESTLET=1 ATDRIVE_RESULTS_DIR=results/notestlet" "ATDRIVE_POINT_CURVES=1 ATDRIVE_RESULTS_DIR=results/pointcurves"; do
  for lo in 0 2 4 6 8 10 12 14; do env $sw python experiments/run_tau_calibration.py --seeds $lo $((lo+2)) & done; wait
  env $sw python experiments/run_tau_calibration.py --merge
  for lo in 0 4 8 12; do env $sw python experiments/run_adaptive.py --seeds $lo $((lo+4)) & done; wait
  env $sw python experiments/run_adaptive.py --merge
done
python experiments/run_system_ablation.py              # full-system ablation (reads the three trees above)
for lo in 0 2 4 6 8 10 12 14; do python experiments/run_system_comparison.py --seeds $lo $((lo+2)) & done; wait
python experiments/run_system_comparison.py --merge    # complete-system comparison
python experiments/run_ranking_quality.py              # ranking quality (reads results/syscmp.json)
for lo in 0 2 4 6 8 10 12 14; do python experiments/run_cat_objective.py --seeds $lo $((lo+2)) & done; wait
python experiments/run_cat_objective.py --merge        # policy-matrix trajectories (results/cat_objective.json)
python experiments/run_policy_matrix.py                # adaptive policies under one IRT + the factorial
for lo in 0 4 8 12; do python experiments/run_route_discrimination.py --seeds $lo $((lo+4)) & done; wait
python experiments/run_route_discrimination.py --merge # route-level discrimination (reads up_frontier.json, adaptive.json)
python experiments/run_us.py                           # Table 3A + 3A(b)
python experiments/run_ups.py                          # Table 3B (+ the speed-ablated prior)
for lo in 0 2 4 6 8 10 12 14; do ATDRIVE_DEVICE=cpu python experiments/run_ups_full.py --seeds $lo $((lo+2)) & done; wait
python experiments/run_ups_full.py --merge             # UPS on the full 220-route SR
python experiments/run_nuplan_zeroshot.py              # nuPlan val14 zero-shot (data/nuplan/val14_zeroshot.npz)
python experiments/run_readout_dropin.py
python experiments/run_model_adequacy.py
python experiments/make_figures.py; python experiments/make_icc_figure.py; python experiments/make_uncertainty_figure.py
```
