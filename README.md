# ATDrive — IRT-Based Adaptive Testing for Closed-Loop Driving Evaluation

Official code release. Treat a driving scenario as a test item and a planner
as an examinee. ATDrive is built around a single probabilistic object — an
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
  every scene difficulty uncertain; ATDrive keeps the conditional (exact under the Rasch conditional)
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
  and the realised error at the stop is reported.
- **C4 Scene-conditioned difficulty.** scene -> b via the RelGraph
  scene-graph encoder (learned per draw on the calibration types, shipped as
  per-run out-of-fold predictions), enabling unseen-scene (US) and joint
  (UPS) generalisation. The encoder of record, R2-noLane, HAS NO LANE GRAPH:
  ego, command and agent tracks only. The lane-carrying model is a control.

## The primary protocol

16-planner x 220-route Bench2Drive panel (one planner per model family,
chosen from 22 with complete records to cover the ability range evenly);
per draw (R = 16), 12 calibration : 4 evaluation planners. **UP runs on the
whole benchmark**: the new planner's bank is all 220 routes of 44 scenario types that the simulator
  completed for it (210-220; SR is the success rate over that recorded-route set), of 44 scenario
types, calibrated from the 12 calibration planners on the same routes; the
36 : 8 scenario-type hold-out is kept for US and UPS only. B = number of
routes rolled out (30/55/110/165 = 5 x {6, 11, 22, 33} = 14/25/50/75% of
the benchmark, multiples of the 5 routes per scenario type); calibration-panel sizes
K_cal in {4, 8, 12}. 64 evaluations per cell.

|                          | calibration planners (12) | evaluation planners (4) |
|--------------------------|---------------------------|-------------------------|
| calibration types (36)   | A: calibration            | **UP** evaluation       |
| evaluation types (8)     | **US** evaluation         | **UPS** target          |

UP spans both type rows: its bank is all 220 routes. Only US and UPS read
the 36 : 8 type split.

## Getting started

```bash
pip install -e .[figs]      # numpy, scipy, scikit-learn, torch (+ matplotlib)
pytest tests/ -q            # fast invariants, CPU

# Reproduce the paper. Each script is listed once with the RESULTS.md section it produces and
# asserts (`anchors OK`). Heavy scripts accept --seeds lo hi shards + --merge; on a clone,
# `--merge` with no shards re-asserts the anchors from the tracked results of record (CPU,
# seconds). Recomputing the shards, run_us / run_ups / run_readout_dropin / run_model_adequacy
# and the live tool need a GPU (the calibration falls back to CPU, which moves third decimals).
python experiments/run_up_frontier.py --merge        # Table 1 (fixed budgets)
python experiments/run_tau_calibration.py --merge    # risk scale c (+ matched-cost tau_hat)
python experiments/run_adaptive.py --merge           # Table 2 (risk-target stopping) + "what each budget buys"
python experiments/run_ablation.py --merge           # component ablation (one component off at a time)
python experiments/run_system_ablation.py            # full-system ablation (needs results/notestlet, results/pointcurves)
python experiments/run_system_comparison.py --merge  # complete-system comparison (ATLAS-style, Fluid-style, ATDrive)
python experiments/run_cat_objective.py --merge      # policy-matrix trajectories (step 1 of the policy table)
python experiments/run_policy_matrix.py              # adaptive policies under one IRT + the factorial
python experiments/run_ranking_quality.py            # ranking quality of the complete-system trajectories
python experiments/run_route_discrimination.py --merge  # route-level discrimination (after Table 1)
python experiments/run_readout_dropin.py             # readout drop-in (GPU)
python experiments/run_us.py                         # Table 3A + 3A(b) (GPU)
python experiments/run_ups.py                        # Table 3B + the two control priors (GPU)
python experiments/run_ups_full.py --merge           # UPS retargeted to the full 220-route SR
python experiments/run_nuplan_zeroshot.py            # nuPlan val14 zero-shot retrieval
python experiments/run_av_baselines.py --merge       # FST / GP-adaptive (AV-testing) re-implementations on the Table 1 protocol
ATDRIVE_NUPLAN_BUNDLE=data/nuplan/val14_full_zeroshot.npz python experiments/run_nuplan_zeroshot.py   # the full 1,118-token split
python experiments/run_model_adequacy.py             # model adequacy appendix (GPU)
python experiments/make_figures.py                   # figs/fig_cost_error, fig_kb_map
python experiments/make_icc_figure.py                # figs/fig_icc
python experiments/make_uncertainty_figure.py        # figs/fig_uncertainty
python experiments/eval_us_predictions.py [npz ...]  # score any US difficulty predictions; no args = the encoder of record (GPU)
# K_cal = 15 supplement: prefix any UP script with ATDRIVE_UP_LOO=1 ATDRIVE_KCALS=15 ATDRIVE_RESULTS_DIR=results/loo15 (RESULTS.md, K_cal = 15)

# Use it inside a real closed-loop evaluation (UP): pick routes, ingest outcomes, stop on risk
python tools/b2d_adaptive_eval.py --dry-run VAD --eps 0.03          # simulate from the matrix (GPU)
python tools/b2d_adaptive_eval.py --name my_planner --agent ... --agent-config ... --eps 0.03
```

## What is in the box

```
atdrive/        the library (PROTOCOL.md has the maths)
experiments/    one entry point per RESULTS.md section (listed above) + build_data.py (provenance)
tools/          b2d_adaptive_eval.py — ATDrive inside a real Bench2Drive evaluation (atdrive/live.py)
encoder/        the scene-encoder training harness (RelGraph R2 / R2-noLane), its data, the nuPlan stage-2 drivers
data/
  matrices/     16 x 220 pass/fail panel of record + the 22-planner matrix it was drawn from
  features/     scene-descriptor sets used as US baselines (cmdkin, gtrisk, ...)
  b2d/          traffic-feature table and kin/density baselines
  encoder/      per-run out-of-fold difficulty predictions + the learned residual SD
                per draw (3 independent runs; no ensembling) for the encoder of record,
                the lane-free RelGraph R2-noLane (relgraph_r2nolane_s*.npz), and for the
                six controls of Table 3A(b): the lane-carrying R2 (relgraph_r2_s*.npz)
                and noroute / sroute / sa2l / nospeed, and the speed ablation of the
                encoder of record itself (relgraph_r2nolane_nospeed_s*.npz)
  nuplan/       the 584-scenario nuPlan val14 panel (a log-availability subset of the 1,118-token split) and encoder predictions of the zero-shot test;
                val14_full_zeroshot.npz + full_val14/ the full-split panel (1,118 tokens, 10-planner score matrix, scenario / log / map table)
  live/         cached leave-one-planner-out risk scales for the live evaluator (keyed by bank)
results/        merged results of record (tracked; per-shard intermediates are not) — RESULTS.md quotes them
figs/           the figures the make_*.py scripts write
tests/          fast invariants
```

## Honest caveats

- Every 64-evaluation cell rests on n_eff = 16 planner clusters; stars are uncorrected paired
  95% intervals. Differences below about .005 SR-MAE are inside the paired 95% intervals
  at 64 evaluations per cell; the tables mark which cells are.
- ATDrive has the lowest error in 7 of 12 cells of Table 1 (macro .0262 vs
  .0293 for ATLAS-style, .0303 for Fluid and .0307 for type-stratified Random);
  the five it does not win are ties inside the intervals, the largest K12 B30,
  where Fluid is lower by .007. Random-policy rows are expected errors over five orders.
- Route-level AUROC on the routes an order did not buy cannot separate
  ATDrive from the type-stratified order (pooled +.0009, 95% CI
  [-.019, +.019]); once a zero-rollout predictor is subtracted the
  type-stratified order is ahead on both route-level scores, and with the
  evaluation set held fixed the five orders barely separate (RESULTS.md,
  route-level discrimination).
- At matched cost the stopping rule buys nothing: with the Delta-R1 order
  fixed, the risk stop and a fixed length of the same mean cost differ by
  -.001 [-.004, +.001] SR-MAE. What separates the adaptive policies is the
  selection rule (Fisher costs +.006 [+.001, +.012] at an identical budget);
  the risk stop's value is the error target it states, not a lower error
  (RESULTS.md, adaptive policies under one IRT).
- At the low budgets the SR number is not yet reportable even when the
  ranking is: at B = 30 only 36-44% of estimates land within 3 SR points
  while 90-94% of within-draw planner pairs are already ordered correctly.
  The stopping rule never selects a budget that small (it spends 70-129).
- The stopping rule buys no accuracy at matched cost (paired deltas -.0047 to +.0034 against a fixed
  budget of the same mean length); what it provides is a stopping time for an error target stated in
  SR units, with c fixed on calibration planners only. The calibration gap (mean
  realised error minus mean scaled risk at the stop) is negative in every
  cell, so the reported risk over-states the error it is bounding.
- Exact ties in the acquisition score (routes of one type with identical
  posteriors) are broken by bank order; this arbitrary but documented
  choice; breaking exact ties at random moves the K12 B30 cell from .0477 to .034-.041 and the
  B55/B110 cells by up to .003, so single-cell differences of that size are not interpretable.
- UPS (Table 3B) is a per-cell result: the transport lowers the predictive
  NLL of the unseen cells at every budget, but its block-SR error sits on
  the scene-prior floor (.092) and does not beat the planner's own probed
  success rate at B >= 55 (.090 / .087) on this panel. Retargeted to the
  full 220-route SR, the scene prior contributes nothing measurable: a
  scene-free prior moves the error by at most .0022 in the readout and
  .0026 in the acquisition, and 23 of those 24 readout intervals and all
  acquisition intervals contain zero; the one exception (Random order,
  priorT(const), B = 55: +.0015 [+.0002, +.0028]) goes against the encoder
  (RESULTS.md, UPS retargeted to the full 220-route SR).
- **The encoder of record has no lane graph.** R2-noLane deletes the whole map
  side of the scene graph (lane tokens, lane_feat, lane-lane edges, agent-lane
  candidates, route relation) and keeps ego, command and agents; the
  lane-carrying R2 that earlier releases shipped is now a control row. Lane-free
  is better in all three runs on US (AUROC .761 +- .006 vs .751 +- .003,
  scene-MAE .181 +- .007 vs .192 +- .003, rho +.545 +- .024 vs +.490 +- .016),
  better on the UPS block-SR MAE (.0922 / .0920 / .0926 vs .0950 / .0936 / .0950)
  and better on the nuPlan zero-shot transfer, and **worse in exactly one place:
  the UPS per-cell NLL (.5974 / .5963 / .5985 against the lane-carrying
  .5857 / .5848 / .5863)**. Against the hand-crafted descriptor stack the
  lane-free encoder is slightly ahead on AUROC and rank correlation (+.012 +- .024,
  within run noise) and still behind on scene-MAE; its contribution is the input
  (raw tracks), not extra accuracy.
- Every published baseline of Table 1 was also run through **its own official
  implementation** (`experiments/official/`, `results/up_official.json`): ATDrive is
  lower in 12 of 12 cells against every method except catR (11 of 12, losing one cell
  by .0003), and the re-implementations reported in Table 1 are if anything more
  generous to the baselines than their own code is. Three of the seven are not
  reproducible as published (unseeded medoids, a `set.seed(NULL)` inside catR's item
  selection, and a selection rule with no randomness at all), and ATLAS's 3PL is
  unidentified at this panel size (RESULTS.md, "Table 1 through the baselines' OWN code").
- Two efficient-testing methods of the driving literature, re-implemented from
  their papers on the same protocol (neither has public code): the fixed-design
  FST (Li et al., T-ITS 2025) reaches macro .0485 and the GP-adaptive sampling
  of Gong et al. (T-ITS 2023), on the route descriptor, .0390 against ATDrive's
  .0318 — ATDrive is lower in every cell, with the paired interval excluding zero
  only at K_cal = 12, B >= 55 for the GP method (RESULTS.md, "AV-testing
  baselines re-implemented from their papers").
- Zero-shot on nuPlan val14, against a label-shuffle null that carries the
  arm's own variance structure (20 fixed permutations x 3 training seeds),
  the lane-free encoder of record clears on the performance drop at BOTH
  budgets (no shuffled labeling of 20 reaches it, p .048 at top-5% and
  top-10%; all three seeds above the single-run 95th percentile). Of the
  lane-carrying controls the speed-ablated one clears at top-5% and is
  marginal at top-10% (p .095), and the speed-kept one does not clear at
  either (p .14 / .19). The whole-panel rank correlation stays marginal for
  all three arms (p .095) (RESULTS.md, nuPlan val14 zero-shot retrieval).
  On the full official Val14 split (1,118 scenarios, all 328 logs, 10
  planners; `results/nuplan_zeroshot_full.json`) the encoder of record clears
  at top-5% (p .048, 0/20 labelings; failure enrichment 1.74) and is marginal
  at top-10% (p .095, 1/20; 1.62). The signal sits on the 584 tokens of the
  panel above (enrichment 2.17 / 1.96 scored within them) and is weak on the
  534 added tokens (1.31 / 1.34), 94 of which are Singapore scenes
  (RESULTS.md, nuPlan Val14 zero-shot retrieval on the full split).
- The RelGraph encoder ships as predictions; its training code depends on
  Bench2Drive raw rollouts (not redistributable) and is staged for a
  separate release; everything downstream of the predictions (US scoring,
  the UPS prior, the nuPlan test) is reproducible here.

## Limitations

- **Panel.** n_eff = 16 planners, one per model family, each an evaluation planner in 2-6 draws;
  the claims generalise to planners of these families in the SR range .135-.777.
- **Panel selection.** The 16 were chosen from 22 by family and SR spread after the 22-planner
  results existed (PROTOCOL section 1); the 22-planner matrix ships as a robustness panel
  (`ATDRIVE_RESPONSE_CSV=data/matrices/b2d_e2e22_response_matrix.csv`). Protocol constants were
  fixed in this order: v3.0.0 (08-29) tie convention; v5.0.0 (09-01) eps grid, 90th percentile,
  T0 = 10; v5.1.0 rounded tie-break; v6.0.0 (09-03) 16-planner rule, K_cal {4, 8, 12}; v7.0.0
  (09-04) budgets {30, 55, 110, 165}. None precedes all of its results in git.
- **Estimand.** SR is over the routes the simulator completed for the planner (210-220 of 220);
  missingness is not at random; the pass/220 convention scales every error by at most 4.6%.
- **Development optimism, symmetric.** Descriptor stacks were feature-selected by leave-one-type-out
  rho on this panel; the encoder line was iterated against the same rho on this panel; only the
  R0 / R1 / R2 ladder was fixed in advance. Both predictor families need one PDM-Lite probe
  rollout per route.
- **Stopping rule.** c at K_cal = 4 rests on four LOO trajectories per draw (range 1.25-3.42); inside
  each LOO fold sigma_b and sigma_g are held at the panel fit. The live tool calibrates c on
  t in [10, 110] (cache entries) against the paper's [10, 220], giving 2.16-2.55 vs 1.95-2.12
  (conservative); its dry-runs are n = 3, and a route the simulator cannot finish terminates the
  run rather than entering the estimate.
- **Tie-break.** Sensitivity was measured for the K_cal = 12 cell only; K_cal = 4 / 8, where all-pass
  and all-fail ties are more common, were not run.
- **Testlet grid.** UG = [-3, 3] spans >= 2 sigma_g for every fitted value (max 1.5; 4.6%
  truncation), so the sigma_g = 2.0 grid candidate is effectively 1.49.
- **Encoder.** Shipped as out-of-fold predictions; the training harness, the Bench2Drive scene
  tensors and every launcher of record are in `encoder/` (`encoder/README.md`; the nuPlan-derived
  tensors are not shipped). nuPlan zero-shot inputs ship as `data/nuplan/val14_zeroshot.npz` and, for the full
  1,118-token split, `data/nuplan/val14_full_zeroshot.npz`. The
  lane-free encoder became canonical after the lane-carrying one had been scored on every table,
  i.e. it was selected on the same panel these numbers are reported on.
- **2PL baselines** use log a ~ N(0, .5^2) and are prior-shrunk at K_cal = 4 (`run_model_adequacy.py`).

## Citation

```bibtex
(to appear)
```
