# Reproducibility

## Anchors ledger (unified protocol)

Every experiment entry point ends with hard assertions against the published
numbers and prints `anchors OK`. If an environment change breaks the
computation, the run fails loudly instead of drifting silently.

| script | asserted anchors (tolerance) |
|---|---|
| `run_up_main.py` | SRVar 29.0 rollouts (+-0.2), MAE .0463 (+-.002), coverage 48/48 exact; Random 40.7; EIG 28.7; SRVar fixed B=29 .0443 |
| `run_up_baselines.py` | Random+CI 40.7; Fluid-Fisher 27.3; metabench-greedy 38.9; Random@29 .0584; Fluid@29 .0375; metabench@69 .0285; Random-100 .0217 |
| `run_atlas_bridge.py` | ours SE(theta)@+-10% = 0.405 (+-.01); ATLAS tau=0.3 63.7 rollouts (+-1.0); tau=0.1 pool exhaustion 100% exact |
| `run_scarcity.py` | J13 SRVar 29.0 / Random 40.7; J4 Fluid MAE .0956 (+-.004) vs ours .0630 (+-.004) |
| `run_us.py` | null 0.710/.207; LLTM+e 0.764/+0.510; two-stage 0.760/+0.487; sigma-hat 0.593; PV share 16.4%; hc +0.486; kin +0.428; encoder d64 AUROC .753, rho +0.469 |
| `run_ups.py` | posterior-a variant 24.7; standalone SRVar 21.0 / EIG 21.2; decomposition .1034/.0350; hybrid D=20 .0470; composition ours .1035 |
| `run_sel_diversity.py` | SRVar Jaccard(S30) 0.141 (+-.01); Spearman(theta gap, overlap) < -0.7 |
| `run_plugin_ablation.py` | marginalised arm = Table 5 ours rows (J13 29.0/.0463/48-48; J4 .0630) |
| `run_factorial_2x2.py` | cells B/D = Table 3 (EIG 28.7; SRVar 29.0/.0463/48-48); `--max-steps` default 120 |
| `run_random_fpc.py` | IRT-free reference (no anchor: pure numpy, deterministic given the split streams) |
| `run_budget_frontier.py` | SRVar@29 .0443; Fluid@29 .0375; Random+IRT@29 .0584; metabench@69 .0285 |
| `run_model_adequacy.py` | diagnostic (no anchor); 1PL/2PL/3PL held-out NLL + split-half rel(log a) |

`tests/test_unified_anchors.py` pins everything cheap and CPU-deterministic
(panel invariants, the draw-0 split, grid constants, kernel determinism).
The legacy layer keeps its own tests (`test_invariants.py`,
`test_kernel_equiv.py`, `test_train_compiles.py`).

## RNG registry (unified protocol)

Random streams are protocol constants. Fresh `np.random.RandomState`
instances are created at the documented points; nothing shares state across
arms, so run order between arms is irrelevant.

| stream | seed | used for |
|---|---|---|
| split | 1000 + draw | planner / type holdout (`scirt/splits.py`) |
| SRVar stop draws | 13 | posterior-predictive CI of the SRVar arm |
| EIG stop draws | 7 | posterior-predictive CI of the EIG arm |
| baseline stop draws | 11 | Fluid / ATLAS-sel / metabench arms |
| Random arm | 300 + 20*draw + planner | selection order *and* its stop draws |
| fixed-budget permutation | 100 + 20*draw + planner | Random / Random-strat / Random-100 |
| randomesque | 1 | ATLAS top-5 sampling (fresh per trajectory) |
| scarcity subsample | 9000 + 100*draw + 10*J_cal + rep | calibration-panel subsets |
| UPS decomposition draws | 500 + 20*draw + planner | decomposition + hybrid |
| UPS random-B | 700 + 20*draw + planner | composition baseline |
| plausible values | 400 + draw | b-hat posterior draws (M = 20) |
| plug-in ablation draws | 19 (plug-in) / 13 (marginalised) | SR-CI stop draws |
| adequacy held-out cells | 2000 + draw | 10% held-out cells + planner halves |
| global | numpy 0 / torch 0 | set at the top of every entry point |

Calibration kernels are RNG-free: zeros initialisation, Adam lr 0.05,
800 iterations (LLTM+e: 1500), fixed regularisation, theta-mean centering
(`scirt/calibration.py`, `scirt/lltm.py`). Selection rules other than
randomesque are deterministic given the posterior.

## Environment and determinism

The unified-protocol anchors were produced with Python 3.9, torch on a
single CUDA GPU, numpy / scipy / scikit-learn as pinned in
`pyproject.toml`. CUDA float32 Adam reductions are not bit-identical to CPU
(measured up to ~2e-3 on a fitted difficulty vector; rank correlation
0.999997); the anchor tolerances (the last digit of the published rounding)
absorb that, and anchors have reproduced across repeated GPU runs. The
kernels accept `device='cpu'`, but the published numbers are GPU runs — on
CPU expect the same values within the stated tolerances, not bit-equality.

Note the legacy layer pins the *opposite* default (CPU) for its own
historical anchors; the two layers keep their own conventions.

## Data provenance

`experiments/build_data.py` packaged every artifact in `data/` from the
research tree, checking md5 equality where a source copy exists and
re-deriving the route->scenario-type map from the raw CARLA checkpoint JSONs
to verify the shipped CSV. The response panel is the raw 16 x 220 pass/fail
matrix; nothing in `data/` is a model output except `data/encoder/*` —
per-run out-of-fold difficulty predictions of the trajectory encoder on the
unified split (d64/d96 x seeds 0-2, single runs — prediction ensembling is
banned project-wide), shipped so Tables 1-2 reproduce without retraining.
Full retraining on the unified split: `train/train_encoder_unified.py`
(verbatim port of the research script that produced the shipped
artifacts; recipe constants in its docstring) with
`data/encoder/b2d_tensors.npz`; `train/train_encoder_b2d.py` is the legacy
LOTO trainer. Bench2Drive rollout annotations themselves are not
redistributed. Measured fidelity of the port: a full seed-0 / d64 retrain
on the development GPU reproduced the shipped artifact bit-for-bit
(max |delta b_tilde| = 0 over all 640 held-out routes; identical route
order), because every draw re-seeds torch/numpy and the recipe is fixed.
On other hardware / driver stacks expect the same values up to training
noise (rho within about +-0.01), not bit-equality. Score any run with
`experiments/eval_us_predictions.py`, which reproduces the published
encoder summaries (0.753 / .189 / +0.469) from the shipped artifacts.

## Known gaps

- The NavSim panel (Table 7) is computed from a 17 x 12,146 out-of-fold
  artifact set not yet packaged here; its runner and artifacts are staged
  for a follow-up commit. The numbers of record are in RESULTS.md.
- The legacy layer reproduces the paper's robustness appendix (220-route
  LOPO snapshot, 2PL bank, target-EIG); the still-older 219-route pipeline
  lives on the `full-reproduction` branch.
