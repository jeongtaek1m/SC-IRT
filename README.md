# SC-IRT: Scene-Conditioned Item Response Theory

[[Paper (coming soon)]](#citation) · [[Tutorial]](tutorials/quickstart.ipynb) · [[Protocol]](PROTOCOL.md) · [[Results]](RESULTS.md) · [[Reproducibility]](REPRODUCIBILITY.md)

**How hard is a driving scene — before any planner has driven it?**

SC-IRT calibrates per-scene *difficulty* from the pass/fail record of a panel of
end-to-end driving planners (Bench2Drive, 16 planners x 220 routes), then trains
an interaction encoder that predicts that difficulty from the scene alone —
GT agent tracks in, one scalar out, supervised directly by the panel's failures.
No feature engineering, no scenario labels, no camera required.

The headline: the learned encoder recovers held-out-type difficulty at
**rho +0.50** (single run) against a noise ceiling of 0.90, where the best simple descriptor
reaches +0.19 — and the same difficulty scores drive adaptive testing that
evaluates a new planner in **~27 closed-loop rollouts instead of 220**.

## Getting started

```bash
git clone https://github.com/jeongtaek1m/SC-IRT.git
cd SC-IRT && pip install -e .
```

Score the released encoder — or any difficulty predictor — in four lines:

```python
import scirt

bt = scirt.encoder_predictions()        # {route_id: difficulty}, all out-of-fold
print(scirt.evaluate(bt))
# {'rho_scene': 0.503, 'auroc': 0.759, 'scene_mae': 0.177, 'n_routes': 220, ...}
```

Bring your own descriptor: any `{route_id: score}` dict works —
`scirt.evaluate` refits planner ability per held-out scenario type with your
difficulty frozen, and pools every metric across the 44 folds. Each headline
number comes with its floor: the **planner-only null** `P = sigmoid(theta_j)`,
the model that says all scenes are equally hard. Chance (0.5 AUROC) is not the
right comparison and is not reported.

```python
scirt.reference()                       # full-panel reference difficulty (Rasch)
scirt.noise_ceiling()                   # {'split_half': 0.68, ..., 'ceiling': 0.90}
```

And the tinyBenchmarks use case, for driving — evaluate a **new planner** from a
handful of closed-loop rollouts instead of the full bank:

```python
responses = {}                          # {route_id: 0|1} as you roll out
for _ in range(25):
    r = scirt.next_route(responses)     # most informative next route (target-EIG)
    responses[r] = run_my_planner(r)    # your closed-loop rollout
scirt.estimate_planner(responses)       # {'theta':…, 'sr_hat':…, 'ci95':…}  (p-IRT)
```

Reaching +-10% on the success rate takes ~27 rollouts instead of 220, and the
95% interval is a posterior-predictive quantile that covers 15 of 16 held-out
planners.

A runnable walk-through of all of the above: [tutorials/quickstart.ipynb](tutorials/quickstart.ipynb).

## Verifying the numbers

```bash
pytest            # pins the kernel, the headline row (0.759 / 0.177 / +0.503),
                  # and the few-rollout estimator against the shipped artifact
```

The pinned per-experiment scripts, reference outputs and the paper's comparison
experiments (descriptor table, adaptive testing, rank fusion) live on the
[`full-reproduction`](../../tree/full-reproduction) branch.

> **Caveat**: `full-reproduction` still reflects the pre-2026-08-17 protocol
> (219-route bank, 2PL calibration throughout, the retired 0.762/0.173/+0.520
> headline). `main` is the current protocol; treat the branch as historical
> until it is re-ported.

## Training the encoder yourself

The evaluation above needs nothing but this repository. Retraining needs the
Bench2Drive rollout annotations (not redistributed) and a GPU:

```bash
python train/build_tensors_b2d.py --anno_root <rollouts> --cmd_feats <kin npz dir> --out b2d_tensors.npz
for d_ep in "64 30" "96 60"; do set -- $d_ep
  for s in 0 1 2; do
    python train/train_encoder_b2d.py --tensors b2d_tensors.npz --d $1 --epochs $2 --seed $s --out runs/k_d${1}e${2}s${s}.npz
  done
done
python train/bundle_runs.py runs/k_*.npz --out interact_b2d_w2a_final.npz
```

Training is GPU-tier reproducibility: seeds move pooled rho by ~0.01 and
cross-device bit-identity is not promised. The shipped artifact is the
reference; the assembler rebuilds it bit-for-bit on every numeric key.

## What is in the box

```
scirt/         the library — calibration kernel, encoder, ability posterior, API
train/         tensor builder, LOTO trainer, run bundler (no ensembling)
tutorials/     runnable quickstart notebook
data/          response matrix, route->type map, kin input, encoder artifact (~0.3 MB)
tests/         kernel-equivalence and number-pinning tests
```

Model in one line: per-agent bidirectional GRUs over 6-second GT box tracks,
a 2-layer transformer mixing agents with an ego query, attention pooling over
windows, kinematic statistics embedded *inside* the model, and a linear head
emitting scalar difficulty — trained with cell-level BCE through
`P(planner j passes scene i) = sigmoid(theta_j − b(scene_i))`, theta frozen
per fold. Details and every equation: [PROTOCOL.md](PROTOCOL.md).

## Honest caveats

- Difficulty is calibrated against *this* panel; a different planner population
  defines a (correlated but not identical) difficulty.
- The reference difficulty is itself a 16-rater estimate, so it is a diagnostic
  anchor and not truth: the primary metric ranks predictions against observed
  failure rates, and no predictor can exceed the 0.899 reliability ceiling.
- **Two regimes, two parameterisations.** Unseen scenes use Rasch
  `sigmoid(theta - b)`; the calibrated bank uses 2PL `sigmoid(a_i(theta - b_i))`.
  Discrimination is used only where it is response-calibrated — a held-out scene
  has no responses, so `a_i` cannot be estimated there at all. This is not a
  free choice per table: log-discrimination has split-half reliability 0.08 on
  this panel (so it is unpredictable from a scene), yet on the calibrated bank
  it still cuts rollouts by 3.1-5.1 with a 95% CI excluding zero
  (-5.12 [-7.31,-2.94] at +-10%, -3.12 [-5.38,-1.25] at +-5%).
- Adaptive selection wins at small budgets and *loses* at large ones. Below
  ~40 rollouts it roughly halves the error against random sampling; past ~60,
  random gives the better success-rate estimate, because a representative
  sample reconstructs a mean better than an informative one.
- This branch is the method only. Ablations and baseline comparisons — the
  descriptor table, kin-fusion and window variants, adaptive testing, rank
  fusion, the NavSim scale-up — live on `full-reproduction`.

## Citation

```bibtex
@article{scirt2026,
  title   = {Scene-Conditioned Item Response Theory for End-to-End Driving Evaluation},
  author  = {<authors>},
  journal = {<venue, under review>},
  year    = {2026}
}
```

Released under the [MIT License](LICENSE).
