# SC-IRT: Scene-Conditioned Item Response Theory

[[Paper (coming soon)]](#citation) · [[Protocol]](PROTOCOL.md) · [[Results]](RESULTS.md) · [[Reproducibility]](REPRODUCIBILITY.md)

**How hard is a driving scene — before any planner has driven it?**

SC-IRT calibrates per-scene *difficulty* from the pass/fail record of a panel of
end-to-end driving planners (Bench2Drive, 16 planners x 219 routes), then trains
an interaction encoder that predicts that difficulty from the scene alone —
GT agent tracks in, one scalar out, supervised directly by the panel's failures.
No feature engineering, no scenario labels, no camera required.

The headline: the learned encoder recovers held-out-type difficulty at
**rho +0.52** against a noise ceiling of 0.90, where the best simple descriptor
reaches +0.19 — and the same difficulty scores drive adaptive testing that
evaluates a new planner in **~26 closed-loop rollouts instead of 219**.

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
# {'auroc': 0.762, 'mae': 0.173, 'rho': 0.520, 'n_routes': 219}
```

Bring your own descriptor: any `{route_id: score}` dict works —
`scirt.evaluate` refits planner ability per held-out scenario type with your
difficulty frozen, and pools AUROC (cell ranking), MAE (per-scene pass-rate
reconstruction) and Spearman rho against the frozen anchor.

```python
gold = scirt.gold()                     # the frozen 2PL anchor itself
scirt.noise_ceiling()                   # {'split_half': 0.69, ..., 'ceiling': 0.90}
```

## Reproducing the paper numbers

```bash
bash run_all.sh                         # ~10 min on CPU, fully deterministic
python tools/compare_outputs.py expected expected_local
```

Four steps, in a fixed order: freeze the gold anchor -> measure the panel's
noise ceiling -> score the encoder -> seed stability and rank decomposition.
Reference outputs ship in `expected/`; the runtime is pinned to CPU/1 thread
because the published numbers are float32 optimisation results
(see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

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
python train/assemble_ensemble.py runs/k_*.npz --out interact_b2d_w2a_final.npz
```

Training is GPU-tier reproducibility: seeds move pooled rho by ~0.01 and
cross-device bit-identity is not promised. The shipped artifact is the
reference; the assembler rebuilds it bit-for-bit on every numeric key.

## What is in the box

```
scirt/         the library — 2PL calibration kernel, encoder, ability MAP, API
experiments/   the four pinned scripts behind the released numbers
train/         tensor builder, LOTO trainer, ensemble assembler
data/          response matrix, route->type map, kin input, encoder artifact (~0.3 MB)
expected/      reference outputs for diffing
tests/         kernel-equivalence and invariant tests
```

Model in one line: per-agent bidirectional GRUs over 6-second GT box tracks,
a 2-layer transformer mixing agents with an ego query, attention pooling over
windows, kinematic statistics embedded *inside* the model, and a linear head
emitting scalar difficulty — trained with cell-level BCE through
`P(planner j passes scene i) = sigmoid(theta_j − b(scene_i))`, theta frozen
per fold. Details and every equation: [PROTOCOL.md](PROTOCOL.md).

Comparison experiments from the paper (hand-crafted descriptor table, adaptive
testing, rank fusion, NavSim scale-up) live on the `full-reproduction` branch.

## Honest caveats

- Difficulty is calibrated against *this* panel; a different planner population
  defines a (correlated but not identical) difficulty.
- The gold anchor is itself a 16-rater estimate: no predictor can exceed the
  0.904 ceiling, and reported rho should be read against it.
- Training supervision is Rasch (a=1): discrimination has split-half
  reliability 0.08 on this panel — noise, not signal. Calibration stays 2PL.

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
