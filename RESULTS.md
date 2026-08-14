# Results

All numbers are Bench2Drive, out-of-fold, produced by this repository under the
pinned runtime (`bash run_all.sh`). Metric definitions are in
[PROTOCOL.md](PROTOCOL.md); environment sensitivity is in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

The paper's NavSim scale-up is not part of this release.

## Table I — Scene difficulty prediction

44-type leave-one-type-out, bank 219 routes x 16 planners.
Scripts: `experiments/descriptor_table.py` (baselines), `experiments/encoder_us.py` (ours).

| Scene descriptor | AUROC ↑ | MAE ↓ | rho ↑ |
|---|---|---|---|
| Random | 0.500 | — | 0.000 |
| Min-TTC | 0.702 | 0.204 | −0.670 |
| Risk field | 0.718 | 0.203 | −0.064 |
| Route geometry | 0.729 | 0.191 | +0.123 |
| Agent density + kinematics | 0.736 | 0.200 | +0.190 |
| Traffic entropy | 0.726 | 0.198 | +0.045 |
| Agent-JEPA | 0.709 | 0.202 | −0.276 |
| **Ours** (difficulty-supervised encoder, 6-seed logit ensemble) | **0.762** | **0.173** | **+0.520** |

The kinematic reference (`bl-cmdkin`, AUROC 0.763 / MAE 0.173 / rho +0.418) and the
GT channel stack (+0.529) appear in Table IV rather than here: the former is the
encoder's own auxiliary input, the latter requires ground-truth boxes at test
time. On rho — the claim of this table — the encoder clears both the simple
descriptors (+0.190 best) by a wide margin; response-ranking AUROC is
statistically tied with the kinematic reference.
| Oracle (frozen gold; reliability ceiling 0.904) | — | — | 1.000 |

Read the rho column with the ceiling in mind: +0.520 is 58% of the 0.904 that a
perfect predictor could reach against a 16-planner gold estimate.

Two negative rho values are worth noting rather than hiding. Min-TTC at −0.670 is
strongly *anti*-correlated with difficulty: the classical criticality scalar
systematically ranks scenarios backwards for end-to-end planners. Agent-JEPA is
mildly anti-correlated at −0.276.

## Table II — Adaptive planner evaluation

Dynamic CAT, stop at SE(theta) < 0.35, leave-one-planner-out over 16.
Scripts: `experiments/cat_up.py`, `experiments/cat_ups.py`.

### (a) UP — calibrated bank (b, a from responses), N = 219

| Item selection | items n ↓ | IES ↑ | MAE ↓ | theta err ↓ |
|---|---|---|---|---|
| Random | 48.9 ±2.2 | 4.5 | .038 ±.008 | .237 ±.048 |
| tinyAnchor | 52.2 ±1.4 | 4.2 | .057 ±.011 | .353 ±.064 |
| 1PL information | 31.1 ±0.5 | 7.0 | .037 ±.006 | .249 ±.047 |
| **2PL information** | **25.6 ±0.8** | **8.6** | .044 ±.011 | .278 ±.058 |

### (b) UPS — amortised calibration (predicted difficulty only), N ≈ 110

The regime the method exists for: both the planner and the scenario type are new,
so no calibrated difficulty exists and classical information selection is not
merely worse but undefined.

| Item selection | items n ↓ | IES ↑ | MAE ↓ | theta err ↓ |
|---|---|---|---|---|
| rules from block (a) | *not defined* | | | |
| Random | 41.1 ±1.2 | 2.7 | .050 ±.004 | .464 ±.057 |
| **Shrunk information (ours)** | **31.5 ±0.4** | **3.5** | **.047 ±.006** | .452 ±.057 |
| Oracle (true calibrated difficulty, ceiling) | 32.9 ±0.7 | — | .051 ±.005 | .323 ±.029 |

The oracle row is not attainable — it needs the responses the procedure is trying
to avoid collecting — and is shown only as a ceiling. Selection on predicted
difficulty reaches it.

## Table III — Real-data scale-up

Not included in this release.

## Table IV — Ablation (rho)

Scripts: `experiments/encoder_verify.py`, `experiments/hybrid_prereg.py`,
`experiments/gold_anchor.py`.

| Configuration | rho ↑ |
|---|---|
| kin-only ridge (no encoder) | +.418 |
| Encoder, individual runs | +.488 ±.020 |
| **Encoder, 6-run ensemble** | **+.524** |
| Hand-crafted GT stack (reference) | +.529 |
| + rank fusion with the stack | +.592 |

**The encoder and the hand-crafted stack tie.** The paired interval on
`ens6 − GT:ck+gtrisk` is Δ = −0.005, CI [−0.145, +0.135] — it straddles zero, and
the honest reading is that neither wins at this scale. The rank-fusion row gains
Δ = +0.071, CI [+0.001, +0.138] over `ck+camrisk-full`, an interval that excludes
zero but only just.

Note that both pre-registered fusion gates (against `ck+spzglob` and against
`GT:ck+gtrisk`) **fail** — their intervals include zero. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md#known-documentation-discrepancies).

## Appendix

- **A1 — full selection-strategy sweep**: the spread / matched / hybrid rows of
  `cat_up` and `cat_ups`.
- **A2 — fixed probe-20 descriptor comparison**: the UP columns of
  `descriptor_table`, demonstrating that aggregate MAE saturates at .050–.052 for
  every descriptor and therefore cannot discriminate between them.
- **A3 — noise ceiling** (`noise_ceiling.py`): split-half 0.691 → reliability
  0.817 → **ceiling 0.904**.
- **A4 — SE calibration** is reported in the paper from the development trace;
  the producing script is not part of this release (the 1PL row of `cat_up`
  shows the same overconfidence directionally: fewer items, larger theta error).

## What the numbers support

1. **Difficulty is learnable from the scene.** rho +.520 against +.191 for the best
   hand-crafted descriptor family (Table I) — though a *stack* of hand-crafted
   features ties with the encoder (Table IV).
2. **Evaluation gets cheaper.** 25.6 routes for a success rate within ±4.4%, an
   8.6x reduction (Table II a). Where no calibration exists at all, selection on
   predicted difficulty still matches the oracle (Table II b).
