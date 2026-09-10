# US descriptor baselines, rebuilt

Table 3A compares scene-difficulty descriptors against the learned encoder. An
audit found several rows were not what their labels claimed; these scripts
rebuild them, running the official implementation where one exists and the
published definition where none does.

| script | row | source | provenance |
|---|---|---|---|
| `min_ttc.py` | Min-TTC | Hayward (1972); definition as in Westhofen et al. (2022) §5.2.1; TET/TIT from Minderhoud & Bovy (2001). No official code (commonroad-crime was consulted and rejected: it needs lanelets or the recorded future) | constant-velocity extrapolation of both actors with oriented bounding boxes, first footprint intersection, solved in closed form (Minkowski difference ray clip) |
| `edrf_riskfield.py` | EDRF-based risk field | EDRF (Jiang et al., arXiv:2410.14996, IEEE conference version), Table II / Table III constants. No official code exists. The previous citation (Westhofen et al. 2022) was wrong: that paper is a survey and defines no field | 6 s horizon, ego Laplace field, analytic spatial maximum; the multimodal predictor is replaced by the single realised future (p = 1) — a substitution of the model's core input, so the row is "EDRF-based features", not EDRF; whether that substitution helps or hurts the descriptor is not established |
| `traffic_entropy.py` | Traffic entropy (ours) | model: **official code** NVlabs/catk (Zhang et al., CVPR 2025, arXiv:2412.05334) running the `pre_bc_E31` checkpoint it was trained for, tokeniser `agent_vocab_555_s2`; SMART tokenisation from Wu et al. (NeurIPS 2024). Neither paper proposes the entropy (or any quantity of the model) as a difficulty measure: the descriptor is OUR construction on their released model and is not a literature baseline | loaded `strict=True` (811/811 tensors) instead of the earlier `strict=False` (808/818) into a different codebase with a mismatched vocabulary |
| `agentjepa.py` | Agent-JEPA | Jaiswal (arXiv:2606.28383). Official code exists — github.com/hellojais/mindrive-jepa (MIT, found 2026-09-10 after this rebuild; the statement "no official code" in the provenance JSONs is wrong) — and a run through it is pending | re-implemented from Sec. 3 including the auxiliary position loss the earlier replication omitted, trained to completion (the shipped features came from an epoch-0 checkpoint) |
| `cmdkin.py`, `inhouse_provenance.py` | Route geometry, Agent density + kin., Kinematics, Hand-crafted risk | no published method — designed in this project | `cmdkin.py` reconstructs the lost producer and reproduces the shipped npz bit for bit; `inhouse_provenance.py` documents every column |
| `score_one.py` | — | — | scores any rebuilt descriptor through the Table 3A machinery next to the shipped row |

Every descriptor, and the learned encoder, reads the same input: the PDM-Lite
reference rollout of the route. PDM-Lite is not one of the 16 evaluated planners
(`atdrive.b2d.EXCLUDED_PLANNER`), so no planner response enters a descriptor —
but these are **probe-rollout descriptors**, not scene-only ones, and the paper
must say so. Column-level provenance is in `results/us_official_provenance/`.
