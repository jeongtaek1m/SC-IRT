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
| `jepa_official_b2d.py` (+ `.yaml`) | Agent-JEPA (official code) — the row of record | Jaiswal (arXiv:2606.28383); **official code** github.com/hellojais/mindrive-jepa (MIT), checkout `/data2/jeongtae/official_baselines/mindrive-jepa`, unmodified | `preprocess` converts the bank's 10 Hz rollouts (`b2d_eval_sensors/route_*/anno`) into the official reader's scenario dicts (5 s windows, stride 10 frames, 5,445 windows) and hands them to the official `SceneTokenizer`; the official trainer with its default config on one GPU (best.pt = epoch 6 of 50 by its own validation rule); the official surprise score; `aggregate` = mean surprise over a route's windows (ours). Null-level result (.699 / .215 / +.005) |
| `agentjepa.py` | Agent-JEPA (our re-impl.; two secondary rows) | same paper; the statement "no official code" in the provenance JSONs is wrong (the official code was found 2026-09-10 after this rebuild) | re-implemented from Sec. 3 including the auxiliary position loss the earlier replication omitted, on Bench2Drive expert clips; the best-val checkpoint's +.151 is the route-length artefact documented in `results/us_official_provenance/`, the full schedule is at null |
| `reeval_amortized.py` | REEval amortized calibration | Truong et al. (ICLR 2025, arXiv:2503.13335); **official code** github.com/sangttruong/reeval, `calibration/calibration.ipynb` cell 8 and its `trainer` copied verbatim | z = w . e + b fitted jointly with the Rasch likelihood (150 nuisance thetas, L-BFGS), then the thetas; the 73-d rollout descriptor is the content embedding (the paper: LLM text embeddings); read out by the amortized model itself (no Ridge, no ATDrive calibration); `results/us_reeval_amortized.json` |
| `cmdkin.py`, `inhouse_provenance.py` | Route geometry, Agent density + kin., Kinematics, Hand-crafted risk | no published method — designed in this project | `cmdkin.py` reconstructs the lost producer and reproduces the shipped npz bit for bit; `inhouse_provenance.py` documents every column |
| `score_one.py` | — | — | scores any rebuilt descriptor through the Table 3A machinery next to the shipped row |

Every descriptor, and the learned encoder, reads the same input: the PDM-Lite
reference rollout of the route. PDM-Lite is not one of the 16 evaluated planners
(`atdrive.b2d.EXCLUDED_PLANNER`), so no planner response enters a descriptor —
but these are **probe-rollout descriptors**, not scene-only ones, and the paper
must say so. Column-level provenance is in `results/us_official_provenance/`.
