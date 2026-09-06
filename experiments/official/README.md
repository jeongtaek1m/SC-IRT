# Official-code baselines

Every published baseline of Table 1 run through **its own implementation**, on the
ATDrive protocol (`experiments/run_up_official.py`; contract in `base.py`;
protocol cells in `data.py`, identical draws / K_cal subsamples / banks to
`run_up_frontier.py`, so every comparison is paired at the evaluation level).

| wrapper | method | official repository | what is called |
|---|---|---|---|
| `tinybench.py` | tinyBenchmarks (Maia Polo et al., ICML 2024, arXiv:2402.14992) | felipemaiapolo/tinyBenchmarks | `tutorials/irt.py:train_irt_model` (py-irt CLI, multidim 2PL, 2000 epochs, hierarchical priors, seed 42), the anchor-points notebook's KMeans + `pairwise_distances`, `estimate_ability_parameters`; readouts anchor / p-IRT / gp-IRT |
| `fluid.py` | Fluid Benchmarking (Hofmann et al., COLM 2025, arXiv:2509.11106) | allenai/fluid-benchmarking | `irt/fit_irt_model.py` (py-irt `TwoParamLogistic`, hierarchical, 1000 epochs), `engine.run_fluid_benchmarking` (MFI selection), `estimators.ability_estimate` |
| `anchorpoints.py` | Anchor Points (Vivek et al., EACL 2024, arXiv:2309.08638) | rvivek3/AnchorPoints | `optimal_valset_validation.py:anchor_points_weighted` core (corrcoef, `kmedoids.fasterpam`, cluster-size weights) |
| `disco.py` | DISCO (Rubinstein et al., 2025, arXiv:2510.07959) | arubique/disco-public | `selection.py` pds/jsd + `sample_by_disagreement`, `irt.py` py-irt fit and D selection, `acc.py:compute_acc_pirt` and the fitted signature estimators |
| `metabench.py` + `metabench_reduce.R` | metabench (Kipnis et al., ICLR 2025, arXiv:2407.12844) | adkipnis/metabench | `analysis/utils.R:run.mirt` / `get.theta`, `reduce.R:collect.item.info` / `get.info.quantiles` / `select.items`, the GAM readout |
| `atlas.py` + `atlas_cat.R` | ATLAS (Li et al., 2025, arXiv:2511.04689) | Peiyu-Georgia-Li/ATLAS | `01_fit_irt.r` (mirt 3PL EM), `03_atlas_cat.r:run_atlas` (catR `nextItem` MFI/EAP/randomesque, `thetaEst`, `semTheta`), `04_pirt_accuracy.r:compute_pirt_accuracy` |
| `catr.py` + `catr_static.R` | the classical information orders (Birnbaum 1968; Lord 1980; van der Linden 1998) | catR 3.17 (Magis & Raiche) | `mirt` 2PL fit, `catR::Ii` (Total-Fisher), `catR::MWI` (prior-weighted / MPWI), `thetaEst`, `Pi` |

Each wrapper documents, in a numbered **ADAPTATIONS** block, every place where the
official code could not be called as-is and why. Each has a `selftest_<name>.py`
that runs real protocol cells and asserts the leak test (flipping the outcome of
every item the method did not select must not change its output), determinism,
budget semantics and, where applicable, the prefix property.

The commit that added this directory records what the runs found; the numbers of
record are in `results/up_official.json` and the RESULTS.md section that cites it.

Repository checkouts and the Python / R environments they need are outside this
repository (`/data2/jeongtae/official_baselines`, commit ids in
`results/up_official.json:provenance`), because they carry their own licences.
