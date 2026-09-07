# The scene encoder — training harness of RelGraph R2 / R2-noLane

Everything that produced the per-draw difficulty predictions shipped in
`data/encoder/*.npz` and the nuPlan zero-shot arms of `data/nuplan/val14_zeroshot.npz`.
The code is copied verbatim from the working trees it ran in (paths below); the
repo package `atdrive/` supplies the Rasch calibration it trains against
(`harness/scirt_rasch.py` -> `atdrive.calibration.calibrate_dense`).

Environment of record: python 3.9.23, torch 2.0.1+cu117, numpy 1.23.4, scipy 1.13.1
(conda env `smart`); one run of `r2_graph.py --domain b2d` takes 15-25 min on one GPU
(16 draws x 30 epochs), two-stage about the same, a nuPlan stage-2 run 5-10 min.

## harness/ — the in-domain encoders (`/data2/jeongtae/relgraph_e16sel`)

| file | role |
|---|---|
| `r2_graph.py` | R2Net: ego / command / agent (/ lane) tokens, relational attention, IRT-likelihood head; `--ablate-lane` = **R2-noLane, the encoder of record** (strips lanes, lane_feat, lane-lane edges, agent-lane candidates and the ego-route relation before any tensor is built); `--ablate-route`; `--early-stop [--proper-init]` two-stage recipe |
| `r0_ego.py` | the reference harness r2_graph builds on: data loading (`b2d_relgraph_v2.npz`), the 16-draw unified split, the frozen 30-epoch recipe, the out-of-fold readout |
| `b2d_earlystop.py` | the two-stage (`--early-stop`) ledger: inner validation split, e* selection, stage-2 refit, output routing to `es/` and `es_pinit/` |
| `b2d_extract.py`, `b2d_route.py`, `navsim_extract.py` | graph-tensor extraction from the recorded rollouts / logs (what built `b2d_relgraph_v2.npz`) |
| `b2d_splits.py`, `scirt_rasch.py` | thin wrappers on `atdrive.splits.unified_split` and `atdrive.calibration.calibrate_dense` |
| `transfer_common.py` | `--full-train --transfer-to` plumbing (cross-domain arms; not used by any table of record) |
| `drop_channel_run.py`, `drop_channel_nolane.py` | the channel controls: ego-speed removed from both ego paths, on the lane-carrying model and on R2-noLane |
| `verify_nolane.py` | asserts that `--ablate-lane` leaves no lane information in any tensor the model sees |
| `b2d_relgraph_v2.npz` | the Bench2Drive scene tensors (4.0 MB; md5 b4f8c967...), extracted from our own recorded rollouts of the 220 routes — the only data input of the in-domain encoders besides the response matrix `data/matrices/b2d_e2e16sel_response_matrix.csv` |

Runs of record (three seeds each, one GPU per seed; `run_*.sh` are the launchers used):

```
python r2_graph.py --domain b2d --gpu G --seed k --ablate-lane        # R2-noLane   -> r2nolane_b2d_s{k}.npz   (encoder of record)
python r2_graph.py --domain b2d --gpu G --seed k                      # R2, lane graph kept (control)
python r2_graph.py --domain b2d --gpu G --seed k --ablate-route       # R2 without the ego-route relation
python drop_channel_nolane.py --drop speed --seed k --gpu G --outdir nolane_nospeed --tag nlnospeed   # R2-noLane, ego speed removed
python drop_channel_run.py    --drop speed --seed k --gpu G ...                                        # R2 (lane kept), ego speed removed
python r2_graph.py --domain b2d --gpu G --seed k --ablate-lane --early-stop   # two-stage recipe control (RESULTS.md, "Recipe control")
```

`experiments/build_data.py` turns each run output into the shipped `data/encoder/relgraph_*_s{k}.npz`
(per-draw evaluation-type routes + b_tilde + sigma) and verifies the shipped file against the raw
export bit for bit. md5 of the run outputs behind the encoder of record:
`r2nolane_b2d_s0.npz e96bbf8d`, `s1 5c71de09`, `s2 ca853a3c`.

The paths `RG = /data2/jeongtae/relgraph_e16sel` in `r2_graph.py`, `r0_ego.py`, `transfer_common.py`
are the working tree of record; to run from this directory set them to `encoder/harness` (or
symlink). The NAVSIM inputs (`navsim_relgraph_v2.npz` 65 MB, `navtest_tensors_v2.npz` 28 MB)
are not shipped — no table of record depends on them.

## nuplan/ — the zero-shot arms (`run_nuplan_zeroshot.py`)

The nuPlan arms are trained by the *frozen* stage-2 driver `frozen/chdrop.py` (with its own
copies of `r0_ego.py`, `r2_graph.py`, `hrel.py`, `b2d_earlystop.py`, `transfer_common.py`,
`zs21_common.py`; working tree `/data2/jeongtae/relgraph/transfer/zs21/stage2`, snapshot here
verbatim) through three wrappers that never write into the frozen tree:

| wrapper | arms | what it swaps |
|---|---|---|
| `chdrop_sel.py` | C0e (lane kept, speed kept), A2e (lane kept, speed removed), nulls C4r2n / C4r2e | the response matrix -> the 16-planner panel of record; the Rasch fit -> `atdrive.calibration.calibrate_dense` |
| `chdrop_perm.py` | permutation-fixed nulls | the label-permutation seed decoupled from the training seed (`--perm P`) |
| `chdrop_nolane.py` | **NLe (R2-noLane, the encoder of record)**, NLAe (R2-noLane, speed removed), nulls C4nl / C4nla | the two swaps above plus `shim/r2_graph.py` first on `sys.path` — the frozen-lineage R2 with `--ablate-lane` (`verify_nolane_frozen.py` checks it) |

`jobs/` holds the exact job lists and the per-GPU workers that ran them
(`<name>|<driver args>`, one serial worker per GPU; `launch_*.sh` = the launchers):
`f_nolane_q*.txt` NLe + C4nl (20 permutations x 3 seeds), `f_nla_q*.txt` NLAe + C4nla,
`f_sel*_gpu*.txt` / `f_perm_gpu*.txt` C0e, A2e and their nulls. `BASE` in the workers and
launchers is rewritten to this directory; everything else is as run. The outputs
(`<arm>_b2d2nuplan_s<k>.npz`, `<null>_p<p>_b2d2nuplan_s<k>.npz`) are read by
`experiments/build_data.py::export_nuplan` into `data/nuplan/val14_zeroshot.npz`, which
`run_nuplan_zeroshot.py` scores. NLAe / C4nla were run but are not in the shipped bundle.

Data the frozen driver reads and that is NOT shipped (nuPlan-derived): `val14_tensors.npz`
(3.6 MB, md5 ee96671c), `b2d_tensors.npz` (1.7 MB, 37796a05), the `frozen30/` r0 checkpoints;
paths are the constants at the top of `frozen/chdrop.py` and `frozen/r0_ego.py`.
