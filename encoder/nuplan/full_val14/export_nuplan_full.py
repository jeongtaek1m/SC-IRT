"""Bundle the FULL val14 panel (1,118 scenarios x 10 planners) with the lane-free encoder's zero-shot
predictions and its permutation-fixed null, in the schema of data/nuplan/val14_zeroshot.npz
(run_nuplan_zeroshot.py reads it with ATDRIVE_NUPLAN_BUNDLE=<path>)."""
import sys, csv, numpy as np, pandas as pd
sys.path.insert(0, '/home/jeongtae/SC-IRT'); sys.path.insert(0, '/data2/jeongtae/relgraph_e16sel')
from scirt_rasch import rasch                      # atdrive.calibration.calibrate_dense, as for the 584 panel
W = '/data2/jeongtae/nuplan_full_val14'; NSEED, NPERM, PERM_SEED0 = 3, 20, 0
import re
s = open('/home/jeongtae/SC-IRT/experiments/build_data.py').read(); m = re.search(r'NUPLAN_PERM_SEED0\s*=\s*(\d+)', s)
if m: PERM_SEED0 = int(m.group(1))
M = pd.read_csv(f'{W}/nuplan_val14_k10_1118_response_matrix.csv', index_col=0)      # rows planners, cols tokens
meta = pd.read_csv(f'{W}/val14_1118_scenario_log_map.csv', dtype=str).set_index('scenario')
# scene order = the logged-ego readout order at widx == 0 of the target tensor
z = np.load(f'{W}/val14_1118_ego_logged.npz', allow_pickle=True); w0 = np.asarray(z['widx']) == 0
tok = np.array([str(t) for t in z['token'][w0]]); pos = {t: i for i, t in enumerate(tok)}
C = M[tok].values.astype(float)                                                      # planners x scenes
Y = np.where(np.isfinite(C), (C < 0.5).astype(np.float32), np.nan).astype(np.float32)
fail = np.nanmean(Y, 0).astype(np.float32)
_, b_ref = rasch(Y, it=800)                                                          # response-calibrated difficulty (in sample)
out = dict(tok=tok, logs=meta.loc[tok, 'log_name'].values.astype(str), Y=Y, fail=fail, b_ref=np.asarray(b_ref, np.float32),
           planners=np.array(M.index.astype(str)), cls=C)
def readout(path):
    a = np.load(path, allow_pickle=True); wi = np.asarray(a['widx']) == 0
    v = np.full(len(tok), np.nan, np.float32); v[[pos[str(t)] for t in a['tgt_groups'][wi]]] = np.asarray(a['pred_logged'], np.float32)[wi]
    assert np.isfinite(v).all(); return a, v
for k in range(NSEED):
    a, out[f'pred_NLe_s{k}'] = readout(f'{W}/stage2/NLe_b2d2nuplan_s{k}.npz'); assert int(a['seed']) == k and bool(a['ablate_lane']) and not bool(a['stage2_label_shuffle'])
for p in range(NPERM):
    perm = np.random.default_rng(PERM_SEED0 + p).permutation(220)
    for k in range(NSEED):
        a, out[f'pred_C4nl_p{p}_s{k}'] = readout(f'{W}/stage2/C4nl_p{p}_b2d2nuplan_s{k}.npz')
        assert int(a['seed']) == k and bool(a['label_shuffle']) and np.array_equal(a['label_perm'], perm) and bool(a['ablate_lane']), (p, k)
dst = '/home/jeongtae/SC-IRT/data/nuplan/val14_full_zeroshot.npz'; np.savez(dst, **out)
print(f'wrote {dst}: {len(tok)} scenes x {Y.shape[0]} planners, finite cells {int(np.isfinite(C).sum())}, base fail {np.nanmean(Y):.4f}, rho(fail, b_ref) {pd.Series(fail).corr(pd.Series(np.asarray(b_ref)), method="spearman"):+.3f}')
