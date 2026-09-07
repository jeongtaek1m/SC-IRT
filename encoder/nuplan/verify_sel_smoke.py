import numpy as np, csv
B='stage2out_sel'   # relative to encoder/nuplan (was the scratch run directory)
a=np.load(f'{B}/C4r2n_p0_b2d2nuplan_s0.npz',allow_pickle=True)
assert int(a['seed'])==0 and bool(a['label_shuffle']) and bool(a['stage2_label_shuffle'])
assert np.array_equal(a['label_perm'], np.random.default_rng(90000).permutation(220)), 'perm 0 is not default_rng(90000).permutation(220)'
assert np.isfinite(a['pred_logged']).all() and a['pred_logged'].shape==(1168,)
rows=list(csv.reader(open('/home/jeongtae/SC-IRT/data/matrices/b2d_e2e16sel_response_matrix.csv')))
assert len(rows)==17 and len(rows[0])==221
print('SMOKE OK: perm 0 fixed, training seed 0, 1168 finite logged predictions')
