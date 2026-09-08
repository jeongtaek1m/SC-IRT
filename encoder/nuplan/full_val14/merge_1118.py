"""Concatenate the 584-scene record tensors with the new 534-scene tensors -> 1,118-scene files
in the exact schemas the frozen driver reads (r0_ego.NUPLAN_EGO['routed'|'logged'], NUPLAN_NPZ)."""
import numpy as np
W = '/data2/jeongtae/nuplan_full_val14'
def cat(a, b, keys_scalar=()):
    out = {}
    for k in a.files:
        if k in keys_scalar or a[k].ndim == 0: assert str(a[k]) == str(b[k]), k; out[k] = a[k]; continue
        assert a[k].shape[1:] == b[k].shape[1:] and a[k].dtype == b[k].dtype, (k, a[k].shape, b[k].shape, a[k].dtype, b[k].dtype)
        out[k] = np.concatenate([a[k], b[k]])
    return out
A = np.load('/data2/jeongtae/navsim_interact/val14_tensors.npz', allow_pickle=True); B = np.load(f'{W}/val14n534_tensors.npz', allow_pickle=True)
T = cat(A, B); assert len(set(T['token'].tolist())) == 1118 and len(T['token']) == 2 * 1118; np.savez_compressed(f'{W}/val14_1118_tensors.npz', **T); print('tensors:', {k: T[k].shape for k in ('token', 'agents', 'ego', 'kin')})
A = np.load('/data2/jeongtae/relgraph/transfer/nuplan/val14_ego_logged.npz', allow_pickle=True); B = np.load(f'{W}/val14n534_ego_logged.npz', allow_pickle=True)
E = cat(A, B); assert np.array_equal(E['token'], T['token']) and np.array_equal(E['widx'], T['widx']); np.savez_compressed(f'{W}/val14_1118_ego_logged.npz', **E); print('ego logged:', E['ego'].shape)
A = np.load('/data2/jeongtae/relgraph/nuplan_val14_relgraph_v2.npz', allow_pickle=True); B = np.load(f'{W}/nuplan_val14n534_relgraph_v2.npz', allow_pickle=True)
G = cat(A, B, keys_scalar=('domain',)); ids = [f'{t}_{w}' for t, w in zip(T['token'], T['widx'])]
assert G['item_id'].tolist() == ids, 'graph rows must follow the ego-tensor rows'
np.savez_compressed(f'{W}/nuplan_val14_1118_relgraph_v2.npz', **G); print('relgraph:', G['lanes'].shape, G['item_id'].shape, 'domain', G['domain'])
