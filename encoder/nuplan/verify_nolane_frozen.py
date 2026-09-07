#!/usr/bin/env python3
"""Proof that /data2/jeongtae/relgraph_nolane_shim/r2_graph.py is (a) the FROZEN
/data2/jeongtae/relgraph/r2_graph.py on its default path and (b) a genuinely lane-free encoder
under --ablate-lane, on BOTH domains the B2D->nuPlan transfer touches.

The shim exists because the nuPlan transfer driver (transfer/zs21/stage2/chdrop.py) imports
r2_graph from the frozen tree, and the frozen tree must never be written.  A wrapper puts this
directory ahead of /data2/jeongtae/relgraph on sys.path, so `import r2_graph` resolves here
while r0_ego, b2d_earlystop, transfer_common, hrel and zs21_common still come from the frozen
tree (this file's r2_graph keeps RG = '/data2/jeongtae/relgraph').

  0  SOURCE     the shim differs from the frozen file only by the ablation: every frozen line
                that is not in the shim verbatim is one of the 19 listed here, and each of those
                is present in the shim in the recorded, ablation-plumbed form.
  1  DEFAULT    with --ablate-lane OFF the shim IS the frozen file, numerically: every tensor
                load_graph puts on the device, every graph_stats constant, and encode() over
                every window of b2d and of nuplan, are bit-identical (max|d| = 0) between the
                frozen module and the shim module holding the same weights.
  2  ABLATION   on b2d AND on nuplan: against a POISONED tensor file (lane geometry, lane_feat,
                a2l_rel, route_rel randomised; L2L / A2L wiring rewired at random; ego, command,
                agents, agent_mask, item_id copied verbatim) the ablated arm's model inputs and
                encode() output are bit-identical, while the default arm's move.  Live lane
                tokens / L2L edges / A2L candidates / route-valid windows are reported before
                and after.
  3  GUARD      the one-line readout guard `rd = rd * nmask.any(1)[:, None]` can only bite on a
                window with NO lanes and NO live agents.  Those windows are counted in the b2d,
                navsim and nuplan tensors, for the default arm and for the ablated arm.
  4  SHADOW     importing r2_graph with this directory first on sys.path really resolves here
                and r0_ego / transfer_common / b2d_earlystop still resolve to the frozen tree.

Everything runs on the CPU; no GPU is taken.  Nothing under /data2/jeongtae/relgraph is written.

  /home/jeongtae/miniconda3/envs/smart/bin/python verify_nolane_frozen.py
"""
import importlib.util
import os
import subprocess
import sys
from importlib.machinery import SourceFileLoader

import numpy as np
import torch
import torch.nn as nn

FROZ_DIR = '/data2/jeongtae/relgraph'
SHIM_DIR = '/data2/jeongtae/relgraph_nolane_shim'
FROZ_PY = f'{FROZ_DIR}/r2_graph.py'
SHIM_PY = f'{SHIM_DIR}/r2_graph.py'
SCRATCH = os.environ.get('NOLANE_SCRATCH', 'nolane_poison')   # where the poisoned copies are written
DEV = 'cpu'
OK = []


def check(name, cond, extra=''):
    OK.append(bool(cond))
    print(f'  [{"PASS" if cond else "FAIL"}] {name}{("   " + extra) if extra else ""}', flush=True)


def load_module(name, path):
    ld = SourceFileLoader(name, path)
    sp = importlib.util.spec_from_loader(name, ld)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    ld.exec_module(m)
    return m


# ══ 0. SOURCE ═══════════════════════════════════════════════════════════════
# The 19 frozen lines that the port rewrites, each with the shim line that replaces it.  A frozen
# line outside this table that is missing from the shim would be a silent deletion.
REWRITES = {
 '"""R2 — the full relational-graph rung of the R-ladder, and its two frozen negative controls.':
 '"""R2 — the full relational-graph rung of the R-ladder, its two frozen negative controls,',
 'FOUR ARMS, ONE ARCHITECTURE.  Every arm below is the SAME R2Net with the same parameter count':
 'FIVE ARMS, ONE ARCHITECTURE.  Every arm below is the SAME R2Net with the same parameter count',
 "               nuplan_ego='logged'):":
 "               nuplan_ego='logged', ablate_lane=False):",
 "    es, ed, er, em, worst = build_aug_edges(d['l2l_src'].astype(np.int32),":
 '    es, ed, er, em, worst = build_aug_edges(l2l_src.astype(np.int32),',
 "                                            d['l2l_dst'].astype(np.int32),":
 '                                            l2l_dst.astype(np.int32),',
 "                                            d['l2l_type'].astype(np.int32), d['l2l_mask'])":
 '                                            l2l_type.astype(np.int32), l2l_mask)',
 "    g.lanes = t(d['lanes'])                       # (N,M,P,4) fp16":
 '    g.lanes = t(lanes)                            # (N,M,P,4) fp16',
 "    g.lane_feat = t(d['lane_feat'])": '    g.lane_feat = t(lane_feat)',
 "    g.lane_mask = t(d['lane_mask'])": '    g.lane_mask = t(lane_mask)',
 "    g.a2l_idx = t(a2l_idx.astype(np.int64)); g.a2l_rel = t(a2l_rel); g.a2l_mask = t(d['a2l_mask'])":
 "    g.a2l_idx = t(a2l_idx.astype(np.int64)); g.a2l_rel = t(a2l_rel); g.a2l_mask = t(a2l_mask)",
 "    g, ids = load_graph('b2d', torch, dev, a.shuffle, a.shuffle_seed, a.ablate_route)":
 "                        ablate_lane=a.ablate_lane)",
 '             b_ref=b_ref, static=static, per_draw=per_draw, **led.fields())':
 '             arm=np.array(tag), ablate_lane=bool(a.ablate_lane),',
 "    g, ids = load_graph('navsim', torch, dev, a.shuffle, a.shuffle_seed, a.ablate_route)":
 "                           ablate_lane=a.ablate_lane)",
 '             per_fold=per_fold)':
 '             per_fold=per_fold, arm=np.array(tag), ablate_lane=bool(a.ablate_lane),',
 '    g, ids = load_graph(src, torch, dev)':
 '    g, ids = load_graph(src, torch, dev, ablate_lane=a.ablate_lane)',
 "    g_t, ids_t = load_graph(tgt, torch, dev, nuplan_ego='routed')":
 "    g_t, ids_t = load_graph(tgt, torch, dev, nuplan_ego='routed', ablate_lane=a.ablate_lane)",
 '                   + (f\'_dag{"".join(a.drop_agent_channels)}\' if a.drop_agent_channels else \'\'))':
 '                   + (f\'_dag{"".join(a.drop_agent_channels)}\' if a.drop_agent_channels else \'\')',
 '                    or a.proper_init or a.shuffle or a.ablate_route or a.zs_suffix), \\':
 '                    or a.proper_init or a.shuffle or a.ablate_route or a.ablate_lane',
 "    assert not (a.shuffle is not None and a.ablate_route), 'one control at a time'":
 '    assert sum([a.shuffle is not None, a.ablate_route, a.ablate_lane]) <= 1, \\',
}


def section0():
    print('\n=== 0. SOURCE: the port is additive ===', flush=True)
    fz = open(FROZ_PY).read().splitlines()
    sh = open(SHIM_PY).read().splitlines()
    shset = set(sh)
    missing = [l for l in fz if l not in shset]
    print(f'  frozen {len(fz)} lines -> shim {len(sh)} lines (+{len(sh) - len(fz)})', flush=True)
    print(f'  frozen lines absent from the shim verbatim: {len(missing)}', flush=True)
    for l in missing:
        r = REWRITES.get(l)
        print(f'    - {l.strip()[:88]}\n      -> {(r or "*** UNEXPECTED ***").strip()[:88]}',
              flush=True)
    check('every frozen line the port rewrites is a recorded ablation rewrite',
          set(missing) <= set(REWRITES), f'{len(missing)} rewritten, {len(REWRITES)} recorded')
    check('every recorded replacement line is present in the shim',
          all(v in shset for v in REWRITES.values()))
    check('no frozen line is deleted without a replacement',
          all(REWRITES.get(l) is not None for l in missing))
    ad = [l for l in sh if l not in set(fz)]
    print(f'  shim lines not in the frozen file: {len(ad)} '
          f'(the ablation: docstring, strip_lanes / verify_lane_ablation, the load_graph block, '
          f'the readout guard, the flag, the tag, the npz metadata)', flush=True)
    check('RG still points at the frozen tree, so every OTHER module comes from there',
          f"RG = '{FROZ_DIR}'" in open(SHIM_PY).read())


# ══ modules ═════════════════════════════════════════════════════════════════
print(__doc__.split('\n\n')[0], flush=True)
print(f'torch {torch.__version__}  device {DEV}\n', flush=True)
section0()

sys.path.insert(0, FROZ_DIR)
import r0_ego                                                            # noqa: E402
import b2d_earlystop as es                                               # noqa: E402
FR = load_module('r2_graph_frozen_ref', FROZ_PY)
SH = load_module('r2_graph_nolane_shim', SHIM_PY)

NPZ = {'b2d': r0_ego.B2D_NPZ, 'navsim': r0_ego.NAVSIM_NPZ, 'nuplan': r0_ego.NUPLAN_NPZ}
ATTR = {'b2d': 'B2D_NPZ', 'navsim': 'NAVSIM_NPZ', 'nuplan': 'NUPLAN_NPZ'}
TENSORS = ('lanes', 'lane_feat', 'lane_mask', 'route_rel', 'route_valid', 'e_src', 'e_dst',
           'e_rel', 'e_mask', 'a2l_idx', 'a2l_rel', 'a2l_mask', 'agents', 'agent_mask',
           'ego_w', 'command')
EGO_T = ('agents', 'agent_mask', 'ego_w', 'command')
POISONED_T = ('lanes', 'lane_feat', 'route_rel', 'e_src', 'e_dst', 'e_rel', 'a2l_idx', 'a2l_rel')
LANE_KEYS = ('lanes', 'lane_feat', 'a2l_rel', 'route_rel')
WIRE_KEYS = ('l2l_src', 'l2l_dst', 'l2l_type', 'a2l_idx')
KEEP_KEYS = ('agents', 'agent_mask', 'ego', 'command', 'item_id',
             'lane_mask', 'l2l_mask', 'a2l_mask', 'route_rel_valid')


def load(mod, domain, path, **kw):
    """load_graph reads the tensor path off r0_ego at call time, so point it at the copy."""
    old = getattr(r0_ego, ATTR[domain])
    setattr(r0_ego, ATTR[domain], path)
    try:
        return mod.load_graph(domain, torch, DEV, **kw)
    finally:
        setattr(r0_ego, ATTR[domain], old)


def tdiff(ga, gb):
    return [k for k in TENSORS if not torch.equal(getattr(ga, k), getattr(gb, k))]


def stdiff(sa, sb):
    return [k for k in sa if not (torch.equal(sa[k][0], sb[k][0])
                                  and torch.equal(sa[k][1], sb[k][1]))]


def model(mod, seed=0):
    torch.manual_seed(seed)
    m = mod.build_r2(torch, nn, 64).to(DEV)
    m.eval()
    return m


def enc(m, g, st, chunk=512):
    with torch.no_grad():
        return torch.cat([m.encode(g, torch.arange(i, min(i + chunk, g.N)), st)
                          for i in range(0, g.N, chunk)])


# ══ 1. DEFAULT PATH ═════════════════════════════════════════════════════════
print('\n=== 1. DEFAULT PATH: the shim IS the frozen file ===', flush=True)
m_f, m_s = model(FR), model(SH)
check('R2Net is identical: same parameter count and same init hash',
      es.init_hash(m_f) == es.init_hash(m_s)
      and sum(p.numel() for p in m_f.parameters()) == sum(p.numel() for p in m_s.parameters()),
      f'{sum(p.numel() for p in m_f.parameters()):,} params, hash {es.init_hash(m_f)}')
for dom in ('b2d', 'nuplan'):
    gf, idf = load(FR, dom, NPZ[dom])
    gs, ids_ = load(SH, dom, NPZ[dom])
    stf = FR.graph_stats(gf, np.arange(gf.N), torch, DEV)
    sts = SH.graph_stats(gs, np.arange(gs.N), torch, DEV)
    check(f'[{dom}] every load_graph tensor bit-identical frozen vs shim (default arm)',
          not tdiff(gf, gs) and list(idf) == list(ids_), f'{len(TENSORS)} tensors, {gf.N} windows')
    check(f'[{dom}] graph_stats constants bit-identical frozen vs shim', not stdiff(stf, sts))
    zf, zs_ = enc(m_f, gf, stf), enc(m_s, gs, sts)
    check(f'[{dom}] encode() bit-identical frozen vs shim over all {gf.N} windows',
          torch.equal(zf, zs_), f'max|d| {float((zf - zs_).abs().max()):.3e}')
    del gf, gs, zf, zs_


# ══ 2. THE ABLATION, BOTH DOMAINS ═══════════════════════════════════════════
def make_poison(src, dst, mod, seed=1234):
    d = np.load(src, allow_pickle=True)
    rng = np.random.default_rng(seed)
    out = {k: d[k] for k in d.files}
    for k in LANE_KEYS:
        out[k] = rng.normal(0.0, 3.0, d[k].shape).astype(d[k].dtype)
    out['l2l_src'] = rng.integers(0, mod.M, d['l2l_src'].shape).astype(d['l2l_src'].dtype)
    out['l2l_dst'] = rng.integers(0, mod.M, d['l2l_dst'].shape).astype(d['l2l_dst'].dtype)
    out['l2l_type'] = rng.integers(0, 4, d['l2l_type'].shape).astype(d['l2l_type'].dtype)
    out['a2l_idx'] = rng.integers(0, mod.M, d['a2l_idx'].shape).astype(d['a2l_idx'].dtype)
    np.savez(dst, **out)
    same = [k for k in KEEP_KEYS if k in d.files
            and np.array_equal(np.asarray(d[k]), np.asarray(out[k]))]
    diff = [k for k in LANE_KEYS + WIRE_KEYS
            if not np.array_equal(np.asarray(d[k]), np.asarray(out[k]))]
    return same, diff


print('\n=== 2. THE ABLATION IS REAL, ON BOTH DOMAINS ===', flush=True)
os.makedirs(SCRATCH, exist_ok=True)
for dom in ('b2d', 'nuplan'):
    print(f'\n  ── {dom} ──', flush=True)
    poison = f'{SCRATCH}/{dom}_poisoned.npz'
    same, diff = make_poison(NPZ[dom], poison, SH)
    check(f'[{dom}] every lane number and every lane index randomised',
          set(diff) == set(LANE_KEYS + WIRE_KEYS), f'{", ".join(diff)}')
    check(f'[{dom}] ego / command / agents / masks / ids copied byte for byte',
          set(same) == set(k for k in KEEP_KEYS if k in np.load(NPZ[dom]).files))
    g_ar, _ = load(SH, dom, NPZ[dom], ablate_lane=True)     # ablated,   real
    g_ap, _ = load(SH, dom, poison, ablate_lane=True)       # ablated,   poisoned
    g_ur, _ = load(SH, dom, NPZ[dom])                       # default,   real
    g_up, _ = load(SH, dom, poison)                         # default,   poisoned
    rows = np.arange(g_ar.N)
    st_ar, st_ap = SH.graph_stats(g_ar, rows, torch, DEV), SH.graph_stats(g_ap, rows, torch, DEV)
    st_ur, st_up = SH.graph_stats(g_ur, rows, torch, DEV), SH.graph_stats(g_up, rows, torch, DEV)
    print(f'    live lane tokens        {int(g_ur.lane_mask.sum()):>8d} -> '
          f'{int(g_ar.lane_mask.sum())}', flush=True)
    print(f'    live L2L edges (augm.)  {int(g_ur.e_mask.sum()):>8d} -> '
          f'{int(g_ar.e_mask.sum())}', flush=True)
    print(f'    live A2L candidates     {int(g_ur.a2l_mask.sum()):>8d} -> '
          f'{int(g_ar.a2l_mask.sum())}', flush=True)
    print(f'    route-valid windows     {int((g_ur.route_valid > 0.5).sum()):>8d} -> '
          f'{int((g_ar.route_valid > 0.5).sum())}   (of {g_ar.N})', flush=True)
    check(f'[{dom}] no lane token, L2L edge, A2L candidate or route row survives',
          int(g_ar.lane_mask.sum()) == 0 and int(g_ar.e_mask.sum()) == 0
          and int(g_ar.a2l_mask.sum()) == 0 and int((g_ar.route_valid > 0.5).sum()) == 0)
    da, du = tdiff(g_ar, g_ap), tdiff(g_ur, g_up)
    check(f'[{dom}] ABLATED model inputs bit-identical under a fully random map', not da,
          f'{len(TENSORS)} tensors compared')
    check(f'[{dom}] DEFAULT model inputs do move (the test is not vacuous)',
          set(du) >= set(POISONED_T), f'{len(du)} of {len(TENSORS)} tensors move: {du}')
    check(f'[{dom}] ego / command / agent tensors identical ABLATED vs DEFAULT on the real graph',
          all(torch.equal(getattr(g_ar, k), getattr(g_ur, k)) for k in EGO_T))
    check(f'[{dom}] graph_stats constants bit-identical under --ablate-lane',
          not stdiff(st_ar, st_ap))
    z_ar, z_ap = enc(m_s, g_ar, st_ar), enc(m_s, g_ap, st_ap)
    z_ur, z_up = enc(m_s, g_ur, st_ur), enc(m_s, g_up, st_up)
    check(f'[{dom}] encode(): ABLATED output bit-identical real vs poisoned',
          torch.equal(z_ar, z_ap), f'max|d| {float((z_ar - z_ap).abs().max()):.3e}')
    check(f'[{dom}] encode(): DEFAULT output does change real vs poisoned',
          not torch.equal(z_ur, z_up), f'max|d| {float((z_ur - z_up).abs().max()):.3e}')
    check(f'[{dom}] encode(): ABLATED differs from DEFAULT on the real graph',
          not torch.equal(z_ar, z_ur), f'max|d| {float((z_ar - z_ur).abs().max()):.3e}')
    check(f'[{dom}] the ablated encoder is finite and not constant',
          bool(torch.isfinite(z_ar).all()) and float(z_ar.std()) > 1e-6,
          f'sd {float(z_ar.std()):.4f} vs default {float(z_ur.std()):.4f}')
    del g_ar, g_ap, g_ur, g_up, z_ar, z_ap, z_ur, z_up

# ══ 3. THE READOUT GUARD ════════════════════════════════════════════════════
print('\n=== 3. THE READOUT GUARD BITES ONLY ON AN EMPTY NODE SET ===', flush=True)
print('  encode() node mask is cat([lane_mask, agent_mask.any(2)]), so the guard multiplier is'
      '\n  < 1 only where a window has NO lane AND NO live agent.', flush=True)
tot = 0
for dom in ('b2d', 'navsim', 'nuplan'):
    d = np.load(NPZ[dom], allow_pickle=True)
    nolane = ~d['lane_mask'].any(1)
    noag = ~d['agent_mask'].any(2).any(1)
    n_empty = int((nolane & noag).sum())
    tot += n_empty
    print(f'    {dom:7s} N {len(nolane):>6d} | zero-lane windows {int(nolane.sum()):>5d} | '
          f'zero-live-agent windows {int(noag.sum()):>5d} | BOTH (guard active, default arm) '
          f'{n_empty} | BOTH under --ablate-lane {int(noag.sum())}', flush=True)
check('the guard is a no-op on every window of every domain in the DEFAULT arm',
      tot == 0, f'{tot} windows with an empty node set across b2d + navsim + nuplan')

# ══ 4. THE SHIM SHADOWS THE FROZEN MODULE ═══════════════════════════════════
print('\n=== 4. THE SHIM SHADOWS THE FROZEN MODULE ===', flush=True)
probe = ('import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); '
         'import r2_graph, r0_ego, transfer_common, b2d_earlystop; '
         'print(r2_graph.__file__); print(r0_ego.__file__); print(transfer_common.__file__); '
         'print(b2d_earlystop.__file__); print(hasattr(r2_graph, "strip_lanes"))'
         % (FROZ_DIR, SHIM_DIR))
out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)
for l in out.stdout.strip().split('\n'):
    print(f'    {l}', flush=True)
lines = out.stdout.strip().split('\n')
check('r2_graph resolves to the shim', lines[0] == SHIM_PY, lines[0])
check('r0_ego / transfer_common / b2d_earlystop still resolve to the frozen tree',
      all(l.startswith(FROZ_DIR + '/') for l in lines[1:4]))
check('the imported r2_graph carries the ablation', lines[4] == 'True')

print(f'\n=== {sum(OK)}/{len(OK)} checks passed ===', flush=True)
sys.exit(0 if all(OK) else 1)
