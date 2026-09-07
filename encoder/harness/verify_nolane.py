#!/usr/bin/env python3
"""Proof that r2_graph.py --ablate-lane (R2-noLane) is a genuinely LANE-FREE encoder.

The decisive experiment is a POISONED graph: a byte-for-byte copy of b2d_relgraph_v2.npz in
which every lane/map number has been replaced by random values (and the L2L / A2L wiring
rewired at random), while ego, command, agents, agent_mask and item_id are copied verbatim.
If a random map changes nothing, no map signal survives.

  1  INPUT     every tensor load_graph puts on the device, and every standardisation constant
               graph_stats derives from them, is BIT-IDENTICAL between the real graph and the
               poisoned graph under --ablate-lane, and DIFFERS without it.
  2  FORWARD   with one fixed set of weights, encode() and forward() are bit-identical between
               real and poisoned under --ablate-lane, and differ without it.  The ego sub-path
               (q_mlp / phi) and the agent sub-path (aout) are bit-identical between the ABLATED
               and the UNABLATED model on the real graph -- the ablation does not perturb them.
  3  GRADIENT  after a backward pass through the full forward, every parameter on the lane
               pathway has EXACTLY zero gradient, while every ego/agent/head parameter does not.
  4  SANITY    the two B2D windows with no live agents (the only place a zero-lane graph can
               produce an empty attention set) give finite, zero readouts, and there is no
               NaN/Inf anywhere in the ablated forward.

Everything here runs on the CPU: no GPU is taken.  The one-draw training smoke test is run
separately (see the header of the output).

  /home/jeongtae/miniconda3/envs/smart/bin/python verify_nolane.py
"""
import os, sys

import numpy as np
import torch
import torch.nn as nn

RG = '/data2/jeongtae/relgraph_e16sel'
POISON = os.environ.get('NOLANE_POISON_NPZ',
                        '/tmp/claude-1016/-home-jeongtae-SCIRT/'
                        'bab07df1-81c1-4668-91c3-429018c80dc5/scratchpad/b2d_poisoned.npz')
sys.path.insert(0, RG)
import r0_ego
import r2_graph as R

DEV = 'cpu'
OK = []


def check(name, cond, extra=''):
    OK.append(bool(cond))
    print(f'  [{"PASS" if cond else "FAIL"}] {name}{("   " + extra) if extra else ""}', flush=True)


# ── the poisoned graph ───────────────────────────────────────────────────────
LANE_KEYS = ('lanes', 'lane_feat', 'a2l_rel', 'route_rel')
WIRE_KEYS = ('l2l_src', 'l2l_dst', 'l2l_type', 'a2l_idx')
KEEP_KEYS = ('agents', 'agent_mask', 'ego', 'command', 'item_id', 'domain',
             'lane_mask', 'l2l_mask', 'a2l_mask', 'route_rel_valid')


def make_poison(src, dst, seed=1234):
    """Randomise every lane NUMBER and every lane WIRING index; copy everything else verbatim.
    The masks are kept so that the UNABLATED arm still runs on a structurally valid graph and
    the comparison isolates lane content + lane correspondence."""
    d = np.load(src, allow_pickle=True)
    rng = np.random.default_rng(seed)
    out = {k: d[k] for k in d.files}
    for k in LANE_KEYS:
        a = d[k]
        out[k] = rng.normal(0.0, 3.0, a.shape).astype(a.dtype)
    out['l2l_src'] = rng.integers(0, R.M, d['l2l_src'].shape).astype(d['l2l_src'].dtype)
    out['l2l_dst'] = rng.integers(0, R.M, d['l2l_dst'].shape).astype(d['l2l_dst'].dtype)
    out['l2l_type'] = rng.integers(0, 4, d['l2l_type'].shape).astype(d['l2l_type'].dtype)
    out['a2l_idx'] = rng.integers(0, R.M, d['a2l_idx'].shape).astype(d['a2l_idx'].dtype)
    np.savez(dst, **out)
    same = [k for k in KEEP_KEYS if np.array_equal(np.asarray(d[k]), np.asarray(out[k]))]
    diff = [k for k in LANE_KEYS + WIRE_KEYS
            if not np.array_equal(np.asarray(d[k]), np.asarray(out[k]))]
    return d, out, same, diff


def load(npz_path, ablate_lane):
    r0_ego.B2D_NPZ = npz_path                      # load_graph reads the path at call time
    g, ids = R.load_graph('b2d', torch, DEV, ablate_lane=ablate_lane)
    r0_ego.B2D_NPZ = f'{RG}/b2d_relgraph_v2.npz'
    return g, ids


TENSORS = ('lanes', 'lane_feat', 'lane_mask', 'route_rel', 'route_valid', 'e_src', 'e_dst',
           'e_rel', 'e_mask', 'a2l_idx', 'a2l_rel', 'a2l_mask', 'agents', 'agent_mask',
           'ego_w', 'command')
LANE_T = ('lanes', 'lane_feat', 'lane_mask', 'route_rel', 'route_valid', 'e_src', 'e_dst',
          'e_rel', 'e_mask', 'a2l_idx', 'a2l_rel', 'a2l_mask')
# the tensors the poison is expected to move.  lane_mask / a2l_mask / route_valid are NOT in it
# on purpose: the poison randomises lane CONTENT and lane WIRING and keeps the mask structure,
# so the default arm still runs on a structurally valid graph and the comparison isolates the
# map information rather than the graph's shape.
POISONED_T = ('lanes', 'lane_feat', 'route_rel', 'e_src', 'e_dst', 'e_rel', 'a2l_idx', 'a2l_rel')
EGO_T = ('agents', 'agent_mask', 'ego_w', 'command')


def tdiff(ga, gb):
    return [k for k in TENSORS if not torch.equal(getattr(ga, k), getattr(gb, k))]


def stdiff(sa, sb):
    return [k for k in sa if not (torch.equal(sa[k][0], sb[k][0])
                                  and torch.equal(sa[k][1], sb[k][1]))]


# ── parameter groups ─────────────────────────────────────────────────────────
LANE_MOD = ('seg', 'lane', 'route', 'mp1', 'mp2', 'edge', 'Wq', 'Wk', 'Wke', 'Wv', 'Wve',
            'Wo', 'Wv2', 'Wve2', 'Wo2', 'ln_l')          # consume or feed ONLY lane tokens
EGO_MOD = ('phi', 'qphi', 'q_mlp', 'astep', 'aout', 'ln_a', 'Wk2', 'Wv3', 'zout', 'head')


def main():
    print(__doc__.split('\n\n')[0], flush=True)
    print(f'torch {torch.__version__}  device {DEV}\n', flush=True)

    print('=== 0. build the poisoned graph file ===', flush=True)
    real_npz = f'{RG}/b2d_relgraph_v2.npz'
    d0, dp, same, diff = make_poison(real_npz, POISON)
    print(f'  poisoned copy -> {POISON}', flush=True)
    print(f'  randomised   : {", ".join(diff)}', flush=True)
    print(f'  copied verbatim: {", ".join(same)}', flush=True)
    check('every lane number / lane index actually changed',
          set(diff) == set(LANE_KEYS + WIRE_KEYS))
    check('ego / command / agents / masks / ids copied byte for byte',
          set(same) == set(KEEP_KEYS))

    print('\n=== 1. INPUT LEVEL ===', flush=True)
    g_ar, ids = load(real_npz, True)              # ablated,   real
    g_ap, _ = load(POISON, True)                  # ablated,   poisoned
    g_ur, _ = load(real_npz, False)               # unablated, real
    g_up, _ = load(POISON, False)                 # unablated, poisoned
    N = g_ar.N
    rows_all = np.arange(N)
    st_ar = R.graph_stats(g_ar, rows_all, torch, DEV)
    st_ap = R.graph_stats(g_ap, rows_all, torch, DEV)
    st_ur = R.graph_stats(g_ur, rows_all, torch, DEV)
    st_up = R.graph_stats(g_up, rows_all, torch, DEV)

    dif_ab = tdiff(g_ar, g_ap)
    dif_un = tdiff(g_ur, g_up)
    print(f'  --ablate-lane : tensors differing real vs poisoned = {dif_ab or "NONE"}', flush=True)
    print(f'  default arm   : tensors differing real vs poisoned = {dif_un}', flush=True)
    check('ABLATED model inputs are bit-identical under a fully random map', not dif_ab,
          f'({len(TENSORS)} tensors compared)')
    check('UNABLATED model inputs do change (the test is not vacuous)',
          set(dif_un) >= set(POISONED_T),
          f'({len(dif_un)} of {len(TENSORS)} tensors move; the mask tensors are held fixed '
          f'by design)')
    check('ego / command / agent tensors identical between ABLATED and DEFAULT on the real graph',
          all(torch.equal(getattr(g_ar, k), getattr(g_ur, k)) for k in EGO_T))
    print(f'  standardisation constants differing real vs poisoned: '
          f'ablated {stdiff(st_ar, st_ap) or "NONE"} / default {stdiff(st_ur, st_up)}', flush=True)
    check('graph_stats constants bit-identical under --ablate-lane', not stdiff(st_ar, st_ap))
    check('agent standardisation constants identical between ABLATED and DEFAULT',
          not stdiff({'ag': st_ar['ag']}, {'ag': st_ur['ag']}))
    print(f'  live lane tokens {int(g_ar.lane_mask.sum())}, live L2L edges '
          f'{int(g_ar.e_mask.sum())}, live A2L candidates {int(g_ar.a2l_mask.sum())}, '
          f'route_valid windows {int((g_ar.route_valid > 0.5).sum())}  (ablated)', flush=True)
    check('no lane token, no L2L edge, no A2L candidate and no route row survives',
          int(g_ar.lane_mask.sum()) == 0 and int(g_ar.e_mask.sum()) == 0
          and int(g_ar.a2l_mask.sum()) == 0 and int((g_ar.route_valid > 0.5).sum()) == 0)

    print('\n=== 2. FORWARD LEVEL ===', flush=True)
    torch.manual_seed(0)
    m = R.build_r2(torch, nn, 64).to(DEV)
    torch.manual_seed(0)
    m2 = R.build_r2(torch, nn, 64).to(DEV)        # same seed -> same weights, both arms
    import b2d_earlystop as es
    check('R2Net is unchanged by the arm: same parameter count and same init hash',
          es.init_hash(m) == es.init_hash(m2)
          and sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in m2.parameters()),
          f'{sum(p.numel() for p in m.parameters()):,} params, hash {es.init_hash(m)}')
    m.eval()

    noag = np.where(~np.asarray(g_ar.agent_mask.cpu()).any(2).any(1))[0]
    B, WPR = 8, 3
    rows = np.concatenate([np.arange(B * WPR - len(noag)), noag])[:B * WPR]
    rows_t = torch.tensor(rows)
    ego = {}
    for nm, mod in (('q_mlp', m.q_mlp), ('aout', m.aout)):
        mod.register_forward_hook(lambda mo, i, o, nm=nm: ego.__setitem__(nm, o.detach().clone()))

    def enc(g, st):
        ego.clear()
        with torch.no_grad():
            z = m.encode(g, rows_t, st)
        return z, {k: v for k, v in ego.items()}

    z_ar, h_ar = enc(g_ar, st_ar)
    z_ap, _ = enc(g_ap, st_ap)
    z_ur, h_ur = enc(g_ur, st_ur)
    z_up, _ = enc(g_up, st_up)
    check('encode(): ABLATED output bit-identical real vs poisoned', torch.equal(z_ar, z_ap),
          f'max|d| {float((z_ar - z_ap).abs().max()):.3e}')
    check('encode(): DEFAULT output does change real vs poisoned', not torch.equal(z_ur, z_up),
          f'max|d| {float((z_ur - z_up).abs().max()):.3e}')
    check('encode(): ABLATED differs from DEFAULT on the real graph (the arm is a real ablation)',
          not torch.equal(z_ar, z_ur), f'max|d| {float((z_ar - z_ur).abs().max()):.3e}')
    check('agent sub-path (aout) bit-identical ABLATED vs DEFAULT, real graph',
          torch.equal(h_ar['aout'], h_ur['aout']),
          f'max|d| {float((h_ar["aout"] - h_ur["aout"]).abs().max()):.3e}')
    check('ego sub-path (qphi -> q_mlp) bit-identical ABLATED vs DEFAULT, real graph',
          torch.equal(h_ar['q_mlp'], h_ur['q_mlp']),
          f'max|d| {float((h_ar["q_mlp"] - h_ur["q_mlp"]).abs().max()):.3e}')

    # full forward, ego sequence included
    g_ = torch.Generator().manual_seed(7)
    X = torch.randn(B, 24, r0_ego.NIN, generator=g_)
    XM = torch.ones(B, 24, dtype=torch.bool)
    wb = torch.tensor(np.repeat(np.arange(B), WPR))
    ww = torch.tensor(np.tile(np.arange(WPR), B))

    def fwd(g, st):
        with torch.no_grad():
            return m(X, XM, g, rows_t, wb, ww, WPR, st)

    f_ar, f_ap, f_ur, f_up = fwd(g_ar, st_ar), fwd(g_ap, st_ap), fwd(g_ur, st_ur), fwd(g_up, st_up)
    check('forward(): ABLATED prediction bit-identical real vs poisoned', torch.equal(f_ar, f_ap),
          f'max|d| {float((f_ar - f_ap).abs().max()):.3e}')
    check('forward(): DEFAULT prediction does change real vs poisoned', not torch.equal(f_ur, f_up),
          f'max|d| {float((f_ur - f_up).abs().max()):.3e}')
    check('forward(): ABLATED output is finite', bool(torch.isfinite(f_ar).all()))

    print('\n=== 3. GRADIENT LEVEL ===', flush=True)
    m.zero_grad(set_to_none=True)
    m.train()
    out = m(X, XM, g_ar, rows_t, wb, ww, WPR, st_ar)
    out.pow(2).sum().backward()
    lane_g, ego_g, bad = [], [], []
    for n_, p in m.named_parameters():
        head = n_.split('.')[0]
        gmax = 0.0 if p.grad is None else float(p.grad.abs().max())
        if head == 'typ':                          # (2, d): row 0 lane type, row 1 agent type
            gl = 0.0 if p.grad is None else float(p.grad[0].abs().max())
            ge = 0.0 if p.grad is None else float(p.grad[1].abs().max())
            lane_g.append(('typ[0] lane-type embedding', gl)); ego_g.append(('typ[1]', ge))
            continue
        (lane_g if head in LANE_MOD else ego_g if head in EGO_MOD else bad).append((n_, gmax))
    check('every parameter is classified as lane-pathway or ego/agent/head',
          not bad, f'unclassified: {[b[0] for b in bad]}')
    worst_l = max(lane_g, key=lambda kv: kv[1])
    zero_e = [k for k, v in ego_g if v == 0.0]
    print(f'  lane-pathway parameters   : {len(lane_g)} tensors, max |grad| over all of them '
          f'{worst_l[1]:.3e}  (worst: {worst_l[0]})', flush=True)
    print(f'  ego/agent/head parameters : {len(ego_g)} tensors, min max|grad| '
          f'{min(v for _, v in ego_g):.3e}, max {max(v for _, v in ego_g):.3e}', flush=True)
    check('NO lane-pathway parameter receives a non-zero gradient under --ablate-lane',
          worst_l[1] == 0.0)
    check('the ego/agent/head parameters DO train (the loss is not degenerate)',
          not zero_e, f'zero-grad ego params: {zero_e}')
    m.zero_grad(set_to_none=True)
    m.train()
    m(X, XM, g_ur, rows_t, wb, ww, WPR, st_ur).pow(2).sum().backward()
    lw = max((0.0 if p.grad is None else float(p.grad.abs().max()))
             for n_, p in m.named_parameters() if n_.split('.')[0] in LANE_MOD)
    check('the same parameters DO get gradient in the DEFAULT arm (they are not dead code)',
          lw > 0.0, f'max |grad| {lw:.3e}')

    print('\n=== 4. SANITY: the empty-node windows ===', flush=True)
    print(f'  B2D windows with zero live agents: {list(noag)}  '
          f'(these have zero nodes at all once the lanes are gone)', flush=True)
    with torch.no_grad():
        z_no = m.encode(g_ar, torch.tensor(noag), st_ar)
    check('empty-node windows give a finite readout', bool(torch.isfinite(z_no).all()),
          f'|z| max {float(z_no.abs().max()):.4f}')
    m.eval()
    with torch.no_grad():
        z_full = m.encode(g_ar, torch.arange(N), st_ar)
    check('no NaN / Inf anywhere in the ablated encoder over all '
          f'{N} B2D windows', bool(torch.isfinite(z_full).all()))
    with torch.no_grad():
        z_full_un = m.encode(g_ur, torch.arange(N), st_ur)
    print(f'  window embedding spread: ablated sd {float(z_full.std()):.4f} vs default '
          f'{float(z_full_un.std()):.4f}  (the ablated encoder is not a constant)', flush=True)
    check('the ablated encoder still varies across windows', float(z_full.std()) > 1e-6)

    print('\n=== 5. REGRESSION: the lane-carrying arms are untouched ===', flush=True)
    ref = os.environ.get('NOLANE_REF_IMPL', f'{RG}/r2_graph.py.bak_prelane')
    if not os.path.exists(ref):
        print(f'  SKIP: no pre-ablation reference implementation at {ref}', flush=True)
    else:
        import importlib.util
        from importlib.machinery import SourceFileLoader
        ld = SourceFileLoader('r2_graph_pre_nolane', ref)     # the .bak suffix is not importable
        sp = importlib.util.spec_from_loader(ld.name, ld)
        OLD = importlib.util.module_from_spec(sp); ld.exec_module(OLD)
        check('the reference implementation predates --ablate-lane',
              'ablate_lane' not in OLD.load_graph.__code__.co_varnames)
        g_o, _ = OLD.load_graph('b2d', torch, DEV)
        st_o = OLD.graph_stats(g_o, rows_all, torch, DEV)
        check('DEFAULT-arm graph tensors identical to the pre-ablation code',
              all(torch.equal(getattr(g_ur, k), getattr(g_o, k)) for k in TENSORS))
        check('DEFAULT-arm graph_stats identical to the pre-ablation code',
              not stdiff(st_ur, st_o))
        torch.manual_seed(0); m_o = OLD.build_r2(torch, nn, 64).to(DEV); m_o.eval()
        torch.manual_seed(0); m_n = R.build_r2(torch, nn, 64).to(DEV); m_n.eval()
        with torch.no_grad():
            zo = m_o.encode(g_o, torch.arange(N), st_o)
            zn = m_n.encode(g_ur, torch.arange(N), st_ur)
        check('DEFAULT-arm encode() BIT-IDENTICAL over all '
              f'{N} windows (the one-line readout guard is a no-op there)',
              torch.equal(zn, zo), f'max|d| {float((zn - zo).abs().max()):.3e}')

    print(f'\n=== {sum(OK)}/{len(OK)} checks passed ===', flush=True)
    return 0 if all(OK) else 1


if __name__ == '__main__':
    sys.exit(main())
