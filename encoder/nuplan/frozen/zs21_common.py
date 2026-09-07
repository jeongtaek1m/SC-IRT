#!/usr/bin/env python3
"""PROTOCOL_R.md section 21.1 — the window-native (R*-W) machinery shared by r2w.py (R2-W,
arms A1..A4) and r0w.py (R0-W, control C1).

WHY A SHARED MODULE.  Section 21.3 compares R2-W with R0-W, so every part that is NOT the
encoder — window selection, the input cap, the two compositions, the recipe, the split, the
output format — must be bit-identical between the two arms or the comparison is between two
implementations rather than between "graph" and "no graph".  This is the same reason
transfer_common.py exists for the section-18 arms, and it is modelled on that file: the arm
supplies only its model and its per-window forward.

WHAT IS REUSED, NOT RETYPED (the frozen files are imported as libraries; none is edited):
  r0_ego.build_b2d          routes / types / Y / static, and the STITCHED 2 Hz speed track that
                            the window-selection rule needs (X[..., 0] under MK).
  r0_ego.build              the SeqNet whose .phi is the ego branch of both arms.
  r0_ego.build_navsim       NavSim target row order (assert only).
  hrel.fixed_k_select       the section-19.1 Z2 / section-21.1 K-window rule, verbatim
                            (hrel.py:296-312).  This module only ADDS the 2 m anchor collapse
                            that section 21.1 requires on top of it.
  transfer_common.window_ego_feats   window-local ego features (source AND target).
  transfer_common.inner_split / FullTrainPlan / plan_fields / predict / device
  transfer_common.subsample_logs     (unused here; B2D source only)
  r2_graph.load_graph       the frozen graph loader (also used by the section-5b gate's
                            production round trip).
  r2_graph.graph_stats      train-split graph standardisation.
  r2_graph.dist_pool        the [mean, max, softmin, std] pool, VERBATIM r0_ego.SeqNet's.
  r2_graph.build_r2         the frozen R2Net; R2-W keeps its encode() and its ego phi and
                            replaces only the head (see r2w.py).
  b2d_earlystop             TwoStage / Stage / Selector / Ledger / HeldOutGuard / erase_heldout /
                            assert_masked / fingerprint / init_hash — the whole B2D selection and
                            leakage apparatus, unmodified.
  b2d_splits.unified_split, R_DRAWS   the 16 draws of Table 1.
  scirt.encoder.rasch       theta / b_ref.

WHAT COULD NOT BE REUSED AND IS RE-IMPLEMENTED HERE (each with its diff stated):
  fit_stage_w   transfer_common.fit_stage's B2D branch with ONE substitution: the cell
                probability.  fit_stage closes over a (B,) vector b and computes
                p = sigmoid(b + sigma*eps - theta); the section-21.1 compositions need a (B, K)
                matrix of window difficulties.  The closure cannot be reached from outside, so
                the loop is copied and the four lines that build `p` are replaced.  Every other
                line — optimiser, epoch loop, shuffling, clipping, logging, the sigma prior, the
                plan.update call — is character-for-character transfer_common.fit_stage.
                fit_stage_diff() prints the diff.
  cell_nll_w    b2d_earlystop.cell_nll with the same one substitution, same reason.
  apply_cap     new (section 21.1 cap 16/10); nothing frozen does this.
  collapse_anchors  new (section 21.1 K_eff <= 8); hrel's rule REPEATS the nearest window
                instead, which section 21.1 forbids (critic #9).

DISCIPLINE.  Nothing in this module reads a nuPlan / navtest / navhard response.  The B2D
response matrix is whatever r0_ego.build_b2d reads by default (section 21.4).
"""
import os
import re
import sys

import numpy as np

RG = '/data2/jeongtae/relgraph'
ZS21 = f'{RG}/transfer/zs21'
for p in (RG, '/home/jeongtae/SC-IRT', '/home/jeongtae/SCIRT/b2d_irt'):
    if p not in sys.path:
        sys.path.insert(0, p)

import b2d_earlystop as es                            # noqa: E402
import hrel                                           # noqa: E402
import r0_ego                                         # noqa: E402
import r2_graph as r2                                 # noqa: E402
import transfer_common as tc                          # noqa: E402

# ── section 21.1 frozen constants (source-only; no target quantile is ever consulted) ────────
K_WINDOWS = 8               # windows per B2D route at normalised arclength 0, 1/7, ..., 1
ANCHOR_MERGE_M = 2.0        # anchors closer than this along the route collapse to one window
CAP_LANES = 16              # k-nearest lanes by ego-anchor distance
CAP_AGENTS = 10             # k-nearest agents by ego-anchor distance
DROP_AGENT_CHANNELS = ('hlen', 'hwid', 'isveh')       # section 21.1 drop (r2_graph's own flag
                                                      # semantics: stats on all 8, slice after
                                                      # standardisation)
STD_EPS = 1e-12             # the eps r0_ego.SeqNet's std block lacks; a constant 12-step window
                            # makes var exactly 0 and sqrt'(0) = inf (draft 21.3.2)
COMPOSITIONS = ('survival', 'mil')


# ══ (a) window selection ════════════════════════════════════════════════════════════════════
def route_rows_and_speed():
    """r0_ego.build_b2d + the graph row order, in ONE place.

    -> dict with routes / types / Y / fail / static (r0_ego.build_b2d, verbatim),
       rows[i]   = ALL graph rows of route i, sorted by window index (= r2_graph.route_windows)
       speed[i]  = the route's stitched UNIQUE 2 Hz speed track, which is exactly the
                   `sp` that hrel.fixed_k_select consumes (r0_ego.build_b2d's X channel 0 under
                   MK; step_feats puts speed first, so no re-stitching is typed here).
    """
    X, MK, routes, types, Y, static = r0_ego.build_b2d()
    d = np.load(r0_ego.B2D_NPZ, allow_pickle=True)
    ids = np.array([str(x) for x in d['item_id']])
    rows = tc.b2d_windows_by_route(ids, routes)
    speed = [np.ascontiguousarray(X[i][MK[i]][:, 0]) for i in range(len(routes))]
    for i in range(len(routes)):
        assert len(speed[i]) == r0_ego.STRIDE * (len(rows[i]) - 1) + 12, \
            'stitched length does not match the window count — the fixed-K rule would misalign'
    return dict(routes=routes, types=types, Y=Y, fail=np.nanmean(Y, 0), static=static,
                rows=rows, speed=speed, ids=ids, X_stitched=X, MK_stitched=MK)


def collapse_anchors(sel, sp, W, merge_m=ANCHOR_MERGE_M):
    """section 21.1: the K selected windows with anchors closer than `merge_m` COLLAPSED to one,
    so a route contributes K_eff <= K DISTINCT windows and never a repeat.

    hrel.fixed_k_select repeats the nearest window when W < K or when anchors coincide; section
    21.1 forbids the repeat (critic #9), so this is the only step added on top of the frozen
    rule.  Separation is measured along the same cumulative arclength fixed_k_select itself uses
    (cumsum |speed| * DT at the anchor steps), so the two share one definition of "where the
    anchor is".  Measured on B2D: the 2,656 windows reduce to 1,105 spatially distinct ones under
    this rule, and to the SAME 1,105 under Euclidean dead-reckoned anchor distance, so the choice
    of metric is immaterial at 2 m.  (The draft's 1,092 comes from a different scratchpad
    reconstruction and is not reproduced here — UNVERIFIED.)
    `sel` is non-decreasing because fixed_k_select argmins increasing targets against a
    non-decreasing arclength, so the sequential comparison below sees every anchor in order."""
    anchors = r0_ego.STRIDE * np.arange(W) + r2.ANCHOR_T
    cum = np.concatenate([[0.0], np.cumsum(np.abs(sp, dtype=np.float64) * r0_ego.DT)])
    s = cum[anchors]
    keep = []
    for w in np.asarray(sel, np.int64):
        if not keep or abs(s[w] - s[keep[-1]]) > merge_m:
            keep.append(int(w))
    return np.array(keep, np.int64)


def select_windows(src, arm_all_windows=False, K=K_WINDOWS, verbose=True):
    """-> W: list of graph-row arrays, one per route.

    arm_all_windows (arm A3): every window of the route, duplicates included.
    otherwise:               hrel.fixed_k_select(K) then collapse_anchors -> K_eff <= K."""
    rows, speed = src['rows'], src['speed']
    W, keff = [], []
    for i in range(len(rows)):
        idx = rows[i]
        if arm_all_windows:
            W.append(idx); keff.append(len(idx)); continue
        sel, _ = hrel.fixed_k_select(speed[i], len(idx), K)
        sel = collapse_anchors(sel, speed[i], len(idx))
        assert len(set(sel.tolist())) == len(sel), 'a window repeats after the anchor collapse'
        W.append(idx[sel]); keff.append(len(sel))
    keff = np.array(keff)
    if verbose:
        h = np.bincount(keff, minlength=K + 1)
        print(f'[zs21] window set: {"ALL windows (arm A3)" if arm_all_windows else f"K={K} + {ANCHOR_MERGE_M} m anchor collapse"} '
              f'-> {int(keff.sum())} window rows over {len(W)} routes, K_eff mean {keff.mean():.3f} '
              f'min {keff.min()} max {keff.max()}', flush=True)
        if not arm_all_windows:
            print('[zs21] K_eff histogram ' +
                  '  '.join(f'{k}:{int(h[k])}' for k in range(1, K + 1) if h[k]), flush=True)
    return W, keff


def assert_fixed_k_matches_hrel(src, n_routes=20, K=K_WINDOWS):
    """section 21.1 asks for the hrel.py:296-312 rule REUSED or reimplemented identically.
    It is reused (select_windows calls it), so this asserts the weaker but still meaningful
    thing: on the first `n_routes` routes the pre-collapse selection is exactly hrel's, and the
    collapse only ever removes members of that selection."""
    for i in range(min(n_routes, len(src['rows']))):
        sel, used_arc = hrel.fixed_k_select(src['speed'][i], len(src['rows'][i]), K)
        keep = collapse_anchors(sel, src['speed'][i], len(src['rows'][i]))
        assert len(sel) == K
        assert set(keep.tolist()) <= set(sel.tolist()), 'collapse invented a window'
        assert list(keep) == sorted(set(keep.tolist())), 'collapse reordered or duplicated'
    print(f'[zs21] fixed-K equality check on {min(n_routes, len(src["rows"]))} routes: '
          f'selection == hrel.fixed_k_select, collapse is a subset — PASS', flush=True)


# ══ (b) input cap ═══════════════════════════════════════════════════════════════════════════
def _lane_dist_and_content(d):
    """Lane distance to the ego anchor and the tie-break content key.

    Distance = min over the N_SEG segment START points the encoder actually reads
    (r2_graph encode(): g.lanes[..., :N_SEG, :]), in the ego-anchor frame whose origin IS the
    anchor (KEYS.md).  Verified on all three tensors: no valid lane carries a zero-padded
    polyline slot, so the min is over real geometry.
    Content = that same slice plus lane_feat plus route_rel, used ONLY to break exact distance
    ties lexicographically.  Without it the kept set would depend on lane INDEX, and the
    section-5b permutation gate would (correctly) fail: measured boundary ties are 18/1394 (B2D),
    620/10986 (NavSim), 51/1137 (nuPlan).  With it, ZERO ties survive on any of the three."""
    lanes = np.asarray(d['lanes'], np.float32)[:, :, :r2.N_SEG, :]
    dist = np.hypot(lanes[..., r2.LN_X], lanes[..., r2.LN_Y]).min(2)
    content = np.concatenate([lanes.reshape(lanes.shape[0], lanes.shape[1], -1),
                              np.asarray(d['lane_feat'], np.float32),
                              np.asarray(d['route_rel'], np.float32)], -1)
    return dist, content


def _agent_dist_and_content(d):
    """Agent distance = |(dx, dy)| at the ANCHOR step (r2_graph.ANCHOR_T).  Verified on all three
    tensors: every agent that is live anywhere is live at the anchor step, so this is defined for
    every agent the encoder can see (`av = agent_mask.any(2)`)."""
    ag = np.asarray(d['agents'], np.float32)
    dist = np.hypot(ag[:, :, r2.ANCHOR_T, r2.AG_DX], ag[:, :, r2.ANCHOR_T, r2.AG_DY])
    content = ag.reshape(ag.shape[0], ag.shape[1], -1)
    return dist, content


def _knearest(mask, dist, content, k):
    """Boolean keep-mask: the k valid entries nearest the anchor, ties broken lexicographically
    on content so the selection is a function of the SET, not of the axis order."""
    N, S = mask.shape
    big = np.float32(1e30)
    dd = np.where(mask, dist, big).astype(np.float64)
    keys = [content[:, :, c].ravel() for c in range(content.shape[-1] - 1, -1, -1)]
    keys += [dd.ravel(), np.repeat(np.arange(N), S)]      # last key = most significant
    # lexsort returns FLAT positions; the most significant key is the row id, so block n holds
    # exactly row n's entries and the column index is the offset inside the block
    order = np.lexsort(keys).reshape(N, S) - (np.arange(N) * S)[:, None]
    rank = np.empty((N, S), np.int64)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(S), (N, S)), 1)
    keep = mask & (rank < k)
    # the cut must be unambiguous: the k-th and (k+1)-th entries may not be identical
    amb = 0
    for n in range(N):
        if int(mask[n].sum()) <= k:
            continue
        a, b = order[n, k - 1], order[n, k]
        if dd[n, a] == dd[n, b] and np.array_equal(content[n, a], content[n, b]):
            amb += 1
    assert amb == 0, f'{amb} windows have an ambiguous cap boundary — the cap is not set-invariant'
    return keep


def cap_masks(d, n_lane=CAP_LANES, n_agent=CAP_AGENTS):
    """-> (keep_lane (N,M) bool, keep_agent (N,A) bool, stats dict).  Reads the frozen npz only."""
    lm = np.asarray(d['lane_mask'], bool)
    am_any = np.asarray(d['agent_mask'], bool).any(2)
    ld, lc = _lane_dist_and_content(d)
    ad, ac = _agent_dist_and_content(d)
    keep_l = _knearest(lm, ld, lc, n_lane)
    keep_a = _knearest(am_any, ad, ac, n_agent)
    st = dict(n_windows=int(lm.shape[0]),
              lanes_before=float(lm.sum(1).mean()), lanes_after=float(keep_l.sum(1).mean()),
              agents_before=float(am_any.sum(1).mean()), agents_after=float(keep_a.sum(1).mean()),
              frac_lanes_removed=float(1 - keep_l.sum() / max(lm.sum(), 1)),
              frac_agents_removed=float(1 - keep_a.sum() / max(am_any.sum(), 1)),
              frac_windows_lane_cut=float((lm.sum(1) > n_lane).mean()),
              frac_windows_agent_cut=float((am_any.sum(1) > n_agent).mean()))
    return keep_l, keep_a, st


def apply_cap(g, torch, keep_l, keep_a, tag=''):
    """Turn the removed nodes OFF in every mask the frozen encoder reads, consistently:
    lane_mask / agent_mask, the augmented L2L edge mask (an edge dies if either endpoint dies)
    and a2l_mask (a candidate dies if its agent or its lane dies).  The tensors themselves are
    NOT compacted: r2_graph.encode is mask-driven everywhere (hl *= LM, softmax masked_fill,
    index_add_ gated by AM & av, the readout's nmask), so a masked node contributes exactly
    nothing and this is equivalent to deleting it — while keeping M / A / EA and therefore the
    frozen loader, the frozen graph_stats and the section-5b gate usable unchanged."""
    dev = g.lane_mask.device
    kl = torch.tensor(np.ascontiguousarray(keep_l), device=dev)
    ka = torch.tensor(np.ascontiguousarray(keep_a), device=dev)
    g.lane_mask = g.lane_mask & kl
    g.agent_mask = g.agent_mask & ka[:, :, None]
    N, A, K = g.a2l_idx.shape
    lane_ok = torch.gather(kl, 1, g.a2l_idx.clamp(min=0).reshape(N, A * K)).reshape(N, A, K)
    g.a2l_mask = g.a2l_mask & lane_ok & ka[:, :, None]
    src_ok = torch.gather(kl, 1, g.e_src.clamp(min=0))
    dst_ok = torch.gather(kl, 1, g.e_dst.clamp(min=0))
    g.e_mask = g.e_mask & src_ok & dst_ok
    assert not bool((g.a2l_mask & ~g.agent_mask[:, :, r2.ANCHOR_T][..., None]).any()), \
        'the cap left an a2l candidate on an agent that is off'
    print(f'[zs21 cap{tag}] lanes/window {float(kl.sum(1).float().mean()):.2f} '
          f'agents/window {float(ka.sum(1).float().mean()):.2f} '
          f'| live L2L edges/window {float(g.e_mask.sum(1).float().mean()):.1f} '
          f'| live A2L candidates/window {float(g.a2l_mask.sum((1, 2)).float().mean()):.1f}',
          flush=True)
    return g


NPZ_OF = {'b2d': lambda: r0_ego.B2D_NPZ, 'navsim': lambda: r0_ego.NAVSIM_NPZ,
          'nuplan': lambda: r0_ego.NUPLAN_NPZ}


def load_graph_capped(domain, torch, dev, cap=True, nuplan_ego='routed', tag=''):
    """r2_graph.load_graph, then the section-21.1 cap.  -> (g, ids, cap_stats or None)."""
    g, ids = r2.load_graph(domain, torch, dev, nuplan_ego=nuplan_ego)
    if not cap:
        print(f'[zs21 cap{tag}] --no-cap: the frozen node sets are used unchanged', flush=True)
        return g, ids, None
    d = np.load(NPZ_OF[domain](), allow_pickle=True)
    keep_l, keep_a, st = cap_masks(d)
    apply_cap(g, torch, keep_l, keep_a, tag=f' {domain}{tag}')
    st['domain'] = domain
    return g, ids, st


def agent_keep_channels(drop):
    """r2_graph's own --drop-agent-channels semantics: names -> the ag_keep index list that
    build_r2 slices with AFTER standardisation (stats stay on all 8 channels)."""
    if not drop:
        return None, []
    drop = sorted(set(drop))
    assert set(drop) <= set(r2.AG_NAMES), f'unknown agent channel; names are {r2.AG_NAMES}'
    return [i for i, n in enumerate(r2.AG_NAMES) if n not in drop], drop


# ══ (d) compositions ════════════════════════════════════════════════════════════════════════
def survival_offset(keff):
    """logit(0.5 ** (1/K_eff)) — the section-21.1 initial offset.  b_k := raw_k - offset makes
    P(route pass) = 0.5 at raw = theta = 0 for EVERY K_eff.  MEASURED CAVEAT (verification
    2026-09-02): that exact neutrality holds only at raw head output == 0.  The real untrained A1
    head outputs about -0.11, and at init rho(b~ readout, K_eff) = +0.61 with the offset vs +0.99
    without it -- the offset removes most of the section-15.7 window-count cue, not all of it.  On
    the trained models the residue is small (rho(pred_src, K_eff) = -0.16 A1 / -0.21 A2 / -0.28 C1
    against rho(K_eff, b_ref) = -0.10, and the partial rho vs b_ref given K_eff barely moves), but
    the in-domain survival readout is NOT K_eff-free.  The transfer readout is unaffected: every
    target route has K=1, where b~ == b_k.  Nor is the composition count-invariant: duplicating
    every window shifts the offset-corrected route logit by +0.05 at fixed b_k (attention-MIL is
    exactly count-invariant, +0.000)."""
    q = 0.5 ** (1.0 / np.maximum(np.asarray(keff, np.float64), 1.0))
    return np.log(q / (1.0 - q))


def survival_logp(torch, bt, wm, sg, gx, THE):
    """P(route pass | theta_j, eps_g) = prod_k sigmoid(theta_j - b_k - sigma*eps_g), with eps
    SHARED inside the route (one GH node per route, marginalised exactly as
    transfer_common.fit_stage marginalises the single-b likelihood).

    Sign convention is the frozen one: z_k = (b_k + sigma*eps) - theta, so a window's FAIL
    probability is sigmoid(z_k) and its pass probability is sigmoid(-z_k).
      log P(pass)  = sum_k logsigmoid(-z_k)                              -> S, exact
      log P(fail)  = log(-expm1(S))                                      -> stable form
    S is clamped strictly below 0 (by one float ulp-scale constant) so -expm1(S) > 0 and the
    backward pass of the log can never see 0.

    bt (B, K) window difficulties, wm (B, K) bool window mask, THE (J,), gx (G,).
    -> (log_pfail, log_ppass), each (J, B, G)."""
    z = bt[None, :, :, None] + sg * gx[None, None, None, :] - THE[:, None, None, None]
    lp = torch.nn.functional.logsigmoid(-z) * wm[None, :, :, None]      # masked window: log 1 = 0
    S = lp.sum(2).clamp(max=-1e-7)
    return torch.log(-torch.expm1(S) + 1e-7), torch.log(torch.exp(S) + 1e-7)


def survival_route_b(torch, bt, wm, sg, gx, lgw):
    """The route readout b~ of the survival composition, in the SAME units as the frozen b~:
    the frozen model's b~ is logit P(item fails | theta = 0), so the route's is
        b~_route = logit( E_eps[ 1 - prod_k sigmoid(-(b_k + sigma*eps)) ] ).
    At K_eff = 1 this reduces to b_k marginalised over eps, i.e. exactly the frozen quantity, so
    the same number is comparable across arms and reduces to the transfer readout on a
    single-window target.  -> (B,)"""
    z = bt[:, :, None] + sg * gx[None, None, :]
    lp = torch.nn.functional.logsigmoid(-z) * wm[:, :, None]
    S = lp.sum(1).clamp(max=-1e-7)                                       # (B, G)
    w = torch.exp(lgw)[None, :]
    pf = ((-torch.expm1(S)) * w).sum(1).clamp(1e-7, 1 - 1e-7)            # (B,)
    return torch.log(pf / (1 - pf))


def mil_attention(torch, nn, d_in, d):
    """Gated attention (Ilse et al. 2018) over a route's windows: a_k >= 0, sum_k a_k = 1.
    b_route = sum_k a_k b_k feeds the FROZEN cell likelihood unchanged, so arm A2 trains through
    transfer_common.fit_stage verbatim.  Window count is invariant by construction."""
    class Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.V = nn.Linear(d_in, d)
            self.U = nn.Linear(d_in, d)
            self.w = nn.Linear(d, 1)

        def forward(self, h, wm):
            e = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
            e = e.masked_fill(~wm, -1e9)
            return torch.softmax(e, 1)
    return Gate()


def attn_entropy(a, wm):
    """Mean entropy of a_k in nats, and the mean log K_eff it is measured against."""
    p = np.where(wm, a, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), 1)
    k = wm.sum(1)
    return float(h.mean()), float(np.log(np.maximum(k, 1)).mean())


# ══ (e) recipe ══════════════════════════════════════════════════════════════════════════════
def cell_nll_w(torch, bt, wm, cols, Yd, Md, THE, gx, lgw, sg, comp):
    """b2d_earlystop.cell_nll with ONE substitution: the cell probability comes from the
    section-21.1 composition instead of a single b.  Everything else — the GH weights, the
    per-observed-cell normalisation, the exclusion of the 0.05*(log sigma)^2 prior — is that
    function line for line.  comp='mil' takes the frozen path exactly (bt is already (B,))."""
    s = torch.tensor(np.asarray(cols, np.int64), device=Yd.device)
    yy = Yd[:, s]; mm = Md[:, s]
    if comp == 'mil':
        z = (bt[None, :, None] + sg * gx[None, None, :]) - THE[:, None, None]
        p = torch.sigmoid(z)
        llc = (yy[:, :, None] * torch.log(p + 1e-7)
               + (1 - yy[:, :, None]) * torch.log(1 - p + 1e-7)) * mm[:, :, None]
    else:
        lf, lp = survival_logp(torch, bt, wm, sg, gx, THE)
        llc = (yy[:, :, None] * lf + (1 - yy[:, :, None]) * lp) * mm[:, :, None]
    return float(-torch.logsumexp(llc.sum(0) + lgw[None, :], 1).sum() / mm.sum())


FIT_STAGE_DIFF = """\
fit_stage_w vs transfer_common.fit_stage (B2D branch only; the NavSim branch is not copied
because section 21 has a B2D source only):
  KEPT VERBATIM   hermegauss(15) + weight normalisation; ls = tensor(-0.5, requires_grad);
                  AdamW(params + [ls], lr=1e-3, weight_decay=0.1); starts() keeps the last
                  partial batch; the per-epoch shuffle from np.random.default_rng(seed);
                  clip_grad_norm_(params, 1.0); the + 0.05*ls**2 sigma prior; the
                  -logsumexp(llc.sum(0) + lgw).sum() / mm.sum() normalisation; plan.update and
                  the identical stdout lines; iv_nll chunking; plan.end_stage(stg, sigma()).
  REPLACED        the four lines
                      z = (bt[None,:,None] + sg*gx[None,None,:]) - THE[:,None,None]
                      p = torch.sigmoid(z)
                      llc = (yy[:,:,None]*log(p+1e-7) + (1-yy[:,:,None])*log(1-p+1e-7))*mm[:,:,None]
                  by the section-21.1 composition (survival: log P(fail) = log(-expm1(sum_k
                  logsigmoid(-z_k))); mil: the SAME four lines on b_route = sum_k a_k b_k).
  REPLACED        es.cell_nll(...) by cell_nll_w(...), which is es.cell_nll with the same one
                  substitution.
  ADDED           fwd(sel) returns (b_window (B,K), window_mask (B,K)) instead of (B,).
"""


def fit_stage_diff():
    return FIT_STAGE_DIFF


def fit_stage_w(torch, nn, m, fwd, stg, plan, Yd, Md, THE, bs, seed, dev, tag, comp,
                chunk=64):
    """See FIT_STAGE_DIFF.  Returns the final sigma."""
    from numpy.polynomial.hermite_e import hermegauss
    idx = np.where(stg.train)[0]
    iv_c = np.where(stg.iv)[0]
    shuf = np.random.default_rng(seed)
    params = list(m.parameters())
    gxn, gwn = hermegauss(15); gwn = gwn / gwn.sum()
    gx = torch.tensor(gxn, dtype=torch.float32, device=dev)
    lgw = torch.log(torch.tensor(gwn, dtype=torch.float32, device=dev))
    ls = torch.tensor(-0.5, device=dev, requires_grad=True)
    opt = torch.optim.AdamW(params + [ls], lr=1e-3, weight_decay=0.1)
    starts = lambda: range(0, len(idx), bs)                       # last partial batch kept

    def batch_loss(sel):
        bt, wm = fwd(sel)
        s = torch.tensor(sel, device=dev)
        yy = Yd[:, s]; mm = Md[:, s]
        sg = torch.exp(ls)
        if comp == 'mil':
            z = (bt[None, :, None] + sg * gx[None, None, :]) - THE[:, None, None]
            p = torch.sigmoid(z)
            llc = (yy[:, :, None] * torch.log(p + 1e-7)
                   + (1 - yy[:, :, None]) * torch.log(1 - p + 1e-7)) * mm[:, :, None]
        else:
            lf, lp = survival_logp(torch, bt, wm, sg, gx, THE)
            llc = (yy[:, :, None] * lf + (1 - yy[:, :, None]) * lp) * mm[:, :, None]
        return -torch.logsumexp(llc.sum(0) + lgw[None, :], 1).sum() / mm.sum() \
            + 0.05 * ls.pow(2)

    def iv_nll():
        with torch.no_grad():
            outs = [fwd(iv_c[i:i + chunk]) for i in range(0, len(iv_c), chunk)]
            bv = torch.cat([o[0] for o in outs])
            wv = torch.cat([o[1] for o in outs]) if outs[0][1] is not None else None
            return cell_nll_w(torch, bv, wv, iv_c, Yd, Md, THE, gx, lgw,
                              torch.exp(ls).detach(), comp)

    sigma = lambda: float(torch.exp(ls))
    for ep in range(stg.epochs):
        m.train(); shuf.shuffle(idx); tl = nb = 0
        for i0 in starts():
            loss = batch_loss(idx[i0:i0 + bs])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tl += float(loss); nb += 1
        if stg.select:
            m.eval(); nll = iv_nll()
            plan.update(stg, ep, tl / max(nb, 1), nll, sigma())
            print(f'  [{tag}] s{stg.no} ep {ep:2d} loss {tl/max(nb,1):.4f} '
                  f'innerval nll {nll:.4f} sigma {sigma():.3f}', flush=True)
        elif ep in (0, 4, stg.epochs - 1):
            print(f'  [{tag}] s{stg.no} ep {ep} loss {tl/max(nb,1):.4f} sigma {sigma():.3f}',
                  flush=True)
    plan.end_stage(stg, sigma())
    return sigma()


# ══ (f) label shuffle ═══════════════════════════════════════════════════════════════════════
def shuffle_route_labels(Y, seed):
    """C2 / A3-shuffle: permute the ROUTE labels (the columns of Y) with a per-seed RNG, leaving
    every feature, split and type in place.  The route keeps its scenario type and its windows
    and receives another route's 16 planner responses.  Scoring still uses the TRUE fail / b_ref,
    so a passing arm here would be reading something that cannot exist."""
    rng = np.random.default_rng(90000 + int(seed))
    perm = rng.permutation(Y.shape[1])
    print(f'[zs21] LABEL SHUFFLE seed {seed}: route label permutation, '
          f'fixed points {int((perm == np.arange(len(perm))).sum())}/{len(perm)}', flush=True)
    return Y[:, perm], perm


# ══ (h) output ══════════════════════════════════════════════════════════════════════════════
def out_path(name):
    os.makedirs(ZS21, exist_ok=True)
    p = f'{ZS21}/{name}'
    for forbidden in ('/frozen30/', '/es/', '/es_pinit/', '/probe/'):
        assert forbidden not in p, p
    return p


def save(path, **kw):
    np.savez(path, **kw)
    print(f'WROTE {path}', flush=True)


def nuplan_logged_ego(g_t, ids_t, torch, dev):
    """section 20.2 S2, verbatim from r2_graph.run_full: swap the LOGGED val14 ego into both the
    ego branch input and g.ego_w.  No nuPlan response is read."""
    e = np.load(r0_ego.NUPLAN_EGO['logged'], allow_pickle=True)
    gid = np.array([f'{str(tk)}_{int(w)}' for tk, w in zip(e['token'], e['widx'])])
    assert np.array_equal(np.array(list(ids_t)), gid), 'logged-ego rows are not the graph rows'
    Xw_l = tc.window_ego_feats(e['ego'].astype(np.float32), e['cmd'].astype(np.float32))
    g_t.ego_w = torch.tensor(Xw_l, device=dev)
    return Xw_l


def nuplan_ids(ids_t):
    wtok = np.array([s.rsplit('_', 1)[0] for s in ids_t])
    wdx = np.array([int(s.rsplit('_', 1)[1]) for s in ids_t])
    return wtok, wdx


# ══ the two loops (the arm supplies only make_model + window_b) ══════════════════════════════
def _readout(torch, arm, bt, wm, sg, gx, lgw):
    """Route-level b~ from the per-window b_k, per the arm's composition."""
    if arm.comp == 'mil':
        return bt                                           # already the route scalar
    return survival_route_b(torch, bt, wm, sg, gx, lgw)


def make_fwd(torch, np_, arm, m, Xd, g, W, st, dev, offs, guard=None, what=''):
    """One closure for both loops.  sel = route indices.
    -> (b (B,) for mil / (B,Kmax) for survival, window mask (B,Kmax) or None).

    The per-window head runs on the route's OWN window rows only; there is no pooling over
    windows anywhere in this function — the composition is applied by the caller's likelihood
    (survival) or by the arm's attention (mil)."""
    Kmax = max(len(w) for w in W)

    def fwd(sel):
        sel = np.asarray(sel, np.int64)
        if guard is not None:
            guard.routes(sel, what)
        rows = np.concatenate([W[i] for i in sel])
        wb = np.concatenate([np.full(len(W[i]), b) for b, i in enumerate(sel)])
        ww = np.concatenate([np.arange(len(W[i])) for i in sel])
        if guard is not None:
            guard.rows(rows, what + ' rows')
        rt = torch.tensor(rows, device=dev)
        bk, hk = m.window_b(Xd[rt], g, rt, st)
        B = len(sel)
        wbt = torch.tensor(wb, device=dev); wwt = torch.tensor(ww, device=dev)
        bm = torch.zeros(B, Kmax, dtype=torch.bool, device=dev)
        bm[wbt, wwt] = True
        bb = torch.zeros(B, Kmax, device=dev, dtype=bk.dtype)
        bb = bb.index_put((wbt, wwt), bk)
        if arm.comp == 'mil':
            hh = torch.zeros(B, Kmax, hk.shape[-1], device=dev, dtype=hk.dtype)
            hh = hh.index_put((wbt, wwt), hk)
            a = m.gate(hh, bm)
            m.last_attn = (a.detach().cpu().numpy(), bm.detach().cpu().numpy())
            return (a * bb * bm).sum(1), bm
        off = torch.tensor(offs[sel], dtype=bb.dtype, device=dev)[:, None]
        return bb - off, bm                                  # section 21.1 survival offset
    return fwd


def run_oof(arm, a):
    """Gate 0 (section 21.0): the B2D 16-draw unified_split OOF, on the SAME split, the same
    b_ref and the same metric as Table 1 / section 12.1, under the es_pinit recipe
    (two-stage checkpoint selection + proper-init) the C0 baseline was re-run with."""
    import torch, torch.nn as nn
    from numpy.polynomial.hermite_e import hermegauss
    from scipy.stats import spearmanr
    from scirt.encoder import rasch
    from b2d_splits import unified_split, R_DRAWS

    dev, devname = tc.device(torch)
    src = route_rows_and_speed()
    assert_fixed_k_matches_hrel(src)
    W, keff = select_windows(src, arm.all_windows)
    routes, types, static = src['routes'], src['types'], src['static']
    Y_true = src['Y']; fail = src['fail']
    _, b_ref = rasch(Y_true, it=800)
    Y = Y_true
    perm = np.arange(len(routes))
    if a.label_shuffle:
        Y, perm = shuffle_route_labels(Y_true, a.seed)
    print(f'[b2d] rho(observed failure rate, full-panel Rasch b) '
          f'{spearmanr(fail, b_ref).correlation:+.4f}', flush=True)

    g = ids = capst = None
    if arm.needs_graph:
        g, ids, capst = load_graph_capped('b2d', torch, dev, cap=a.cap)
        assert list(ids) == list(src['ids']), 'graph rows are not the r0_ego rows'
        Xw = g.ego_w.cpu().numpy()
    else:
        d = np.load(r0_ego.B2D_NPZ, allow_pickle=True)
        Xw = tc.window_ego_feats(d['ego'].astype(np.float32), d['command'].astype(np.float32))
    offs = survival_offset(keff)

    gxn, gwn = hermegauss(15); gwn = gwn / gwn.sum()
    gx = torch.tensor(gxn, dtype=torch.float32, device=dev)
    lgw = torch.log(torch.tensor(gwn, dtype=torch.float32, device=dev))
    R, J = len(routes), Y.shape[0]
    utypes = sorted(set(types))
    pred = np.full((R_DRAWS, R), np.nan)
    per_draw = []
    led = es.Ledger(R_DRAWS, a.epochs, True)

    for draw in range(min(a.draws, R_DRAWS)):
        hp, ht = unified_split(draw, utypes, J)
        keepJ = np.array([j for j in range(J) if j not in hp])
        te = np.isin(types, list(ht)); tr = ~te
        plan = es.TwoStage(draw, types, tr, a.epochs, True)
        guard = es.HeldOutGuard(np.where(te)[0],
                                np.concatenate([W[i] for i in np.where(te)[0]]), hp, keepJ)
        guard.selftest()
        sg = float('nan')
        for stg in plan.stages():
            trn = stg.train
            torch.manual_seed(a.seed); np.random.seed(a.seed)
            th_f, _ = rasch(Y[keepJ][:, trn])
            torch.manual_seed(a.seed)                 # proper-init: re-seed AFTER rasch
            plan.note_theta(stg, th_f)
            rows_tr = guard.rows(np.concatenate([W[i] for i in np.where(trn)[0]]), 'ego/graph stats')
            flat = Xw[rows_tr].reshape(-1, r0_ego.NIN)
            mu = flat.mean(0); sd = flat.std(0) + 1e-6
            Xd = torch.tensor((Xw - mu) / sd, device=dev)
            st = r2.graph_stats(g, rows_tr, torch, dev) if arm.needs_graph else None
            m = arm.make_model(torch, nn, a.d, dev)
            if draw == 0:
                print(f'  [init] draw 0 s{stg.no} seed {a.seed} proper_init=True '
                      f'weight-hash {es.init_hash(m)}', flush=True)
            if draw == 0 and stg.no == 1:
                print(f'[b2d] {arm.tag} params {sum(p.numel() for p in m.parameters()):,}',
                      flush=True)
            Yk = es.erase_heldout(Y[keepJ], te)
            Yd = torch.tensor(np.nan_to_num(Yk), dtype=torch.float32, device=dev)
            Md = torch.tensor((~np.isnan(Yk)).astype(np.float32), device=dev)
            THE = torch.tensor(th_f, dtype=torch.float32, device=dev)
            es.assert_masked(torch, Yd, Md, te, dev)
            print(plan.head(stg), flush=True)
            fwd = make_fwd(torch, np, arm, m, Xd, g, W, st, dev, offs, guard,
                           f'{stg.name} forward')
            sg = fit_stage_w(torch, nn, m, fwd, stg, plan, Yd, Md, THE, a.bs, a.seed, dev,
                             f'b2d draw {draw}', arm.comp)
        print(plan.line(int(te.sum())), flush=True)
        guard.end_training()
        m.eval()
        fwd_h = make_fwd(torch, np, arm, m, Xd, g, W, st, dev, offs, None, '')
        ii = guard.heldout(np.where(te)[0])
        with torch.no_grad():
            outs = [fwd_h(ii[i:i + 32]) for i in range(0, len(ii), 32)]
            pr = torch.cat([_readout(torch, arm, o[0], o[1],
                                     torch.tensor(sg, device=dev), gx, lgw) for o in outs])
        pred[draw, ii] = pr.cpu().numpy()
        led.record(draw, plan, guard)
        print(guard.leak_line(draw), flush=True)
        per_draw.append(spearmanr(pred[draw, ii], fail[ii]).correlation)
        print(f'  [b2d draw {draw}] held-out {te.sum()} routes  '
              f'rho_scene {per_draw[-1]:+.4f}', flush=True)

    per_draw = np.array(per_draw)
    out = out_path(f'{arm.tag}_b2d_oof_s{a.seed}.npz')
    save(out, pred=pred, routes=routes, item_id=routes, types=types, Y=Y_true, fail=fail,
         b_ref=b_ref, static=static, per_draw=per_draw, keff=keff, arm=arm.tag,
         comp=arm.comp, cap=bool(a.cap), all_windows=bool(arm.all_windows),
         label_shuffle=bool(a.label_shuffle), label_perm=perm, seed=a.seed, device=devname,
         drop_agent_channels=np.array(arm.drop), **led.fields())
    Pl, Fl, Bl = [], [], []
    for dd in range(len(per_draw)):
        k = np.isfinite(pred[dd])
        Pl.append(pred[dd][k]); Fl.append(fail[k]); Bl.append(b_ref[k])
    Pp, Ff, Bb = map(np.concatenate, (Pl, Fl, Bl))
    print(f'{arm.tag.upper()}_B2D_OOF seed={a.seed}  '
          f'PRIMARY macro per-draw rho(fail) {per_draw.mean():+.4f} +/- '
          f'{per_draw.std(ddof=1):.4f}  | pooled rho_fail {spearmanr(Pp, Ff).correlation:+.4f}  '
          f'pooled rho_ref {spearmanr(Pp, Bb).correlation:+.4f}  '
          f'({len(Pp)} held-out cells)', flush=True)


def run_full(arm, a):
    """section 18.1 / 21.1: ONE model per seed on ALL 220 B2D routes, then forwarded on the
    target tensors.  Target responses are never read here."""
    import torch, torch.nn as nn
    from numpy.polynomial.hermite_e import hermegauss
    from scipy.stats import spearmanr
    from scirt.encoder import rasch

    dev, devname = tc.device(torch)
    src = route_rows_and_speed()
    assert_fixed_k_matches_hrel(src)
    W, keff = select_windows(src, arm.all_windows)
    routes, types = src['routes'], src['types']
    Y_true = src['Y']; fail = src['fail']
    Y = Y_true
    perm = np.arange(len(routes))
    if a.label_shuffle:
        Y, perm = shuffle_route_labels(Y_true, a.seed)
    _, b_src = rasch(Y_true, it=800)

    g = ids = capst = None
    if arm.needs_graph:
        g, ids, capst = load_graph_capped('b2d', torch, dev, cap=a.cap)
        assert list(ids) == list(src['ids']), 'graph rows are not the r0_ego rows'
        Xw = g.ego_w.cpu().numpy()
    else:
        d = np.load(r0_ego.B2D_NPZ, allow_pickle=True)
        Xw = tc.window_ego_feats(d['ego'].astype(np.float32), d['command'].astype(np.float32))
    offs = survival_offset(keff)

    gxn, gwn = hermegauss(15); gwn = gwn / gwn.sum()
    gx = torch.tensor(gxn, dtype=torch.float32, device=dev)
    lgw = torch.log(torch.tensor(gwn, dtype=torch.float32, device=dev))
    itr, iv = tc.inner_split('b2d', types)
    plan = tc.FullTrainPlan(types, itr, iv, a.epochs)
    Yd = torch.tensor(np.nan_to_num(Y), dtype=torch.float32, device=dev)
    Md = torch.tensor((~np.isnan(Y)).astype(np.float32), device=dev)
    tag = f'{arm.tag} b2d->({a.transfer_to}) s{a.seed}'
    sg = float('nan')
    for stg in plan.stages():
        trn = stg.train
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        th_f, _ = rasch(Y[:, trn])
        torch.manual_seed(a.seed)                     # proper-init
        plan.note_theta(stg, th_f)
        rows_tr = np.concatenate([W[i] for i in np.where(trn)[0]])
        flat = Xw[rows_tr].reshape(-1, r0_ego.NIN)
        mu = flat.mean(0); sd = flat.std(0) + 1e-6
        Xd = torch.tensor((Xw - mu) / sd, device=dev)
        st = r2.graph_stats(g, rows_tr, torch, dev) if arm.needs_graph else None
        m = arm.make_model(torch, nn, a.d, dev)
        print(f'  [init] s{stg.no} seed {a.seed} proper_init=True weight-hash {es.init_hash(m)}',
              flush=True)
        if stg.no == 1:
            print(f'[b2d] {arm.tag} params {sum(p.numel() for p in m.parameters()):,}', flush=True)
        THE = torch.tensor(th_f, dtype=torch.float32, device=dev)
        print(plan.head(stg), flush=True)
        fwd = make_fwd(torch, np, arm, m, Xd, g, W, st, dev, offs, None, '')
        sg = fit_stage_w(torch, nn, m, fwd, stg, plan, Yd, Md, THE, a.bs, a.seed, dev, tag,
                         arm.comp)
    print(plan.line(0), flush=True)
    m.eval()
    chunks, attn = [], []
    with torch.no_grad():
        for i in range(0, len(routes), 32):
            o = fwd(np.arange(i, min(i + 32, len(routes))))
            chunks.append(_readout(torch, arm, o[0], o[1], torch.tensor(sg, device=dev), gx, lgw))
            if arm.comp == 'mil':
                attn.append(m.last_attn)
    pred_src = torch.cat(chunks).cpu().numpy()
    rho_src = float(spearmanr(pred_src, b_src).correlation)
    print(f'  [{tag}] e* {plan.best_ep}, stage 2 {plan.stage2_epochs} epochs on {len(routes)} '
          f'routes | source in-sample rho_ref {rho_src:+.4f} rho_fail '
          f'{spearmanr(pred_src, fail).correlation:+.4f}', flush=True)
    attn_stats = {}
    if arm.comp == 'mil' and attn:
        A = np.concatenate([x[0] for x in attn]); Mm = np.concatenate([x[1] for x in attn])
        ah, lk = attn_entropy(A, Mm)
        amax = float(np.where(Mm, A, 0.0).max(1).mean())
        attn_stats = dict(attn_entropy_nats=ah, attn_mean_log_keff=lk, attn_mean_max_a=amax)
        print(f'  [{tag}] attention over all {len(A)} routes: a_k entropy {ah:.4f} nats vs mean '
              f'log K_eff {lk:.4f} (uniform would match); mean max a_k {amax:.4f}', flush=True)

    base = dict(arm=arm.tag, comp=arm.comp, src='b2d', seed=a.seed, device=devname, bs=a.bs,
                epochs=a.epochs, pred_src=pred_src, src_item_id=routes, src_groups=types,
                src_b_ref=b_src, rho_src_insample=rho_src, x_mu=mu, x_sd=sd,
                sub_logs=np.array([]), cap=bool(a.cap), all_windows=bool(arm.all_windows),
                label_shuffle=bool(a.label_shuffle), label_perm=perm, keff=keff,
                drop_agent_channels=np.array(arm.drop), **attn_stats, **tc.plan_fields(plan))
    if arm.needs_graph:
        base.update({f'gs_{k}_{n}': v.cpu().numpy() for k, (mu_, sd_) in st.items()
                     for n, v in (('mu', mu_), ('sd', sd_))})
    if capst:
        base.update({f'cap_src_{k}': v for k, v in capst.items() if k != 'domain'})

    for tgt in a.transfer_to.split(','):
        out = dict(base); out['tgt'] = tgt
        if arm.needs_graph:
            g_t, ids_t, cst = load_graph_capped(tgt, torch, dev, cap=a.cap,
                                                nuplan_ego='routed', tag=' target')
            if cst:
                out.update({f'cap_tgt_{k}': v for k, v in cst.items() if k != 'domain'})
            Xw_t = g_t.ego_w.cpu().numpy()
        else:
            dd = np.load(NPZ_OF[tgt](), allow_pickle=True)
            ids_t = np.array([str(x) for x in dd['item_id']])
            g_t = None
            if tgt == 'nuplan':
                e = np.load(r0_ego.NUPLAN_EGO['routed'], allow_pickle=True)
                Xw_t = tc.window_ego_feats(e['ego'].astype(np.float32),
                                           e['cmd'].astype(np.float32))
            else:
                Xw_t = tc.window_ego_feats(dd['ego'].astype(np.float32),
                                           dd['command'].astype(np.float32))
        ones = [np.array([r]) for r in range(len(ids_t))]

        def fwd_t(Xt):
            Xd_t = torch.tensor((Xt - mu) / sd, device=dev)
            f = make_fwd(torch, np, arm, m, Xd_t, g_t, ones, st, dev, np.zeros(len(ids_t)))
            with torch.no_grad():
                o = [f(np.arange(i, min(i + 512, len(ids_t)))) for i in range(0, len(ids_t), 512)]
            return torch.cat([x[0][:, 0] if arm.comp != 'mil' else x[0] for x in o]).cpu().numpy()

        pred_w = fwd_t(Xw_t)                     # section 21.1: the target window's own b_k
        if tgt == 'nuplan':
            wtok, wdx = nuplan_ids(ids_t)
            if arm.needs_graph:
                Xw_l = nuplan_logged_ego(g_t, ids_t, torch, dev)
            else:
                el = np.load(r0_ego.NUPLAN_EGO['logged'], allow_pickle=True)
                gid = np.array([f'{str(t)}_{int(w)}' for t, w in zip(el['token'], el['widx'])])
                assert np.array_equal(np.array(list(ids_t)), gid)
                Xw_l = tc.window_ego_feats(el['ego'].astype(np.float32),
                                           el['cmd'].astype(np.float32))
            pred_w_l = fwd_t(Xw_l)
            out.update(pred_routed=pred_w, pred_logged=pred_w_l, pred=pred_w,
                       item_id=ids_t, tgt_groups=wtok, widx=wdx)
        else:
            # log ids come from the NavSim TENSOR file, joined by token.  r2_graph.run_full calls
            # r0_ego.build_navsim here, which also opens the navtest PDMS matrix and throws the
            # responses away; this arm never opens it at all (section 21 discipline).
            t = np.load(r0_ego.NAVSIM_LOGS, allow_pickle=True)
            lbt = {str(nm): str(l) for nm, l in zip(t['names'], t['log'])}
            assert set(map(str, ids_t)) <= set(lbt), 'a navtest token has no log'
            out.update(pred=pred_w, item_id=ids_t,
                       tgt_groups=np.array([lbt[str(x)] for x in ids_t]))
        print(f'  [{tag}] target {tgt}: {len(out["pred"])} window preds', flush=True)
        save(out_path(f'{arm.tag}_b2d2{tgt}_s{a.seed}.npz'), **out)
