#!/usr/bin/env python3
"""R2 — the full relational-graph rung of the R-ladder, its two frozen negative controls,
and the lane-free architectural ablation.

FIVE ARMS, ONE ARCHITECTURE.  Every arm below is the SAME R2Net with the same parameter count
and the same training recipe; they differ only in what the graph tensors contain, so a
difference between them cannot be a capacity difference.

  R2           (default)                    ego + command, agents, lanes + lane_feat, L2L edges,
                                            A2L with a2l_rel, route_rel masked by route_rel_valid
  R2-noRoute   --ablate-route               identical, route_rel masked out ENTIRELY
  R2-noLane    --ablate-lane                identical, the WHOLE lane/map side removed: no lane
                                            tokens, no lane geometry / lane_feat, no L2L, no
                                            A2L, no route_rel.  ego + command + agents only.
  S-route      --shuffle route --shuffle-seed K     identical, route correspondence destroyed
  S-a2l        --shuffle a2l   --shuffle-seed K     identical, A2L correspondence replaced by a
                                            geometry-matched one with RECOMPUTED relations

Everything outside the model input/architecture is r0_ego.py: split construction, gold targets,
the IRT loss, the training recipe, both frozen metrics and the output format all come from that
harness (imported, not re-typed).  The ego branch is r0_ego.build(...).phi plus a verbatim copy
of r0_ego.SeqNet.forward's [mean, max, softmin, std] pooling, so R2 - R0 is exactly the graph.

The graph is a per-WINDOW object; B2D's gold is per-ROUTE.  So the window vectors of a route are
pooled with the same distribution pooling before the head.  NavSim is one window per scored scene,
so that pool is over a single element (degenerate, hence the eps in dist_pool).

────────────────────────────────────────────────────────────────────────────────────────────
THE CONTROLS.  PROTOCOL_R.md section 2 was AMENDED (2026-08-26, before any R1/R2 number was
seen) and the draft control this file used to implement was SCRAPPED.  The scrapped one permuted
a2l_idx while leaving a2l_rel in place, which hands the model lane token l7 carrying the relation
vector computed for l1 — an internally INCONSISTENT graph.  A reviewer then says the drop comes
from geometrically inconsistent edge attributes, not from the necessity of correct
correspondence, and the control collapses exactly when R2 wins big.  Neither control below can
be attacked that way: every relation vector the model sees is the true geometry of the lane
token it is attached to.

S-route (PRIMARY control) — `--shuffle route`
  A2L, lane geometry, lane_feat and L2L are left BYTE-FOR-BYTE untouched.  The route_rel ROWS are
  permuted across lanes, within a window, within the junction / non-junction class, each lane's
  whole 5-channel row moving as ONE unit (never channel by channel: that would manufacture route
  feature combinations that do not exist and inject noise instead of breaking correspondence).
  Windows with route_rel_valid = False are skipped — there is no route there to destroy, and
  their route_rel is all-zero anyway, so the permutation is a no-op by construction.
  No inconsistent edge attribute can arise, because route_rel is a property of a LANE, not the
  geometry of an agent-lane pair.  The question it attacks directly: is the correct
  Agent-Lane-Route wiring necessary, or does the route feature DISTRIBUTION alone carry the gain?

S-a2l (SECONDARY control) — `--shuffle a2l`
  Each agent's candidates are reassigned to different lanes AND the relation vector is RECOMPUTED
  for the new lane from the stored polyline.  Per candidate slot with true lane l_true:
      l' = argmin_l D( r(a, l), r(a, l_true) )
      over l in the same window, same junction/non-junction class, EXCLUDING every lane in that
      agent's own true candidate set and every lane already given to that agent.
  The exclusion of the agent's WHOLE true set (not just l_true, which is all the protocol text
  literally requires) is deliberate and is the difference between a control and a no-op: the
  model reads the K axis as a SET through a softmax, so if l' were allowed to be another of the
  agent's own true candidates the "shuffled" set could come back a permutation of itself
  carrying its own true relations, and nothing would have been destroyed.
  The input is ALWAYS the recomputed r(a, l'), never the old a2l_rel moved across.

  D — the distance between relation vectors, stated explicitly because the protocol demands it:
      D(r, r0)^2 = (d_lat - d_lat0)^2 + (d_lon - d_lon0)^2
                 + PSI_SCALE^2 * [ (cos dpsi - cos dpsi0)^2 + (sin dpsi - sin dpsi0)^2 ]
      metres on the two translational channels; the heading term is the CHORD distance on the
      unit circle, |(cos, sin) - (cos0, sin0)| = 2 |sin(dpsi/2)|, converted to metres by
      PSI_SCALE = 5 m per unit chord.  So a 30 deg heading error costs 2.6 m, 90 deg costs 7.1 m
      and a full reversal costs 10 m — i.e. a same-heading lane a couple of metres to the side
      always beats an opposite-heading lane at the same place.  Using the chord rather than the
      angle keeps D a metric on the stored representation and never has to unwrap an angle.

  THE LOCALITY BOUND, and why argmin D alone is not enough.  The pool is additionally restricted
  to lanes whose polyline passes within A2L_ALT_RADIUS = 15 m of the agent's anchor.  This is not
  decoration: measured on B2D without it, argmin D moves the median candidate from 3.4 m to
  21.9 m from its agent, with a 116 m tail and 58% of replacements beyond 15 m.  The reason is
  that (d_lat, d_lon) is a LANE-FRAME coordinate pair, not a position: when the agent's
  projection clamps to a lane's END, d_lat keeps only the component perpendicular to that last
  tangent and d_lon saturates at the lane's arclength, so an agent 100 m beyond the end of a
  co-linear lane has d_lat ~ 0, a small d_lon gap and therefore a tiny D.  The extractor never
  meets those lanes because it forms candidates only within A2L_RADIUS = 5 m; the replacement
  search has to be told the same thing.  15 m is three to four lane widths — the same local road
  complex, a genuinely different lane with different topology and different route relations —
  and it sits well inside the 45 m that PROTOCOL_R section 2 names as the point where the
  control stops being interpretable.  Both distances are reported before and after.

  Preserved:  candidate degree per agent (elementwise), the global candidate-count distribution,
              junction class of every candidate, local geometric plausibility, relation marginals
  Destroyed:  lane identity, lane topology correspondence, lane-route correspondence
  Reported (PROTOCOL_R section 2 requires all of these, and S-a2l is read as a secondary
  sensitivity analysis if any of them looks bad):  replacement success rate, same-lane
  replacement rate (must be 0), d_lat / d_lon / dpsi change distributions, agent candidate degree
  before/after, lane IN-degree before/after, and the agent -> assigned-lane distance before and
  after, because a control that moves an agent onto a lane 45 m away is a different experiment.

Both controls: shuffle seeds 0, 1, 2, fixed.  Both verify their invariants IN CODE and print
them, and both print positive proof that the correspondence actually changed.

────────────────────────────────────────────────────────────────────────────────────────────
R2-noLane (LANE-FREE ENCODER) — `--ablate-lane`
  Not a correspondence control but an ARCHITECTURAL ablation: the entire map side of the graph
  is deleted and what remains is ego + command + agents.  The published structural controls say
  the lane side is inert on this bank — dropping the ego-route relation IMPROVES rho (+.030),
  shuffling route correspondence (+.022) and agent-lane correspondence (+.011) do nothing beyond
  seed noise — so the question this arm attacks is whether a strictly simpler encoder with NO
  map input at all reproduces the US / UPS / nuPlan numbers.  A simpler model with the same
  numbers is the better model.

  DESTROYED, in load_graph BEFORE any tensor is built, so nothing derived from a lane can reach
  the model through a later step (strip_lanes / verify_lane_ablation):
      lane_mask     -> all False       no lane token participates in any attention or any pool
      lanes         -> all zero        lane geometry (x, y, dx_seg, dy_seg)
      lane_feat     -> all zero        is_junction, arclen
      l2l_mask      -> all False       and l2l_src / l2l_dst / l2l_type -> -1.  build_aug_edges
                                       then runs on the EMPTIED arrays and yields 0 edges, so the
                                       augmented-edge tensors are not merely unused, they are
                                       provably lane-independent
      a2l_idx       -> all -1          the A2L edge set
      a2l_rel       -> all zero        its relation attributes
      a2l_mask      -> all False
      route_rel     -> all zero        and route_rel_valid -> 0 everywhere (as in R2-noRoute)
  UNTOUCHED, byte for byte and asserted in code: agents, agent_mask, ego, command, item_id.  The
  ego branch (phi, qphi, q_mlp) and the agent branch (astep, aout) therefore consume exactly the
  same numbers as the default arm.

  ARCHITECTURE.  R2Net is NOT rebuilt.  Every lane-side module is still constructed, so the
  parameter count, the initialisation and therefore init_hash are identical to the default arm;
  the modules simply receive an all-masked input.  With zero lane tokens the graph path
  degenerates gracefully everywhere it already carried a mask (hl * lane_mask = 0; the L2L
  relation loop `continue`s on every relation because no edge is live; the A2L context is gated
  by a2l_mask.any(); the L2A index_add has no live source), with ONE exception that had to be
  fixed for this arm:

    encode()'s command-conditioned readout softmaxes over the fused node set [lanes | agents].
    A window with no lanes AND no live agents (B2D has exactly 2 of 2656 such windows) leaves
    that node mask empty; the -1e9 fill is then uniform rather than zero, so the readout returns
    the mean of unmasked garbage AND hands the lane-side parameters a non-zero gradient.  The
    fix is one line in encode(), `rd = rd * nmask.any(1)[:, None]`.  For every window with at
    least one valid node it multiplies by exactly 1.0, so the DEFAULT arm is bit-identical (with
    lanes present no window ever has an empty node set); for an empty node set it returns 0 —
    the same value the companion mean-pool `sm` already returns there.  Nothing else in the
    model changed.

  Output: r2nolane_b2d_s<seed>.npz / r2nolane_navsim_s<seed>.npz, and the npz records
  arm='r2nolane' and ablate_lane=True, so it cannot be mistaken for the default encoder.
  Verified end to end by verify_nolane.py (input, forward and gradient level).

Usage:
  r2_graph.py --domain b2d|navsim --gpu G --seed S
  r2_graph.py --domain b2d|navsim --gpu G --seed S --ablate-route
  r2_graph.py --domain b2d|navsim --gpu G --seed S --ablate-lane
  r2_graph.py --domain b2d|navsim --gpu G --seed S --shuffle route --shuffle-seed K
  r2_graph.py --domain b2d|navsim --gpu G --seed S --shuffle a2l   --shuffle-seed K
  r2_graph.py --domain b2d|navsim --gpu G --shuffle a2l --shuffle-seed 0 --verify-only
  r2_graph.py --domain b2d|navsim --gpu G --ablate-lane --verify-only
"""
import argparse, math, os, re, sys

import json
import numpy as np

RG = '/data2/jeongtae/relgraph_e16sel'
sys.path.insert(0, RG)
import r0_ego                                     # reference harness — splits, gold, recipe
from visual_window import build_visual_window, load_window_visual, check_feature_cache, build_window_fusion   # window arms (2026-09-12)
import b2d_earlystop as es                        # the ONE B2D checkpoint-selection rule

# ── frozen channel symbols (KEYS.md). NEVER a numeric index. ──────────────────
LN_X, LN_Y, LN_DX, LN_DY = 0, 1, 2, 3             # lanes (N,M,P,4)
LF_JUNCTION, LF_ARCLEN = 0, 1                     # lane_feat (N,M,2)
L2L_SUCC, L2L_LEFT, L2L_RIGHT, L2L_OPP = 0, 1, 2, 3
A2L_DLAT, A2L_DLON, A2L_COS, A2L_SIN = 0, 1, 2, 3
R_ON_ROUTE, R_REACH, R_SHARES, R_XSECT, R_ORDER = 0, 1, 2, 3, 4
AG_DX, AG_DY, AG_COS, AG_SIN, AG_SPEED, AG_HLEN, AG_HWID, AG_ISVEH = 0, 1, 2, 3, 4, 5, 6, 7
EGO_SPEED, EGO_COS, EGO_SIN = r0_ego.EGO_SPEED, r0_ego.EGO_COS, r0_ego.EGO_SIN

M, P, A, K, T = 128, 10, 48, 8, 12
N_SEG = P - 1                                     # the P-th dx_seg/dy_seg slot is ZERO PADDING
N_LANE_CH, N_LF, N_RR, N_A2L, N_AG = 4, 2, 5, 4, 8
ANCHOR_T = 3                                      # ego-anchor frame, anchor = t 3
NIN = r0_ego.NIN
DT = r0_ego.DT

# message-passing relations: the 4 stored types + PREDECESSOR as the SUCCESSOR transpose
REL_SUCC, REL_PRED, REL_LEFT, REL_RIGHT, REL_OPP = 0, 1, 2, 3, 4
N_REL = 5
EA = 768                                          # augmented edge slots; asserted below

PSI_SCALE = 5.0                                   # metres per unit heading-chord, see docstring
A2L_ALT_RADIUS = 15.0                             # S-a2l locality bound, m; see docstring
A2L_RADIUS = 5.0                                  # the extractor's own candidate radius (KEYS.md)


# ── data ─────────────────────────────────────────────────────────────────────
class G:
    """Frozen tensors, on GPU, fp16. Nothing here is ever written back to disk."""
    pass


def build_aug_edges(src, dst, typ, em):
    """LEFT/RIGHT are stored in BOTH directions already (verified: every LEFT edge has its
    reverse RIGHT edge), so they are used as stored. SUCCESSOR is one-way, so PREDECESSOR is
    its transpose (KEYS.md). OPPOSITE is only partly symmetric in the files, so it is
    symmetrised as a deduplicated undirected relation."""
    N = len(src)
    os_ = np.full((N, EA), -1, np.int16)
    od = np.full((N, EA), -1, np.int16)
    orl = np.full((N, EA), -1, np.int8)
    om = np.zeros((N, EA), bool)
    worst = 0
    for n in range(N):
        e = em[n]
        s = src[n][e].astype(np.int64); t = dst[n][e].astype(np.int64); ty = typ[n][e]
        S, D, R = [], [], []
        m0 = ty == L2L_SUCC
        S += [s[m0], t[m0]]; D += [t[m0], s[m0]]
        R += [np.full(m0.sum(), REL_SUCC), np.full(m0.sum(), REL_PRED)]
        for tt, rr in ((L2L_LEFT, REL_LEFT), (L2L_RIGHT, REL_RIGHT)):
            mm = ty == tt
            S.append(s[mm]); D.append(t[mm]); R.append(np.full(mm.sum(), rr))
        m3 = ty == L2L_OPP
        a, b = s[m3], t[m3]
        key = np.unique(np.minimum(a, b) * M + np.maximum(a, b))
        lo, hi = key // M, key % M
        S.append(np.r_[lo, hi]); D.append(np.r_[hi, lo])
        R.append(np.full(2 * len(key), REL_OPP))
        S = np.concatenate(S); D = np.concatenate(D); R = np.concatenate(R)
        worst = max(worst, len(S))
        assert len(S) <= EA, f'augmented edge overflow {len(S)} > {EA}'
        os_[n, :len(S)] = S; od[n, :len(S)] = D; orl[n, :len(S)] = R; om[n, :len(S)] = True
    return os_, od, orl, om, worst


def window_ego_feats(ego, cmd):
    """VERBATIM the per-step ego features r0_ego.build_navsim builds, applied window-locally.
    [speed, acc, yawrate, |acc|, |yawrate|, cmd(4)]"""
    sp = ego[:, :, EGO_SPEED]
    psi = np.arctan2(ego[:, :, EGO_SIN], ego[:, :, EGO_COS])
    acc = np.gradient(sp, DT, axis=1)
    yr = np.concatenate([np.zeros((len(sp), 1), np.float32),
                         r0_ego.wrap(np.diff(psi, axis=1)) / DT], 1)
    return np.concatenate([sp[..., None], acc[..., None], yr[..., None],
                           np.abs(acc)[..., None], np.abs(yr)[..., None],
                           np.repeat(cmd[:, None, :], T, 1)], -1).astype(np.float32)


# ══ S-route ══════════════════════════════════════════════════════════════════
def shuffle_route(route_rel, route_valid, lane_mask, lane_feat, seed):
    """PROTOCOL_R section 2, S-route.  Permute route_rel ROWS across lanes, within a window,
    within the junction / non-junction class.  The whole 5-channel row moves as one unit.
    route_rel_valid = False windows are skipped (no route to destroy; their rows are all zero).

    Returns (new_route_rel, came_from) where came_from[n, l] is the lane index whose row now
    sits at slot l (itself where nothing moved).  Nothing else in the file is read or written."""
    rng = np.random.default_rng(seed)
    out = route_rel.copy()                          # fp16 -> fp16, bit-exact row moves
    junc = np.asarray(lane_feat[:, :, LF_JUNCTION], np.float32) > 0.5
    came = np.broadcast_to(np.arange(route_rel.shape[1]), route_rel.shape[:2]).copy()
    for n in range(len(route_rel)):
        if not route_valid[n]:
            continue
        for cls in (False, True):
            ids = np.where(lane_mask[n] & (junc[n] == cls))[0]
            if len(ids) > 1:
                src = rng.permutation(ids)
                out[n, ids] = route_rel[n, src]
                came[n, ids] = src
    return out, came


def verify_route_shuffle(old, new, came, route_valid, lane_mask, lane_feat, tag):
    """A control whose route feature DISTRIBUTION shifted is not a correspondence control."""
    print(f'--- S-route invariants [{tag}] ---', flush=True)
    junc = np.asarray(lane_feat[:, :, LF_JUNCTION], np.float32) > 0.5
    o = np.asarray(old, np.float32); nw = np.asarray(new, np.float32)
    lm = lane_mask
    rv = route_valid.astype(bool)
    v = lm & rv[:, None]                              # lanes the control can touch
    moved = came != np.arange(old.shape[1])[None, :]

    same_invalid = np.array_equal(old[~rv], new[~rv])
    print(f'  route_rel_valid=False windows untouched (bitwise)     : {same_invalid}  '
          f'({int((~rv).sum())} windows)', flush=True)
    pad_ok = np.array_equal(old[~lm], new[~lm])
    print(f'  padding lane slots untouched (bitwise)                : {pad_ok}', flush=True)

    # the permutation is a bijection within (window, class): the row multiset must be identical
    bad = sum(int(not np.array_equal(np.sort(o[n, ids], 0), np.sort(nw[n, ids], 0)))
              for n in np.where(rv)[0]
              for ids in (np.where(lm[n] & (junc[n] == c))[0] for c in (False, True))
              if len(ids))
    print(f'  per-(window,class) row multiset preserved exactly     : {bad == 0}  '
          f'({bad} violations)', flush=True)
    jsrc = np.take_along_axis(junc, came, 1)
    nviol_j = int((jsrc[v] != junc[v]).sum())
    print(f'  every row stays inside its junction class             : {nviol_j == 0}  '
          f'({nviol_j} violations)', flush=True)
    print(f'  destination is always a VALID lane                    : '
          f'{int((v & ~np.take_along_axis(lm, came, 1)).sum())} violations', flush=True)

    names = ['on_route_corridor', 'route_reachable_3hop', 'shares_downstream',
             'polyline_intersects_route', 'route_order_norm']
    print('  channel marginals over touchable lanes    old -> new', flush=True)
    for c, nm in enumerate(names):
        print(f'    {nm:26s} mean {o[:, :, c][v].mean():+.6f} -> {nw[:, :, c][v].mean():+.6f}   '
              f'sd {o[:, :, c][v].std():.6f} -> {nw[:, :, c][v].std():.6f}', flush=True)

    # PROOF the correspondence actually changed
    onr_o = o[:, :, R_ON_ROUTE] > 0.5
    onr_n = nw[:, :, R_ON_ROUTE] > 0.5
    n_onr = int(onr_o[v].sum())
    kept = int((onr_o & onr_n)[v].sum())
    # expected retention under a UNIFORM permutation within each (window, class) group: each of
    # the group's n_onr on-route lanes draws its new row uniformly from the group's n rows.  The
    # right null is this, NOT the global on_route rate -- the permutation is deliberately
    # confined to the group, and on-route lanes are concentrated inside one class of one window.
    exp = sum((int((onr_o[n] & g).sum()) ** 2) / max(int(g.sum()), 1)
              for n in np.where(rv)[0]
              for g in (lm[n] & (junc[n] == c) for c in (False, True)) if g.any())
    print(f'  lanes whose route_rel row came from another lane      : '
          f'{float(moved[v].mean()):.4f}  ({int(moved[v].sum())} of {int(v.sum())})', flush=True)
    print(f'  on_route_corridor lanes still on_route after          : '
          f'{kept}/{n_onr} = {kept / max(n_onr, 1):.4f}   vs {exp / max(n_onr, 1):.4f} expected '
          f'under a uniform within-(window,class) permutation, vs {float(onr_o[v].mean()):.4f} '
          f'the global on_route rate', flush=True)
    row_same = np.all(np.isclose(o, nw), -1)
    print(f'  lanes whose 5-channel row VALUE is unchanged          : '
          f'{float(row_same[v].mean()):.4f}   (high is EXPECTED and harmless: most lanes carry '
          f'the identical all-zero off-route row, so permuting identical rows is invisible; the '
          f'row that carries the signal is the on_route one above)', flush=True)
    assert same_invalid and pad_ok and bad == 0 and nviol_j == 0
    print('  A2L / lanes / lane_feat / L2L untouched               : True (never written)',
          flush=True)


# ══ S-a2l ════════════════════════════════════════════════════════════════════
def window_rel(lanes_n, lmask_n, apt, apsi, sgn_lat, sgn_psi):
    """Recompute the A2L relation of every live agent against EVERY valid lane of one window,
    from the stored polyline alone — no map API.  Exactly the extractor's definition:

      segments are lanes[..., :P-1, :]  (the P-th dx_seg/dy_seg slot is zero padding)
      the agent's ANCHOR position is projected onto the segments; the nearest one is used
      d_lat  signed perpendicular offset from that segment (left positive)
      d_lon  arclength along the lane MINUS s*, the arclength of the lane point closest to the
             ego anchor — the same origin KEYS.md mandates for route_order_norm, so d_lon > 0
             means the agent is further along that lane than the ego is
      dpsi   agent BODY yaw - lane tangent yaw, stored as (cos, sin)

    sgn_lat / sgn_psi are the per-domain handedness calibration measured in calibrate_signs().

    -> rel (n_a, M, 4), dist (n_a, M) point distance to the polyline; invalid lanes get inf."""
    lp = np.asarray(lanes_n, np.float64)
    m_ = len(lp)
    segA = lp[:, :N_SEG, LN_X:LN_Y + 1]                       # (M, S, 2)
    seg = lp[:, :N_SEG, LN_DX:LN_DY + 1]                      # (M, S, 2)
    seg_len = np.hypot(seg[..., 0], seg[..., 1])              # (M, S)
    seg_th = np.arctan2(seg[..., 1], seg[..., 0])
    seg_s0 = np.concatenate([np.zeros((m_, 1)), np.cumsum(seg_len, 1)[:, :-1]], 1)
    L2 = np.maximum((seg * seg).sum(-1), 1e-12)
    ar = np.arange(m_)

    # s* : the point of each lane closest to the EGO anchor, which is the origin of this frame
    t0 = np.clip((-segA * seg).sum(-1) / L2, 0.0, 1.0)        # (M, S)
    p0 = segA + t0[..., None] * seg
    k0 = np.hypot(p0[..., 0], p0[..., 1]).argmin(1)
    s_star = seg_s0[ar, k0] + t0[ar, k0] * seg_len[ar, k0]    # (M,)

    w = apt[:, None, None, :] - segA[None]                    # (n_a, M, S, 2)
    t = np.clip((w * seg[None]).sum(-1) / L2[None], 0.0, 1.0)
    dv = w - t[..., None] * seg[None]                         # agent - projection
    dd = np.hypot(dv[..., 0], dv[..., 1])                     # (n_a, M, S)
    k = dd.argmin(2)                                          # (n_a, M)

    ia = np.arange(len(apt))[:, None]
    im = ar[None, :]
    th = seg_th[im, k]
    dvk = dv[ia, im, k]
    dlat = sgn_lat * (-np.sin(th) * dvk[..., 0] + np.cos(th) * dvk[..., 1])
    dlon = seg_s0[im, k] + t[ia, im, k] * seg_len[im, k] - s_star[None, :]
    dpsi = sgn_psi * r0_ego.wrap(apsi[:, None] - th)
    rel = np.stack([dlat, dlon, np.cos(dpsi), np.sin(dpsi)], -1)
    dist = dd[ia, im, k]
    bad = ~np.asarray(lmask_n, bool)[None, :]
    rel = np.where(bad[..., None], 0.0, rel)
    dist = np.where(bad, np.inf, dist)
    return rel, dist


def rel_dist(r, r0):
    """D of the docstring: metres, with the heading chord scaled by PSI_SCALE m per unit."""
    d = r - r0
    return np.sqrt(d[..., A2L_DLAT] ** 2 + d[..., A2L_DLON] ** 2
                   + PSI_SCALE ** 2 * (d[..., A2L_COS] ** 2 + d[..., A2L_SIN] ** 2))


def agent_anchor(ag, n, live):
    apt = np.asarray(ag[n, live, ANCHOR_T, LN_X:LN_Y + 1], np.float64)
    apsi = np.arctan2(np.asarray(ag[n, live, ANCHOR_T, AG_SIN], np.float64),
                      np.asarray(ag[n, live, ANCHOR_T, AG_COS], np.float64))
    return apt, apsi


def _recompute_err(d, rows, sgn_lat, sgn_psi):
    """|recomputed r(a, l_true) - STORED a2l_rel| over a sample of windows."""
    lanes, lmask = d['lanes'], d['lane_mask']
    ai, ar_, amk = d['a2l_idx'], d['a2l_rel'], d['a2l_mask']
    e = []
    for n in rows:
        live = np.nonzero(amk[n].any(1))[0]
        if not len(live):
            continue
        apt, apsi = agent_anchor(d['agents'], n, live)
        rel, _ = window_rel(lanes[n], lmask[n], apt, apsi, sgn_lat, sgn_psi)
        for j, a in enumerate(live):
            ks = np.nonzero(amk[n, a])[0]
            lt = ai[n, a, ks].astype(np.int64)
            e.append(np.abs(rel[j, lt] - np.asarray(ar_[n, a, ks], np.float64)))
    e = np.concatenate(e) if e else np.zeros((1, N_A2L))
    return float(e.max()), np.median(e, 0), np.percentile(e, 99, 0), len(e)


def calibrate_signs(d, rows):
    """The frozen B2D window tensor lives in CARLA's LEFT-handed world and is mirrored in y
    relative to the NavSim tensor (that clash is recorded in both extractors, never patched).
    A mirror flips the sign of BOTH d_lat and dpsi.  Rather than assume which side of it a given
    file sits on, MEASURE it: recompute r(a, l_true) under each convention and keep the one that
    reproduces the STORED a2l_rel.  The residual under the winning convention is the noise floor
    of the whole control — the replacement relations are only as trustworthy as the recomputation
    of the true ones, so it is printed next to the displacements the control actually makes."""
    out = {}
    best = None
    for sl in (+1.0, -1.0):
        for sp in (+1.0, -1.0):
            err = _recompute_err(d, rows, sl, sp)
            out[(sl, sp)] = err
            if best is None or err[0] < out[best][0]:
                best = (sl, sp)
    return best, out


def shuffle_a2l(d, seed, tag):
    """PROTOCOL_R section 2, S-a2l.  Reassign every candidate to a geometry-matched DIFFERENT
    lane and RECOMPUTE the relation vector for it.  -> (new_a2l_idx, new_a2l_rel, diag)."""
    rng = np.random.default_rng(seed)
    lanes, lmask, lfeat = d['lanes'], d['lane_mask'], d['lane_feat']
    ag, amk = d['agents'], d['a2l_mask']
    a2l_idx, a2l_rel = d['a2l_idx'], d['a2l_rel']
    N = len(lanes)
    junc_all = np.asarray(lfeat[:, :, LF_JUNCTION], np.float32) > 0.5

    probe = np.unique(np.linspace(0, N - 1, min(N, 400)).astype(int))
    (sgn_lat, sgn_psi), errs = calibrate_signs(d, probe)
    print(f'--- S-a2l relation recomputation [{tag}] ---', flush=True)
    print(f'  handedness calibration, MEASURED not assumed, on {len(probe)} probe windows:',
          flush=True)
    for (sl, sp), (mx, med, p99, ne) in sorted(errs.items(), key=lambda kv: kv[1][0]):
        mark = ' <- used' if (sl, sp) == (sgn_lat, sgn_psi) else ''
        print(f'    d_lat sign {sl:+.0f}  dpsi sign {sp:+.0f}   max |err| {mx:9.4f}   '
              f'median/channel [{med[0]:.4f} {med[1]:.4f} {med[2]:.4f} {med[3]:.4f}]{mark}',
              flush=True)
    mx, med, p99, ne = errs[(sgn_lat, sgn_psi)]
    print(f'  recomputed r(a, l_true) vs STORED a2l_rel over {ne} real candidates:', flush=True)
    for c, nm in enumerate(['d_lat  (m) ', 'd_lon  (m) ', 'cos dpsi   ', 'sin dpsi   ']):
        print(f'    {nm} median |err| {med[c]:.5f}   p99 {p99[c]:.5f}', flush=True)
    print(f'  max |err| over all channels {mx:.5f}   <- fp16 storage of lanes / agents / a2l_rel '
          f'is the floor here; the control below moves candidates by METRES', flush=True)

    new_idx = a2l_idx.copy()
    new_rel = a2l_rel.copy()
    repl = np.zeros(a2l_idx.shape, bool)              # slots that actually got a new lane
    n_slot = n_ok = n_same = n_nopool = 0
    dchg, d_old, d_new = [], [], []
    for n in range(N):
        live = np.nonzero(amk[n].any(1))[0]
        if not len(live):
            continue
        apt, apsi = agent_anchor(ag, n, live)
        rel, dist = window_rel(lanes[n], lmask[n], apt, apsi, sgn_lat, sgn_psi)
        valid = np.asarray(lmask[n], bool)
        junc = junc_all[n]
        pos = {int(a): j for j, a in enumerate(live)}
        # seeded processing order: WHICH agent claims a contended lane first is the only place a
        # shuffle seed can enter an assignment that is otherwise a deterministic argmin
        for a in rng.permutation(live):
            j = pos[int(a)]
            ks = np.nonzero(amk[n, a])[0]
            banned = np.zeros(M, bool)
            banned[a2l_idx[n, a, ks].astype(np.int64)] = True   # the agent's WHOLE true set
            near = valid & (dist[j] <= A2L_ALT_RADIUS)          # the locality bound
            for k in rng.permutation(ks):
                n_slot += 1
                lt = int(a2l_idx[n, a, k])
                r0 = rel[j, lt]
                d_old.append(dist[j, lt])
                ok = near & (junc == junc[lt]) & ~banned
                if not ok.any():
                    n_nopool += 1                     # no local same-class lane: slot KEPT
                    d_new.append(dist[j, lt])
                    continue
                lp = int(np.where(ok, rel_dist(rel[j], r0[None, :]), np.inf).argmin())
                banned[lp] = True
                n_ok += 1
                n_same += int(lp == lt)
                repl[n, a, k] = True
                new_idx[n, a, k] = lp
                new_rel[n, a, k] = rel[j, lp].astype(new_rel.dtype)
                dchg.append(np.abs(rel[j, lp] - r0))
                d_new.append(dist[j, lp])
    dchg = np.stack(dchg) if dchg else np.zeros((1, N_A2L))
    return new_idx, new_rel, dict(n_slot=n_slot, n_ok=n_ok, n_same=n_same, n_nopool=n_nopool,
                                  dchg=dchg, repl=repl, sgn=(sgn_lat, sgn_psi),
                                  d_old=np.array(d_old), d_new=np.array(d_new),
                                  rec_err=errs[(sgn_lat, sgn_psi)])


def verify_a2l_shuffle(d, new_idx, new_rel, diag, tag):
    """Print every invariant PROTOCOL_R section 2 names for S-a2l, and assert the ones whose
    failure would make the control uninterpretable."""
    print(f'--- S-a2l invariants [{tag}] ---', flush=True)
    old, oldr = d['a2l_idx'], d['a2l_rel']
    v = d['a2l_mask']
    lmask, lfeat, lanes = d['lane_mask'], d['lane_feat'], d['lanes']
    junc = np.asarray(lfeat[:, :, LF_JUNCTION], np.float32) > 0.5
    N = len(old)
    rows = np.arange(N)[:, None, None]
    q = [50, 90, 99, 100]

    ok_deg = np.array_equal(old >= 0, new_idx >= 0)
    print(f'  candidate degree preserved elementwise                : {ok_deg}', flush=True)
    co = np.bincount(v.sum(-1).ravel(), minlength=K + 1)
    cn = np.bincount((new_idx >= 0).sum(-1).ravel(), minlength=K + 1)
    print(f'  global candidate-count histogram old == new           : {np.array_equal(co, cn)}'
          f'   {co.tolist()}', flush=True)

    jo = junc[rows, np.clip(old, 0, None).astype(np.int64)]
    jn = junc[rows, np.clip(new_idx, 0, None).astype(np.int64)]
    nviol_j = int((jo[v] != jn[v]).sum())
    print(f'  junction class of each candidate preserved            : '
          f'{float((jo[v] == jn[v]).mean()):.6f}  ({nviol_j} violations)', flush=True)
    lv = lmask[rows, np.clip(new_idx, 0, None).astype(np.int64)]
    nviol_v = int((v & ~lv).sum())
    print(f'  new index is always a VALID lane                      : {nviol_v} violations',
          flush=True)

    rp = diag['repl']
    dup = inown = 0
    for n in range(N):
        for i in range(A):
            if not v[n, i].any():
                continue
            w = new_idx[n, i][v[n, i]]
            dup += int(len(np.unique(w)) != len(w))
            ts = set(old[n, i][v[n, i]].tolist())
            inown += sum(int(x) in ts for x in new_idx[n, i][rp[n, i]])
    print(f'  duplicate lane inside an agent candidate set          : {dup}', flush=True)
    print(f'  REPLACED candidate landing in the agent\'s OWN true set: {inown}  '
          f'(excluded by construction; the KEPT slots below of course still hold their own '
          f'original lane)', flush=True)

    print(f'  replacement success rate                              : '
          f'{diag["n_ok"] / max(diag["n_slot"], 1):.6f}   '
          f'({diag["n_ok"]} of {diag["n_slot"]} slots.  The other {diag["n_nopool"]} had no lane '
          f'within {A2L_ALT_RADIUS:g} m of the agent that was of the same junction class and '
          f'outside the agent\'s own candidate set, and KEPT their original candidate)', flush=True)
    print(f'  same-lane replacement rate  (must be 0)               : '
          f'{diag["n_same"] / max(diag["n_ok"], 1):.6f}   ({diag["n_same"]})', flush=True)
    print(f'  candidates whose lane index actually changed          : '
          f'{float((old[v] != new_idx[v]).mean()):.6f}', flush=True)

    dc = diag['dchg']
    print(f'  |change| in the relation vector  p50/p90/p99/max', flush=True)
    for c, nm in enumerate(['d_lat  (m) ', 'd_lon  (m) ', 'cos dpsi   ', 'sin dpsi   ']):
        print(f'    {nm} {np.percentile(dc[:, c], q).round(4).tolist()}', flush=True)
    dpsi_deg = np.degrees(2 * np.arcsin(np.clip(
        np.hypot(dc[:, A2L_COS], dc[:, A2L_SIN]) / 2, 0, 1)))
    print(f'    dpsi   (deg) {np.percentile(dpsi_deg, q).round(2).tolist()}', flush=True)

    print(f'  relation MARGINALS over all candidates   before -> after', flush=True)
    for c, nm in enumerate(['d_lat', 'd_lon', 'cos dpsi', 'sin dpsi']):
        a_ = np.asarray(oldr, np.float32)[..., c][v]
        b_ = np.asarray(new_rel, np.float32)[..., c][v]
        print(f'    {nm:9s} mean {a_.mean():+.4f} -> {b_.mean():+.4f}   '
              f'sd {a_.std():.4f} -> {b_.std():.4f}   '
              f'p50 {np.median(a_):+.4f} -> {np.median(b_):+.4f}', flush=True)

    ino = np.zeros((N, M), np.int32); inn = np.zeros((N, M), np.int32)
    rr = np.broadcast_to(np.arange(N)[:, None, None], old.shape)[v]
    np.add.at(ino, (rr, old[v].astype(np.int64)), 1)
    np.add.at(inn, (rr, new_idx[v].astype(np.int64)), 1)
    lm = lmask
    print(f'  lane IN-degree over valid lanes   mean {ino[lm].mean():.4f} -> {inn[lm].mean():.4f}'
          f'   sd {ino[lm].std():.4f} -> {inn[lm].std():.4f}   '
          f'max {int(ino[lm].max())} -> {int(inn[lm].max())}', flush=True)
    print(f'    in-degree histogram 0..7  old {np.bincount(ino[lm], minlength=8)[:8].tolist()}',
          flush=True)
    print(f'    in-degree histogram 0..7  new {np.bincount(inn[lm], minlength=8)[:8].tolist()}',
          flush=True)

    # THE check that decides whether this is still the same experiment.  Distance is measured
    # to the nearest point of the SEGMENTS (the extractor's own measure), not to the stored
    # polyline vertices, which would over-report by up to half a segment.
    do, dn = diag['d_old'], diag['d_new']
    print(f'  agent -> assigned-lane distance  ORIGINAL p50/p90/p99/max '
          f'{np.percentile(do, q).round(2).tolist()} m   '
          f'(extractor candidate radius {A2L_RADIUS:g} m)', flush=True)
    print(f'  agent -> assigned-lane distance  REPLACED p50/p90/p99/max '
          f'{np.percentile(dn, q).round(2).tolist()} m   '
          f'(locality bound {A2L_ALT_RADIUS:g} m)', flush=True)
    over = float((dn > A2L_ALT_RADIUS + 1e-6).mean())
    print(f'  candidates further than the {A2L_ALT_RADIUS:g} m locality bound  : {over:.6f}  '
          f'(0 expected: kept slots are within {A2L_RADIUS:g} m, replaced ones are bounded)',
          flush=True)
    assert ok_deg and np.array_equal(co, cn) and nviol_j == 0 and nviol_v == 0 and dup == 0
    assert diag['n_same'] == 0, 'same-lane replacement is not a control'
    assert inown == 0, 'a replacement landed inside the agent own true candidate set'
    assert over == 0.0, 'a candidate escaped the locality bound'
    print('  lanes / lane_feat / L2L / route_rel untouched         : True (never written)',
          flush=True)


# ══ R2-noLane ════════════════════════════════════════════════════════════════
def strip_lanes(d, route_rel, a2l_idx, a2l_rel, rv, tag):
    """R2-noLane: delete the ENTIRE map side of the graph (module docstring).  Returns the
    replacement arrays for everything a lane can reach; the .npz is never written.  The arrays
    that carry ego / command / agents are not returned because they are not touched at all."""
    z = np.zeros_like
    new = dict(lanes=z(d['lanes']), lane_feat=z(d['lane_feat']), lane_mask=z(d['lane_mask']),
               l2l_src=np.full_like(d['l2l_src'], -1), l2l_dst=np.full_like(d['l2l_dst'], -1),
               l2l_type=np.full_like(d['l2l_type'], -1), l2l_mask=z(d['l2l_mask']),
               a2l_idx=np.full_like(a2l_idx, -1), a2l_rel=z(a2l_rel), a2l_mask=z(d['a2l_mask']),
               route_rel=z(route_rel), route_rel_valid=z(rv))
    verify_lane_ablation(d, new, route_rel, a2l_idx, a2l_rel, rv, tag)
    return new


def verify_lane_ablation(d, new, route_rel, a2l_idx, a2l_rel, rv, tag):
    """An ablation that leaves one live lane token, one live edge or one non-zero lane number is
    not a lane-free encoder.  Everything below is checked, not assumed."""
    print(f'--- R2-noLane invariants [{tag}] ---', flush=True)
    N = len(d['lane_mask'])
    print(f'  live lane tokens          {int(d["lane_mask"].sum()):>9d} -> '
          f'{int(new["lane_mask"].sum()):d}   '
          f'(windows with >=1 lane {int(d["lane_mask"].any(1).sum())} -> '
          f'{int(new["lane_mask"].any(1).sum())})', flush=True)
    print(f'  live L2L edges            {int(d["l2l_mask"].sum()):>9d} -> '
          f'{int(new["l2l_mask"].sum()):d}', flush=True)
    print(f'  live A2L candidates       {int(d["a2l_mask"].sum()):>9d} -> '
          f'{int(new["a2l_mask"].sum()):d}', flush=True)
    print(f'  route_rel_valid windows   {int((np.asarray(rv) > 0.5).sum()):>9d} -> '
          f'{int((new["route_rel_valid"] > 0.5).sum()):d}   (of {N})', flush=True)
    for k, before in (('lanes', d['lanes']), ('lane_feat', d['lane_feat']),
                      ('a2l_rel', a2l_rel), ('route_rel', route_rel)):
        b = np.asarray(before, np.float64)
        print(f'  {k:10s} non-zero entries {int((b != 0).sum()):>9d} -> '
              f'{int((np.asarray(new[k], np.float64) != 0).sum()):d}   '
              f'|max| {np.abs(b).max():.4f} -> {np.abs(np.asarray(new[k], np.float64)).max():.4f}',
          flush=True)
    print(f'  a2l_idx values            min {int(np.asarray(a2l_idx).min())} max '
          f'{int(np.asarray(a2l_idx).max())} -> all {int(new["a2l_idx"].max())}', flush=True)
    for k in ('lanes', 'lane_feat', 'lane_mask', 'l2l_mask', 'a2l_mask', 'a2l_rel', 'route_rel',
              'route_rel_valid'):
        assert not np.asarray(new[k]).any(), f'{k} still carries a non-zero value'
    for k in ('l2l_src', 'l2l_dst', 'l2l_type', 'a2l_idx'):
        assert (np.asarray(new[k]) == -1).all(), f'{k} still carries a live index'
    # every replacement has the dtype and shape of the array it replaces: the model never sees a
    # differently-typed tensor in this arm
    for k, v in new.items():
        ref = route_rel if k == 'route_rel' else a2l_idx if k == 'a2l_idx' else \
            a2l_rel if k == 'a2l_rel' else rv if k == 'route_rel_valid' else d[k]
        assert v.shape == np.shape(ref) and v.dtype == np.asarray(ref).dtype, k
    print('  every lane/map array is zeroed or -1, with the original dtype and shape : True',
          flush=True)
    print('  ego / command / agents / agent_mask / item_id never read here            : True',
          flush=True)


# ── loader ───────────────────────────────────────────────────────────────────
def load_graph(domain, torch, dev, shuffle=None, shuffle_seed=None, ablate_route=False,
               ablate_lane=False):
    """Read the frozen tensors, apply AT MOST ONE control, and put the graph on the GPU.
    The .npz is NEVER written back.  The path is read off r0_ego at call time so a test harness
    can point this loader at a subset file."""
    npz = r0_ego.B2D_NPZ if domain == 'b2d' else r0_ego.NAVSIM_NPZ
    d = np.load(npz, allow_pickle=True)
    ids = np.array([str(x) for x in d['item_id']])
    a2l_idx, a2l_rel = d['a2l_idx'], d['a2l_rel']
    route_rel = d['route_rel']
    rv = d['route_rel_valid'].astype(np.float32)

    # every candidate belongs to an agent that is live at the anchor step — the assumption the
    # A2L definition and therefore the recomputation both rest on
    assert not (d['a2l_mask'] & ~d['agent_mask'][:, :, ANCHOR_T][..., None]).any(), \
        'a candidate belongs to an agent that is not live at the anchor step'

    if shuffle == 'route':
        assert shuffle_seed is not None
        route_rel, came = shuffle_route(route_rel, d['route_rel_valid'], d['lane_mask'],
                                        d['lane_feat'], shuffle_seed)
        verify_route_shuffle(d['route_rel'], route_rel, came, d['route_rel_valid'],
                             d['lane_mask'], d['lane_feat'],
                             f'{domain} shuffle-seed {shuffle_seed}')
    elif shuffle == 'a2l':
        assert shuffle_seed is not None
        a2l_idx, a2l_rel, diag = shuffle_a2l(d, shuffle_seed,
                                             f'{domain} shuffle-seed {shuffle_seed}')
        verify_a2l_shuffle(d, a2l_idx, a2l_rel, diag, f'{domain} shuffle-seed {shuffle_seed}')
    elif shuffle is not None:
        raise ValueError(shuffle)

    # the map side, as local names, so R2-noLane can replace it BEFORE anything is derived
    lanes, lane_feat, lane_mask = d['lanes'], d['lane_feat'], d['lane_mask']
    l2l_src, l2l_dst, l2l_type, l2l_mask = (d['l2l_src'], d['l2l_dst'], d['l2l_type'],
                                            d['l2l_mask'])
    a2l_mask = d['a2l_mask']
    if ablate_lane:
        assert shuffle is None and not ablate_route, 'one control at a time'
        nw = strip_lanes(d, route_rel, a2l_idx, a2l_rel, rv, domain)
        lanes, lane_feat, lane_mask = nw['lanes'], nw['lane_feat'], nw['lane_mask']
        l2l_src, l2l_dst, l2l_type, l2l_mask = (nw['l2l_src'], nw['l2l_dst'], nw['l2l_type'],
                                                nw['l2l_mask'])
        a2l_idx, a2l_rel, a2l_mask = nw['a2l_idx'], nw['a2l_rel'], nw['a2l_mask']
        route_rel, rv = nw['route_rel'], nw['route_rel_valid']

    es, ed, er, em, worst = build_aug_edges(l2l_src.astype(np.int32),
                                            l2l_dst.astype(np.int32),
                                            l2l_type.astype(np.int32), l2l_mask)
    print(f'[{domain}] augmented L2L edges: max {worst} / {EA} slots '
          f'(SUCC + PRED(transpose) + LEFT + RIGHT + OPPOSITE symmetrised)', flush=True)
    g = G()
    t = lambda x: torch.tensor(x, device=dev)
    g.lanes = t(lanes)                            # (N,M,P,4) fp16
    g.lane_feat = t(lane_feat)
    g.lane_mask = t(lane_mask)
    g.route_rel = t(route_rel)
    if ablate_route:
        rv = np.zeros_like(rv)                    # R2-noRoute: the route stream is OFF everywhere
    g.route_valid = t(rv)                         # a MASK only — never an input feature
    g.e_src = t(es.astype(np.int64)); g.e_dst = t(ed.astype(np.int64))
    g.e_rel = t(er.astype(np.int64)); g.e_mask = t(em)
    g.a2l_idx = t(a2l_idx.astype(np.int64)); g.a2l_rel = t(a2l_rel); g.a2l_mask = t(a2l_mask)
    g.agents = t(d['agents']); g.agent_mask = t(d['agent_mask'])
    g.ego_w = t(window_ego_feats(d['ego'].astype(np.float32), d['command'].astype(np.float32)))
    g.command = t(d['command'].astype(np.float32))
    g.N = len(ids)
    return g, ids


def graph_stats(g, rows, torch, dev):
    """Train-split standardisation, exactly as r0_ego standardises X per draw/fold."""
    acc = {k: [0.0, None, None] for k in ('lane', 'lf', 'rr', 'a2l', 'ag')}

    def add(k, x):                                  # x (n, C) float32
        c, s, q = acc[k]
        acc[k] = [c + len(x), (0 if s is None else s) + x.sum(0),
                  (0 if q is None else q) + (x * x).sum(0)]

    for i0 in range(0, len(rows), 1024):
        r = torch.tensor(rows[i0:i0 + 1024], device=dev)
        lm = g.lane_mask[r]
        add('lane', g.lanes[r][:, :, :N_SEG, :].float()[lm].reshape(-1, N_LANE_CH))
        add('lf', g.lane_feat[r].float()[lm])
        rvm = lm & (g.route_valid[r][:, None] > 0.5)          # stats over VALID route rows only
        if rvm.any():
            add('rr', g.route_rel[r].float()[rvm])
        am = g.a2l_mask[r]
        add('a2l', g.a2l_rel[r].float()[am])
        gm = g.agent_mask[r]
        add('ag', g.agents[r].float()[gm])
    dim = dict(lane=N_LANE_CH, lf=N_LF, rr=N_RR, a2l=N_A2L, ag=N_AG)
    out = {}
    for k, (c, s, q) in acc.items():
        if c == 0:
            # R2-noRoute gates every route row off, so no route statistic exists to take. The
            # route stream is multiplied by route_valid = 0 there, so mu/sd are never read; a
            # neutral (0, 1) keeps the arm from dividing by an empty accumulator.
            out[k] = (torch.zeros(dim[k], device=dev), torch.ones(dim[k], device=dev))
            continue
        mu = s / c
        out[k] = (mu, (q / c - mu * mu).clamp(min=0).sqrt() + 1e-6)
    return out


# ── model ────────────────────────────────────────────────────────────────────
def dist_pool(torch, z, mask, tau, eps=0.0):
    """VERBATIM r0_ego.SeqNet.forward's [mean, max, softmin, std] pooling. eps>0 only for the
    per-route WINDOW pool, where a 1-window route makes var exactly 0 and sqrt'(0) = inf."""
    m = mask[..., None].float()
    n = m.sum(1).clamp(min=1)
    mean = (z * m).sum(1) / n
    mx = z.masked_fill(~mask[..., None], -1e9).max(1).values
    neg = torch.where(mask[..., None], -z / tau, torch.full_like(z, -float('inf')))
    smin = -tau * torch.logsumexp(neg, 1)
    var = ((z - mean[:, None]) ** 2 * m).sum(1) / n
    return torch.cat([mean, mx, smin, (var + eps).sqrt()], -1)


def build_r2(torch, nn, d=64, tau=0.5, heads=4, visual_dim=0, visual_only=False, motion_dim=0,
             viswin_dim=0, viswin_dv=128, viswin_layers=2, ego_extra=False, fuse_dim=0, fuse_tokens=()):
    F = nn.functional
    dh = d // heads

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, d), nn.SiLU(), nn.Linear(d, o))

    class L2L(nn.Module):
        """R-GCN layer: per-relation mean of transformed neighbours, summed over relations."""
        def __init__(self):
            super().__init__()
            self.W = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(N_REL)])
            self.self_w = nn.Linear(d, d)
            self.ln = nn.LayerNorm(d)

        def forward(self, hl, es, ed, er, em):
            n = hl.shape[0]
            flat = hl.reshape(n * M, d)
            off = (torch.arange(n, device=hl.device) * M)[:, None]
            s = (es + off).clamp(min=0); t = (ed + off).clamp(min=0)
            out = torch.zeros_like(flat)
            one = torch.ones(1, 1, device=hl.device)
            for r in range(N_REL):
                sel = em & (er == r)
                if not bool(sel.any()):
                    continue
                si, ti = s[sel], t[sel]
                cnt = torch.zeros(n * M, 1, device=hl.device)
                cnt.index_add_(0, ti, one.expand(len(ti), 1))
                out.index_add_(0, ti, self.W[r](flat[si]) / cnt[ti].clamp(min=1))
            u = self.self_w(hl) + out.reshape(n, M, d)
            return self.ln(hl + F.silu(u))

    class R2Net(nn.Module):
        def __init__(self):
            super().__init__()
            # ego branch — r0_ego's own module, verbatim
            self.phi = r0_ego.build(torch, nn, d, tau).phi
            self.tau = tau
            # window-local ego query (same architecture, own weights) — query only, no value path
            self.qphi = r0_ego.build(torch, nn, d, tau).phi
            self.q_mlp = mlp(2 * d + 4, d)
            # lanes
            self.seg = mlp(N_LANE_CH, d)
            self.lane = mlp(2 * d + N_LF, d)
            self.route = mlp(N_RR, d)
            # agents
            self.astep = mlp(N_AG, d)
            self.aout = nn.Linear(2 * d, d)
            # L2L
            self.mp1, self.mp2 = L2L(), L2L()
            # A2L / L2A
            self.edge = mlp(N_A2L, d)
            self.Wq, self.Wk, self.Wke = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
            self.Wv, self.Wve = nn.Linear(d, d), nn.Linear(d, d)
            self.Wo = nn.Linear(d, d); self.ln_a = nn.LayerNorm(d)
            self.Wv2, self.Wve2, self.Wo2 = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
            self.ln_l = nn.LayerNorm(d)
            # command-conditioned readout over the fused set
            self.typ = nn.Parameter(torch.zeros(2, d))
            self.Wk2, self.Wv3 = nn.Linear(d, d), nn.Linear(d, d)
            self.zout = mlp(2 * d, d)
            # head: r0_ego's head shape, widened for [ego 4d | graph 4d] (+ visual 4d / motion-FM 4d when those branches are attached)
            self.visual_only = visual_only
            self.ego_extra = ego_extra                       # --ego: the route-level ego branch added to a no-track arm
            self.fuse_tokens = tuple(fuse_tokens)
            fuse_no_track = bool(fuse_dim) and 'track' not in self.fuse_tokens      # fusion arm without the track token: no z_ego either
            hin = (0 if visual_only else d * 8) + (d * 4 if visual_dim else 0) + (d * 4 if motion_dim else 0) + (d * 4 if (visual_only and ego_extra) else 0)
            if fuse_dim:                                                                # fusion arm: [z_ego ; pooled fused windows]
                hin = d * 8 if 'track' in self.fuse_tokens else d * 4 + (d * 4 if ego_extra else 0)
            self.head = nn.Sequential(nn.LayerNorm(hin), nn.Linear(hin, d), nn.SiLU(),
                                      nn.Linear(d, 1))
            # visual branch (2026-09-12): frozen pretrained per-frame features of the three cameras, concatenated
            # per frame, a shared step MLP of phi's shape, then the same [mean, max, softmin, std] route pooling.
            # Built LAST so that every module above keeps the initial weights of the encoder of record.
            self.vphi = nn.Sequential(nn.Linear(visual_dim, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU()) if visual_dim else None
            # motion-FM branch (2026-09-12): frozen pretrained per-window features of the SMART/CAT-K traffic model
            # (experiments/motion_fm_features.py), same step MLP and same route pooling as the visual branch.
            # Built LAST, after vphi, for the same reason.
            self.mphi = nn.Sequential(nn.Linear(motion_dim, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU()) if motion_dim else None
            # window-level visual-temporal arm (--visual-window, 2026-09-12): the frames of track window w -> P + e_t ->
            # Transformer -> mean -> fused with z_w -> [mean, max] over windows -> its own head (visual_window.py).
            self.viswin = build_visual_window(torch, nn, viswin_dim, d_v=viswin_dv, d_track=d, layers=viswin_layers) if viswin_dim else None
            # token-fusion window arm (--fuse-window): [visual window token, track token z_w, SMART token] + modality
            # embeddings -> 1-layer fusion Transformer -> mean -> 64-d window feature -> the record's route pooling
            self.fusewin = build_window_fusion(torch, nn, fuse_dim, fuse_tokens, d=d, smart_dim=motion_dim or 256) if fuse_dim else None

        def encode(self, g, rows, st):
            """One window -> one d-vector.  The lane axis and the agent axis are both SETS: no
            positional encoding on either, ever (PROTOCOL_R 5b, verified numerically)."""
            n = len(rows)
            LM = g.lane_mask[rows]
            LS = ((g.lanes[rows][:, :, :N_SEG, :].float() - st['lane'][0]) / st['lane'][1])
            LF = (g.lane_feat[rows].float() - st['lf'][0]) / st['lf'][1]
            s = self.seg(LS)
            hl = self.lane(torch.cat([s.mean(2), s.max(2).values, LF], -1))
            RV = g.route_valid[rows]
            RR = (g.route_rel[rows].float() - st['rr'][0]) / st['rr'][1]
            hl = hl + self.route(RR) * RV[:, None, None]
            hl = hl * LM[..., None]

            AGM = g.agent_mask[rows]
            AG = (g.agents[rows].float() - st['ag'][0]) / st['ag'][1]
            if getattr(self, 'ssl_mask', None) is not None:     # --ssl pre-training: the masked (agent, step) cells are hidden
                AGM = AGM & ~self.ssl_mask
                AG = AG * AGM[..., None]
            a = self.astep(AG) * AGM[..., None]
            mt = AGM[..., None].float()
            amean = (a * mt).sum(2) / mt.sum(2).clamp(min=1)
            amax = a.masked_fill(~AGM[..., None], -1e9).max(2).values
            av = AGM.any(2)
            amax = torch.where(av[..., None], amax, torch.zeros_like(amax))
            ha = self.aout(torch.cat([amean, amax], -1)) * av[..., None]

            es, ed, er, em = g.e_src[rows], g.e_dst[rows], g.e_rel[rows], g.e_mask[rows]
            hl = self.mp1(hl, es, ed, er, em) * LM[..., None]

            J = g.a2l_idx[rows].clamp(min=0)
            AM = g.a2l_mask[rows]
            AR = (g.a2l_rel[rows].float() - st['a2l'][0]) / st['a2l'][1]
            e = self.edge(AR)
            hlj = torch.gather(hl, 1, J.reshape(n, A * K, 1).expand(-1, -1, d)).reshape(n, A, K, d)
            q = self.Wq(ha).reshape(n, A, 1, heads, dh)
            kk = (self.Wk(hlj) + self.Wke(e)).reshape(n, A, K, heads, dh)
            vv = (self.Wv(hlj) + self.Wve(e)).reshape(n, A, K, heads, dh)
            lg = (q * kk).sum(-1) / math.sqrt(dh)
            lg = lg.masked_fill(~AM[..., None], -1e9)
            al = torch.softmax(lg, 2)
            ctx = (al[..., None] * vv).sum(2).reshape(n, A, d)
            has = AM.any(2)
            ha = self.ln_a(ha + self.Wo(ctx) * has[..., None]) * av[..., None]
            if getattr(self, 'ssl_mask', None) is not None:
                self._ha = ha                                 # per-agent embeddings for the SSL decoder

            off = (torch.arange(n, device=hl.device) * M)[:, None, None]
            gi = (J + off).reshape(-1)
            sel = (AM & av[..., None]).reshape(-1)
            msg = (self.Wv2(ha)[:, :, None, :].expand(-1, -1, K, -1) + self.Wve2(e)).reshape(-1, d)
            buf = torch.zeros(n * M, d, device=hl.device)
            cnt = torch.zeros(n * M, 1, device=hl.device)
            gi_s = gi[sel]
            buf.index_add_(0, gi_s, msg[sel])
            cnt.index_add_(0, gi_s, torch.ones(len(gi_s), 1, device=hl.device))
            hl = self.ln_l(hl + self.Wo2((buf / cnt.clamp(min=1)).reshape(n, M, d))) * LM[..., None]

            hl = self.mp2(hl, es, ed, er, em) * LM[..., None]

            EW = g.ego_w[rows]
            ze = self.qphi(EW)
            qv = self.q_mlp(torch.cat([ze.mean(1), ze.max(1).values, g.command[rows].float()], -1))

            nodes = torch.cat([hl + self.typ[0], ha + self.typ[1]], 1)
            nmask = torch.cat([LM, av], 1)
            kk2 = self.Wk2(nodes).reshape(n, M + A, heads, dh)
            vv2 = self.Wv3(nodes).reshape(n, M + A, heads, dh)
            q2 = qv.reshape(n, 1, heads, dh)
            lg2 = (q2 * kk2).sum(-1) / math.sqrt(dh)
            lg2 = lg2.masked_fill(~nmask[..., None], -1e9)
            at = torch.softmax(lg2, 1)
            if getattr(self, 'capture', None) is not None:          # --dump-attn: readout attention over the
                self.capture.append((rows.detach().cpu().numpy(),   # agent nodes, (n, A, heads); never on the
                                     at[:, M:, :].detach().cpu().numpy()))   # record path (capture is None)
            rd = (at[..., None] * vv2).sum(1).reshape(n, d)
            # R2-noLane only: a window with no lanes AND no live agents has an empty node set,
            # where the -1e9 fill softmaxes to UNIFORM instead of zero.  x1.0 for every window
            # that has a valid node, so the lane-carrying arms are bit-identical; 0 for an empty
            # set, which is what the mean-pool `sm` below already returns there.
            rd = rd * nmask.any(1)[:, None]
            nm = nmask[..., None].float()
            sm = (nodes * nm).sum(1) / nm.sum(1).clamp(min=1)
            zw_ = self.zout(torch.cat([rd, sm], -1))
            if getattr(self, 'ssl_mask', None) is not None:
                self._zw = zw_
            return zw_

        def forward(self, x, xmask, g, rows, wb, ww, nW, st, xv=None, xvmask=None, xm=None, xmmask=None):
            """x/xmask: r0's ego sequence for this batch. rows: the batch's window rows.
            wb/ww: which (batch item, window slot) each row lands in. xv/xvmask: the batch's visual
            feature sequences (B, Lv, visual_dim) when a visual branch is attached; xm/xmmask: the
            batch's motion-FM sequences (B, Lm, motion_dim) when that branch is attached."""
            if getattr(self, 'viswin', None) is not None:        # --visual-window: xv/xvmask are the window frames
                r = self.viswin.window(xv, xvmask, self.encode(g, rows, st))
                return self.viswin.route(r, wb, ww, x.shape[0], nW)
            if getattr(self, 'fusewin', None) is not None:       # --fuse-window: xv/xvmask window frames, xm the SMART window rows
                zw = self.encode(g, rows, st) if 'track' in self.fuse_tokens else None
                r = self.fusewin(xv, xvmask, zw, xm)
                B = x.shape[0]
                buf = torch.zeros(B, nW, r.shape[-1], device=r.device, dtype=r.dtype)
                bm = torch.zeros(B, nW, dtype=torch.bool, device=r.device)
                buf[wb, ww] = r
                bm[wb, ww] = True
                zg = dist_pool(torch, buf, bm, self.tau, eps=1e-12)                   # the record's route pooling
                if 'track' in self.fuse_tokens:
                    return self.head(torch.cat([dist_pool(torch, self.phi(x), xmask, self.tau), zg], -1)).squeeze(-1)
                parts = [zg] + ([dist_pool(torch, self.phi(x), xmask, self.tau)] if self.ego_extra else [])
                return self.head(torch.cat(parts, -1)).squeeze(-1)
            parts = []
            if self.vphi is not None:
                parts.append(dist_pool(torch, self.vphi(xv), xvmask, self.tau, eps=1e-12))
            if self.mphi is not None:
                parts.append(dist_pool(torch, self.mphi(xm), xmmask, self.tau, eps=1e-12))
            if self.visual_only:
                if self.ego_extra:                              # explicit ego status: phi over the route ego sequence, Pool
                    parts.append(dist_pool(torch, self.phi(x), xmask, self.tau))
                return self.head(torch.cat(parts, -1)).squeeze(-1)
            zego = dist_pool(torch, self.phi(x), xmask, self.tau)
            zw = self.encode(g, rows, st)
            B = x.shape[0]
            buf = torch.zeros(B, nW, zw.shape[-1], device=zw.device, dtype=zw.dtype)
            bm = torch.zeros(B, nW, dtype=torch.bool, device=zw.device)
            buf[wb, ww] = zw
            bm[wb, ww] = True
            if getattr(self, 'capture', None) is not None and buf.requires_grad:   # --dump-attn: window saliency
                buf.retain_grad(); self._buf = buf
            zg = dist_pool(torch, buf, bm, self.tau, eps=1e-12)
            return self.head(torch.cat([zego, zg] + parts, -1)).squeeze(-1)

    return R2Net()


def ssl_pretrain(torch, nn, m, g, st, W, tr_routes, dev, seed, check_rows, max_epochs=120, patience=6,
                 mask_frac=0.3, seg=4, bs=64):
    """Track SSL initialisation of the encoder (2026-09-12, user's design): on the stage's TRAINING routes only,
    hide a contiguous segment of `seg` steps of a random `mask_frac` of the live agents (>= 8 valid steps) of every
    window, encode the window with those cells masked, and reconstruct the hidden cells' standardised
    [dx, dy, cos, sin, v] from [the agent's embedding h_k ; the window vector z_w ; a learned step embedding]
    with a small decoder (MSE over the masked cells). The pretext runs through `encode()` only, so it trains the
    WINDOW encoder (agent MLP, window ego query, readout attention, z_out; 51,456 parameters of the lane-free
    model); the route-level ego branch `phi` and the difficulty head are not on that path, receive no
    gradient and keep their initial weights (the inert lane-side modules receive zero gradients and only
    the optimiser's decoupled weight decay, which cannot change any output). The epoch is chosen by the
    reconstruction loss on an
    inner validation split (10% of the training routes, by route), max `max_epochs`, `patience`; the decoder is
    discarded. Returns (best epoch, best inner-val loss)."""
    import copy
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(tr_routes))
    nv = max(8, len(perm) // 10)
    va, tr = perm[:nv], perm[nv:]
    T = g.agents.shape[2]
    d = m.phi[0].out_features
    dec = nn.Sequential(nn.Linear(2 * d + 16, 128), nn.SiLU(), nn.Linear(128, 5)).to(dev)
    step_emb = nn.Parameter(torch.zeros(T, 16, device=dev))
    nn.init.normal_(step_emb, std=0.02)
    opt = torch.optim.AdamW(list(m.parameters()) + list(dec.parameters()) + [step_emb], lr=1e-3, weight_decay=0.1)
    gen = torch.Generator(device=dev)

    def batch_loss(route_ids, g_seed):
        rows = check_rows(np.concatenate([W[i] for i in route_ids]), 'ssl batch')
        rt = torch.tensor(rows, device=dev)
        AGM = g.agent_mask[rt]
        n, A = AGM.shape[0], AGM.shape[1]
        gen.manual_seed(int(g_seed))
        live = AGM.sum(2) >= 8
        pick = live & (torch.rand(n, A, device=dev, generator=gen) < mask_frac)
        starts = torch.randint(0, T - seg + 1, (n, A), device=dev, generator=gen)
        tidx = torch.arange(T, device=dev)[None, None, :]
        M = pick[..., None] & (tidx >= starts[..., None]) & (tidx < starts[..., None] + seg) & AGM
        m.ssl_mask = M
        zw = m.encode(g, rt, st)
        ha = m._ha
        m.ssl_mask = None
        AG = (g.agents[rt].float() - st['ag'][0]) / st['ag'][1]
        b, k, t = M.nonzero(as_tuple=True)
        if len(b) == 0:
            return zw.sum() * 0.0
        pred = dec(torch.cat([ha[b, k], zw[b], step_emb[t]], -1))
        return ((pred - AG[b, k, t, :5]) ** 2).mean()

    best = (float('inf'), None, -1)
    bad = 0
    for ep in range(max_epochs):
        m.train(); dec.train()
        order = rng.permutation(tr)
        for i0 in range(0, len(order), bs):
            loss = batch_loss(order[i0:i0 + bs], seed * 1000 + ep * 10 + i0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(m.parameters()) + list(dec.parameters()), 1.0); opt.step()
        m.eval(); dec.eval()
        with torch.no_grad():
            vl = float(np.mean([float(batch_loss(va[i0:i0 + bs], 777 + i0)) for i0 in range(0, len(va), bs)]))
        if vl < best[0] - 1e-4:
            best, bad = (vl, copy.deepcopy(m.state_dict()), ep), 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best[1])
    m.ssl_mask = None
    return best[2], best[0]


# ── B2D ──────────────────────────────────────────────────────────────────────
def route_windows(ids, routes):
    """Same route ordering and same within-route ordering as r0_ego.build_b2d."""
    mm = [re.match(r'route_(\d+)_(\d+)$', s) for s in ids]
    rid = np.array([m.group(1) for m in mm]); widx = np.array([int(m.group(2)) for m in mm])
    out = []
    for r in routes:
        sel = np.where(rid == r)[0]
        out.append(sel[np.argsort(widx[sel])])
    return out


def run_b2d(a, tag):
    import torch, torch.nn as nn
    from numpy.polynomial.hermite_e import hermegauss
    from scipy.stats import spearmanr
    from scirt_rasch import rasch
    from b2d_splits import unified_split, R_DRAWS

    dev = 'cuda'
    X, MK, routes, types, Y, static = r0_ego.build_b2d()
    g, ids = load_graph('b2d', torch, dev, a.shuffle, a.shuffle_seed, a.ablate_route,
                        a.ablate_lane)
    if a.verify_only:
        return
    W = route_windows(ids, routes)
    nW = max(len(w) for w in W)
    XV = MKV = XM = MKM = XW = MW = None
    FEAT_PROV = {}                                            # provenance of every frozen-feature cache read (saved in the npz)
    if a.visual_window or a.fuse_window:                      # window-aligned frames for the window arms
        vdir = a.visual_window or a.fuse_window
        FEAT_PROV['visual_window'] = check_feature_cache(vdir, ('model', 'stride', 'views'))
        XW, MW = load_window_visual(vdir, ids, a.visual_tokens, a.visual_views)
        print(f'[b2d] visual-window arm: {a.visual_window}: {XW.shape[0]} windows x {XW.shape[1]} frames x {XW.shape[2]}-d, '
              f'd_v {a.viswin_dim}, {a.viswin_layers} layer(s), PCA {a.visual_pca or "off"}', flush=True)
    if a.visual:                                              # frozen visual features (experiments/visual_features.py)
        FEAT_PROV['visual'] = check_feature_cache(a.visual, ('model', 'stride', 'views'))
        seqs = []
        for r in routes:
            z = np.load(f'{a.visual}/route_{r}.npz')
            vsel = slice(0, 1) if a.visual_views == 'front' else slice(None)     # --visual-views front: rgb_front only
            parts = [z['cls'][:, vsel].astype(np.float32)] + ([z['patch'][:, vsel].astype(np.float32)] if a.visual_tokens == 'cls+patch' else [])
            assert (z['frame'] == np.arange(len(z['frame'])) * int(z['stride'])).all(), f'route_{r}: visual frames are not every {int(z["stride"])}th frame'
            seqs.append(np.concatenate(parts, -1).reshape(len(z['frame']), -1))   # (Lv, n_views x C) per 2-Hz frame
        Lv = max(len(q) for q in seqs)
        XV = np.zeros((len(routes), Lv, seqs[0].shape[1]), np.float32); MKV = np.zeros((len(routes), Lv), bool)
        for i, q in enumerate(seqs):
            XV[i, :len(q)] = q; MKV[i, :len(q)] = True
        print(f'[b2d] visual branch: {a.visual} tokens {a.visual_tokens}: {XV.shape[2]}-d per frame, frames/route p50 '
              f'{int(np.median([len(q) for q in seqs]))} max {Lv}' + (' (VISUAL ONLY: no track branch)' if a.visual_only else ''), flush=True)
    if a.motion_fm:                                           # frozen motion-model features (experiments/motion_fm_features.py)
        FEAT_PROV['motion_fm'] = check_feature_cache(a.motion_fm, ('ckpt', 'stride_frames', 'n_window_frames', 'hidden_dim'))
        seqs = [np.load(f'{a.motion_fm}/route_{r}.npz')['feat'].astype(np.float32) for r in routes]
        Lm = max(len(q) for q in seqs)
        XM = np.zeros((len(routes), Lm, seqs[0].shape[1]), np.float32); MKM = np.zeros((len(routes), Lm), bool)
        for i, q in enumerate(seqs):
            XM[i, :len(q)] = q; MKM[i, :len(q)] = True
        print(f'[b2d] motion-FM branch: {a.motion_fm}: {XM.shape[2]}-d per window, windows/route p50 '
              f'{int(np.median([len(q) for q in seqs]))} max {Lm}' + (' (NO TRACK BRANCH)' if a.visual_only else ''), flush=True)
    XS = None
    if a.fuse_window and 'smart' in a.fuse_tokens.split(','):  # SMART window feature of every graph row: window w -> SMART
        assert XM is not None, '--fuse-tokens smart needs --motion-fm'  # window starting at the same frame 20w (clamped)
        ridx = {r: i for i, r in enumerate(routes)}
        XS = np.zeros((len(ids), XM.shape[2]), np.float32)
        for n_, s_ in enumerate(ids):
            r_, w_ = re.match(r'route_(\d+)_(\d+)$', s_).groups()
            i_ = ridx[r_]; XS[n_] = XM[i_, min(int(w_), int(MKM[i_].sum()) - 1)]
    print(f'[b2d] {len(routes)} routes -> {sum(len(w) for w in W)} window graphs, '
          f'windows/route p50 {int(np.median([len(w) for w in W]))} max {nW}', flush=True)
    R, J = len(routes), Y.shape[0]
    fail = np.nanmean(Y, 0)
    _, b_ref = rasch(Y, it=800)
    print(f'[b2d] rho(observed failure rate, full-panel Rasch b) '
          f'{spearmanr(fail, b_ref).correlation:+.4f}', flush=True)

    gxn, gwn = hermegauss(15); gwn = gwn / gwn.sum()
    gx = torch.tensor(gxn, dtype=torch.float32, device=dev)
    lgw = torch.log(torch.tensor(gwn, dtype=torch.float32, device=dev))
    utypes = sorted(set(types))
    pred = np.full((R_DRAWS, R), np.nan)
    per_draw = []
    ssl_log = []                                             # (draw, stage, best epoch, inner-val loss) of --ssl
    led = es.Ledger(R_DRAWS, a.epochs, a.early_stop)           # sigma / e* / curve / leak counts

    for draw in range(min(a.draws, R_DRAWS)):
        hp, ht = unified_split(draw, utypes, J)
        keepJ = np.array([j for j in range(J) if j not in hp])
        te = np.isin(types, list(ht)); tr = ~te
        plan = es.TwoStage(draw, types, tr, a.epochs, a.early_stop)
        guard = es.HeldOutGuard(np.where(te)[0],
                                np.concatenate([W[i] for i in np.where(te)[0]]), hp, keepJ)
        guard.selftest()
        # ONE guard for BOTH stages: opened here, closed only after stage 2's last epoch.
        for stg in plan.stages():
            trn = stg.train                     # stage 1: A_train.  stage 2: the FULL A block.
            torch.manual_seed(a.seed); np.random.seed(a.seed)  # same seed -> same fresh init
            th_f, b_f = rasch(Y[keepJ][:, trn])   # stage 1 theta_inner / stage 2 theta_outer
            BH = torch.full((R,), float('nan'), device=dev)    # --match: the stage Rasch difficulty of the training routes
            BH[torch.tensor(np.where(trn)[0], device=dev)] = torch.tensor(b_f, dtype=torch.float32, device=dev)
            if a.proper_init:
                torch.manual_seed(a.seed)       # re-seed AFTER rasch: rasch reseeds the torch RNG
                                                # internally, so without this every seed builds the
                                                # SAME initial weights.  Stage 1 and stage 2 still
                                                # start from identical weights within a run.
            plan.note_theta(stg, th_f)
            mu = X[trn][MK[trn]].mean(0); sd = X[trn][MK[trn]].std(0) + 1e-6
            Xn = ((X - mu) / sd) * MK[..., None]
            def pool_reduce(Xf, mode, n_views=(3 if a.visual_views == 'all' else 1)):   # --visual-pool: mean over the cameras (+ channel average-pool)
                if mode == 'none':
                    return Xf
                C = Xf.shape[-1] // n_views
                Xp = Xf.reshape(*Xf.shape[:-1], n_views, C).mean(-2)           # (..., C): the 3 views averaged
                if mode.startswith('view+ch'):
                    k = int(mode[len('view+ch'):])
                    Xp = Xp.reshape(*Xp.shape[:-1], C // k, k).mean(-1)         # (..., C/k): k adjacent channels averaged
                return np.ascontiguousarray(Xp, dtype=np.float32)
            def pca_reduce(Xf, Mf, fit_rows, K):               # --visual-pca: PCA fitted on the TRAINING frames only
                Xf = pool_reduce(Xf, a.visual_pool)
                if not K:
                    return Xf
                Ftr = Xf[fit_rows][Mf[fit_rows]]
                mu_ = Ftr.mean(0)
                _, _, Vt = np.linalg.svd(Ftr - mu_, full_matrices=False)
                return ((Xf - mu_) @ Vt[:K].T * Mf[..., None]).astype(np.float32)
            if XV is not None:                                # visual features: same rule, training routes only
                XVp = pca_reduce(XV, MKV, np.where(trn)[0], a.visual_pca)
                muv = XVp[trn][MKV[trn]].mean(0); sdv = XVp[trn][MKV[trn]].std(0) + 1e-6
                XVd = torch.tensor(((XVp - muv) / sdv) * MKV[..., None], device=dev); MKVd = torch.tensor(MKV, device=dev)
            if XW is not None:                                # window frames: PCA / standardisation on the training windows only
                trw = np.concatenate([W[i] for i in np.where(trn)[0]])
                XWp = pca_reduce(XW, MW, trw, a.visual_pca)
                muw = XWp[trw][MW[trw]].mean(0); sdw = XWp[trw][MW[trw]].std(0) + 1e-6
                XWd = torch.tensor(((XWp - muw) / sdw) * MW[..., None], device=dev); MWd = torch.tensor(MW, device=dev)
            if XM is not None:                                # motion-FM features: same rule, training routes only
                mum = XM[trn][MKM[trn]].mean(0); sdm = XM[trn][MKM[trn]].std(0) + 1e-6
                XMd = torch.tensor(((XM - mum) / sdm) * MKM[..., None], device=dev); MKMd = torch.tensor(MKM, device=dev)
            if XS is not None:                                # SMART window rows: standardised with the training windows
                trw_ = np.concatenate([W[i] for i in np.where(trn)[0]])
                mus = XS[trw_].mean(0); sds = XS[trw_].std(0) + 1e-6
                XSd = torch.tensor((XS - mus) / sds, device=dev)
            # graph_stats follows the stage's training rows, exactly as theta and mu/sd do
            st = graph_stats(g, guard.rows(np.concatenate([W[i] for i in np.where(trn)[0]]),
                                           'graph_stats'), torch, dev)
            m = build_r2(torch, nn, a.d, visual_dim=(XVd.shape[2] if XV is not None else 0), visual_only=a.visual_only,
                         motion_dim=(XM.shape[2] if XM is not None else 0),
                         viswin_dim=(XWd.shape[2] if (XW is not None and a.visual_window) else 0), viswin_dv=a.viswin_dim, viswin_layers=a.viswin_layers,
                         ego_extra=a.ego, fuse_dim=(XWd.shape[2] if (XW is not None and a.fuse_window) else 0),
                         fuse_tokens=tuple(a.fuse_tokens.split(',')) if a.fuse_window else ()).to(dev)
            if draw == 0:
                print(f'  [init] draw 0 s{stg.no} seed {a.seed} '
                      f'proper_init={a.proper_init} weight-hash {es.init_hash(m)}',
                      flush=True)
            if draw == 0 and stg.no == 1:
                print(f'[b2d] {tag} params {sum(p.numel() for p in m.parameters()):,}',
                      flush=True)
            if a.ssl:                                     # --ssl: track-SSL initialisation, then the SAME IRT recipe below
                ep_b, vl_b = ssl_pretrain(torch, nn, m, g, st, W, np.where(trn)[0], dev, a.seed * 100 + draw, guard.rows,
                                          max_epochs=a.ssl_max_epochs, mask_frac=a.ssl_mask, seg=a.ssl_seg)
                ssl_log.append((draw, stg.no, ep_b, vl_b))
                print(f'  [b2d draw {draw}] s{stg.no} track-SSL init: best epoch {ep_b} inner-val recon {vl_b:.4f}', flush=True)
            ls = torch.tensor(-0.5, device=dev, requires_grad=True)
            opt = torch.optim.AdamW(list(m.parameters()) + [ls], lr=1e-3, weight_decay=0.1)
            THE = torch.tensor(th_f, dtype=torch.float32, device=dev)
            Yk = es.erase_heldout(Y[keepJ], te) if a.early_stop else Y[keepJ]
            Yd = torch.tensor(np.nan_to_num(Yk), dtype=torch.float32, device=dev)
            Md = torch.tensor((~np.isnan(Yk)).astype(np.float32), device=dev)
            Xd = torch.tensor(Xn, device=dev); Md_ = torch.tensor(MK, device=dev)
            idx = guard.routes(np.where(trn)[0], f'{stg.name} train columns')
            iv_c = guard.routes(np.where(stg.iv)[0], f'{stg.name} inner-val columns')
            if a.early_stop:
                es.assert_masked(torch, Yd, Md, te, dev)
            print(plan.head(stg), flush=True)

            def fwd(sel):
                s = torch.tensor(sel, device=dev)
                rows = np.concatenate([W[i] for i in sel])
                wb = torch.tensor(np.concatenate([np.full(len(W[i]), b)
                                                  for b, i in enumerate(sel)]), device=dev)
                ww = torch.tensor(np.concatenate([np.arange(len(W[i])) for i in sel]), device=dev)
                if XW is not None:                                # window-aligned frames of this batch's rows
                    rt = torch.tensor(rows, device=dev)
                    if a.fuse_window:
                        return m(Xd[s], Md_[s], g, rt, wb, ww, nW, st, XWd[rt], MWd[rt], xm=(XSd[rt] if XS is not None else None))
                    return m(Xd[s], Md_[s], g, rt, wb, ww, nW, st, XWd[rt], MWd[rt])
                return m(Xd[s], Md_[s], g, torch.tensor(rows, device=dev), wb, ww, nW, st,
                         *((XVd[s], MKVd[s]) if XV is not None else ()),
                         **({'xm': XMd[s], 'xmmask': MKMd[s]} if XM is not None else {}))

            for ep in range(stg.epochs):
                m.train(); np.random.shuffle(idx); tl = nb = 0
                for i0 in range(0, len(idx), a.bs):
                    sel = guard.routes(idx[i0:i0 + a.bs], f'{stg.name} train batch')
                    bt = fwd(sel); sg = torch.exp(ls)
                    z = (bt[None, :, None] + sg * gx[None, None, :]) - THE[:, None, None]
                    p = torch.sigmoid(z)
                    s = torch.tensor(sel, device=dev)
                    yy = Yd[:, s]; mm = Md[:, s]
                    llc = (yy[:, :, None] * torch.log(p + 1e-7)
                           + (1 - yy[:, :, None]) * torch.log(1 - p + 1e-7)) * mm[:, :, None]
                    loss = -torch.logsumexp(llc.sum(0) + lgw[None, :], 1).sum() / mm.sum() \
                        + 0.05 * ls.pow(2)
                    if a.match > 0:                                   # ablation arm, see --match
                        loss = loss + a.match * ((bt - BH[s]) ** 2).mean()
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                    tl += float(loss); nb += 1
                if stg.select:
                    m.eval(); sg = torch.exp(ls).detach()
                    with torch.no_grad():
                        bv = torch.cat([fwd(guard.routes(iv_c[i:i + 32], 'inner-val forward'))
                                        for i in range(0, len(iv_c), 32)])
                    nll = es.cell_nll(torch, bv, guard.routes(iv_c, 'inner-val likelihood'),
                                      Yd, Md, THE, gx, lgw, sg)
                    plan.update(stg, ep, tl / max(nb, 1), nll, float(sg))
                    print(f'  [b2d draw {draw}] s{stg.no} ep {ep:2d} loss {tl/max(nb,1):.4f} '
                          f'innerval nll {nll:.4f} sigma {float(sg):.3f}', flush=True)
                elif ep in (0, 4, stg.epochs - 1):
                    print(f'  [b2d draw {draw}] s{stg.no} ep {ep} loss {tl/max(nb,1):.4f} '
                          f'sigma {float(torch.exp(ls)):.3f}', flush=True)
            plan.end_stage(stg, float(torch.exp(ls)))
        if a.early_stop:
            print(plan.line(int(te.sum())), flush=True)
        guard.end_training()                     # held-out readable HERE, after STAGE 2 only
        m.eval()
        with torch.no_grad():
            # ONE pass over the held-out block; the chunking below is that single pass, so
            # guard.heldout() is called once with the whole index set, not once per chunk.
            ii = guard.heldout(np.where(te)[0])
            pr = np.concatenate([fwd(ii[i:i + 32]).cpu().numpy() for i in range(0, len(ii), 32)])
            pred[draw, ii] = pr
        if a.dump_attn and not a.visual_only and not a.fuse_window:    # visualisation only: same weights, same held-out routes.  Saved: the agent
            m.capture = []                       # READOUT attention (not DINO patch / temporal attention) + window saliency
            if m.viswin is not None:
                m.viswin.capture_buf = True      # the window arm keeps its window tensor in viswin.route
            sal = {}
            for i in ii:                         # one route per pass: d f_phi / d z_w for every window w
                out_i = fwd(np.array([i]))
                out_i.sum().backward()
                bufmod = m.viswin if m.viswin is not None else m
                sal[int(i)] = bufmod._buf.grad[0, :len(W[i])].norm(dim=-1).cpu().numpy()
            rows_c = np.concatenate([c[0] for c in m.capture]); at_c = np.concatenate([c[1] for c in m.capture])
            m.capture = None
            if m.viswin is not None:
                m.viswin.capture_buf = False
            tr_idx = np.where(trn)[0]                                       # in-sample f_phi of the training routes,
            with torch.no_grad():                                           # with the draw's Rasch fit and sigma_r
                pr_tr = np.concatenate([fwd(tr_idx[i:i + 32]).cpu().numpy() for i in range(0, len(tr_idx), 32)])
            dump_path = a.dump_attn.replace('.npz', f'_draw{draw}.npz')       # one file per draw
            np.savez(dump_path, draw=draw, heldout=ii, routes=np.array(routes), pred=pr,
                     train_idx=tr_idx, pred_train=pr_tr, theta=th_f, b_hat=b_f, keepJ=keepJ,
                     sigma=float(torch.exp(ls).detach()),
                     rows=rows_c, attn=at_c.astype(np.float32),
                     sal_route=np.array([int(i) for i in sal]), sal=np.array([sal[int(i)] for i in sal], dtype=object),
                     window_rows=np.array([W[i] for i in ii], dtype=object), item_id=np.array(ids))
            print(f'  [b2d draw {draw}] wrote attention dump {dump_path}: {len(rows_c)} windows, {len(sal)} routes', flush=True)
        led.record(draw, plan, guard)
        print(guard.leak_line(draw), flush=True)
        per_draw.append(spearmanr(pr, fail[te]).correlation)
        print(f'  [b2d draw {draw}] held-out {te.sum()} routes  '
              f'rho_scene {per_draw[-1]:+.4f}', flush=True)

    out = es.out_path(RG, f'{tag}_b2d_s{a.seed}.npz', a.early_stop, a.proper_init)
    per_draw = np.array(per_draw)
    np.savez(out, pred=pred, routes=routes, item_id=routes, types=types, Y=Y, fail=fail,
             b_ref=b_ref, static=static, per_draw=per_draw,
             arm=np.array(tag), ablate_lane=bool(a.ablate_lane),
             config=np.array(json.dumps(vars(a), default=str)), feature_provenance=np.array(json.dumps(FEAT_PROV, default=str)),
             ssl_log=np.array(ssl_log, dtype=float),
             ablate_route=bool(a.ablate_route), **led.fields())
    Pl, Fl, Bl, Sl = [], [], [], []
    for dd in range(R_DRAWS):
        k = np.isfinite(pred[dd])
        Pl.append(pred[dd][k]); Fl.append(fail[k]); Bl.append(b_ref[k]); Sl.append(static[k])
    Pp, Ff, Bb, Ss = map(np.concatenate, (Pl, Fl, Bl, Sl))
    print(f'{tag.upper()}_B2D seed={a.seed}  rho_scene {spearmanr(Pp, Ff).correlation:+.4f}  '
          f'rho_ref {spearmanr(Pp, Bb).correlation:+.4f}  '
          f'static {spearmanr(Pp[Ss], Ff[Ss]).correlation:+.4f}  '
          f'non-static {spearmanr(Pp[~Ss], Ff[~Ss]).correlation:+.4f}  '
          f'(pooled {len(Pp)} held-out cells)  '
          f'per-draw mean {per_draw.mean():+.4f} +/- {per_draw.std(ddof=1):.4f}', flush=True)
    print(f'WROTE {out}', flush=True)


# ── NavSim ───────────────────────────────────────────────────────────────────
def run_navsim(a, tag):
    import torch, torch.nn as nn
    from scipy.stats import spearmanr
    from scirt_rasch import rasch

    dev = 'cuda'
    X, MK, toks, logs, Y = r0_ego.build_navsim()
    g, ids = load_graph('navsim', torch, dev, a.shuffle, a.shuffle_seed, a.ablate_route,
                        a.ablate_lane)
    if a.verify_only:
        return
    assert list(ids) == list(toks), 'graph rows must be r0 rows'
    N = len(toks)
    nW = 1
    fail = np.nanmean(Y, 0)
    _, b_ref = rasch(Y, it=800)
    print(f'[navsim] rho(observed failure rate, full-panel Rasch b) '
          f'{spearmanr(fail, b_ref).correlation:+.4f}', flush=True)

    ulogs = np.array(sorted(set(logs)))
    chunks = np.array_split(np.random.default_rng(0).permutation(len(ulogs)), a.nfolds)
    pred = np.full(N, np.nan)
    per_fold = []
    Yd = torch.tensor(np.nan_to_num(Y), device=dev)
    Md = torch.tensor((~np.isnan(Y)).astype(np.float32), device=dev)

    for fold in range(a.nfolds):
        te_logs = set(ulogs[chunks[fold]])
        rest = [l for l in ulogs if l not in te_logs]
        iv_logs = set(np.random.default_rng(100 + fold).choice(rest, 15, replace=False))
        te = np.array([l in te_logs for l in logs])
        iv = np.array([l in iv_logs for l in logs])
        tr = ~te & ~iv
        torch.manual_seed(a.seed + 1000); np.random.seed(a.seed)
        th_tr, _ = rasch(Y[:, tr])
        THE = torch.tensor(th_tr, dtype=torch.float32, device=dev)
        mu = X[tr][MK[tr]].mean(0); sd = X[tr][MK[tr]].std(0) + 1e-6
        Xd = torch.tensor(((X - mu) / sd) * MK[..., None], device=dev)
        Mk = torch.tensor(MK, device=dev)
        st = graph_stats(g, np.where(tr)[0], torch, dev)
        m = build_r2(torch, nn, a.d).to(dev)
        if fold == 0:
            print(f'[navsim] {tag} params {sum(p.numel() for p in m.parameters()):,}', flush=True)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
        idx_tr = np.where(tr)[0]
        steps = a.epochs * max(len(idx_tr) // a.bs, 1)
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(s / 200 + 1e-2, 0.5 * (1 + math.cos(math.pi * s / max(steps, 1)))))
        shuf = np.random.default_rng(a.seed)

        def fwd(sel):
            s = torch.tensor(sel, device=dev)
            z = torch.arange(len(sel), device=dev)
            return m(Xd[s], Mk[s], g, s, z, torch.zeros_like(z), nW, st)

        def pred_on(mask):
            ii = np.where(mask)[0]
            with torch.no_grad():
                return np.concatenate([fwd(ii[i:i + 512]).cpu().numpy()
                                       for i in range(0, len(ii), 512)])

        def nll_of(bv, mask):
            ii = torch.tensor(np.where(mask)[0], device=dev)
            p = torch.sigmoid(torch.tensor(bv, dtype=torch.float32, device=dev)[None, :]
                              - THE[:, None])
            yy = Yd[:, ii]; mm = Md[:, ii]
            return float((-(yy * torch.log(p + 1e-7)
                            + (1 - yy) * torch.log(1 - p + 1e-7)) * mm).sum() / mm.sum())

        best = (9e9, None, -1)
        for ep in range(a.epochs):
            m.train(); shuf.shuffle(idx_tr); tl = nb = 0
            for i0 in range(0, len(idx_tr) - a.bs + 1, a.bs):
                sel = idx_tr[i0:i0 + a.bs]
                s = torch.tensor(sel, device=dev)
                p = torch.sigmoid(fwd(sel)[None, :] - THE[:, None])
                yy = Yd[:, s]; mm = Md[:, s]
                loss = (-(yy * torch.log(p + 1e-7)
                          + (1 - yy) * torch.log(1 - p + 1e-7)) * mm).sum() / mm.sum()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); sch.step(); tl += float(loss); nb += 1
            m.eval()
            nll = nll_of(pred_on(iv), iv)
            if nll < best[0]:
                best = (nll, pred_on(te), ep)
            if ep in (0, 4, a.epochs - 1):
                print(f'  [navsim fold {fold}] ep {ep} train nll {tl/max(nb,1):.4f} '
                      f'innerval nll {nll:.4f}', flush=True)
        pred[te] = best[1]
        per_fold.append(spearmanr(best[1], fail[te]).correlation)
        print(f'  [navsim fold {fold}] train {tr.sum()} innerval {iv.sum()} test {te.sum()} '
              f'| best ep {best[2]} nll {best[0]:.4f} | rho_scene '
              f'{spearmanr(best[1], fail[te]).correlation:+.4f} '
              f'rho_ref {spearmanr(best[1], b_ref[te]).correlation:+.4f}', flush=True)

    out = f'{RG}/{tag}_navsim_s{a.seed}.npz'
    per_fold = np.array(per_fold)
    np.savez(out, pred=pred, tokens=toks, item_id=toks, logs=logs, fail=fail, b_ref=b_ref, Y=Y,
             per_fold=per_fold, arm=np.array(tag), ablate_lane=bool(a.ablate_lane),
             ablate_route=bool(a.ablate_route))
    k = np.isfinite(pred)
    print(f'{tag.upper()}_NAVSIM seed={a.seed}  '
          f'rho_scene {spearmanr(pred[k], fail[k]).correlation:+.4f}  '
          f'rho_ref {spearmanr(pred[k], b_ref[k]).correlation:+.4f}  '
          f'(out-of-fold {k.sum()}/{N})  '
          f'per-fold mean {per_fold.mean():+.4f} +/- {per_fold.std(ddof=1):.4f}', flush=True)
    print(f'WROTE {out}', flush=True)


# ── section 18: ONE model on ALL source items, forwarded on the other domain ──────────────
def run_full(a):
    """--full-train --transfer-to DOM [--subsample-logs N].  Additive: run_b2d / run_navsim are
    untouched.  Schedule, e* selection and recipe come from transfer_common (es.TwoStage
    underneath); this function supplies R2's graph, ego tensors and forward only.  The target
    graph is loaded AFTER training; target responses are never read or saved."""
    import torch, torch.nn as nn
    from scipy.stats import spearmanr
    from scirt_rasch import rasch
    import transfer_common as tc

    assert a.shuffle is None and not a.ablate_route and not a.ablate_lane, \
        'section 18 arms are R0 / R2 / H-rel only'
    dev, devname = tc.device(torch)
    src, tgt = a.domain, a.transfer_to
    tag = f'r2 {tc.src_tag(a)}->{tgt} s{a.seed}'
    sub_logs = np.array([])
    g, ids = load_graph(src, torch, dev)
    if src == 'b2d':
        X, MK, items, grp, Y, _ = r0_ego.build_b2d()
        W = route_windows(ids, items); nW = max(len(w) for w in W)
    else:
        X, MK, items, grp, Y = r0_ego.build_navsim()
        assert list(ids) == list(items), 'graph rows must be r0 rows'
        rows_all = np.arange(len(ids))
        if a.subsample_logs:
            keep, sub_logs = tc.subsample_logs(grp, a.subsample_logs)
            X, MK, items, grp, Y = X[keep], MK[keep], items[keep], grp[keep], Y[:, keep]
            rows_all = np.where(keep)[0]          # graph rows of the kept windows; g stays whole
        W = [np.array([r]) for r in rows_all]; nW = 1
    N = len(items)
    fail = np.nanmean(Y, 0)
    _, b_src = rasch(Y, it=800)                   # SOURCE full-panel reference, sanity rho only
    itr, iv = tc.inner_split(src, grp)
    plan = tc.FullTrainPlan(grp, itr, iv, a.epochs)
    Yd = torch.tensor(np.nan_to_num(Y), dtype=torch.float32, device=dev)
    Md = torch.tensor((~np.isnan(Y)).astype(np.float32), device=dev)
    chunk = 32 if src == 'b2d' else 512

    def make_fwd(m, Xd, Mk, g_, W_, nW_, st_):
        def fwd(sel):
            s = torch.tensor(sel, device=dev)
            rows = np.concatenate([W_[i] for i in sel])
            wb = torch.tensor(np.concatenate([np.full(len(W_[i]), b) for b, i in enumerate(sel)]),
                              device=dev)
            ww = torch.tensor(np.concatenate([np.arange(len(W_[i])) for i in sel]), device=dev)
            return m(Xd[s], Mk[s], g_, torch.tensor(rows, device=dev), wb, ww, nW_, st_)
        return fwd

    Mk = torch.tensor(MK, device=dev)
    for stg in plan.stages():
        trn = stg.train                           # stage 1: inner-train.  stage 2: ALL items.
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        th_f, _ = rasch(Y[:, trn])                # theta on this stage's SOURCE columns
        torch.manual_seed(a.seed)                 # proper-init (section 12): re-seed AFTER rasch
        plan.note_theta(stg, th_f)
        mu = X[trn][MK[trn]].mean(0); sd = X[trn][MK[trn]].std(0) + 1e-6
        Xd = torch.tensor(((X - mu) / sd) * MK[..., None], device=dev)
        st = graph_stats(g, np.concatenate([W[i] for i in np.where(trn)[0]]), torch, dev)
        m = build_r2(torch, nn, a.d).to(dev)
        print(f'  [init] s{stg.no} seed {a.seed} proper_init=True weight-hash {es.init_hash(m)}',
              flush=True)
        THE = torch.tensor(th_f, dtype=torch.float32, device=dev)
        print(plan.head(stg), flush=True)
        fwd = make_fwd(m, Xd, Mk, g, W, nW, st)
        tc.fit_stage(src, torch, nn, m, fwd, stg, plan, Yd, Md, THE, a.bs, a.seed, dev, tag,
                     chunk=chunk)
    print(plan.line(0), flush=True)
    pred_src = tc.predict(torch, m, fwd, N, chunk)
    rho_src = float(spearmanr(pred_src, b_src).correlation)
    print(f'  [{tag}] e* {plan.best_ep}, stage 2 {plan.stage2_epochs} epochs on {N} items | '
          f'source in-sample rho_ref {rho_src:+.4f} rho_fail '
          f'{spearmanr(pred_src, fail).correlation:+.4f}', flush=True)

    # ── target: loaded only now; ego mu/sd and graph_stats are the SOURCE stage-2 statistics ──
    out = dict(arm='r2', src=tc.src_tag(a), tgt=tgt, seed=a.seed, device=devname, bs=a.bs,
               epochs=a.epochs, pred_src=pred_src, src_item_id=items, src_groups=grp,
               src_b_ref=b_src, rho_src_insample=rho_src, x_mu=mu, x_sd=sd, sub_logs=sub_logs,
               **{f'gs_{k}_{n}': v.cpu().numpy() for k, (mu_, sd_) in st.items()
                  for n, v in (('mu', mu_), ('sd', sd_))},
               **tc.plan_fields(plan))
    g_t, ids_t = load_graph(tgt, torch, dev)

    def fwd_t(Xt, MKt, W_, nW_, ch):
        Xd_t = torch.tensor(((Xt - mu) / sd) * MKt[..., None], device=dev)
        Mk_t = torch.tensor(MKt, device=dev)
        return tc.predict(torch, m, make_fwd(m, Xd_t, Mk_t, g_t, W_, nW_, st), len(Xt), ch)
    Xw = g_t.ego_w.cpu().numpy()                  # window-local ego features, NavSim-style rows
    ones = [np.array([r]) for r in range(len(ids_t))]
    pred_w = fwd_t(Xw, np.ones(Xw.shape[:2], bool), ones, 1, 512)
    if tgt == 'navsim':
        _, _, toks, logs, _ = r0_ego.build_navsim()   # its Y is discarded, never saved
        assert list(ids_t) == list(toks)
        out.update(pred=pred_w, item_id=ids_t, tgt_groups=logs)
    else:
        Xr, MKr, routes, types, _, _ = r0_ego.build_b2d()   # R2's OWN B2D item: stitched ego
        W_b = route_windows(ids_t, routes)                  # sequence + window set pooled
        out.update(pred=pred_w, item_id=ids_t,              # inside the model
                   pred_route=fwd_t(Xr, MKr, W_b, max(len(w) for w in W_b), 32),
                   pred_route_winmean=tc.route_mean(pred_w, W_b, len(routes)),
                   routes=routes, tgt_groups=types)
    print(f'  [{tag}] target {tgt}: {len(out["pred"])} window preds'
          + (f', {len(out["pred_route"])} route preds' if 'pred_route' in out else ''), flush=True)
    tc.save(tc.out_path('r2', a), **out)


def backbone_tag(feature_dir):
    """Short tag of the frozen backbone a feature directory was written with (from its files' `model` / `ckpt`)."""
    import glob
    f = sorted(glob.glob(f'{feature_dir}/route_*.npz'))[0]
    z = np.load(f)
    name = str(z['model']) if 'model' in z.files else str(z['ckpt'])
    known = {'vit_small_patch16_dinov3.lvd1689m': 'dS', 'vit_large_patch16_dinov3.lvd1689m': 'dL',
             'vit_small_patch14_dinov2.lvd142m': 'd2S', 'vit_large_patch14_dinov2.lvd142m': 'd2L'}
    if name in known:
        return known[name]
    import hashlib
    return hashlib.md5(name.encode()).hexdigest()[:6]


def arm_tag(a, tag):
    """The output tag encodes EVERY setting that changes what the encoder reads or computes, so two different
    configurations can never share a result file:
      base arm  _vis / _visonly / _mfm / _mfmonly / _vismfm / _vismfmtrk, _viswin[d<dv>l<layers>]
      + _cp (visual tokens cls+patch) + _front (rgb_front only) + _vpool[c<k>] + _pca<K> + _ego
      + backbone: --arm-suffix if given, else derived from the feature files (dS / dL / ... for the visual dir)
      (the motion-FM checkpoint is validated by check_feature_cache and recorded in the npz's feature_provenance)."""
    if a.dump_attn:
        tag = f'{tag}_attnviz'
    if a.fuse_window:
        pass                                         # the fusion tag below names its tokens (vis / track / smart)
    elif a.visual and a.motion_fm:
        tag = f'{tag}_{"vismfm" if a.visual_only else "vismfmtrk"}'
    elif a.visual:
        tag = f'{tag}_{"visonly" if a.visual_only else "vis"}'
    elif a.motion_fm:
        tag = f'{tag}_{"mfmonly" if a.visual_only else "mfm"}'
    if a.visual_window:
        tag = f'{tag}_viswin' + ('' if (a.viswin_dim, a.viswin_layers) == (128, 2) else f'd{a.viswin_dim}l{a.viswin_layers}')
    if a.fuse_window:
        tag = f'{tag}_fusewin_' + ''.join(t[0] for t in a.fuse_tokens.split(','))      # vt / vs / vts
    if a.ssl:
        tag = f'{tag}_ssl' + ('' if (a.ssl_mask, a.ssl_seg, a.ssl_max_epochs) == (0.3, 4, 120) else f'm{a.ssl_mask:g}s{a.ssl_seg}e{a.ssl_max_epochs}')
    uses_visual = bool(a.visual or a.visual_window or a.fuse_window)
    if uses_visual and a.visual_tokens != 'cls':
        tag = f'{tag}_cp'
    if uses_visual and a.visual_views == 'front':
        tag = f'{tag}_front'
    if uses_visual and a.visual_pool != 'none':
        tag = f'{tag}_vpool' + ('' if a.visual_pool == 'view' else 'c' + a.visual_pool[len('view+ch'):])
    if uses_visual and a.visual_pca:
        tag = f'{tag}_pca{a.visual_pca}'
    if a.ego and (a.visual_only or (a.fuse_window and 'track' not in a.fuse_tokens.split(','))):
        tag = f'{tag}_ego'
    if a.arm_suffix:
        tag = f'{tag}_{a.arm_suffix}'
    elif uses_visual:
        tag = f'{tag}_{backbone_tag(a.visual or a.visual_window or a.fuse_window)}'
    return tag                                   # the motion cache's checkpoint is validated and recorded (feature_provenance), not tagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True, choices=['b2d', 'navsim'])
    ap.add_argument('--gpu', default='2')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--bs', type=int, default=None)
    ap.add_argument('--nfolds', type=int, default=5)
    ap.add_argument('--draws', type=int, default=99)
    ap.add_argument('--match', type=float, default=0.0,
                    help='ablation arm (2026-09-11): add match * mean_i (f_phi(x_i) - b_hat_i)^2 over the training\n'
                         'routes of the batch, b_hat = the stage Rasch difficulty of the same fit that gives theta_hat.\n'
                         'Output tag <tag>_match<value>; never the encoder of record.')
    ap.add_argument('--shuffle', default=None, choices=['route', 'a2l'],
                    help='S-route / S-a2l correspondence control (PROTOCOL_R section 2, amended)')
    ap.add_argument('--shuffle-seed', type=int, default=None, help='shuffle seed 0, 1 or 2')
    ap.add_argument('--ablate-route', action='store_true',
                    help='R2-noRoute: keep Agent<->Lane, mask route_rel entirely.')
    ap.add_argument('--ablate-lane', action='store_true',
                    help='R2-noLane: the LANE-FREE encoder. Removes the whole map side of the '
                         'graph before any tensor is built - lane tokens (lane_mask all False), '
                         'lane geometry, lane_feat, the L2L edges, the A2L edges and a2l_rel, '
                         'and route_rel / route_rel_valid. Ego, command, agents, agent masks and '
                         'the agent-agent structure are untouched byte for byte, and R2Net is '
                         'not rebuilt, so the parameter count and init_hash match the default '
                         'arm. Writes r2nolane_<domain>_s<seed>.npz with arm and ablate_lane '
                         'recorded in the file.')
    ap.add_argument('--visual', default=None,
                    help='two-branch arm (2026-09-12): directory of per-route frozen visual features written by\n'
                         'experiments/visual_features.py; adds a visual branch to the head input. Tag <tag>_vis.')
    ap.add_argument('--visual-only', action='store_true',
                    help='with --visual and/or --motion-fm: DROP THE TRACK BRANCH, the head reads the frozen\n'
                         'branches only (tags _visonly / _mfmonly / _vismfm)')
    ap.add_argument('--visual-tokens', default='cls', choices=['cls', 'cls+patch'])
    ap.add_argument('--visual-views', default='all', choices=['all', 'front'], help='cameras used by the visual branches: all three or rgb_front only (tag _front)')
    ap.add_argument('--visual-pool', default='none', choices=['none', 'view', 'view+ch4', 'view+ch8'],
                    help='parameter-free reduction of the frozen visual features before the projection: mean over the 3 cameras,\n'
                         'optionally followed by an average-pool of k adjacent channels (tags _vpool, _vpoolc4, _vpoolc8)')
    ap.add_argument('--visual-pca', type=int, default=0, help='reduce the frozen visual features to K dims by a PCA fitted per draw on the training frames (0 = off)')
    ap.add_argument('--visual-window', default=None,
                    help='window-level visual-temporal arm (2026-09-12): directory of per-route frozen visual features; the\n'
                         '12 frames of each track window -> P + e_t -> Transformer -> mean, fused with z_w -> [mean, max] over\n'
                         'windows -> head (encoder/harness/visual_window.py). Tag <tag>_viswin.')
    ap.add_argument('--viswin-dim', type=int, default=128)
    ap.add_argument('--ssl', action='store_true', help='track-SSL initialisation (masked agent-track reconstruction on the training routes, epoch by inner validation) before the IRT loss; tag _ssl')
    ap.add_argument('--ssl-max-epochs', type=int, default=120)
    ap.add_argument('--ssl-mask', type=float, default=0.3, help='share of live agents masked per window')
    ap.add_argument('--ssl-seg', type=int, default=4, help='masked segment length in 2-Hz steps')
    ap.add_argument('--fuse-window', default=None,
                    help='token-fusion window arm (2026-09-12): directory of per-route frozen visual features; per window the\n'
                         'tokens of --fuse-tokens (+ modality embeddings) go through a 1-layer fusion Transformer, the mean\n'
                         'is the 64-d window feature and the record route pooling follows. Tag <tag>_fusewin_<tokens>.')
    ap.add_argument('--fuse-tokens', default='vis,track', help='comma list of vis, track, smart (smart needs --motion-fm)')
    ap.add_argument('--viswin-layers', type=int, default=2)
    ap.add_argument('--ego', action='store_true', help='with --visual-only: add the route-level ego branch (speed, acc, yaw rate, command) to the frozen branches (tag _ego)')
    ap.add_argument('--arm-suffix', default='', help='free text appended to the output tag (e.g. the visual backbone)')
    ap.add_argument('--motion-fm', default=None,
                    help='foundation-model arm (2026-09-12): directory of per-route frozen motion features written\n'
                         'by experiments/motion_fm_features.py (frozen SMART/CAT-K); adds a motion branch to the\n'
                         'head input. Tags <tag>_mfm / _mfmonly, and _vismfm / _vismfmtrk together with --visual.')
    ap.add_argument('--dump-attn', default=None,
                    help='visualisation (2026-09-11): after each draw, save the readout attention over the agent\n'
                         'nodes and d f_phi / d z_w per window for the held-out routes to this npz. Output tag\n'
                         '<tag>_attnviz; never the encoder of record.')
    ap.add_argument('--verify-only', action='store_true',
                    help='build the graph, print the control invariants, and stop')
    ap.add_argument('--early-stop', action='store_true',
                    help='B2D only: two-stage nested CHECKPOINT SELECTION BY INNER-VALIDATION '
                         'NLL. Stage 1 carves an inner-val split of whole scenario types out of '
                         'the training block, fits theta on the A_train responses, runs all 30 '
                         'epochs and takes e* = argmin inner-val NLL. Stage 2 refits theta on '
                         'the FULL outer-training responses, re-initialises the model with the '
                         'same seed and trains on the FULL outer-training block for exactly e*+1 '
                         'epochs. The held-out block is forwarded once, after stage 2. So the '
                         'only difference from the frozen arm is 30 -> e*+1 epochs. Nothing is '
                         'stopped early; all 30 epochs are always run. Writes to RG/es/, never '
                         'over frozen30. NAVSIM already selects on inner-val and is not run.')
    ap.add_argument('--proper-init', action='store_true',
                    help='B2D two-stage only: re-seed the torch RNG AFTER the rasch call so '
                         'the model initialisation actually varies with --seed. Without this '
                         'flag scirt.encoder.rasch reseeds torch internally and every seed '
                         'gets the identical initial weights, so the seeds differ only in '
                         'minibatch order. Writes to RG/es_pinit/, never over RG/es/. '
                         'Off by default: the existing es/ results stay bit-reproducible.')
    ap.add_argument('--full-train', action='store_true',
                    help='PROTOCOL_R section 18: ONE model per seed on ALL source items. Stage 1 '
                         'selects e* by inner-val NLL on a grouped inner split (B2D: 8 of 44 '
                         'types; NavSim: 1/5 of the logs), stage 2 refits on every source item '
                         'for e*+1 epochs with theta refitted on every source column (the '
                         'section-12 nested rule). Re-seeds after rasch (proper-init) always. '
                         'Writes only under RG/transfer/. Requires --transfer-to.')
    ap.add_argument('--transfer-to', default=None, choices=['b2d', 'navsim'],
                    help='after --full-train, forward the OTHER domain tensor under the SOURCE '
                         'normalisation statistics and save window-level (and, for a B2D '
                         'target, route-level) predictions plus the source in-sample prediction.')
    ap.add_argument('--subsample-logs', type=int, default=None,
                    help='NavSim source only: keep whole logs (fixed seed) until ~N windows '
                         '(section 18.2 T3, N=2656). The kept logs are recorded in the output.')
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    if a.full_train or a.transfer_to or a.subsample_logs:
        import transfer_common as tc
        tc.check_args(a)
        assert not a.early_stop, '--full-train carries its own nested selection; drop --early-stop'
        a.epochs = a.epochs or (30 if a.domain == 'b2d' else 40)
        a.bs = a.bs or (64 if a.domain == 'b2d' else 256)
        run_full(a)
        return
    assert not (a.early_stop and a.domain == 'navsim'), \
        'NAVSIM already selects the best-inner-val checkpoint; --early-stop is B2D only'
    assert not (a.proper_init and not a.early_stop), \
        '--proper-init is only defined for the two-stage --early-stop protocol'
    assert sum([a.shuffle is not None, a.ablate_route, a.ablate_lane]) <= 1, \
        'one control at a time'
    assert (a.shuffle is None) == (a.shuffle_seed is None), \
        '--shuffle and --shuffle-seed go together'
    tag = 'r2'
    if a.shuffle == 'route':
        tag = f'sroute{a.shuffle_seed}'
    elif a.shuffle == 'a2l':
        tag = f'sa2l{a.shuffle_seed}'
    if a.ablate_route:
        tag = 'r2noroute'
    if a.ablate_lane:
        tag = 'r2nolane'
    if a.match > 0:
        tag = f'{tag}_match{a.match:g}'
    tag = arm_tag(a, tag)
    if a.domain == 'b2d':
        a.epochs = a.epochs or 30; a.bs = a.bs or 64
        run_b2d(a, tag)
    else:
        a.epochs = a.epochs or 40; a.bs = a.bs or 256
        run_navsim(a, tag)


if __name__ == '__main__':
    main()
