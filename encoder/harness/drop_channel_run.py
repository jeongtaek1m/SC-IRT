#!/usr/bin/env python3
"""Ego-channel deletion for the 16-planner B2D OOF loop (r2_graph, RelGraph R2).

WHY.  The scene encoder's ego input has 9 channels and channel 0 is speed, read off an offline
reference trajectory.  A reviewer can ask whether the encoder is trading on speed rather than on
scene structure.  This driver reruns the FROZEN r2_graph B2D out-of-fold loop with one ego
channel removed, so the Table 3A (US) and Table 3B (UPS) rows can be recomputed without it.

WHAT IS PATCHED, AND WHERE.  Nothing under this directory is edited; r2_graph.py, r0_ego.py,
b2d_earlystop.py and hrel.py are imported and used verbatim.  Exactly three things are injected:

  1. r2_graph.build_r2 is wrapped.  The R2Net it returns gets its `forward` replaced by a closure
     that zeroes x[..., idx] before calling the original forward.  x there is the STANDARDISED
     ego tensor ((X - mu)/sd)*MK, so mu/sd are still computed on all 9 channels of the training
     block and the channel is removed afterwards -- the same convention as
     /data2/jeongtae/relgraph/transfer/zs21/stage2/chdrop.py.  Replacing an instance attribute is
     enough because nn.Module.__call__ reads self.forward, so state_dict, init_hash, the
     parameter list and the optimiser are bit-for-bit the frozen path's.

  2. R2Net.qphi gets the SAME treatment.  This is the difference that matters here.  R2Net has a
     second, window-local ego path -- `ze = self.qphi(g.ego_w[rows])`, which builds the attention
     query -- and g.ego_w is RAW (r2_graph.py:613 stores window_ego_feats output straight to the
     GPU, never touched by mu/sd), so a post-standardisation rule alone does NOT reach it and
     speed survives there.  --drop-egow (ON by default in this script, unlike chdrop.py) closes
     that path too.  --no-drop-egow reproduces chdrop.py's literal "ego branch only" arm.

  3. b2d_earlystop.out_path is redirected into --outdir.  Without --early-stop the frozen rule is
     `return f'{rg}/{name}'`, which is the shipped r2_b2d_s{seed}.npz itself; the redirect is what
     keeps this driver from overwriting a frozen result.  The frozen function is still called
     first, so its own asserts still run.

  Plus one repair that is not an injection: scirt_rasch.py (frozen) does
  `from scirt.calibration import calibrate_dense`, and the SC-IRT package has since been renamed
  scirt -> driveat, so that import now raises.  install_scirt_alias() puts driveat into
  sys.modules under the old name.  driveat/calibration.py differs from the last committed
  scirt/calibration.py by a docstring only, so this is the shipped numerics.

WHICH CHANNEL IS SPEED.  r0_ego.py:64 step_feats -> np.c_[sp, acc, yr, |acc|, |yr|, cmd] and
r2_graph.py:176 window_ego_feats builds the identical order, so index 0 is speed in BOTH the
standardised route-level tensor and the raw window-level tensor.  --selfcheck asserts this
numerically against the ego[..., EGO_SPEED] column of the frozen npz rather than trusting it.

VERIFICATION.  Every patched forward, once per distinct tensor shape, prints max |x| over the
dropped columns (must be exactly 0) and a bitwise torch.equal on the kept columns (must be True).
The records are written into the output npz as drop_check.

THE ARGV THE SHIPPED RUNS USED.  Reconstructed from /data2/jeongtae/relgraph_e16sel/r2_b2d_s0.npz,
NOT assumed:
    early_stop            = False        -> no --early-stop  (and therefore no --proper-init,
    es_best_ep            = all -1          which main() asserts requires --early-stop)
    es_stage2_epochs      = all 30       -> --epochs default 30
    es_sigma_stage1       = all NaN
    es_curve              = all NaN
    iv_fingerprint        = all ''
    pred                  = (16, 220)    -> R_DRAWS 16 (b2d_splits.py), --draws default 99 capped
    file at RG/ top level                -> out_path's `not early_stop` branch
    tag in the file name  = 'r2'         -> no --shuffle, no --ablate-route
so the shipped command was
    r2_graph.py --domain b2d --gpu G --seed S
with d 64, epochs 30, bs 64 all left at their defaults.  That is what this driver replays.
(The es_* fields are present in the shipped npz because es.Ledger always writes them; their
values say early stopping was OFF.)

The output npz is the frozen loop's own file with drop_* provenance appended, so SC-IRT's
experiments/build_data.py:export_relgraph reads it unchanged.

Usage
  drop_channel_run.py --selfcheck --gpu 1                       # no training, no writes
  drop_channel_run.py --drop speed --seed 0 --gpu 1 --draws 1 --epochs 2 --tag smoke
  drop_channel_run.py --drop speed --seed 0 --gpu 1             # one full seed of the arm
  drop_channel_run.py --frozen-replay --seed 0 --gpu 1          # the unpatched matched control
"""
import argparse
import json
import os
import sys

import numpy as np

RG = '/data2/jeongtae/relgraph_e16sel'
for _p in (RG, '/home/jeongtae/SC-IRT'):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# r0_ego.py:57 NIN = 9 and :64 step_feats -> np.c_[sp, acc, yr, |acc|, |yr|, cmd];
# r2_graph.py:123 NIN = r0_ego.NIN and :176 window_ego_feats builds the identical order.
CH_NAMES = ['speed', 'acc', 'yawrate', 'absacc', 'absyawrate', 'cmd0', 'cmd1', 'cmd2', 'cmd3']
CH_ALIAS = {'|acc|': 'absacc', 'abs_acc': 'absacc',
            '|yawrate|': 'absyawrate', 'abs_yawrate': 'absyawrate',
            'yr': 'yawrate', 'sp': 'speed'}


def install_scirt_alias():
    """scirt_rasch.py is frozen and imports the pre-rename package name."""
    try:
        import scirt.calibration                                          # noqa: F401
        return 'scirt (native)'
    except ImportError:
        pass
    import atdrive
    import atdrive.calibration
    for old in ('scirt', 'driveat'):                  # both former package names still resolve
        sys.modules[old] = atdrive
        sys.modules[old + '.calibration'] = atdrive.calibration
    return f'scirt/driveat -> atdrive alias ({atdrive.__file__})'


def parse_drop(s):
    """'speed' / 'acc,|acc|' / 'cmd' (= the 4 command channels) -> (names, indices)."""
    if not s or not s.strip():
        return [], []
    names = []
    for tok in s.split(','):
        t = tok.strip()
        if not t:
            continue
        if t == 'cmd':
            names += ['cmd0', 'cmd1', 'cmd2', 'cmd3']
            continue
        t = CH_ALIAS.get(t, t)
        assert t in CH_NAMES, f'unknown ego channel {tok!r}; names are {CH_NAMES} (+cmd)'
        names.append(t)
    names = sorted(set(names), key=CH_NAMES.index)
    return names, [CH_NAMES.index(n) for n in names]


# ── the injection ────────────────────────────────────────────────────────────
class Zeroer:
    """Replaces a module's `forward` with one that zeroes x[..., idx] first."""

    def __init__(self, idx, label, max_lines=8):
        self.idx = list(idx)
        self.label = label
        self.max_lines = max_lines
        self.lines = 0
        self.shapes = set()
        self.calls = 0
        self.checks = []

    def patch(self, mod, argpos=0):
        if not self.idx:
            return mod                       # empty drop list = no patch at all = frozen path
        orig = mod.forward

        def fwd(*args, **kw):
            args = list(args)
            x = args[argpos]
            y = x.clone()
            y[..., self.idx] = 0.0
            self._report(x, y)
            args[argpos] = y
            return orig(*args, **kw)
        mod.forward = fwd
        return mod

    def _report(self, x, y):
        """One line per distinct tensor shape: the dropped columns are exactly 0 and the kept
        columns are the frozen path's tensor, compared bitwise."""
        import torch
        self.calls += 1
        shp = tuple(x.shape)
        seen = shp in self.shapes
        self.shapes.add(shp)
        if seen or self.lines >= self.max_lines:
            return
        self.lines += 1
        keep = [i for i in range(x.shape[-1]) if i not in self.idx]
        dmax = float(y[..., self.idx].abs().max())
        same = bool(torch.equal(x[..., keep], y[..., keep]))
        k = y[..., keep]
        rec = dict(label=self.label, call=self.calls, shape=list(shp), dropped=self.idx,
                   dropped_max_abs=dmax, dropped_exactly_zero=bool(dmax == 0.0),
                   kept_bit_identical=same,
                   kept_mean=float(k.mean()), kept_sd=float(k.std()))
        self.checks.append(rec)
        print(f'  [drop {self.label}] forward #{self.calls} x{list(shp)} '
              f'dropped cols {self.idx} max|x| {dmax:.3e} exactly_zero={dmax == 0.0} | '
              f'kept cols {keep} bit-identical={same} '
              f'mean {float(k.mean()):+.4f} sd {float(k.std()):.4f}', flush=True)

    def assert_clean(self):
        for r in self.checks:
            assert r['dropped_exactly_zero'], f'{self.label}: dropped column not exactly 0: {r}'
            assert r['kept_bit_identical'], f'{self.label}: kept columns were altered: {r}'


def install(idx, egow_idx):
    """Wrap r2_graph.build_r2.  Returns (zeroers, the original factory)."""
    import r2_graph
    z = Zeroer(idx, 'r2.ego')                    # standardised route-level ego -> R2Net.forward(x)
    ze = Zeroer(egow_idx, 'r2.egow')             # RAW window-local ego -> R2Net.qphi(g.ego_w)
    orig = r2_graph.build_r2

    def build_r2(torch, nn, d=64, tau=0.5, heads=4):
        m = orig(torch, nn, d, tau, heads)       # R2Net.__init__ takes .phi and .qphi from
        z.patch(m)                               # r0_ego.build(...), so the r0 factory is NOT
        ze.patch(m.qphi)                         # patched here -- only this instance is
        return m
    r2_graph.build_r2 = build_r2
    return {'ego': z, 'egow': ze}, orig


# ── the no-training check ────────────────────────────────────────────────────
def selfcheck(a):
    """No training, no writes.  On the REAL B2D graph and the REAL ego tensors:
      (a) channel 0 of both ego tensors really is speed;
      (b) the unpatched factory is run TWICE to measure the arithmetic floor of the device;
      (c) with an empty drop list the wrapper is a no-op -- same init hash AND a forward that
          matches the unpatched one to within that floor (on CPU the floor is 0, so this is a
          bit-identity check);
      (d) with --drop speed the output moves well outside the floor, and the per-forward
          verification lines print.

    WHY THE DEVICE MATTERS.  The frozen R2Net's L2L layer and its L2A scatter both use
    Tensor.index_add_, which on CUDA accumulates with atomics and is therefore NOT run-to-run
    reproducible: two forwards of the SAME weights on the SAME input differ by ~4e-08.  That is
    the frozen path's own nondeterminism, not the wrapper's, and it makes "bit-identical" an
    impossible bar on GPU.  --check-device cpu (the default) removes it, so the no-op proof is
    exact; --check-device cuda reports the floor instead."""
    import torch
    import torch.nn as nn
    import r0_ego
    import r2_graph
    import b2d_earlystop as bes
    dev = a.check_device

    X, MK, routes, types, Y, static = r0_ego.build_b2d()
    g, ids = r2_graph.load_graph('b2d', torch, dev)
    W = r2_graph.route_windows(ids, routes)
    nW = max(len(w) for w in W)

    # (a) is index 0 speed?
    d = np.load(r0_ego.B2D_NPZ, allow_pickle=True)
    sp_npz = d['ego'].astype(np.float32)[:, :, r0_ego.EGO_SPEED]
    sp_ego_w = g.ego_w[:, :, 0].cpu().numpy()
    assert np.array_equal(sp_npz, sp_ego_w), 'ego_w column 0 is not the npz ego speed column'
    sp_X = X[:, :, 0][MK]
    print(f'[selfcheck] channel 0 == speed: ego_w[:,:,0] == npz ego[...,EGO_SPEED] exactly; '
          f'X[:,:,0] range [{sp_X.min():.3f}, {sp_X.max():.3f}] m/s, '
          f'ego_w[:,:,0] range [{sp_ego_w.min():.3f}, {sp_ego_w.max():.3f}] m/s', flush=True)

    # one forward on the first --check-routes routes, standardised the way run_b2d does it
    sel = np.arange(min(a.check_routes, len(routes)))
    allrows = np.concatenate([W[i] for i in range(len(routes))])
    st = r2_graph.graph_stats(g, allrows, torch, dev)
    mu = X[MK].mean(0)
    sd = X[MK].std(0) + 1e-6
    Xn = ((X - mu) / sd) * MK[..., None]
    Xd = torch.tensor(Xn, device=dev)
    Md = torch.tensor(MK, device=dev)
    s = torch.tensor(sel, device=dev)
    rows = torch.tensor(np.concatenate([W[i] for i in sel]), device=dev)
    wb = torch.tensor(np.concatenate([np.full(len(W[i]), b) for b, i in enumerate(sel)]),
                      device=dev)
    ww = torch.tensor(np.concatenate([np.arange(len(W[i])) for i in sel]), device=dev)

    def one(factory, tag):
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        m = factory(torch, nn, 64).to(dev)
        m.eval()
        h = bes.init_hash(m)
        with torch.no_grad():
            out = m(Xd[s], Md[s], g, rows, wb, ww, nW, st)
        print(f'[selfcheck] {tag:14s} init_hash {h}  out[:4] '
              f'{np.array2string(out[:4].cpu().numpy(), precision=8)}', flush=True)
        return out, h

    orig = r2_graph.build_r2
    base, h_base = one(orig, 'unpatched')
    base2, h_base2 = one(orig, 'unpatched again')
    floor = float((base - base2).abs().max())
    print(f'[selfcheck] DEVICE FLOOR on {dev}: the unpatched factory run twice differs by '
          f'max|diff| {floor:.3e} (index_add_ atomics; 0 on cpu)', flush=True)
    assert h_base == h_base2, 'the unpatched factory is not seed-reproducible'

    zs_e, _ = install([], [])
    empty, h_empty = one(r2_graph.build_r2, 'patched drop=[]')
    r2_graph.build_r2 = orig
    assert zs_e['ego'].calls == 0 and zs_e['egow'].calls == 0, \
        'the empty-drop wrapper installed a closure; it must not patch at all'
    ok_hash = (h_base == h_empty)
    diff_empty = float((base - empty).abs().max())
    ok_out = bool(torch.equal(base, empty))
    print(f'[selfcheck] EMPTY-DROP NO-OP: init_hash identical {ok_hash}, closures installed 0, '
          f'forward bit-identical {ok_out} (max|diff| {diff_empty:.3e} vs floor {floor:.3e})',
          flush=True)
    assert ok_hash, 'the empty-drop wrapper changed the initial weights'
    assert ok_out or diff_empty <= floor, \
        f'the empty-drop wrapper moved the output beyond the device floor ({diff_empty:.3e})'

    names, idx = parse_drop(a.drop or 'speed')
    zs, _ = install(idx, idx if a.drop_egow else [])
    dropped, h_drop = one(r2_graph.build_r2, f'drop={",".join(names)}')
    r2_graph.build_r2 = orig
    for z in zs.values():
        z.assert_clean()
    diff_drop = float((base - dropped).abs().max())
    print(f'[selfcheck] DROP {names} idx {idx} drop_egow {a.drop_egow}: closures fired '
          f'ego {zs["ego"].calls} egow {zs["egow"].calls}, init_hash identical to unpatched '
          f'{h_base == h_drop} (same seed, same architecture), '
          f'output max|diff| {diff_drop:.4e} = {diff_drop / max(floor, 1e-30):.0f}x the floor',
          flush=True)
    assert h_base == h_drop, 'the drop wrapper changed the initial weights'
    assert zs['ego'].calls == 1, 'the standardised-ego closure did not fire'
    assert zs['egow'].calls == (1 if a.drop_egow else 0), 'the raw-ego-window closure misfired'
    assert diff_drop > max(floor, 1e-12) * 100, 'dropping speed barely moved the output'
    print('[selfcheck] PASS', flush=True)


# ── the training run ─────────────────────────────────────────────────────────
def run(a, names, idx, driver_cmd):
    import b2d_earlystop as bes
    import r2_graph

    os.makedirs(a.outdir, exist_ok=True)
    _orig_out = bes.out_path
    box = {}

    def out_path(rg, name, early_stop, proper_init=False):
        frozen = _orig_out(rg, name, early_stop, proper_init)   # keep every frozen assert
        q = os.path.join(a.outdir, f'{a.tag}_{os.path.basename(frozen)}')
        assert os.path.abspath(q) != os.path.abspath(frozen), \
            'the redirect resolved back onto the frozen output path'
        assert not os.path.exists(q), f'{q} already exists; refusing to overwrite'
        box['out'] = q
        return q
    bes.out_path = out_path

    zs, _ = install(idx, idx if a.drop_egow else [])

    argv = ['r2_graph.py', '--domain', 'b2d', '--gpu', str(a.gpu), '--seed', str(a.seed)]
    if a.epochs is not None:
        argv += ['--epochs', str(a.epochs)]
    if a.draws is not None:
        argv += ['--draws', str(a.draws)]
    sys.argv = argv
    print(f'[drop] === {" ".join(argv)} ===', flush=True)
    r2_graph.main()

    for z in zs.values():
        z.assert_clean()
    out = box['out']
    z = np.load(out, allow_pickle=True)
    add = dict(drop_names=np.array(names, dtype=object), drop_idx=np.array(idx, int),
               drop_channel_names=np.array(CH_NAMES, dtype=object),
               drop_egow=bool(a.drop_egow), drop_tag=a.tag, drop_seed=a.seed,
               drop_check=json.dumps({k: v.checks for k, v in zs.items()}),
               drop_forward_calls=json.dumps({k: v.calls for k, v in zs.items()}),
               drop_cmd=driver_cmd, drop_replayed_argv=' '.join(argv))
    keep = {k: z[k] for k in z.files}
    keep.update(add)
    np.savez(out, **keep)
    print(f'\n[drop] pred {keep["pred"].shape} sigma {keep["sigma"].shape} '
          f'per_draw {keep["per_draw"].shape} routes {keep["routes"].shape} '
          f'early_stop {bool(keep["early_stop"])}', flush=True)
    print(f'[drop] forwards patched: '
          f'{ {k: v.calls for k, v in zs.items()} }', flush=True)
    print(f'WROTE {out}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drop', default='',
                    help=f'comma list of ego channels to zero; names {",".join(CH_NAMES)} '
                         f'(+ cmd = all four).  Empty = no patch at all = the frozen path.')
    ap.add_argument('--drop-egow', dest='drop_egow', action='store_true', default=True,
                    help='ON BY DEFAULT: also zero the channel in the raw window-local ego query '
                         'g.ego_w -> qphi.  This is the complete removal.')
    ap.add_argument('--no-drop-egow', dest='drop_egow', action='store_false',
                    help='standardised ego branch only (chdrop.py default): speed SURVIVES in '
                         'the raw qphi query.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu', default=None, required=True)
    ap.add_argument('--epochs', type=int, default=None, help='smoke only; shipped runs used 30')
    ap.add_argument('--draws', type=int, default=None, help='smoke only; shipped runs used all 16')
    ap.add_argument('--tag', default=None, help='output file prefix')
    ap.add_argument('--outdir', default=f'{RG}/nospeed')
    ap.add_argument('--selfcheck', action='store_true',
                    help='no training, no writes: prove the wrapper is a no-op when --drop is '
                         'empty and that it bites when it is not')
    ap.add_argument('--check-routes', type=int, default=4, help='--selfcheck forward batch size')
    ap.add_argument('--frozen-replay', action='store_true',
                    help='allow an empty --drop and run the loop with NOTHING patched: the '
                         'matched control for the drop arm, replayed under the current '
                         'environment (which reaches driveat through the scirt alias, unlike the '
                         'shipped runs).  Everything else is identical, so a control that lands '
                         'on the shipped numbers licenses reading the drop arm against them.')
    ap.add_argument('--check-device', default='cpu', choices=['cpu', 'cuda'],
                    help='--selfcheck device.  cpu (default) makes index_add_ deterministic, so '
                         'the empty-drop no-op proof is bit-exact; cuda reports the atomics floor.')
    driver_cmd = ' '.join(sys.argv)          # captured BEFORE run() overwrites sys.argv
    a = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    alias = install_scirt_alias()
    names, idx = parse_drop(a.drop)
    if a.tag is None:
        a.tag = 'r2drop' + (''.join(names) or 'none')
    print(f'[drop] tag {a.tag} seed {a.seed} gpu {a.gpu!r} | drop {names} -> idx {idx} '
          f'of {CH_NAMES} | drop_egow {a.drop_egow} | {alias}', flush=True)

    if a.selfcheck:
        selfcheck(a)
        return
    assert idx or a.frozen_replay, \
        '--drop is empty: nothing would be removed.  Use --selfcheck for the no-op proof, or '\
        '--frozen-replay to run the unpatched control on purpose.'
    run(a, names, idx, driver_cmd)

    try:
        import torch
        if torch.cuda.is_available():
            print(f'[drop] max_cuda_alloc {torch.cuda.max_memory_allocated()/2**20:.1f}MiB '
                  f'reserved {torch.cuda.max_memory_reserved()/2**20:.1f}MiB', flush=True)
    except Exception:
        pass


if __name__ == '__main__':
    main()
