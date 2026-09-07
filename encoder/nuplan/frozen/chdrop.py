#!/usr/bin/env python3
"""PROTOCOL_R.md section 21.9 stage 2 — ego-channel deletion driver.

WHAT IT IS.  One driver for every section-21.9 arm.  It runs the FROZEN loops
(r0_ego.run_b2d / r0_ego.run_full, r2_graph.run_b2d / r2_graph.run_full) as libraries and
injects exactly one thing: the chosen ego channels are set to 0 AFTER the source mu/sd
standardisation, identically for the source tensor and for every target tensor.  No frozen
file is edited (r0_ego.py, r2_graph.py, hrel.py, transfer_common.py, b2d_earlystop.py are
imported, never written).

WHERE THE ZEROING HAPPENS.  At the point where the standardised ego tensor enters the model.
Both frozen loops build the model through a factory (r0_ego.build / r2_graph.build_r2) and
then call it as m(x, mask, ...) with x = ((X - mu)/sd) * MK, so the factory is wrapped and the
returned module's `forward` is replaced by a closure that zeroes x[..., idx] and calls the
original forward.  Consequences, all of them wanted:
  * mu/sd are still computed on ALL 9 channels of the training block (the --drop-agent-channels
    convention of section 19.1 Z4: statistics on everything, the channel removed afterwards);
  * source and EVERY target (nuPlan logged, nuPlan routed, navtest/navhard) go through the same
    forward, so no target can be treated differently by accident;
  * the 16-draw OOF path and the --full-train path share the injection;
  * with --drop '' nothing is patched at all, so the default path is the frozen path.
The wrapper writes only the dropped columns; the kept columns are the frozen path's tensor,
bit for bit.  It prints the check (max |x| over the dropped columns = 0, and the kept-column
mean/sd) for the first forwards it sees, per distinct tensor shape.

R2's TWO EGO INPUTS (recorded, not hidden).  r2_graph.R2Net has the ego branch
`zego = dist_pool(self.phi(x))` — x is the standardised ego tensor, and that is what --drop
zeroes, which is what section 21.9 specifies ("ego 브랜치의 같은 채널").  It ALSO has a second,
window-local ego path `ze = self.qphi(g.ego_w[rows])` used to build the attention query.
g.ego_w is RAW (never touched by mu/sd), so the post-standardisation rule does not reach it and
speed survives there.  --drop-egow (default OFF) additionally zeroes the same channel of the
qphi input, for the caller who wants the channel gone from the whole model.  Which of the two
is the section-21.9 A2 arm is a protocol question, not a code question; the default is the
literal reading and every output records which was used.

Usage
  chdrop.py --arch r0 --drop speed --seed 0 --phase oof  --tag A1 --gpu 2
  chdrop.py --arch r0 --drop speed --seed 0 --phase full --tag A1 --gpu 2
  chdrop.py --arch r2 --drop speed --seed 0 --phase full --tag A2 --gpu 3
  chdrop.py --arch r0 --drop speed --seed 3 --phase full --tag C4 --label-shuffle
  chdrop.py --arch r0 --drop speed --seed 0 --phase full --tag C5 --force-estar a0_estar.json
  chdrop.py --arch r0 --drop speed --seed 0 --phase full --tag C6 --source navsim
"""
import argparse
import json
import os
import sys

import numpy as np

RG = '/data2/jeongtae/relgraph'
S2 = f'{RG}/transfer/zs21/stage2'
for p in (RG, '/home/jeongtae/SC-IRT', '/home/jeongtae/SCIRT/b2d_irt'):
    if p not in sys.path:
        sys.path.insert(0, p)

# r0_ego.py:57 NIN = 9 and :64-69 step_feats -> np.c_[sp, acc, yr, |acc|, |yr|, cmd];
# r2_graph.py:127 NIN = r0_ego.NIN and :180-190 window_ego_feats builds the identical order.
CH_NAMES = ['speed', 'acc', 'yawrate', 'absacc', 'absyawrate', 'cmd0', 'cmd1', 'cmd2', 'cmd3']
CH_ALIAS = {'|acc|': 'absacc', 'abs_acc': 'absacc', 'absacc': 'absacc',
            '|yawrate|': 'absyawrate', 'abs_yawrate': 'absyawrate', 'absyawrate': 'absyawrate',
            'yr': 'yawrate', 'sp': 'speed'}


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
        assert t in CH_NAMES, f'unknown ego channel {tok!r}; names are {CH_NAMES} (+cmd, |acc|, |yawrate|)'
        names.append(t)
    names = sorted(set(names), key=CH_NAMES.index)
    return names, [CH_NAMES.index(n) for n in names]


# ── the injection ────────────────────────────────────────────────────────────
class Zeroer:
    """Replaces a module's `forward` with one that zeroes x[..., idx] first.  nn.Module.__call__
    reads self.forward, so an instance attribute is enough: no wrapper module, so state_dict,
    init_hash, parameter list and optimiser are exactly the frozen path's."""

    def __init__(self, idx, label, max_lines=10):
        self.idx = list(idx)
        self.label = label
        self.max_lines = max_lines
        self.lines = 0
        self.shapes = set()
        self.calls = 0
        self.checks = []

    def reset(self, label=None):
        """New run inside the same process (R0 forwards nuPlan-logged, nuPlan-routed and NavSim
        in three calls): print the check again for each, so every target is evidenced."""
        self.shapes = set()
        self.lines = 0
        if label:
            self.label = label

    def patch(self, mod, argpos=0):
        if not self.idx:
            return mod
        orig = mod.forward

        def fwd(*args, **kw):
            args = list(args)
            x = args[argpos]
            self._check(x, 'before')
            x = x.clone()
            x[..., self.idx] = 0.0
            self._check(x, 'after')
            args[argpos] = x
            return orig(*args, **kw)
        mod.forward = fwd
        return mod

    def _check(self, x, when):
        if when == 'before':
            self.calls += 1
            self.new = tuple(x.shape) not in self.shapes
            self.shapes.add(tuple(x.shape))
            return
        if not (self.new or self.lines < 3) or self.lines >= self.max_lines:
            return
        self.lines += 1
        keep = [i for i in range(x.shape[-1]) if i not in self.idx]
        d = x[..., self.idx].abs().max().item() if self.idx else 0.0
        k = x[..., keep]
        rec = dict(label=self.label, call=self.calls, shape=list(x.shape),
                   dropped_max_abs=float(d), kept_mean=float(k.mean()), kept_sd=float(k.std()))
        self.checks.append(rec)
        print(f'  [chdrop {self.label}] forward #{self.calls} x{list(x.shape)} '
              f'dropped cols {self.idx} max|x| {d:.3e}  kept cols mean {float(k.mean()):+.4f} '
              f'sd {float(k.std()):.4f}', flush=True)


def install(arch, idx, egow_idx):
    """Patch the model factories.  Returns the Zeroer objects (for the provenance record)."""
    import r0_ego
    zs = {}
    if arch == 'r0':
        z = Zeroer(idx, 'r0.ego')
        orig = r0_ego.build

        def build(torch, nn, d=64, tau=0.5):
            return z.patch(orig(torch, nn, d, tau))
        r0_ego.build = build
        zs['ego'] = z
    else:
        import r2_graph
        z = Zeroer(idx, 'r2.ego')
        ze = Zeroer(egow_idx, 'r2.egow')
        orig = r2_graph.build_r2

        def build_r2(torch, nn, d=64, tau=0.5, heads=4, ag_keep=None):
            m = orig(torch, nn, d, tau, heads, ag_keep)   # R2Net.__init__ takes .phi from
            z.patch(m)                                    # r0_ego.build(...).phi, so the r0
            ze.patch(m.qphi)                              # factory is NOT patched here
            return m
        r2_graph.build_r2 = build_r2
        zs['ego'] = z
        zs['egow'] = ze
    return zs


def install_shuffle(seed):
    """C4 / --label-shuffle: permute the B2D route labels with zs21_common's own rule.  The
    frozen loops derive fail / b_ref / theta from build_b2d's Y, so patching the loader shuffles
    every training signal at once.  The TRUE arrays are kept here and written back into the
    output npz, because scoring must stay on the true labels."""
    import r0_ego
    # zs21_common puts /home/jeongtae/SC-IRT ahead of the RG shim on sys.path, and that repo's
    # scirt package has no .encoder.  r2_graph.run_full imports scirt.encoder LAZILY (line 1263),
    # i.e. after this, so prime sys.modules from the shim FIRST and put RG back in front after.
    import scirt.encoder                                                     # noqa: F401
    sys.path.insert(0, f'{RG}/transfer/zs21')
    from zs21_common import shuffle_route_labels
    sys.path.insert(0, RG)
    orig = r0_ego.build_b2d
    keep = {}

    def build_b2d():
        X, MK, routes, types, Y, static = orig()
        Ys, perm = shuffle_route_labels(Y, seed)
        keep['Y_true'] = Y
        keep['perm'] = perm
        return X, MK, routes, types, Ys, static
    r0_ego.build_b2d = build_b2d
    return keep


def install_forced_estar(ep):
    """C5: keep stage 1 (so the arm's OWN e* is still measured and recorded) but hand stage 2
    the supplied epoch index.  es.Selector.best_ep is a read-only property over .best, so the
    single point of override is that tuple, set between the two stages()."""
    import transfer_common as tc
    base = tc.FullTrainPlan
    rec = {}

    class ForcedPlan(base):
        def stages(self):
            it = super().stages()
            s1 = next(it)
            yield s1
            rec['natural'] = int(self.picker.best_ep)
            rec['forced'] = int(ep)
            rec['natural_nll'] = float(self.picker.best[0])
            rec['natural_sigma'] = float(self.picker.sigma)
            print(f'  [chdrop] --force-estar: stage 1 selected e* {rec["natural"]} '
                  f'(inner-val nll {rec["natural_nll"]:.4f}, sigma {rec["natural_sigma"]:.3f}), '
                  f'stage 2 forced to e* {ep}', flush=True)
            # F3 fix: make the forced epoch's own bookkeeping consistent -- best[0] and
            # picker.sigma are what transfer_common.py:96-102 store as the printed nll and
            # es_sigma_stage1.  The NATURAL values are preserved in rec / the npz.
            self.picker.best = (float(self.picker.curve[int(ep), 1]), int(ep))
            self.picker.sigma = float(self.picker.curve[int(ep), 2])
            for s in it:
                yield s
    tc.FullTrainPlan = ForcedPlan
    return rec


# ── output bookkeeping ───────────────────────────────────────────────────────
def meta(a, names, idx, zs, extra=None):
    d = dict(stage2_arm=a.tag, stage2_arch=a.arch, stage2_source=a.source,
             stage2_drop=np.array(names, dtype=object), stage2_drop_idx=np.array(idx, int),
             stage2_channel_names=np.array(CH_NAMES, dtype=object),
             stage2_drop_egow=bool(a.drop_egow),
             stage2_label_shuffle=bool(a.label_shuffle),
             stage2_check=json.dumps({k: v.checks for k, v in zs.items()}),
             stage2_cmd=' '.join(sys.argv))
    d.update(extra or {})
    return d


def only(pattern):
    """The one file the frozen loop just wrote.  Globbed rather than name-guessed: the frozen
    out_path appends section-18.6/19.1/20.2 markers (_logged, _wu*, _dag*) from flags the arm
    never set, so a hardcoded name silently breaks when a default changes."""
    import glob as _g
    hit = sorted(_g.glob(pattern, recursive=True))
    assert len(hit) == 1, f'expected exactly one output for {pattern}, got {hit}'
    return hit[0]


def rewrite(src_path, dst_path, add):
    z = np.load(src_path, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    out.update(add)
    np.savez(dst_path, **out)
    print(f'WROTE {dst_path}', flush=True)
    return out


def per_draw_true(pred, fail):
    from scipy.stats import spearmanr
    v = np.full(pred.shape[0], np.nan)
    for k in range(pred.shape[0]):
        h = np.isfinite(pred[k])
        if h.sum() >= 3:
            v[k] = spearmanr(pred[k][h], fail[h]).correlation
    return v


# ── phases ───────────────────────────────────────────────────────────────────
def run_oof(a, names, idx):
    """Gate 0: the frozen 16-draw unified_split B2D OOF under --early-stop --proper-init, with
    b2d_earlystop.out_path redirected (es_pinit/ is never written).  Same shape as
    transfer/zs21/c0_oof.py and stage2/a0_oof.py."""
    import b2d_earlystop as es
    raw = f'{a.outdir}/raw/{a.tag}_oof_s{a.seed}'
    os.makedirs(raw, exist_ok=True)
    _orig = es.out_path

    def out_path(rg, name, early_stop, proper_init=False):
        p = _orig(rg, name, early_stop, proper_init)      # keeps every assert of the frozen rule
        assert '/es_pinit/' in p, p
        return f'{raw}/{os.path.basename(p)}'
    es.out_path = out_path

    zs = install(a.arch, idx, idx if a.drop_egow else [])
    shuf = install_shuffle(a.seed) if a.label_shuffle else None

    mod = _import(a.arch)
    argv = [f'{a.arch}.py', '--domain', 'b2d', '--gpu', str(a.gpu), '--seed', str(a.seed),
            '--early-stop', '--proper-init']
    if a.bs:
        argv += ['--bs', str(a.bs)]
    if a.draws != 99:
        argv += ['--draws', str(a.draws)]
    sys.argv = argv
    mod.main()

    base = 'r0_b2d_s%d.npz' % a.seed if a.arch == 'r0' else 'r2_b2d_s%d.npz' % a.seed
    srcf = f'{raw}/{base}'
    dst = f'{a.outdir}/{a.tag}_b2d_oof_s{a.seed}.npz'
    z = np.load(srcf, allow_pickle=True)
    add = meta(a, names, idx, zs, dict(arm=a.tag, seed=a.seed, raw_file=srcf))
    if 'item_id' not in z.files:                     # r0_ego's savez omits it, r2_graph's has it
        add['item_id'] = z['routes']
    if shuf is not None:                             # scoring must use the TRUE labels
        Yt = shuf['Y_true']
        ft = np.nanmean(Yt, 0)
        from scirt.encoder import rasch
        _, bt = rasch(Yt, it=800)
        add.update(Y_shuffled=z['Y'], fail_shuffled=z['fail'], b_ref_shuffled=z['b_ref'],
                   Y=Yt, fail=ft, b_ref=np.asarray(bt, float),
                   label_perm=shuf['perm'], label_shuffle=True)
        if 'per_draw' in z.files:
            add['per_draw_shuffled'] = z['per_draw']
        add['per_draw'] = per_draw_true(z['pred'], ft)
    elif 'per_draw' not in z.files:                  # r0: recompute so score_zs21.gate0() works
        add['per_draw'] = per_draw_true(z['pred'], z['fail'])
    rewrite(srcf, dst, add)


def run_transfer(a, names, idx):
    """--full-train + forward to every target.  The frozen run_full is used verbatim; only
    transfer_common.TRANSFER_DIR is redirected, so nothing beside the section-18/19/20 outputs
    can be written.  R0 needs one run per nuPlan ego tensor (its run_full forwards the one named
    by --nuplan-ego); the two are merged here into the stage-1 one-file format after asserting
    they came from the identical model.  R2's run_full already forwards both."""
    import transfer_common as tc
    raw = f'{a.outdir}/raw/{a.tag}_full_s{a.seed}'
    os.makedirs(raw, exist_ok=True)
    tc.TRANSFER_DIR = raw

    zs = install(a.arch, idx, idx if a.drop_egow else [])
    shuf = install_shuffle(a.seed) if a.label_shuffle else None
    forced = None
    if a.force_estar is not None:
        forced = install_forced_estar(a.force_estar)

    mod = _import(a.arch)
    src = a.source

    def call(extra, sub):
        """One frozen --full-train run into its OWN raw subdirectory, so the file it produced is
        identified by globbing that directory rather than by re-deriving the frozen name."""
        d = f'{raw}/{sub}'
        os.makedirs(d, exist_ok=True)
        tc.TRANSFER_DIR = d
        argv = [f'{a.arch}.py', '--domain', src, '--gpu', str(a.gpu), '--seed', str(a.seed),
                '--full-train'] + extra
        if a.bs:
            argv += ['--bs', str(a.bs)]
        sys.argv = argv
        print(f'\n[chdrop] === {" ".join(argv)} ===', flush=True)
        for z in zs.values():
            z.reset()
        mod.main()
        return only(f'{d}/**/*.npz')

    if a.arch == 'r0':
        fl = call(['--transfer-to', 'nuplan', '--nuplan-ego', 'logged'], 'nuplan_logged')
        fr = call(['--transfer-to', 'nuplan', '--nuplan-ego', 'routed'], 'nuplan_routed')
        zl = np.load(fl, allow_pickle=True)
        zr = np.load(fr, allow_pickle=True)
        for k in ('pred_src', 'x_mu', 'x_sd', 'es_best_ep'):
            assert np.array_equal(np.asarray(zl[k]), np.asarray(zr[k])), \
                f'the logged and routed runs are not the same model ({k} differs)'
        assert np.array_equal(zl['item_id'], zr['item_id'])
        out = {k: zl[k] for k in zl.files}
        out.update(pred_logged=zl['pred'], pred_routed=zr['pred'], pred=zl['pred'],
                   nuplan_ego='both', raw_files=np.array([fl, fr], dtype=object))
        out.update(meta(a, names, idx, zs, dict(arm=a.tag, frozen_arm='r0', seed=a.seed)))
        if forced:
            out.update(estar_natural=forced.get('natural', -1), estar_forced=forced.get('forced', -1),
                        estar_natural_nll=forced.get('natural_nll', float('nan')),
                        estar_natural_sigma=forced.get('natural_sigma', float('nan')))
        if shuf is not None:
            out.update(label_shuffle=True, label_perm=shuf['perm'])
        dst = f'{a.outdir}/{a.tag}_{src}2nuplan_s{a.seed}.npz'
        np.savez(dst, **out)
        print(f'WROTE {dst}', flush=True)
    else:
        srcf = call(['--transfer-to', 'nuplan'], 'nuplan')   # r2 writes pred_logged AND pred_routed
        add = meta(a, names, idx, zs, dict(arm=a.tag, frozen_arm='r2', seed=a.seed,
                                           raw_file=srcf))
        if forced:
            add.update(estar_natural=forced.get('natural', -1), estar_forced=forced.get('forced', -1),
                        estar_natural_nll=forced.get('natural_nll', float('nan')),
                        estar_natural_sigma=forced.get('natural_sigma', float('nan')))
        if shuf is not None:
            add.update(label_shuffle=True, label_perm=shuf['perm'])
        rewrite(srcf, f'{a.outdir}/{a.tag}_{src}2nuplan_s{a.seed}.npz', add)

    if src == 'b2d':                                  # navtest / navhard
        srcf = call(['--transfer-to', 'navsim'], 'navsim')
        add = meta(a, names, idx, zs, dict(arm=a.tag, frozen_arm=a.arch, seed=a.seed,
                                           raw_file=srcf))
        if forced:
            add.update(estar_natural=forced.get('natural', -1), estar_forced=forced.get('forced', -1),
                        estar_natural_nll=forced.get('natural_nll', float('nan')),
                        estar_natural_sigma=forced.get('natural_sigma', float('nan')))
        if shuf is not None:
            add.update(label_shuffle=True, label_perm=shuf['perm'])
        rewrite(srcf, f'{a.outdir}/{a.tag}_b2d2navsim_s{a.seed}.npz', add)


def _import(arch):
    import importlib
    return importlib.import_module('r0_ego' if arch == 'r0' else 'r2_graph')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True, choices=['r0', 'r2'])
    ap.add_argument('--drop', default='',
                    help=f'comma list of ego channels to zero after standardisation; '
                         f'names {",".join(CH_NAMES)} (+ cmd = all four, |acc|, |yawrate|). '
                         f"Empty string = no patch at all = the frozen path.")
    ap.add_argument('--drop-egow', action='store_true',
                    help='R2 only, OFF by default: also zero the channel in the window-local ego '
                         'query g.ego_w -> qphi.  section 21.9 says "ego branch", which is the '
                         'standardised x -> phi path; g.ego_w is raw and is not covered by the '
                         'post-standardisation rule.  See the module docstring.')
    ap.add_argument('--label-shuffle', action='store_true',
                    help='C4: permute the B2D route labels (zs21_common.shuffle_route_labels, '
                         'per-seed RNG).  Scoring stays on the true labels.')
    ap.add_argument('--force-estar', default=None,
                    help='C5: JSON (inline or a file path) mapping seed -> e* to force on stage '
                         '2 of --full-train, e.g. {"0":4,"1":16,"2":3}.  Stage 1 still runs and '
                         'its own e* is recorded as estar_natural.')
    ap.add_argument('--source', default='b2d', choices=['b2d', 'navsim'],
                    help='C6: train on NavSim instead of B2D (frozen NavSim recipe: BCE, wd '
                         '0.05, bs 256, 40 epochs, warmup+cosine) and forward to nuPlan.')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--phase', required=True, choices=['oof', 'full'])
    ap.add_argument('--tag', default=None, help='arm tag used in the output file names')
    ap.add_argument('--gpu', default='2')
    ap.add_argument('--bs', type=int, default=None)
    ap.add_argument('--draws', type=int, default=99, help='smoke only: fewer OOF draws')
    ap.add_argument('--outdir', default=S2)
    a = ap.parse_args()

    assert a.gpu in ('', '2', '3'), 'GPUs 0 and 1 belong to another user'
    names, idx = parse_drop(a.drop)
    if a.tag is None:
        a.tag = f'{a.arch}drop' + (''.join(n for n in names) or 'none')
    assert not (a.drop_egow and a.arch != 'r2'), '--drop-egow is an R2 flag'
    assert not (a.label_shuffle and a.source != 'b2d'), '--label-shuffle is the B2D route rule'
    assert not (a.phase == 'oof' and a.source != 'b2d'), 'Gate 0 is the B2D 16-draw OOF'
    if a.force_estar is not None:
        assert a.phase == 'full', '--force-estar acts on the --full-train stage-2 epoch count'
        txt = open(a.force_estar).read() if os.path.exists(a.force_estar) else a.force_estar
        m = {int(k): int(v) for k, v in json.loads(txt).items()}
        assert a.seed in m, f'--force-estar has no entry for seed {a.seed}'
        a.force_estar = m[a.seed]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.makedirs(a.outdir, exist_ok=True)
    print(f'[chdrop] arm {a.tag} arch {a.arch} source {a.source} seed {a.seed} phase {a.phase} '
          f'gpu {a.gpu!r} | drop {names} -> idx {idx} of {CH_NAMES} | drop_egow {a.drop_egow} '
          f'| label_shuffle {a.label_shuffle} | force_estar {a.force_estar}', flush=True)

    if a.phase == 'oof':
        run_oof(a, names, idx)
    else:
        run_transfer(a, names, idx)

    try:
        import torch
        if torch.cuda.is_available():
            print(f'[chdrop] max_cuda_alloc {torch.cuda.max_memory_allocated()/2**20:.1f}MiB '
                  f'reserved {torch.cuda.max_memory_reserved()/2**20:.1f}MiB', flush=True)
    except Exception:
        pass


if __name__ == '__main__':
    main()
