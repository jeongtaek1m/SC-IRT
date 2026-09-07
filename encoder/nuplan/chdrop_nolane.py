#!/usr/bin/env python3
"""chdrop.py on the PANEL OF RECORD with the LANE-FREE encoder (R2-noLane).
usage: chdrop_nolane.py [--perm P] <chdrop args, --arch r2 --phase full>

ADAPTATIONS of the frozen driver.  Five attribute / sys.path swaps on imported frozen modules;
nothing in the frozen tree is written.
 1. r0_ego.B2D_MAT -> data/matrices/b2d_e2e16sel_response_matrix.csv, the 16-planner panel of
    record.  The frozen tree hard-codes the pre-selection e2e16 matrix (MindDrive, SimLingo-
    IVL35-1B, UniAD-Base instead of Drive-pi0-Base, Hydra-NeXt, PGS), deleted from the repo in
    v7.0.0.  Set right after `import r0_ego`, before hrel / r2_graph copy it at their import.
    r2_graph.run_full takes Y from r0_ego.build_b2d(), which reads the module global.
 2. scirt.encoder.rasch -> relgraph_e16sel/scirt_rasch.rasch (atdrive.calibration.calibrate_dense),
    the calibration behind the shipped in-domain encoders; every frozen call site imports rasch
    lazily from scirt.encoder.
 3. --perm P: install_shuffle(P) instead of install_shuffle(training seed).  The frozen driver
    ties the permutation to the training seed, so its ten C4 runs are ten permutations with one
    training seed each; a null matched to a three-seed arm mean needs the permutation held fixed
    while the training seed varies.
 4. NEW - /data2/jeongtae/relgraph_nolane_shim goes FIRST on sys.path, so `import r2_graph`
    resolves to the one-file shim that carries the lane-free ablation, while r0_ego, hrel,
    b2d_earlystop, transfer_common and zs21_common still come from the frozen tree (the shim
    keeps RG = '/data2/jeongtae/relgraph').  chdrop.py itself puts RG on sys.path at ITS import,
    so the shim is inserted afterwards and r2_graph is imported here, once, and asserted to be
    the shim file.  Proof of the ablation: relgraph_nolane_shim/verify_nolane_frozen.py.
 5. NEW - --ablate-lane is appended to the argv chdrop hands the frozen driver.  chdrop builds
    that argv inside run_transfer and has no flag to extend it, so r2_graph.main is wrapped
    instead; the flag reaches BOTH legs of the transfer (nuPlan and NavSim) and, inside
    run_full, both the source graph and every target graph through the same load_graph argument.
    The output npz records arm='r2nolane', ablate_lane=True and the marker _nolane in its name.
"""
import importlib.util
import sys

RG = '/data2/jeongtae/relgraph'
S2 = f'{RG}/transfer/zs21/stage2'
E16 = '/data2/jeongtae/relgraph_e16sel'
SHIM = '/data2/jeongtae/relgraph_nolane_shim'
SEL = '/home/jeongtae/SC-IRT/data/matrices/b2d_e2e16sel_response_matrix.csv'

argv = sys.argv[1:]
perm = None
if '--perm' in argv:
    i = argv.index('--perm')
    perm = int(argv[i + 1])
    del argv[i:i + 2]
    assert '--label-shuffle' in argv, '--perm is for the label-shuffle null'
else:
    assert '--label-shuffle' not in argv, 'a shuffle run must fix its permutation with --perm'
assert 'r2' == argv[argv.index('--arch') + 1], 'R2-noLane is an R2 arm (the lane side is R2-only)'
assert 'full' == argv[argv.index('--phase') + 1], \
    'the lane-free arm is the --full-train transfer; chdrop.run_oof globs the r2_ file name'

sys.path.insert(0, S2)
sys.path.insert(0, RG)
import chdrop                                                      # noqa: E402
import r0_ego                                                      # noqa: E402
assert 'hrel' not in sys.modules and 'r2_graph' not in sys.modules, 'patch must precede their import'
r0_ego.B2D_MAT = SEL
import scirt.encoder as _se                                        # noqa: E402  (the RG shim)
_spec = importlib.util.spec_from_file_location('scirt_rasch', f'{E16}/scirt_rasch.py')
_sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sr)                                      # e16sel dir itself stays OFF sys.path
_se.rasch = _sr.rasch

sys.path.insert(0, SHIM)                       # AFTER chdrop's own RG insert, so the shim wins
import r2_graph as _r2                                             # noqa: E402
assert _r2.__file__ == f'{SHIM}/r2_graph.py', _r2.__file__
assert hasattr(_r2, 'strip_lanes'), 'the imported r2_graph does not carry the ablation'
assert r0_ego.__file__ == f'{RG}/r0_ego.py', r0_ego.__file__
_orig_main = _r2.main


def _main():
    if '--ablate-lane' not in sys.argv:
        sys.argv = sys.argv + ['--ablate-lane']
    print(f'[chdrop_nolane] driver argv {sys.argv}', flush=True)
    return _orig_main()


_r2.main = _main

if perm is not None:
    _orig = chdrop.install_shuffle

    def install_shuffle(seed):
        print(f'[chdrop_nolane] label permutation seed {perm} (training seed {seed})', flush=True)
        return _orig(perm)
    chdrop.install_shuffle = install_shuffle

print(f'[chdrop_nolane] r2_graph {_r2.__file__} | r0_ego {r0_ego.__file__} | labels {SEL} | '
      f'rasch {_sr.__file__} | perm {perm} | --ablate-lane forced ON', flush=True)
sys.argv = ['chdrop.py'] + argv
chdrop.main()
