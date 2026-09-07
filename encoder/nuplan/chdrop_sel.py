#!/usr/bin/env python3
"""chdrop.py on the PANEL OF RECORD, with the label-permutation seed decoupled from the
training seed.   usage: chdrop_sel.py [--perm P] <chdrop args>

Three attribute swaps on imported frozen modules; nothing in the frozen tree is written.
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
"""
import importlib.util
import sys

RG = '/data2/jeongtae/relgraph'
S2 = f'{RG}/transfer/zs21/stage2'
E16 = '/data2/jeongtae/relgraph_e16sel'
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

if perm is not None:
    _orig = chdrop.install_shuffle

    def install_shuffle(seed):
        print(f'[chdrop_sel] label permutation seed {perm} (training seed {seed})', flush=True)
        return _orig(perm)
    chdrop.install_shuffle = install_shuffle

print(f'[chdrop_sel] labels {SEL} | rasch {_sr.__file__} | perm {perm}', flush=True)
sys.argv = ['chdrop.py'] + argv
chdrop.main()
