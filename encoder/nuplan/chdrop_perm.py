#!/usr/bin/env python3
"""chdrop.py with the label-permutation seed DECOUPLED from the training seed.
usage: chdrop_perm.py --perm P <chdrop args, must include --label-shuffle>
The frozen chdrop.py calls install_shuffle(a.seed), so its ten C4 runs are ten different
permutations with one training seed each.  A null matched to a three-seed arm mean needs the
permutation held fixed while the training seed varies; this wrapper only swaps the seed handed
to install_shuffle.  Nothing in the frozen tree is written."""
import sys
S2 = '/data2/jeongtae/relgraph/transfer/zs21/stage2'
sys.path.insert(0, S2)
argv = sys.argv[1:]
i = argv.index('--perm'); perm = int(argv[i + 1]); del argv[i:i + 2]
assert '--label-shuffle' in argv, 'wrapper is only for the label-shuffle null'
import chdrop                                                      # noqa: E402
_orig = chdrop.install_shuffle
def install_shuffle(seed):
    print(f'[chdrop_perm] label permutation seed {perm} (training seed {seed})', flush=True)
    return _orig(perm)
chdrop.install_shuffle = install_shuffle
sys.argv = ['chdrop.py'] + argv
chdrop.main()
