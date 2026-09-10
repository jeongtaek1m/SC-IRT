Import stubs that let the official MFGPreliability package (github.com/umbrellagong/MFGPreliability,
commit 604bfce) import unmodified in this environment. Its single-fidelity path never calls
`emukit.multi_fidelity.convert_lists_to_array` (bi-fidelity only) nor `pyDOE.lhs` (our wrapper
draws the initial routes from the discrete bank), so both are stubbed to raise if ever called.
