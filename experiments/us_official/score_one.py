#!/usr/bin/env python3
"""Score one (or several) rebuilt US descriptor npz files exactly the way
Table 3A does.

Usage
    ATDRIVE_US_EXTRA=/path/a.npz[,/path/b.npz] python score_one.py
    python score_one.py /path/a.npz [/path/b.npz ...]        (equivalent)

Each npz must carry 'stats' (n_routes x d) and 'names' (n_routes, 'route_<id>'
or '<id>'), i.e. the format atdrive.b2d.load_features reads.  Every file given
is appended to run_us.load_descriptor_arms() as one more descriptor arm and the
whole shipped Table 3A is recomputed -- same unified split, same 16 draws, same
two-stage Ridge plug-in (alpha 100 for d > 10 else 10, train-fold z-scoring),
same pooled AUROC / scene-MAE / Spearman rho -- so each new row prints next to
the shipped rows and the shipped anchors still assert.

Results go to a scratch directory (ATDRIVE_RESULTS_DIR) so the repo's
results/us.json is never touched.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments'))
os.environ.setdefault('ATDRIVE_RESULTS_DIR',
                      '/data2/jeongtae/official_baselines/us_features/_score_one_results')

import numpy as np                                                    # noqa: E402
import run_us                                                         # noqa: E402


def load_npz(path):
    """One descriptor npz -> {route_id: float64 vector} (route_ prefix stripped)."""
    d = np.load(path, allow_pickle=True)
    names = [str(x).replace('route_', '') for x in d['names']]
    return {names[i]: d['stats'][i].astype(np.float64) for i in range(len(names))}


def main():
    paths = [p for p in sys.argv[1:] if p.strip()]
    paths += [p for p in os.environ.get('ATDRIVE_US_EXTRA', '').split(',') if p.strip()]
    if not paths:
        sys.exit('give at least one npz on argv or in ATDRIVE_US_EXTRA')

    extra = {}
    for p in paths:
        feat = load_npz(p)
        d = len(next(iter(feat.values())))
        extra[f'EXTRA {Path(p).stem} ({d}d)'] = feat
        print(f'extra arm: {Path(p).stem}  {len(feat)} routes x {d}d  <- {p}', flush=True)

    base = run_us.load_descriptor_arms

    def patched():
        arms = base()
        arms.update(extra)
        return arms

    run_us.load_descriptor_arms = patched
    run_us.main()


if __name__ == '__main__':
    main()
