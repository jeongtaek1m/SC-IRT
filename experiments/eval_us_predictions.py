#!/usr/bin/env python3
"""Score any unseen-scene difficulty predictions in Table-3A units.

    python experiments/eval_us_predictions.py data/encoder/relgraph_r2_s0.npz [more.npz ...]

Each npz holds draw{r}_rt (route ids of block C) and draw{r}_bt (b_tilde).
With several files (e.g. three seeds) the per-file metrics are summarised as
mean +- SD — single runs only; no prediction averaging.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atdrive.us_eval import USEvaluator, load_pred_npz


def main(paths):
    ev = USEvaluator()
    rows = []
    for pth in paths:
        m = ev.evaluate(load_pred_npz(pth))
        rows.append(m)
        print(f'{Path(pth).name:34s} AUROC {m["auroc"]:.3f} ({m["d_auroc"]:+.3f} vs null {m["null"]["auroc"]:.3f})  '
              f'MAE {m["mae"]:.3f} ({m["rel_mae"]:+.1%})  rho {m["rho"]:+.3f}  '
              f'[{m["n_routes"]} routes{", skipped " + str(m["skipped"]) if m["skipped"] else ""}]')
    if len(rows) > 1:
        for k in ('auroc', 'mae', 'rho'):
            v = np.array([r[k] for r in rows])
            print(f'  {k:5s} {v.mean():.3f} +- {v.std(ddof=1):.3f}  (n={len(v)} runs)')


if __name__ == '__main__':
    main(sys.argv[1:] or sorted(str(p) for p in (Path(__file__).resolve().parents[1] / 'data/encoder').glob('relgraph_r2_s[0-9].npz')))
