#!/usr/bin/env python3
"""Appendix A3: how much of the gold difficulty is signal rather than panel noise.

The gold difficulty is itself an estimate from 16 planners, so no predictor —
however good — can correlate with it perfectly. This puts a ceiling on Table I's
rho column, and reporting rho without it invites reading 0.520 as "half way to
the truth" when the attainable maximum is 0.904.

Method: split the panel into two halves of eight planners, calibrate each half
independently, and correlate the two difficulty vectors. Spearman-Brown corrects
that half-panel reliability up to the full 16-planner panel, and the square root
of the reliability is the expected correlation of a perfect predictor.

Bank note: this experiment runs on the full 220-route collection, not the
219-route evaluation bank. Its filter is "at least eight observed responses",
which excludes nothing; the sibling experiments filter on feature availability
instead. Unifying the two would move the published ceiling.
"""

import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scirt import data, irt, runtime  # noqa: E402

runtime.configure()
runtime.set_global_seeds(0)

panel = data.read_response_panel()
J = panel.n_planners

Y = panel.dense_all()
Y = Y[[i for i in range(Y.shape[0]) if (~np.isnan(Y[i])).sum() >= 8]]


def calibrate_half(planner_cols, it=800):
    """Fit a 2PL on one half of the panel and return centred difficulty."""
    M = Y[:, planner_cols]
    fit = irt.fit_irt_map(M, ~np.isnan(M), model="2pl", it=it)
    return irt.center_b(fit)


rng = np.random.RandomState(0)
halves = []
for _ in range(20):
    pm = rng.permutation(J)
    a, b = list(pm[: J // 2]), list(pm[J // 2 :])
    halves.append(spearmanr(calibrate_half(a), calibrate_half(b)).correlation)

halves = np.array(halves)
r_half = halves.mean()
rel_full = 2 * r_half / (1 + r_half)  # Spearman-Brown, half panel -> full panel
ceiling = np.sqrt(rel_full)

print(
    f"split-half ρ (8 vs 8 planners, 20 splits): {r_half:.3f} ± {halves.std():.3f}  "
    f"[range {halves.min():.3f}~{halves.max():.3f}]"
)
print(f"Spearman-Brown reliability of the full 16-planner gold: {rel_full:.3f}")
print(f"→ expected ceiling ρ for a perfect predictor = √{rel_full:.3f} = {ceiling:.3f}")

# Context lines: the first three are transcribed from gold_anchor.py's output;
# the scenparamz-glob line comes from the hybrid_prereg.py baseline arm.
for name, rho in [
    ("cmdkin", 0.418),
    ("cmdkin+camrisk-full", 0.528),
    ("GT ck+gtrisk", 0.529),
    ("cmdkin+scenparamz-glob", 0.554),
]:
    print(
        f"  {name:24s} ρ=+{rho:.3f} → {100*rho/ceiling:.0f}% of ceiling   "
        f"(attenuation-corrected ρ={rho/ceiling:.3f})"
    )
