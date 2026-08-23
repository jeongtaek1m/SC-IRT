"""SC-IRT: scene-conditioned item response theory for driving-policy evaluation.

Treat a driving scenario as a test item and a planner as an examinee. Fitting an
IRT model to the pass/fail panel recovers a per-scenario difficulty; learning to
*predict* that difficulty from the scene alone makes it available for scenarios
no planner has driven, which is what turns a fixed benchmark into an adaptive
one.

Unified protocol (the paper's tables) — one generative model, one split::

    theta_j ~ N(0,1);  b_i | x_i ~ N(b_tilde(x_i), sigma^2);  P = sigmoid(theta - b)

    splits        13/3 planners x 36/8 scene types, R = 16 draws (the split)
    b2d           data loading (repo-local, canonical route order)
    calibration   MAP item-bank calibration (1PL main; 2PL/3PL for baselines)
    curves        theta grid + difficulty-marginalised item curves
    bayes         grid posterior, posterior-predictive SR, credible intervals
    acquisition   SR-variance rule (ours), theta-EIG, Fisher/ATLAS/static rules
    lltm          one-stage LLTM+e descriptor estimator (Table 1 canonical)
    metrics       pooled metrics + the paper's resampling conventions

Experiment entry points live in `experiments/`, one per reported table; each
asserts the published numbers at the end of its run.

Legacy layer (the pre-unified 220-route LOPO snapshot, kept for the paper's
robustness appendix): `api`, `data`, `features`, `irt`, `theta`, `posterior`.
"""

from . import (  # noqa: F401
    acquisition,
    b2d,
    bayes,
    calibration,
    curves,
    data,
    features,
    irt,
    lltm,
    metrics,
    paths,
    posterior,
    runtime,
    splits,
    theta,
)

__version__ = "2.0.0"

from .api import (  # noqa: F401,E402
    calibrated_bank,
    encoder_predictions,
    estimate_planner,
    evaluate,
    next_route,
    noise_ceiling,
    reference,
)
