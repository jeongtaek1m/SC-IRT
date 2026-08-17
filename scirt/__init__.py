"""SC-IRT: scene-conditioned item response theory for driving-policy evaluation.

Treat a driving scenario as a test item and a planner as an examinee. Fitting an
IRT model to the pass/fail panel recovers a per-scenario difficulty; learning to
*predict* that difficulty from the scene alone makes it available for scenarios
no planner has driven, which is what turns a fixed benchmark into an adaptive
one.

Layout::

    runtime    pinned device, threads, dtype, seeds
    paths      repository-root resolution
    data       response panel, route types, evaluation bank
    features   scene descriptors and the reported registry
    irt        MAP calibration kernel and identification policies
    theta      ability estimation for adaptive testing
    posterior  Bayesian ability posterior for the calibrated-bank regime

Experiment entry points live in `experiments/`, one per reported table.
"""

from . import (  # noqa: F401
    data,
    features,
    irt,
    paths,
    posterior,
    runtime,
    theta,
)

__version__ = "1.0.0"

from .api import (  # noqa: F401,E402
    calibrated_bank,
    encoder_predictions,
    estimate_planner,
    evaluate,
    next_route,
    noise_ceiling,
    reference,
)
