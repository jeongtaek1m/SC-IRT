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
    stats      summary statistics

Experiment entry points live in `experiments/`, one per reported table.
"""

from . import (  # noqa: F401
    data,
    features,
    irt,
    paths,
    runtime,
    stats,
    theta,
)

__version__ = "1.0.0"
