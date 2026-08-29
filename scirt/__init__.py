"""SC-IRT: scene-conditioned item response theory for driving-policy evaluation.

Treat a driving scenario as a test item and a planner as an examinee.

    evaluation model   y_ij ~ Bernoulli(sigmoid(theta_j - b_i)),  b_i | A ~ N(b_hat_i, s_i^2)
    acquisition model  a joint 2PL fit of the same panel, used only to choose scenes
    unseen scenes      b_i | x_i ~ N(w^T x_i, sigma^2)  (LLTM+e) or a trajectory encoder

    splits        13/3 planners x 36/8 scene types, R = 16 draws
    b2d           data loading (repo-local, canonical route order)
    calibration   MAP calibration of the evaluation (1PL) and acquisition (2PL) models
    curves        theta grid + difficulty-marginalised item curves
    bayes         grid posterior, MAP-fill readout, posterior L1 risk (stopping)
    acquisition   localize -> cover (UP), theta-EIG (UPS)
    baselines     published selectors and their native readouts
    lltm          one-stage LLTM+e descriptor estimator (US canonical)
    encoder       the trajectory encoder (US learned representation)
    metrics       pooled metrics + the paper's resampling conventions

Experiment entry points live in `experiments/`, one per reported table; the
table-producing ones assert the published numbers at the end of their run.
"""

from . import (  # noqa: F401
    acquisition,
    b2d,
    baselines,
    bayes,
    calibration,
    curves,
    lltm,
    metrics,
    splits,
)

__version__ = "3.0.0"
