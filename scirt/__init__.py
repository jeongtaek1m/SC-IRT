"""SC-IRT: scene-conditioned item response theory for driving-policy evaluation.

Treat a driving scenario as a test item and a planner as an examinee. One
uncertainty-aware Rasch posterior does everything:

    evaluation model   y_sk ~ Bernoulli(sigmoid(theta_k - b_s)),  b_s | A ~ N(b_hat_s, s_s^2)
    acquisition        Delta-R1: expected drop in the posterior L1 risk of SR_hat
    stopping           R1(D_t) <= tau_hat, fixed on the calibration panel
    unseen scenes      b_s | x_s ~ N(w^T x_s, sigma^2)  (LLTM+e; RelGraph ships as predictions)

    splits        random 16/6 planners x 36/8 scene types, R = 16 draws
    b2d           data loading (repo-local, canonical route order)
    calibration   MAP calibration of the 1PL evaluation model (2PL for baselines only)
    curves        theta grid + difficulty-marginalised item curves
    bayes         grid posterior, MAP-fill readout, posterior L1 risk
    acquisition   r1_pick (Delta-R1, UP), eig_pick (theta-EIG, UPS probe)
    baselines     published selectors and their native readouts
    lltm          one-stage LLTM+e descriptor estimator (US canonical)
    us_eval       common US scoring for descriptors, LLTM+e and encoder predictions
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
    us_eval,
)

__version__ = "4.0.0"
