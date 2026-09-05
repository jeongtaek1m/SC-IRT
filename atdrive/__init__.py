"""ATDrive: IRT-based adaptive testing for closed-loop driving evaluation.

Treat a driving scenario as a test item and a planner as an examinee. One
uncertainty-aware Rasch posterior does everything:

    evaluation model   y_sk ~ Bernoulli(sigmoid(theta_k - b_s + u_kg)),  exact grid posterior of b_s | A,  u_kg ~ N(0, sigma_g^2)
    acquisition        Delta-R1: expected drop in the posterior L1 risk of SR_hat
    stopping           c * R1(D_t) <= eps, c fixed on the calibration panel
    unseen scenes      b_s | scene ~ N(b_tilde_s, sigma^2)  (RelGraph R2, shipped as per-run predictions)

    splits        random 12/4 planners x 36/8 scene types, R = 16 draws
    b2d           data loading (repo-local, canonical route order)
    calibration   explicit-prior MAP (EB sigma_b), exact difficulty posteriors, testlet SD
    curves        grids + exact difficulty-marginalised item curves
    bayes         (theta, u) posterior, posterior-median readout, posterior L1 risk
    acquisition   r1_pick (Delta-R1, UP), r1_pick_transfer (Delta-R1 on D, UPS probe), eig_pick (theta-EIG, ablation)
    baselines     published selectors and their native readouts
    us_eval       common US scoring for descriptor baselines and encoder predictions
    metrics       pooled metrics + the paper's resampling conventions
    live          the evaluator to embed in a real closed-loop run (bank, next route, observe, stop)

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
    live,
    metrics,
    splits,
    us_eval,
)

__version__ = "7.0.0"
