"""Fold-internal Rasch calibration that freezes theta for encoder training —
the SC-IRT v5 calibration (explicit priors theta ~ N(0,1), b ~ N(0, sigma_b^2)
with sigma_b by empirical Bayes). Replaces the removed scirt.encoder.rasch.
Y: (J planners, R routes) with fail = 1 and nan for missing cells.
Returns (theta, b), both theta-mean centred (P(pass) = sigmoid(theta - b))."""
import sys
import numpy as np
sys.path.insert(0, '/home/jeongtae/SC-IRT')


def rasch(Y, it=800, seed=0):
    from atdrive.calibration import calibrate_dense      # the repo package (scirt -> driveat -> atdrive)
    Yp = 1.0 - Y
    MK = ~np.isnan(Yp)
    Y0 = np.nan_to_num(Yp)
    J, R = Y.shape
    b, th, _ = calibrate_dense(Y0.T, MK.T, list(range(R)), list(range(J)), it=it)
    return th, b
