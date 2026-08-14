"""Stage-2 explanatory regression: scene features -> item parameters.

The two-stage explanatory IRT baseline calibrates difficulty from responses
(Stage 1, `scirt.irt`) and then regresses those difficulties on scene features
(Stage 2, here). At test time only the features are available, so the regression
supplies the difficulty for a route no planner has driven.

Standardisation statistics come from the training fold only. Computing them over
all routes would leak held-out information into every fold.
"""

import numpy as np
from sklearn.linear_model import Ridge


def zscore_stats(Z):
    """Per-column mean and standard deviation, with an epsilon for constant columns."""
    return Z.mean(0), Z.std(0) + 1e-9


def _design(feat, routes):
    return np.vstack([feat[r] for r in routes])


class RidgeB:
    """A fitted difficulty regression, together with its standardisation statistics.

    Kept as an object rather than a function so that predictions and the residual
    variance used by the shrunk-information selector both come from the *same*
    fit, exactly as in the original code.
    """

    def __init__(self, rg, mean, std):
        self._rg, self._mean, self._std = rg, mean, std

    def _z(self, x):
        return (x - self._mean) / self._std

    def predict(self, feat, routes, *, predict):
        """Predict difficulty for `routes`.

        Args:
            predict: 'per_row' issues one `predict` call per route; 'batch'
                issues a single call for all of them. Required, and not a style
                choice: the two differ by ~1e-7 through BLAS batching, which is
                invisible to a rank correlation but not to the stored artifacts,
                and the CAT strategies that sort by difficulty amplify it into
                different item trajectories.
        """
        if predict == "per_row":
            return {r: float(self._rg.predict(self._z(feat[r])[None])[0]) for r in routes}
        if predict == "batch":
            pred = self._rg.predict(self._z(_design(feat, routes)))
            return {r: float(pred[i]) for i, r in enumerate(routes)}
        raise ValueError(f"predict must be 'per_row' or 'batch', got {predict!r}")

    def residual_var(self, feat, routes, targets):
        """In-fold residual variance, the `s^2` of `selection.shrunk_information`.

        Estimated on the training fold only, which is what keeps the selection
        rule deployable: nothing about the held-out routes enters it.
        """
        resid = np.array(
            [
                float(self._rg.predict(self._z(feat[r])[None])[0]) - t
                for r, t in zip(routes, targets)
            ]
        )
        return float(resid.var())


def fit_ridge_b(feat, train_routes, targets, *, alpha):
    """Fit the feature -> difficulty ridge on the training fold."""
    Z = _design(feat, train_routes)
    m, s = zscore_stats(Z)
    return RidgeB(Ridge(alpha=alpha).fit((Z - m) / s, targets), m, s)


def ridge_b(feat, train_routes, targets, predict_routes, *, alpha, predict):
    """Fit and predict in one call, for sites that need no residual variance."""
    return fit_ridge_b(feat, train_routes, targets, alpha=alpha).predict(
        feat, predict_routes, predict=predict
    )


def ridge_explanatory_2pl(feat, train_routes, b_hat, a_hat, predict_routes, *, alpha=None):
    """Two ridges, giving predicted difficulty and predicted log discrimination.

    The full explanatory-2PL baseline: a cell probability can then be formed for
    a route with no responses at all, as sigmoid(a_tilde * (theta - b_tilde)).

    `alpha` defaults to the dimension rule the original used — 100 above ten
    features, 10 at or below. Every live descriptor is at least 12-dimensional,
    so the branch is currently inert, but a low-dimensional descriptor would
    otherwise be silently over-regularised.
    """
    dim = len(next(iter(feat.values())))
    if alpha is None:
        alpha = 100.0 if dim > 10 else 10.0

    Z = _design(feat, train_routes)
    m, s = zscore_stats(Z)
    Zs = (Z - m) / s

    rb = Ridge(alpha=alpha).fit(Zs, [b_hat[r] for r in train_routes])
    ra = Ridge(alpha=alpha).fit(
        Zs, [np.log(max(a_hat.get(r, 1.0), 1e-3)) for r in train_routes]
    )

    Za = (_design(feat, predict_routes) - m) / s
    b_pred, a_pred = rb.predict(Za), np.exp(ra.predict(Za))
    return (
        {r: float(b_pred[i]) for i, r in enumerate(predict_routes)},
        {r: float(a_pred[i]) for i, r in enumerate(predict_routes)},
    )
