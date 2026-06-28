from astropy.visualization import time_support
from astropy.units.quantity_helper.function_helpers import interp
import nemos as nmo
import numpy as np
import pynapple as nap
from nemos.basis import BSplineEval, CyclicBSplineEval
from sklearn.base import BaseEstimator, TransformerMixin


def interpolate(var, y, other):
    if var == "H" or var == "T":
        return nap.TsdFrame(
            d=np.arctan2(
                np.sin(y).restrict(other.time_support).interpolate(other).values,
                np.cos(y).restrict(other.time_support).interpolate(other).values,
            ),
            t=other.times(),
            time_support=other.time_support,
        )
    else:
        return y.interpolate(other)


class GridBasis(BaseEstimator, TransformerMixin):
    def __init__(self, spacing=40.0, orientation=0.0):
        self.spacing = spacing
        self.orientation = orientation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)

        x = X[:, 0]
        y = X[:, 1]

        k = 2 * np.pi / self.spacing

        directions = [
            self.orientation,
            self.orientation + np.pi / 3,
            self.orientation + 2 * np.pi / 3,
        ]

        features = []

        for theta in directions:
            proj = x * np.cos(theta) + y * np.sin(theta)
            phase = k * proj

            features.append(np.cos(phase))
            features.append(np.sin(phase))

        return np.column_stack(features)

    @property
    def n_features_out_(self):
        return 6


def make_feature(v, x, bounds, y, epoch):
    interpolated = interpolate(v, x, y)
    if interpolated.ndim == 1:
        interpolated = nap.TsdFrame(
            d=interpolated.values,
            t=interpolated.times(),
            time_support=interpolated.time_support,
        )
    return interpolated.restrict(epoch).clip(
        *((None, None) if bounds is None else bounds)
    )


def get_basis(var, bounds):
    range = max(b[1] - b[0] for b in bounds)

    if var == ("P_x", "P_y"):
        basis = (
            BSplineEval(n_basis_funcs=10, label="P_x", bounds=bounds[0])
            * BSplineEval(n_basis_funcs=10, label="P_y", bounds=bounds[1])
        ).to_transformer()
        hyperparams = {
            "P_x__n_basis_funcs": np.arange(5, int(0.5 * range), 1),
            "P_y__n_basis_funcs": np.arange(5, int(0.5 * range), 1),
        }
    elif var == "P":
        basis = CyclicBSplineEval(
            n_basis_funcs=10, label="P", bounds=bounds[0]
        ).to_transformer()
        hyperparams = {
            "n_basis_funcs": np.arange(5, int(0.5 * range), 1),
        }
    elif var == "S":
        basis = BSplineEval(
            n_basis_funcs=10, label="S", bounds=bounds[0]
        ).to_transformer()
        hyperparams = {
            "n_basis_funcs": np.arange(5, int(0.5 * range), 1),
        }
    elif var == "H":
        basis = CyclicBSplineEval(
            n_basis_funcs=10, label="H", bounds=bounds[0]
        ).to_transformer()
        hyperparams = {
            "n_basis_funcs": np.arange(5, int(0.5 * np.degrees(range)), 1),
        }
    elif var == "T":
        basis = CyclicBSplineEval(
            n_basis_funcs=10, label="T", bounds=bounds[0]
        ).to_transformer()
        hyperparams = {
            "n_basis_funcs": np.arange(5, int(0.5 * np.degrees(range)), 1),
        }
    elif var == "grid":
        basis = GridBasis()
        hyperparams = {
            "spacing": np.arange(0.2 * range, 0.5 * range, 2),
            "orientation": np.linspace(
                0,
                np.pi / 3,
                30,
                endpoint=False,
            ),
        }
    else:
        raise ValueError(f"Unknown variable to fit GLM for {var}.")

    return basis, hyperparams


FANCY_LABELS = {"S": "S", "H": "H", "T": "T", ("P_x", "P_y"): "P", "P": "P"}
