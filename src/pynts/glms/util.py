import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pynapple as nap
from nemos.basis import BSplineEval, CyclicBSplineEval
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, TransformerMixin

from pynts.smoothing import gaussian_filter_nan


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


def plot_grid_fit(cluster, session, bin_size_sec, model):
    tc = nap.compute_tuning_curves(
        cluster, np.stack([session["P_x"], session["P_y"]], axis=1), bins=40
    )
    tc = gaussian_filter_nan(tc, (2, 2), keep=False, mode="fill")

    x_centers = tc.coords["0"].values
    y_centers = tc.coords["1"].values

    xx, yy = np.meshgrid(x_centers, y_centers)  # shape (len(y), len(x))
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])

    pred_rate = model.predict(grid_coords) / bin_size_sec
    pred_grid = pred_rate.reshape(len(y_centers), len(x_centers))

    extent = (x_centers.min(), x_centers.max(), y_centers.min(), y_centers.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(
        tc.values[0].T,
        origin="lower",
        extent=extent,
        cmap="viridis",
        aspect="auto",
    )
    axes[0].set_title("Empirical tuning curve")
    fig.colorbar(im0, ax=axes[0], label="rate (Hz)")

    im1 = axes[1].imshow(
        pred_grid,
        origin="lower",
        extent=extent,
        cmap="viridis",
        aspect="auto",
    )
    axes[1].set_title("GLM-predicted tuning")
    fig.colorbar(im1, ax=axes[1], label="predicted rate (Hz)")

    for ax in axes:
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")

    plt.show()


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
            "spacing": np.arange(0.2 * range, 0.6 * range, 1),
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


def wilcoxon_nan(a, b, alternative="greater", zero_method="zsplit", min_pairs=3):
    a, b = np.array(a), np.array(b)
    valid = ~np.isnan(a) & ~np.isnan(b)
    if valid.sum() < min_pairs:
        return np.nan
    return wilcoxon(
        a[valid], b[valid], alternative=alternative, zero_method=zero_method
    )[1]


FANCY_LABELS = {"S": "S", "H": "H", "T": "T", ("P_x", "P_y"): "P", "P": "P"}
