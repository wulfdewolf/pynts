from typing import Self

import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pynapple as nap
import seaborn as sns
from nemos.basis import BSplineEval, CyclicBSplineEval
from numba import njit
from numpy.typing import ArrayLike
from scipy.ndimage import label, maximum_filter
from scipy.optimize import minimize
from scipy.special import logsumexp, xlogy
from scipy.stats import uniform, wilcoxon
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from pynts.smoothing import gaussian_filter_nan
from pynts.util import interpolate, wrap_list


@njit(cache=True)
def _log_gaussian_lattice(
    coordinates: np.ndarray,
    width_ratio: float,
    radius: int,
) -> np.ndarray:
    """Evaluate Gaussian lattice sums without intermediate arrays."""
    output = np.empty((len(coordinates), 1))
    factor = 0.5 / width_ratio**2

    for index in range(len(coordinates)):
        u, v = coordinates[index]
        minimum_distance = np.inf

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                du = u - i
                dv = v - j
                distance = du * du + du * dv + dv * dv
                minimum_distance = min(minimum_distance, distance)

        total = 0.0

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                du = u - i
                dv = v - j
                distance = du * du + du * dv + dv * dv
                total += np.exp(-factor * (distance - minimum_distance))

        output[index, 0] = -factor * minimum_distance + np.log(total)

    return output


class GaussianGridBasis(BaseEstimator, TransformerMixin):
    """A periodic Gaussian lattice with shared field amplitudes."""

    def __init__(
        self,
        field_spacing: float = 40.0,
        orientation: float = 0.0,
        width_ratio: float = 0.15,
        phase_u: float = 0.0,
        phase_v: float = 0.0,
    ):
        self.field_spacing = field_spacing
        self.orientation = orientation
        self.width_ratio = width_ratio
        self.phase_u = phase_u
        self.phase_v = phase_v

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "GaussianGridBasis":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        theta = self.orientation
        lattice = self.field_spacing * np.array(
            [
                [np.cos(theta), np.sin(theta)],
                [np.cos(theta + np.pi / 3), np.sin(theta + np.pi / 3)],
            ]
        )

        coordinates = X[:, :2] @ np.linalg.inv(lattice)
        coordinates -= np.array([self.phase_u, self.phase_v])
        coordinates -= np.floor(coordinates + 0.5)

        radius = int(np.ceil(8 * self.width_ratio)) + 1

        return _log_gaussian_lattice(
            np.ascontiguousarray(coordinates),
            self.width_ratio,
            radius,
        )

    @property
    def n_features_out_(self) -> int:
        return 1


class GaussianGridPoissonRegressor(RegressorMixin, BaseEstimator):
    """Fit nonnegative background and amplitude for fixed log tuning."""

    def __init__(self, alpha: float = 0.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        tuning = np.exp(np.asarray(X, dtype=float)[:, 0])
        y = np.asarray(y, dtype=float)
        mean_tuning = float(tuning.mean())

        def objective(
            parameters: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            background, amplitude = parameters
            prediction = background + amplitude * tuning
            residual = 1.0 - y / prediction

            loss = (
                np.mean(prediction - xlogy(y, prediction))
                + 0.5 * self.alpha * amplitude**2
            )
            gradient = np.array(
                [
                    residual.mean(),
                    np.dot(residual, tuning) / y.size + self.alpha * amplitude,
                ]
            )
            return float(loss), gradient

        mean_count = max(float(y.mean()), 1e-10)
        initial = np.array(
            [
                mean_count / 2,
                mean_count / (2 * max(mean_tuning, 1e-10)),
            ]
        )

        result = minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            bounds=[(1e-12, None), (0.0, None)],
            options={"ftol": 1e-9, "gtol": 1e-6},
        )

        self.background_, self.amplitude_ = result.x
        self.optimization_result_ = result
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        tuning = np.exp(np.asarray(X, dtype=float)[:, 0])
        return self.background_ + self.amplitude_ * tuning


class GridBasis(BaseEstimator, TransformerMixin):
    def __init__(self, field_spacing=40.0, orientation=0.0):
        self.field_spacing = field_spacing
        self.orientation = orientation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)

        x = X[:, 0]
        y = X[:, 1]

        k = 2 * np.pi / self.field_spacing

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


def plot_glm_fit(axs, tc, session, bin_size_sec, model):
    x_centers = tc.coords["0"].values
    y_centers = tc.coords["1"].values

    xx, yy = np.meshgrid(x_centers, y_centers)  # shape (len(y), len(x))
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])

    pred_rate = model.predict(grid_coords) / bin_size_sec
    pred_grid = pred_rate.reshape(len(y_centers), len(x_centers))

    extent = (x_centers.min(), x_centers.max(), y_centers.min(), y_centers.max())

    im0 = axs[0].imshow(
        tc.values[0].T,
        origin="lower",
        extent=extent,
        cmap="turbo",
        aspect="auto",
    )
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    for spine in axs[1].spines.values():
        spine.set_linewidth(2)

    im1 = axs[1].imshow(
        pred_grid,
        origin="lower",
        extent=extent,
        cmap="turbo",
        aspect="auto",
    )
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    for spine in axs[1].spines.values():
        spine.set_color("#B2BEB5")
        spine.set_linewidth(2)


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
    ranges = [b[1] - b[0] for b in bounds]

    if var == ("P_x", "P_y") or var == ["P_x", "P_y"]:
        basis = (
            BSplineEval(
                n_basis_funcs=int(np.ceil(ranges[0] / 5)),
                label="P_x",
                bounds=bounds[0],
            )
            * BSplineEval(
                n_basis_funcs=int(np.ceil(ranges[1] / 5)),
                label="P_y",
                bounds=bounds[1],
            )
        ).to_transformer()
        hyperparams = {"glm__alpha": [0.0001, 0.001, 0.01, 0.1]}
    elif var == "P":
        basis = CyclicBSplineEval(
            n_basis_funcs=int(np.ceil(ranges[0] / 5)), label="P", bounds=bounds[0]
        ).to_transformer()
        hyperparams = {}
    elif var == "S":
        basis = BSplineEval(
            n_basis_funcs=int(np.ceil(ranges[0] / 2)), label="S", bounds=bounds[0]
        ).to_transformer()
        hyperparams = {}
    elif var == "H":
        basis = CyclicBSplineEval(
            n_basis_funcs=60, label="H", bounds=bounds[0]
        ).to_transformer()
        hyperparams = {}
    elif var == "T":
        basis = CyclicBSplineEval(
            n_basis_funcs=60, label="T", bounds=bounds[0]
        ).to_transformer()
        hyperparams = {}
    elif var == "grid_gauss":
        basis = GaussianGridBasis()
        hyperparams = {
            "basis__field_spacing": np.arange(0.2 * max(ranges), 0.8 * max(ranges), 5),
            "basis__orientation": np.linspace(0.0, np.pi / 3, 12, endpoint=False),
            "basis__width_ratio": [0.10, 0.15, 0.2, 0.25],
            "basis__phase_u": np.arange(4) / 4,
            "basis__phase_v": np.arange(4) / 4,
        }
    elif var == "grid":
        basis = GridBasis()
        hyperparams = {
            "field_spacing": np.arange(0.2 * max(ranges), 0.7 * max(ranges), 2),
            "orientation": np.linspace(0.0, np.pi / 3, 15, endpoint=False),
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


def compute_com(
    model,
    bounds,
    resolution_cm: float = 2.0,
    thresh: float = 0.5,
) -> tuple[float, float]:
    bounds = np.asarray(bounds, dtype=float)
    xs = np.arange(bounds[0, 0], bounds[0, 1] + resolution_cm, resolution_cm)
    ys = np.arange(bounds[1, 0], bounds[1, 1] + resolution_cm, resolution_cm)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    rate = model.predict(positions).reshape(xx.shape)
    rate = np.nan_to_num(rate, nan=0.0)
    mask = rate >= thresh * rate.max()
    if not mask.any():
        return np.nan, np.nan
    r_masked = rate * mask
    total = r_masked.sum()
    if total == 0:
        return np.nan, np.nan
    com_x = float((xx * r_masked).sum() / total)
    com_y = float((yy * r_masked).sum() / total)
    return com_x, com_y


def count_fields(
    model,
    bounds: np.ndarray,
    resolution_cm: float = 2.0,
    thresh: float = 0.5,
):
    """
    Count firing fields – and optionally return their mean area – in the
    model-predicted rate map.
    Parameters
    ----------
    model : sklearn.Pipeline
        Fitted GLM pipeline (must implement ``predict`` on positions [N, 2]).
    bounds : array-like, shape (2, 2)
        [[xmin, xmax], [ymin, ymax]] (same units as the model inputs).
    resolution_cm : float, default 2.0
        Spatial sampling resolution (cm).
    return_mean_size : bool, default False
        If True, also return the mean field area (cm²).
    thresh : float, default 0.5
        Rate threshold expressed as a fraction of the peak rate that defines
        the field mask (0.5 ≈ half-height).
    Returns
    -------
    int
        Number of detected fields.
    float, optional
        Mean field area (cm²). Returned only when ``return_mean_size`` is True.
    """
    bounds = np.asarray(bounds, dtype=float)
    # Regular grid covering the arena
    xs = np.arange(bounds[0, 0], bounds[0, 1] + resolution_cm, resolution_cm)
    ys = np.arange(bounds[1, 0], bounds[1, 1] + resolution_cm, resolution_cm)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    # Predicted rate map
    rate = model.predict(positions).reshape(xx.shape)
    # Locate local maxima (field centres)
    peaks = rate == maximum_filter(rate, size=3)
    peaks &= rate > 0.9 * rate.max()
    # Supra-threshold mask that delimits fields
    field_mask = rate >= thresh * rate.max()
    field_labels, _ = label(field_mask)
    # Keep only regions that contain a peak
    peak_labels = np.unique(field_labels[peaks])
    peak_labels = peak_labels[peak_labels != 0]  # remove background
    n_fields = len(peak_labels)
    pixel_area = resolution_cm**2  # cm² per pixel
    areas = np.array(
        [(field_labels == lab).sum() * pixel_area for lab in peak_labels],
        dtype=float,
    )
    diameters = 2 * np.sqrt(areas / np.pi)
    mean_diameter = float(diameters.mean()) if diameters.size else np.nan
    return n_fields, mean_diameter
