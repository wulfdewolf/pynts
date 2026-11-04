from typing import Any, Self

import numpy as np
import pynapple as nap
from matplotlib.axes import Axes
from nemos.basis import BSplineEval, CyclicBSplineEval
from numpy.typing import ArrayLike
from scipy.ndimage import label, maximum_filter
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from pynts.glms.grid_glm import get_gaussian_grid_basis
from pynts.util import interpolate

FANCY_LABELS = {
    "S": "S",
    "H": "H",
    "T": "T",
    ("P_x", "P_y"): "P",
    "P": "P",
}


class GridBasis(BaseEstimator, TransformerMixin):
    """Sinusoidal features along three grid directions."""

    def __init__(
        self,
        field_spacing: float = 40.0,
        orientation: float = 0.0,
    ) -> None:
        self.field_spacing = field_spacing
        self.orientation = orientation

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> Self:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        k = 2 * np.pi / self.field_spacing
        features = []

        for offset in (0.0, np.pi / 3, 2 * np.pi / 3):
            theta = self.orientation + offset
            projection = X[:, 0] * np.cos(theta) + X[:, 1] * np.sin(theta)
            phase = k * projection
            features.extend([np.cos(phase), np.sin(phase)])

        return np.column_stack(features)

    @property
    def n_features_out_(self) -> int:
        return 6


def plot_glm_fit(
    axs: list[Axes] | np.ndarray,
    tc: Any,
    session: dict,
    bin_size_sec: float,
    model: Pipeline,
) -> None:
    """Plot observed and predicted spatial tuning."""
    x_centers = tc.coords["0"].values
    y_centers = tc.coords["1"].values

    xx, yy = np.meshgrid(x_centers, y_centers)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    prediction = model.predict(positions) / bin_size_sec
    predicted_rate = prediction.reshape(xx.shape)

    extent = (
        x_centers.min(),
        x_centers.max(),
        y_centers.min(),
        y_centers.max(),
    )

    for ax, rate in zip(axs, (tc.values[0].T, predicted_rate)):
        ax.imshow(
            rate,
            origin="lower",
            extent=extent,
            cmap="turbo",
            aspect="auto",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in axs[1].spines.values():
        spine.set_color("#B2BEB5")
        spine.set_linewidth(2)


def make_feature(
    v: str,
    x: nap.Tsd | nap.TsdFrame,
    bounds: ArrayLike | None,
    y: nap.Tsd | nap.TsdFrame,
    epoch: nap.IntervalSet,
) -> nap.TsdFrame:
    """Interpolate, restrict, and clip a behavioral feature."""
    interpolated = interpolate(v, x, y)

    if interpolated.ndim == 1:
        interpolated = nap.TsdFrame(
            d=interpolated.values,
            t=interpolated.times(),
            time_support=interpolated.time_support,
        )

    limits = (None, None) if bounds is None else bounds
    return interpolated.restrict(epoch).clip(*limits)


def get_basis(
    var: str | list[str] | tuple[str, ...],
    bounds: ArrayLike,
) -> tuple[BaseEstimator, dict[str, ArrayLike]]:
    """Create a basis and its pipeline parameter grid."""
    bounds = np.asarray(bounds, dtype=float)
    ranges = bounds[:, 1] - bounds[:, 0]

    if isinstance(var, (list, tuple)) and len(var) == 1:
        var = var[0]

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
        return basis, {"glm__alpha": [0.0001, 0.001, 0.01, 0.1]}

    if var == "P":
        basis = CyclicBSplineEval(
            n_basis_funcs=int(np.ceil(ranges[0] / 5)),
            label="P",
            bounds=bounds[0],
        ).to_transformer()
        return basis, {}

    if var == "S":
        basis = BSplineEval(
            n_basis_funcs=int(np.ceil(ranges[0] / 2)),
            label="S",
            bounds=bounds[0],
        ).to_transformer()
        return basis, {}

    if var in ("H", "T"):
        basis = CyclicBSplineEval(
            n_basis_funcs=60,
            label=var,
            bounds=bounds[0],
        ).to_transformer()
        return basis, {}

    if var == "grid_gauss":
        return get_gaussian_grid_basis(bounds)

    if var == "grid":
        hyperparams = {
            "basis__field_spacing": np.arange(
                0.2 * max(ranges),
                0.7 * max(ranges),
                2,
            ),
            "basis__orientation": np.linspace(
                0.0,
                np.pi / 3,
                15,
                endpoint=False,
            ),
        }
        return GridBasis(), hyperparams

    raise ValueError(f"Unknown variable to fit GLM for {var}.")


def wilcoxon_nan(
    a: ArrayLike,
    b: ArrayLike,
    alternative: str = "greater",
    zero_method: str = "zsplit",
    min_pairs: int = 3,
) -> float:
    """Run a paired Wilcoxon test after excluding NaN pairs."""
    a = np.asarray(a)
    b = np.asarray(b)
    valid = ~np.isnan(a) & ~np.isnan(b)

    if valid.sum() < min_pairs:
        return np.nan

    return float(
        wilcoxon(
            a[valid],
            b[valid],
            alternative=alternative,
            zero_method=zero_method,
        ).pvalue
    )


def _spatial_mesh(
    bounds: ArrayLike,
    resolution_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a regular spatial sampling grid."""
    bounds = np.asarray(bounds, dtype=float)
    xs = np.arange(bounds[0, 0], bounds[0, 1] + resolution_cm, resolution_cm)
    ys = np.arange(bounds[1, 0], bounds[1, 1] + resolution_cm, resolution_cm)
    return np.meshgrid(xs, ys, indexing="xy")


def compute_com(
    model: Pipeline,
    bounds: ArrayLike,
    resolution_cm: float = 2.0,
    thresh: float = 0.5,
) -> tuple[float, float]:
    """Compute the rate-weighted center above a relative threshold."""
    xx, yy = _spatial_mesh(bounds, resolution_cm)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    rate = model.predict(positions).reshape(xx.shape)
    rate = np.nan_to_num(rate, nan=0.0)

    weights = np.where(rate >= thresh * rate.max(), rate, 0.0)
    total = weights.sum()

    if total == 0:
        return np.nan, np.nan

    return (
        float((xx * weights).sum() / total),
        float((yy * weights).sum() / total),
    )


def count_fields_from_rate(
    rate: ArrayLike,
    bin_area: float,
    thresh: float = 0.5,
) -> tuple[int, float]:
    """Return field count and mean equivalent circular diameter."""
    rate = np.asarray(rate, dtype=float)
    valid = np.isfinite(rate)

    if not valid.any() or rate[valid].max() <= 0:
        return 0, np.nan

    rate = np.where(valid, rate, -np.inf)
    peak_rate = rate.max()

    peaks = rate == maximum_filter(rate, size=3)
    peaks &= rate > 0.9 * peak_rate

    field_labels, _ = label(valid & (rate >= thresh * peak_rate))
    peak_labels = np.unique(field_labels[peaks])
    peak_labels = peak_labels[peak_labels != 0]

    areas = np.array(
        [(field_labels == field_id).sum() * bin_area for field_id in peak_labels],
        dtype=float,
    )
    diameters = 2 * np.sqrt(areas / np.pi)
    mean_diameter = float(diameters.mean()) if diameters.size else np.nan

    return len(peak_labels), mean_diameter


def _as_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x.values if hasattr(x, "values") else x)
