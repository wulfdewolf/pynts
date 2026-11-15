"""Discounted future spatial features for Poisson GLMs."""

import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike

from pynts.glms.util import get_basis
from pynts.util import wrap_list


def discounted_future_features(
    features: np.ndarray,
    increments: np.ndarray,
    horizon: float,
) -> np.ndarray:
    """Average future features, holding the final feature beyond the recording."""
    features = np.asarray(features, dtype=float)
    increments = np.asarray(increments, dtype=float)

    if features.ndim != 2 or not len(features):
        raise ValueError("features must be a nonempty two-dimensional array.")
    if increments.shape != (len(features) - 1,):
        raise ValueError("Provide one increment per adjacent sample.")
    if np.any(~np.isfinite(increments)) or np.any(increments < 0):
        raise ValueError("increments must be finite and non-negative.")
    if not np.isfinite(horizon) or horizon < 0:
        raise ValueError("horizon must be finite and non-negative.")

    if horizon == 0:
        return features.copy()

    discounts = np.exp(-increments / horizon)
    weights = -np.expm1(-increments / horizon)

    result = np.empty_like(features)
    result[-1] = features[-1]

    for index in range(len(features) - 2, -1, -1):
        result[index] = (
            weights[index] * features[index] + discounts[index] * result[index + 1]
        )

    return result


def make_future_design_matrix(
    session: dict,
    bounds: ArrayLike,
    horizon: float,
    discount_by: str = "distance",
) -> nap.TsdFrame:
    """Discount spatial B-splines within each continuous recording interval."""
    if not np.isfinite(horizon) or horizon < 0:
        raise ValueError("horizon must be finite and non-negative.")
    if discount_by not in {"time", "distance"}:
        raise ValueError("discount_by must be 'time' or 'distance'.")

    bounds = np.asarray(bounds, dtype=float)
    if (
        bounds.shape != (2, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 1] <= bounds[:, 0])
    ):
        raise ValueError("bounds must contain increasing x and y limits.")

    x = session["P_x"]
    y = session["P_y"]
    times = np.asarray(x.times(), dtype=float)

    if not np.array_equal(times, y.times()):
        raise ValueError("P_x and P_y must have matching timestamps.")

    positions = np.column_stack((x.values, y.values)).astype(float)
    if not len(positions) or not np.isfinite(positions).all():
        raise ValueError("Positions must be nonempty and finite.")

    support = x.time_support.intersect(y.time_support)

    basis, _ = get_basis(
        ["P_x", "P_y"],
        bounds=bounds,
        ranges=bounds[:, 1] - bounds[:, 0],
    )
    spatial_features = np.asarray(
        basis.fit_transform(positions),
        dtype=float,
    )
    if not np.isfinite(spatial_features).all():
        raise ValueError("Spatial features are nonfinite; check position bounds.")

    features = np.full_like(spatial_features, np.nan)

    for start, end in zip(support.start, support.end):
        indices = np.flatnonzero((times >= start) & (times <= end))
        if not indices.size:
            continue

        if discount_by == "time":
            increments = np.diff(times[indices])
        else:
            increments = np.linalg.norm(
                np.diff(positions[indices], axis=0),
                axis=1,
            )

        features[indices] = discounted_future_features(
            spatial_features[indices],
            increments,
            horizon,
        )

    return nap.TsdFrame(
        t=times,
        d=features,
        columns=[f"future_{index}" for index in range(features.shape[1])],
        time_support=support,
    )
