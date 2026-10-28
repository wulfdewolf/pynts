import numpy as np
import pandas as pd
import pynapple as nap
from numpy.typing import ArrayLike
from scipy.signal import lfilter


def compute_future_displacement(
    session: dict,
    session_type: str,
    clusters: nap.TsGroup,
    horizon_range: ArrayLike = np.arange(0, 21, 1),
    *,
    moving_threshold_cm: float = 0.025,
) -> pd.DataFrame:
    """Measure discounted future displacement along the current heading."""
    del clusters, session_type

    positions = pd.concat(
        [
            session["P_x"].as_series().rename("x"),
            session["P_y"].as_series().rename("y"),
        ],
        axis=1,
    ).dropna()

    times = positions.index.to_numpy(dtype=float)
    xy = positions.to_numpy(dtype=float)
    dt = float(np.median(np.diff(times)))

    steps = np.diff(xy, axis=0, prepend=xy[:1])
    distances = np.linalg.norm(steps, axis=1)
    moving = distances > moving_threshold_cm

    headings = np.zeros_like(steps)
    headings[moving] = steps[moving] / distances[moving, None]

    horizons = np.asarray(horizon_range, dtype=float)
    if np.any(~np.isfinite(horizons)) or np.any(horizons < 0):
        raise ValueError("Horizons must be finite and non-negative.")

    # Centre coordinates to reduce cancellation in the discounted sums.
    xy = xy - xy.mean(axis=0)
    results = []

    for horizon in horizons:
        if horizon == 0:
            score = 0.0 if moving.any() else np.nan
            n_samples = int(moving.sum())
        else:
            gamma = float(np.exp(-dt / horizon))

            # Future-only sums with weights 1, gamma, gamma**2, ...
            future_positions = lfilter(
                [1.0],
                [1.0, -gamma],
                xy[:0:-1],
                axis=0,
            )[::-1]
            future_weights = lfilter(
                [1.0],
                [1.0, -gamma],
                np.ones(len(xy) - 1),
            )[::-1]

            mean_displacement = future_positions / future_weights[:, None] - xy[:-1]
            forward_displacement = np.einsum(
                "ij,ij->i",
                mean_displacement,
                headings[:-1],
            )

            # Retain samples with at least five horizons of future data.
            valid = moving[:-1] & (times[-1] - times[:-1] >= 5 * horizon)
            n_samples = int(valid.sum())
            score = float(forward_displacement[valid].mean()) if n_samples else np.nan

        results.append(
            {
                "horizon": float(horizon),
                "forward_displacement": score,
                "n_samples": n_samples,
            }
        )

    return pd.DataFrame(results)
