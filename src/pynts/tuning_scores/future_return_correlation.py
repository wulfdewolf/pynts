from typing import Callable

import numpy as np
import pandas as pd
import pynapple as nap
from numpy.typing import ArrayLike
from tqdm import tqdm


def compute_future_return_correlation(
    session: dict,
    session_type: str,
    clusters: nap.TsGroup,
    horizon_range: ArrayLike = np.arange(0, 21, 1),
    is_shuffle: bool = False,
    *,
    n_folds: int = 2,
    n_spatial_bins: int | None = None,
    directional_states: bool = True,
    n_heading_bins: int = 4,
    moving_threshold_cm: float = 0.025,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Compute blocked-CV future-return correlations during movement."""
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
    moving = np.linalg.norm(steps, axis=1) > moving_threshold_cm

    states, n_states = _discretize_positions(
        xy[:, 0],
        xy[:, 1],
        n_spatial_bins,
        directional_states=directional_states,
        n_heading_bins=n_heading_bins,
    )

    horizons = np.asarray(horizon_range, dtype=float)
    if np.any(~np.isfinite(horizons)) or np.any(horizons < 0):
        raise ValueError("Horizons must be finite and non-negative.")

    identity = np.eye(n_states)
    folds = np.array_split(np.arange(len(states)), n_folds)
    fold_data = []

    for fold_index, test_indices in enumerate(folds):
        if len(test_indices) < 2:
            continue

        train_mask = moving.copy()
        train_mask[test_indices] = False

        transition_matrix, observed_states = _estimate_transition_matrix(
            states,
            train_mask,
            n_states,
        )

        test_states = states[test_indices]
        test_moving = moving[test_indices]

        # A usable target requires an adjacent moving sample.
        eligible = np.zeros(len(test_indices), dtype=bool)
        eligible[:-1] = test_moving[:-1] & test_moving[1:]
        valid = eligible & observed_states[test_states]

        fold_data.append(
            (
                fold_index,
                test_states,
                test_moving,
                transition_matrix,
                eligible,
                valid,
            )
        )

    rng = np.random.default_rng(random_state)
    results = []

    for horizon in tqdm(horizons, unit="horizon"):
        gamma = 0.0 if horizon == 0 else float(np.exp(-dt / horizon))

        for (
            fold_index,
            test_states,
            test_moving,
            transition_matrix,
            eligible,
            valid,
        ) in fold_data:
            targets = _compute_state_future_returns(
                test_states,
                test_moving,
                n_states,
                gamma,
            )[valid]

            if gamma == 0 or not valid.any():
                correlation = np.nan
            else:
                unique_states, inverse = np.unique(
                    test_states[valid],
                    return_inverse=True,
                )
                successor_rows = np.linalg.solve(
                    (identity - gamma * transition_matrix).T,
                    identity[:, unique_states],
                ).T

                # Match targets whose first future sample has weight one.
                predictions = (
                    successor_rows[inverse] - identity[test_states[valid]]
                ) / gamma

                if is_shuffle:
                    targets = targets[rng.permutation(len(targets))]

                correlation = _flattened_correlation(predictions, targets)

            n_eligible = int(eligible.sum())
            results.append(
                {
                    "horizon": float(horizon),
                    "fold": fold_index,
                    "correlation": correlation,
                    "n_test_samples": int(valid.sum()),
                    "test_coverage": (
                        float(valid.sum() / n_eligible) if n_eligible else np.nan
                    ),
                }
            )

    return pd.DataFrame(results)


def _compute_state_future_returns(
    states: np.ndarray,
    moving: np.ndarray,
    n_states: int,
    gamma: float,
) -> np.ndarray:
    """Compute future returns within uninterrupted movement bouts."""
    returns = np.zeros((len(states), n_states), dtype=float)

    for time_index in range(len(states) - 2, -1, -1):
        if moving[time_index] and moving[time_index + 1]:
            returns[time_index] = gamma * returns[time_index + 1]
            returns[time_index, states[time_index + 1]] += 1.0

    return returns


def _flattened_correlation(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Correlate flattened prediction and target matrices."""
    if len(predictions) == 0:
        return np.nan

    predictions = predictions.ravel()
    targets = targets.ravel()

    if np.std(predictions) == 0 or np.std(targets) == 0:
        return np.nan

    return float(np.corrcoef(predictions, targets)[0, 1])


def _discretize_positions(
    x: ArrayLike,
    y: ArrayLike,
    n_bins: int | None,
    *,
    directional_states: bool = False,
    n_heading_bins: int = 4,
) -> tuple[np.ndarray, int]:
    """Convert positions to spatial or directional-state indices."""
    if n_bins is not None and n_bins <= 0:
        raise ValueError("n_spatial_bins must be positive.")
    if directional_states and n_heading_bins < 2:
        raise ValueError("n_heading_bins must be at least 2.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if n_bins is None:
        bins_per_unit = 30 / 100
        n_x_bins = int(np.ceil(np.ptp(x) * bins_per_unit))
        n_y_bins = int(np.ceil(np.ptp(y) * bins_per_unit))
        n_x_bins = max(5, 5 * round(n_x_bins / 5))
        n_y_bins = max(5, 5 * round(n_y_bins / 5))
    else:
        n_x_bins = n_bins
        n_y_bins = n_bins

    x_edges = np.linspace(np.min(x), np.max(x), n_x_bins + 1)
    y_edges = np.linspace(np.min(y), np.max(y), n_y_bins + 1)

    x_bins = np.searchsorted(x_edges, x, side="right") - 1
    y_bins = np.searchsorted(y_edges, y, side="right") - 1
    x_bins = np.clip(x_bins, 0, n_x_bins - 1)
    y_bins = np.clip(y_bins, 0, n_y_bins - 1)

    spatial_states = y_bins * n_x_bins + x_bins
    n_spatial_states = n_x_bins * n_y_bins

    if not directional_states:
        return spatial_states, n_spatial_states

    step_x = np.diff(x, prepend=x[0])
    step_y = np.diff(y, prepend=y[0])
    angles = np.mod(np.arctan2(step_y, step_x), 2 * np.pi)

    bin_width = 2 * np.pi / n_heading_bins
    heading_bins = np.floor(
        np.mod(angles + bin_width / 2, 2 * np.pi) / bin_width
    ).astype(int)

    states = spatial_states * n_heading_bins + heading_bins
    return states, n_spatial_states * n_heading_bins


def _estimate_transition_matrix(
    states: np.ndarray,
    train_mask: np.ndarray,
    n_states: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate transitions whose source and target are in training data."""
    valid_transitions = train_mask[:-1] & train_mask[1:]
    source = states[:-1][valid_transitions]
    target = states[1:][valid_transitions]

    counts = np.zeros((n_states, n_states), dtype=float)
    np.add.at(counts, (source, target), 1.0)

    outgoing_counts = counts.sum(axis=1)
    observed_states = outgoing_counts > 0

    transition_matrix = np.zeros_like(counts)
    transition_matrix[observed_states] = (
        counts[observed_states] / outgoing_counts[observed_states, np.newaxis]
    )

    return transition_matrix, observed_states
