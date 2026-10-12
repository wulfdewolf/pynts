from typing import Callable

import numpy as np
import pandas as pd
import pynapple as nap
from numpy.typing import ArrayLike


def compute_future_return_correlation(
    session: dict,
    session_type: str,
    clusters: nap.TsGroup,
    horizon_range: ArrayLike = np.arange(0, 31, 1),
    is_shuffle: bool = False,
    *,
    n_folds: int = 5,
    n_spatial_bins: int | None = None,
    directional_states: bool = False,
    n_heading_bins: int = 4,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Compute blocked-CV successor future-return correlations."""
    del clusters  # Retained for compatibility with related metrics.

    x = session["P_x"].as_series().dropna()
    y = session["P_y"].as_series().dropna()
    times = session["P_x"].dropna().times()
    dt = float(np.median(np.diff(times)))

    states, n_states = _discretize_positions(
        x,
        y,
        n_spatial_bins,
        directional_states=directional_states,
        n_heading_bins=n_heading_bins,
    )

    folds = np.array_split(np.arange(len(states)), n_folds)
    rng = np.random.default_rng(random_state)
    results = []

    for horizon in np.asarray(horizon_range, dtype=float):
        if horizon < 0:
            raise ValueError("Horizons must be non-negative.")

        gamma = 0.0 if horizon == 0 else float(np.exp(-dt / horizon))

        for fold_index, test_indices in enumerate(folds):
            if len(test_indices) < 2:
                continue

            train_mask = np.ones(len(states), dtype=bool)
            train_mask[test_indices] = False

            transition_matrix, observed_states = _estimate_transition_matrix(
                states,
                train_mask,
                n_states,
            )
            successor_matrix = np.linalg.solve(
                np.eye(n_states) - gamma * transition_matrix,
                np.eye(n_states),
            )

            future_predictions = successor_matrix[states[test_indices]]
            future_predictions -= np.eye(n_states)[states[test_indices]]

            future_returns = _compute_state_future_returns(
                states[test_indices],
                n_states,
                gamma,
            )
            valid = observed_states[states[test_indices]]

            if is_shuffle:
                shuffled_order = rng.permutation(len(future_returns))
                future_returns = future_returns[shuffled_order]

            correlation = _flattened_correlation(
                future_predictions[valid],
                future_returns[valid],
            )
            results.append(
                {
                    "horizon": horizon,
                    "fold": fold_index,
                    "correlation": correlation,
                    "n_test_samples": int(valid.sum()),
                    "test_coverage": float(valid.mean()),
                }
            )

    return pd.DataFrame(results)


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
        n_x_bins = max(1, int(np.ceil(np.ptp(x) * bins_per_unit)))
        n_y_bins = max(1, int(np.ceil(np.ptp(y) * bins_per_unit)))
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


def _compute_state_future_returns(
    states: np.ndarray,
    n_states: int,
    gamma: float,
) -> np.ndarray:
    """Compute future-only discounted state-occupancy returns."""
    returns = np.zeros((len(states), n_states), dtype=float)

    for time_index in range(len(states) - 2, -1, -1):
        returns[time_index, states[time_index + 1]] = 1.0
        returns[time_index] += gamma * returns[time_index + 1]

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
