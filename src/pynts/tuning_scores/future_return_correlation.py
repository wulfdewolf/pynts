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
    n_spatial_bins: int = 30,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Compute blocked-CV successor future-return correlations."""
    del clusters  # Retained for compatibility with related metrics.

    x = session["P_x"].as_series().dropna()
    y = session["P_y"].as_series().dropna()

    common_times = x.index.intersection(y.index)
    x = x.loc[common_times].to_numpy(dtype=float)
    y = y.loc[common_times].to_numpy(dtype=float)
    times = common_times.to_numpy(dtype=float)

    if len(times) < n_folds + 1:
        raise ValueError("The trajectory is too short for the requested folds.")

    dt = float(np.median(np.diff(times)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Position timestamps must be strictly increasing.")

    states, n_states = _discretize_positions(
        x,
        y,
        n_spatial_bins,
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

            # Exclude the current state from the prediction.
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
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, int]:
    """Convert positions to flattened spatial-bin indices."""
    x_edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    y_edges = np.linspace(np.min(y), np.max(y), n_bins + 1)

    x_bins = np.searchsorted(x_edges, x, side="right") - 1
    y_bins = np.searchsorted(y_edges, y, side="right") - 1

    x_bins = np.clip(x_bins, 0, n_bins - 1)
    y_bins = np.clip(y_bins, 0, n_bins - 1)

    return y_bins * n_bins + x_bins, n_bins**2


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
