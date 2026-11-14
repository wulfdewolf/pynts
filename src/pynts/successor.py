from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, diags, eye
from scipy.sparse.linalg import spsolve


@dataclass
class SuccessorStates:
    """Visited states and their empirical transition matrix."""

    indices: np.ndarray
    labels: np.ndarray
    counts: np.ndarray
    transition_matrix: csc_matrix
    x_edges: np.ndarray
    y_edges: np.ndarray
    n_map_states: int

    def average(self, values: np.ndarray) -> np.ndarray:
        """Average sample features within each visited state."""
        means = np.zeros(
            (len(self.labels), values.shape[1]),
            dtype=float,
        )
        np.add.at(means, self.indices, values)
        return means / self.counts[:, None]


def build_successor_states(
    positions: np.ndarray,
    arena_size: float,
    n_states_side: int,
    directional_states: bool,
    n_heading_bins: int,
) -> SuccessorStates:
    """Estimate transitions between position–movement-heading states."""
    positions = np.asarray(positions, dtype=float)
    x_edges = np.linspace(0.0, arena_size, n_states_side + 1)
    y_edges = x_edges.copy()

    x_bins = np.searchsorted(x_edges, positions[:, 0], side="right") - 1
    y_bins = np.searchsorted(y_edges, positions[:, 1], side="right") - 1
    x_bins = np.clip(x_bins, 0, n_states_side - 1)
    y_bins = np.clip(y_bins, 0, n_states_side - 1)

    labels = y_bins * n_states_side + x_bins
    n_map_states = n_states_side**2

    if directional_states:
        steps = np.diff(positions, axis=0, prepend=positions[:1])
        displacement = np.linalg.norm(steps, axis=1)
        angles = np.mod(
            np.arctan2(steps[:, 1], steps[:, 0]),
            2 * np.pi,
        )

        moving = displacement > 0
        if moving.any():
            previous = np.maximum.accumulate(
                np.where(moving, np.arange(len(positions)), -1)
            )
            previous[previous < 0] = np.flatnonzero(moving)[0]
            angles = angles[previous]
        else:
            angles[:] = 0.0

        bin_width = 2 * np.pi / n_heading_bins
        headings = np.floor(
            np.mod(angles + bin_width / 2, 2 * np.pi) / bin_width
        ).astype(int)

        labels = labels * n_heading_bins + headings
        n_map_states *= n_heading_bins

    unique_labels, indices = np.unique(labels, return_inverse=True)
    n_states = len(unique_labels)
    counts = np.bincount(indices, minlength=n_states).astype(float)

    transitions = coo_matrix(
        (
            np.ones(len(positions) - 1),
            (indices[:-1], indices[1:]),
        ),
        shape=(n_states, n_states),
    ).tocsr()

    outgoing = np.asarray(transitions.sum(axis=1)).ravel()
    inverse_counts = np.divide(
        1.0,
        outgoing,
        out=np.zeros_like(outgoing),
        where=outgoing > 0,
    )
    transition_matrix = (diags(inverse_counts) @ transitions).tocsc()

    return SuccessorStates(
        indices=indices,
        labels=unique_labels,
        counts=counts,
        transition_matrix=transition_matrix,
        x_edges=x_edges,
        y_edges=y_edges,
        n_map_states=n_map_states,
    )


def compute_successor_features(
    transition_matrix: csc_matrix,
    features: np.ndarray,
    horizon: float,
    dt: float,
) -> np.ndarray:
    """Compute discounted expected feature occupancy."""
    if horizon < 0:
        raise ValueError("horizon must be non-negative.")

    if horizon == 0:
        return features.copy()

    gamma = np.exp(-dt / horizon)
    matrix = eye(transition_matrix.shape[0], format="csc") - gamma * transition_matrix
    result = np.asarray(spsolve(matrix, features))
    return result.reshape(features.shape)
