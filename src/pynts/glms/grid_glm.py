from collections.abc import Callable
from typing import Self

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import xlogy
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.pipeline import Pipeline

ScoreFunction = Callable[[np.ndarray, np.ndarray], float]


@njit(cache=True)
def _log_gaussian_lattice(
    coordinates: np.ndarray,
    width_ratio: float,
    radius: int,
) -> np.ndarray:
    """Evaluate lattice sums for the configured width range."""
    output = np.empty((len(coordinates), 1))
    factor = 0.5 / width_ratio**2

    for index in range(len(coordinates)):
        u, v = coordinates[index]
        total = 0.0

        for i in range(-radius, radius + 1):
            du = u - i

            for j in range(-radius, radius + 1):
                dv = v - j
                distance = du * du + du * dv + dv * dv
                total += np.exp(-factor * distance)

        output[index, 0] = np.log(total)

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
    ) -> None:
        self.field_spacing = field_spacing
        self.orientation = orientation
        self.width_ratio = width_ratio
        self.phase_u = phase_u
        self.phase_v = phase_v

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> Self:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        theta = self.orientation

        lattice = self.field_spacing * np.array(
            [
                [np.cos(theta), np.sin(theta)],
                [
                    np.cos(theta + np.pi / 3),
                    np.sin(theta + np.pi / 3),
                ],
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

    def __init__(self, alpha: float = 0.0) -> None:
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


def get_gaussian_grid_basis(
    bounds: ArrayLike,
) -> tuple[GaussianGridBasis, dict[str, ArrayLike]]:
    """Create the Gaussian basis and its parameter grid."""
    bounds = np.asarray(bounds, dtype=float)
    scale = float(np.max(bounds[:, 1] - bounds[:, 0]))

    hyperparams = {
        "basis__field_spacing": np.arange(0.2 * scale, 0.8 * scale, 10),
        "basis__orientation": np.linspace(0.0, np.pi / 3, 6, endpoint=False),
        "basis__width_ratio": [0.15, 0.3],
        "basis__phase_u": np.arange(3) / 3,
        "basis__phase_v": np.arange(3) / 3,
    }
    return GaussianGridBasis(), hyperparams


def _score_grid_candidate(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, float],
    folds: list[tuple[np.ndarray, np.ndarray]],
    score: ScoreFunction,
) -> float:
    """Evaluate one basis across fixed CV folds."""
    basis = GaussianGridBasis(
        **{key.removeprefix("basis__"): value for key, value in params.items()}
    )
    features = basis.transform(X)
    scores = []

    for train, validation in folds:
        model = GaussianGridPoissonRegressor()
        model.fit(features[train], y[train])
        prediction = model.predict(features[validation])
        scores.append(score(y[validation], prediction))

    return float(np.mean(scores))


def _zoom_grid(
    best_params: dict[str, float],
    hyperparams: dict[str, ArrayLike],
    level: int = 1,
) -> dict[str, list[float]]:
    """Build a finer neighborhood around one candidate."""
    phase_names = {"basis__phase_u", "basis__phase_v"}
    positive_names = {"basis__field_spacing", "basis__width_ratio"}
    grid = {}

    for name, values in hyperparams.items():
        values = np.unique(np.asarray(values, dtype=float))
        center = float(best_params[name])

        if values.size == 1:
            grid[name] = [center]
            continue

        step = float(np.min(np.diff(values))) / 2**level
        neighbors = center + step * np.array([-1.0, 0.0, 1.0])

        if name in phase_names:
            neighbors %= 1.0
        elif name in positive_names:
            neighbors = neighbors[neighbors > 0]

        grid[name] = np.unique(neighbors).tolist()

    return grid


def normalize_grid_orientation(
    orientation: float,
    phase_u: float,
    phase_v: float,
) -> tuple[float, float, float]:
    """Express orientation in [0, pi/3), preserving the lattice."""
    period = np.pi / 3
    turns = int(np.floor(orientation / period))
    orientation %= period

    for _ in range(turns % 6):
        phase_u, phase_v = -phase_v, phase_u + phase_v

    return (
        float(orientation),
        float(phase_u % 1),
        float(phase_v % 1),
    )


def search_gaussian_grid(
    X: np.ndarray,
    y: np.ndarray,
    hyperparams: dict[str, ArrayLike],
    n_jobs: int,
    score: ScoreFunction,
    *,
    beam_width: int = 2,
) -> tuple[Pipeline, float]:
    """Refine several promising candidates and refit the best."""
    folds = list(KFold(n_splits=2).split(X))
    names = sorted(hyperparams)
    scores: dict[tuple[float, ...], float] = {}

    def evaluate(
        grids: list[dict[str, ArrayLike]],
    ) -> list[dict[str, float]]:
        candidates = {}
        for grid in grids:
            for params in ParameterGrid(grid):
                key = tuple(float(params[name]) for name in names)
                if key not in scores:
                    candidates[key] = dict(zip(names, key))
        results = Parallel(n_jobs=n_jobs)(
            delayed(_score_grid_candidate)(X, y, params, folds, score)
            for params in candidates.values()
        )
        scores.update(zip(candidates, results))
        best_keys = sorted(
            scores,
            key=lambda key: -np.inf if np.isnan(scores[key]) else scores[key],
            reverse=True,
        )[:beam_width]
        return [dict(zip(names, key)) for key in best_keys]

    beam = evaluate([hyperparams])
    for level in range(1, 3):
        beam = evaluate(
            [_zoom_grid(params, hyperparams, level=level) for params in beam]
        )
    best_params = beam[0]
    best_key = tuple(best_params[name] for name in names)

    (
        best_params["basis__orientation"],
        best_params["basis__phase_u"],
        best_params["basis__phase_v"],
    ) = normalize_grid_orientation(
        best_params["basis__orientation"],
        best_params["basis__phase_u"],
        best_params["basis__phase_v"],
    )

    model = Pipeline(
        [
            ("basis", GaussianGridBasis()),
            ("glm", GaussianGridPoissonRegressor()),
        ]
    )
    model.set_params(**best_params)
    model.fit(X, y)

    return model, scores[best_key]
