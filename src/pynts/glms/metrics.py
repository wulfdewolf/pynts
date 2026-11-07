import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numba import njit
from numpy.typing import ArrayLike
from scipy.stats import wilcoxon
from sklearn.base import RegressorMixin


@njit(cache=True)
def _poisson_loglik(y: np.ndarray, prediction: np.ndarray) -> float:
    """Mean Poisson log-likelihood for flattened arrays."""
    if y.size == 0:
        return np.nan

    total = 0.0
    impossible = False

    for index in range(y.size):
        observed = y[index]
        expected = prediction[index]

        if (
            not np.isfinite(observed)
            or not np.isfinite(expected)
            or observed < 0
            or expected < 0
        ):
            return np.nan

        if expected == 0:
            if observed > 0:
                impossible = True
        else:
            total += (
                observed * math.log(expected) - expected - math.lgamma(observed + 1)
            )

    if impossible:
        return -np.inf

    return total / y.size


def poisson_loglik_score(
    y: ArrayLike,
    prediction: ArrayLike,
) -> float:
    """Mean Poisson log-likelihood per bin."""
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    if y.shape != prediction.shape:
        raise ValueError("Observed and predicted counts must have equal shapes.")

    return float(
        _poisson_loglik(
            np.ascontiguousarray(y.ravel()),
            np.ascontiguousarray(prediction.ravel()),
        )
    )


@njit(cache=True)
def _poisson_pseudo_r2(
    y: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """McFadden pseudo-R² against the evaluated data's mean."""
    if y.size == 0:
        return np.nan

    model_ll = _poisson_loglik(y, prediction)
    baseline_ll = _poisson_loglik(y, np.full(y.size, y.mean()))

    if not np.isfinite(model_ll) or not np.isfinite(baseline_ll) or baseline_ll == 0:
        return np.nan

    return 1 - model_ll / baseline_ll


def poisson_pseudo_r2(
    y: ArrayLike,
    prediction: ArrayLike,
) -> float:
    """McFadden pseudo-R² against the evaluated data's mean."""
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    if y.shape != prediction.shape:
        raise ValueError("Observed and predicted counts must have equal shapes.")

    return float(
        _poisson_pseudo_r2(
            np.ascontiguousarray(y.ravel()),
            np.ascontiguousarray(prediction.ravel()),
        )
    )


def negative_mse_score(
    y: ArrayLike,
    prediction: ArrayLike,
) -> float:
    """Negative mean squared error."""
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    if y.shape != prediction.shape:
        raise ValueError("Observed and predicted values must have equal shapes.")

    if y.size == 0:
        return np.nan

    return -float(np.mean((y - prediction) ** 2))


def paired_score_pvalue(
    model: ArrayLike,
    null: ArrayLike,
) -> float:
    """Test whether paired model scores exceed null scores."""
    model = np.asarray(model, dtype=float)
    null = np.asarray(null, dtype=float)

    if model.shape != null.shape:
        raise ValueError("Model and null scores must have matching splits.")

    valid = np.isfinite(model) & np.isfinite(null)
    difference = model[valid] - null[valid]

    if difference.size < 3:
        return np.nan

    if np.all(difference == 0):
        return 1.0

    return float(
        wilcoxon(
            difference,
            alternative="greater",
            zero_method="wilcox",
        ).pvalue
    )


def paired_loglik_pvalue(
    model: ArrayLike,
    null: ArrayLike,
) -> float:
    """Backward-compatible alias for paired score comparisons."""
    return paired_score_pvalue(model, null)


def evaluate_glm(
    model: RegressorMixin,
    X: np.ndarray,
    y: np.ndarray,
    test_idx: list[np.ndarray],
    null_mean: float,
    *,
    family: Literal["poisson", "gaussian"] = "poisson",
    null_models: Sequence[RegressorMixin] | None = None,
) -> dict:
    """Evaluate blocks using Poisson likelihood or negative MSE.

    Poisson pseudo-R² uses each block's mean as its baseline.
    Gaussian R² uses the supplied training mean as its baseline.
    Paired tests compare block scores against the supplied nulls.
    """
    if family not in {"poisson", "gaussian"}:
        raise ValueError("Family must be 'poisson' or 'gaussian'.")

    if null_models is not None and not null_models:
        raise ValueError("Provide at least one null model.")

    score = poisson_loglik_score if family == "poisson" else negative_mse_score

    test_pr2 = np.full(len(test_idx), np.nan)
    test_score = np.full(len(test_idx), np.nan)
    null_score = np.full(len(test_idx), np.nan)
    model_null_score = np.full(len(test_idx), np.nan)

    for index, mask in enumerate(test_idx):
        if not mask.any():
            continue

        features = X[mask]
        observed = y[mask]
        prediction = model.predict(features)
        null_prediction = np.full(observed.shape, null_mean, dtype=float)

        test_score[index] = score(observed, prediction)
        null_score[index] = score(observed, null_prediction)

        if family == "poisson":
            test_pr2[index] = poisson_pseudo_r2(observed, prediction)
        elif (
            np.isfinite(test_score[index])
            and np.isfinite(null_score[index])
            and null_score[index] < 0
        ):
            test_pr2[index] = 1 - test_score[index] / null_score[index]

        if null_models is not None:
            model_null_score[index] = float(
                np.mean(
                    [score(observed, null.predict(features)) for null in null_models]
                )
            )

    finite_pr2 = test_pr2[np.isfinite(test_pr2)]
    mean_null_p_value = paired_score_pvalue(test_score, null_score)

    results = {
        "test_pr2": test_pr2,
        "test_score": test_score,
        "null_score": null_score,
        "test_n_bins": np.array([mask.sum() for mask in test_idx], dtype=int),
        "median_test_pr2": (
            float(np.median(finite_pr2)) if finite_pr2.size else np.nan
        ),
        "mean_null_p_value": mean_null_p_value,
        "paired_p_value": mean_null_p_value,
    }

    if null_models is not None:
        model_null_p_value = paired_score_pvalue(
            test_score,
            model_null_score,
        )
        results.update(
            model_null_score=model_null_score,
            model_null_p_value=model_null_p_value,
            paired_p_value=float(np.maximum(mean_null_p_value, model_null_p_value)),
        )

    return results
