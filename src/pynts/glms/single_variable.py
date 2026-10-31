import math
import time
from typing import Optional

import numpy as np
import pynapple as nap
from numba import njit
from numpy.typing import ArrayLike
from scipy.special import gammaln, xlogy
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from pynts.glms.util import (
    GaussianGridPoissonRegressor,
    compute_com,
    count_fields,
    get_basis,
    make_feature,
    plot_glm_fit,
    wilcoxon_nan,
)
from pynts.smoothing import gaussian_filter_nan
from pynts.util import wrap_list


def _as_array(x):
    return np.asarray(x.values if hasattr(x, "values") else x)


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


def fit_glm(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    correlates: str | ArrayLike,
    epoch: Optional[nap.IntervalSet] = None,
    bin_size_sec: float = 0.05,
    bounds: Optional[ArrayLike] = None,
    force_basis=None,
    plot: bool = False,
):
    """
    Fit one neuron's Poisson GLM.

    CV selection:
        Mean Poisson log-likelihood per bin.

    Held-out significance:
        Paired epoch-level mean log-likelihoods, testing model > null.
        The null predicts the training mean in every bin.

    Reporting:
        median_score retains held-out McFadden pseudo-R²

    run_time measures hyperparameter search and its automatic final refit
    """
    correlates = list(wrap_list(correlates))

    if not correlates:
        raise ValueError("Provide at least one correlate.")

    if epoch is None:
        epoch = cluster.time_support.intersect(session[correlates[0]].time_support)

    for correlate in correlates:
        epoch = epoch.intersect(session[correlate].time_support)

    if bounds is None:
        bounds = [
            (
                np.nanmin(_as_array(session[v])),
                np.nanmax(_as_array(session[v])),
            )
            for v in correlates
        ]

    bounds = np.array(bounds, dtype=float, copy=True)
    if bounds.shape != (len(correlates), 2):
        raise ValueError(f"bounds must have shape ({len(correlates)}, 2).")

    bounds[:, 0] = np.floor(bounds[:, 0] / 5) * 5
    bounds[:, 1] = np.ceil(bounds[:, 1] / 5) * 5

    # Shared time axis for counts and behavioral features
    counts = cluster.count(bin_size_sec, ep=epoch)[:, 0]
    y = _as_array(counts).astype(float, copy=False)

    X = np.concatenate(
        [
            _as_array(make_feature(v, session[v], bounds[i], counts, epoch))
            for i, v in enumerate(correlates)
        ],
        axis=1,
    )

    splits = (
        epoch.split((epoch.tot_length() - 0.01) / 20)
        if session.get("trials") is None
        else session["trials"]
    )

    def moving_mask(ep):
        return ~np.isnan(_as_array(ep.intersect(session["moving"]).in_interval(counts)))

    train_idx = moving_mask(splits[:10])
    test_idx = [moving_mask(ep) for ep in splits[10:]]

    if train_idx.sum() < 2:
        return {
            "test_pr2": np.full(len(test_idx), np.nan),
            "test_loglik": np.full(len(test_idx), np.nan),
            "null_loglik": np.full(len(test_idx), np.nan),
            "test_n_bins": np.array([mask.sum() for mask in test_idx], dtype=int),
            "cv_loglik": np.nan,
            "run_time": 0.0,
        }

    # ---------------------------------------------------------------
    # Hyperparameter search
    # ---------------------------------------------------------------
    basis, hyperparams = get_basis(
        force_basis if force_basis is not None else correlates,
        bounds,
    )

    model = Pipeline(
        [
            ("basis", basis),
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            (
                "glm",
                GaussianGridPoissonRegressor()
                if force_basis == "grid_gauss"
                else PoissonRegressor(solver="newton-cholesky"),
            ),
        ]
    )

    search_kwargs = {
        "estimator": model,
        "param_grid": hyperparams,
        "cv": 2,
        "scoring": make_scorer(poisson_loglik_score),
        "n_jobs": 1,
    }
    search = GridSearchCV(**search_kwargs)

    start_time = time.perf_counter()
    search.fit(X[train_idx], y[train_idx])
    run_time = time.perf_counter() - start_time

    best_model = search.best_estimator_

    # ---------------------------------------------------------------
    # Held-out evaluation: one paired value per epoch
    # ---------------------------------------------------------------
    null_mean = y[train_idx].mean()

    pseudo_r2_scores = np.full(len(test_idx), np.nan)
    model_loglik = np.full(len(test_idx), np.nan)
    null_loglik = np.full(len(test_idx), np.nan)

    for test_number, mask in enumerate(test_idx):
        if not mask.any():
            continue

        observed = y[mask]
        prediction = best_model.predict(X[mask])
        null_prediction = np.full(observed.shape, null_mean, dtype=float)

        pseudo_r2_scores[test_number] = poisson_pseudo_r2(observed, prediction)
        model_loglik[test_number] = poisson_loglik_score(observed, prediction)
        null_loglik[test_number] = poisson_loglik_score(observed, null_prediction)

    result = {
        "test_pr2": pseudo_r2_scores.copy(),
        "test_loglik": model_loglik.copy(),
        "null_loglik": null_loglik.copy(),
        "test_n_bins": np.array([mask.sum() for mask in test_idx], dtype=int),
        "cv_loglik": float(search.best_score_),
        "run_time": run_time,
    }

    # ---------------------------------------------------------------
    # Spatial summaries
    # ---------------------------------------------------------------
    if "P_x" in correlates:
        result["n_fields"], result["field_size"] = count_fields(
            best_model, bounds, resolution_cm=4
        )

        if force_basis is None or "P" in force_basis:
            result["com_x"], result["com_y"] = compute_com(
                best_model, bounds, resolution_cm=4
            )

        elif force_basis == "grid":
            fitted_basis = best_model.named_steps["basis"]

            for field in ["orientation", "field_spacing"]:
                result[field] = getattr(fitted_basis, field)
        elif force_basis == "grid_gauss":
            fitted_basis = best_model.named_steps["basis"]

            # Better estimate of size
            if hasattr(fitted_basis, "width_ratio"):
                sigma = fitted_basis.width_ratio * fitted_basis.field_spacing
                result["field_size"] = float(2 * np.sqrt(2 * np.log(2)) * sigma)

            for field in (
                "orientation",
                "field_spacing",
                "width_ratio",
                "phase_u",
                "phase_v",
            ):
                if hasattr(fitted_basis, field):
                    result[field] = getattr(fitted_basis, field)

    if plot:
        import matplotlib.pyplot as plt

        if correlates != ["P_x", "P_y"]:
            raise ValueError("Plotting requires correlates=['P_x', 'P_y'].")

        resolution_cm = 4.0
        edges = [
            np.linspace(
                lower,
                upper,
                max(1, int(np.ceil((upper - lower) / resolution_cm))) + 1,
            )
            for lower, upper in bounds
        ]

        tc = nap.compute_tuning_curves(
            cluster, np.stack([session["P_x"], session["P_y"]], axis=1), bins=edges
        )
        tc = gaussian_filter_nan(tc, (1, 1), keep=True)
        fig, axs = plt.subplots(
            1,
            2,
            figsize=(6, 3),
            constrained_layout=True,
        )
        plot_glm_fit(axs, tc, session, bin_size_sec, best_model)
        axs[0].set_title("Observed: training data")
        axs[1].set_title("Fitted")
        plt.show()

    return result
