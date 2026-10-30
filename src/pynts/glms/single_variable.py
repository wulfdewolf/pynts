import time
from typing import Optional

import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike
from scipy.special import gammaln, xlogy
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, KFold
from sklearn.pipeline import Pipeline

from pynts.glms.util import (
    compute_com,
    count_fields,
    get_basis,
    make_feature,
    wilcoxon_nan,
)
from pynts.util import wrap_list


def _as_array(x):
    return np.asarray(x.values if hasattr(x, "values") else x)


def poisson_loglik_score(y, prediction):
    """Mean Poisson log-likelihood per bin. Higher is better."""
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    if y.shape != prediction.shape:
        raise ValueError("Observed and predicted counts must have equal shapes.")

    if (
        y.size == 0
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(prediction))
        or np.any(y < 0)
        or np.any(prediction < 0)
    ):
        return np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.mean(xlogy(y, prediction) - prediction - gammaln(y + 1)))


def poisson_pseudo_r2(y, prediction):
    """
    McFadden pseudo-R² using the evaluated data's mean as its baseline.

    This baseline differs from the training-mean null used in the
    held-out significance test.
    """
    y = np.asarray(y, dtype=float)

    if y.size == 0:
        return np.nan

    model_ll = poisson_loglik_score(y, prediction)
    baseline_ll = poisson_loglik_score(
        y,
        np.full(y.shape, y.mean(), dtype=float),
    )

    if not np.isfinite(model_ll) or not np.isfinite(baseline_ll) or baseline_ll == 0:
        return np.nan

    return float(1 - model_ll / baseline_ll)


def _finite_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def fit_glm(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    correlates: str | ArrayLike,
    epoch: Optional[nap.IntervalSet] = None,
    bin_size_sec: float = 0.02,
    bounds: Optional[ArrayLike] = None,
    force_basis=None,
):
    """
    Fit one neuron's Poisson GLM.

    CV selection:
        Mean Poisson log-likelihood per bin.

    Held-out significance:
        Paired epoch-level mean log-likelihoods, testing model > null.
        The null predicts the training mean in every bin.

    Reporting:
        median_score retains held-out McFadden pseudo-R².
        mean_loglik_gain_per_bin reports improvement over the null,
        weighting valid test epochs equally.

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
            "median_score": np.nan,
            "mean_loglik_gain_per_bin": np.nan,
            "median_loglik_gain_per_bin": np.nan,
            "n_valid_test_pairs": 0,
            "p_val": np.nan,
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
                PoissonRegressor(solver="newton-cholesky"),
            ),
        ]
    )

    search_space = {
        **{f"basis__{name}": values for name, values in hyperparams.items()},
        "glm__alpha": [0.0001, 0.001, 0.01],
    }

    search_kwargs = {
        "estimator": model,
        "param_grid": search_space,
        "cv": KFold(
            n_splits=2,
            shuffle=True,
            random_state=420,
        ),
        "scoring": make_scorer(poisson_loglik_score),
        "n_jobs": 1,
    }

    if force_basis == "grid":
        search = HalvingGridSearchCV(
            **search_kwargs,
            factor=4,
            min_resources=min(int(train_idx.sum()), 4000),
            resource="n_samples",
            random_state=420,
        )
    else:
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

    valid = np.isfinite(model_loglik) & np.isfinite(null_loglik)
    loglik_gain = model_loglik[valid] - null_loglik[valid]

    if valid.sum() < 3:
        p_val = np.nan
    elif np.all(loglik_gain == 0):
        p_val = 1.0
    else:
        p_val = wilcoxon_nan(
            model_loglik[valid],
            null_loglik[valid],
            alternative="greater",
        )

    result = {
        "median_score": _finite_median(pseudo_r2_scores),
        "mean_loglik_gain_per_bin": (
            float(loglik_gain.mean()) if loglik_gain.size else np.nan
        ),
        "median_loglik_gain_per_bin": _finite_median(loglik_gain),
        "n_valid_test_pairs": int(valid.sum()),
        "p_val": p_val,
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

            if cluster["brain_region"].iloc[0] != "E" and result["n_fields"] < 4:
                result["p_val"] = 1.0

                # Preserve the existing exclusion flag.
                # This is not a population-wide FDR correction.
                result["p_val_fdr"] = 1.0

    return result
