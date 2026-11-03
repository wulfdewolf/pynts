import time

import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.pipeline import Pipeline

from pynts.bounds import get_variable_bounds
from pynts.glms.grid_glm import search_gaussian_grid
from pynts.glms.metrics import (
    evaluate_glm,
    paired_loglik_pvalue,
    poisson_loglik_score,
)
from pynts.glms.util import (
    _as_array,
    compute_com,
    count_fields,
    get_basis,
    make_feature,
    plot_glm_fit,
)
from pynts.smoothing import gaussian_filter_nan
from pynts.util import wrap_list


def _select_model(
    X: np.ndarray,
    y: np.ndarray,
    bounds: np.ndarray,
    basis_name: str | list[str],
    n_jobs: int,
) -> tuple[Pipeline, float]:
    """Select hyperparameters and refit on all training data."""
    basis, hyperparams = get_basis(basis_name, bounds)

    if basis_name == "grid_gauss":
        return search_gaussian_grid(
            X,
            y,
            hyperparams,
            n_jobs,
            score=poisson_loglik_score,
        )

    model = Pipeline(
        [
            ("basis", basis),
            ("imputer", SimpleImputer(strategy="median")),
            ("glm", PoissonRegressor(solver="newton-cholesky")),
        ]
    )
    search_kwargs = {
        "estimator": model,
        "param_grid": hyperparams,
        "cv": 2,
        "scoring": make_scorer(poisson_loglik_score),
        "n_jobs": n_jobs,
    }

    if basis_name == "grid":
        search = HalvingGridSearchCV(
            **search_kwargs,
            factor=6,
            min_resources=min(len(X), 1000),
            return_train_score=False,
            resource="n_samples",
            random_state=0,
        )
    else:
        search = GridSearchCV(**search_kwargs)

    search.fit(X, y)
    return search.best_estimator_, float(search.best_score_)


def fit_glm(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    correlates: str | ArrayLike,
    epoch: nap.IntervalSet | None = None,
    bin_size_sec: float = 0.05,
    bounds: ArrayLike | None = None,
    force_basis: str | None = None,
    n_jobs: int = 1,
    plot: bool = False,
) -> dict:
    """Fit a Poisson model using CV and evaluate held-out epochs.

    Select by mean CV log-likelihood. Test held-out epoch likelihoods
    against a null predicting the training mean. Report McFadden
    pseudo-R² against each evaluated epoch's mean.

    run_time includes hyperparameter search and final refitting.
    """
    correlates = list(wrap_list(correlates))

    if not correlates:
        raise ValueError("Provide at least one correlate.")

    if epoch is None:
        epoch = cluster.time_support.intersect(session[correlates[0]].time_support)

    for correlate in correlates:
        epoch = epoch.intersect(session[correlate].time_support)

    bounds = get_variable_bounds(session, correlates, bounds)

    if bounds.shape != (len(correlates), 2):
        raise ValueError(f"bounds must have shape ({len(correlates)}, 2).")

    # Align spike counts and behavioral features.
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

    def moving_mask(ep: nap.IntervalSet) -> np.ndarray:
        moving_epoch = ep.intersect(session["moving"])
        return ~np.isnan(_as_array(moving_epoch.in_interval(counts)))

    train_idx = moving_mask(splits[:10])
    test_idx = [moving_mask(ep) for ep in splits[10:]]
    test_n_bins = np.array([mask.sum() for mask in test_idx], dtype=int)

    if train_idx.sum() < 2:
        return {
            "test_pr2": np.full(len(test_idx), np.nan),
            "test_loglik": np.full(len(test_idx), np.nan),
            "null_loglik": np.full(len(test_idx), np.nan),
            "test_n_bins": test_n_bins,
            "median_test_pr2": np.nan,
            "paired_p_value": np.nan,
            "cv_loglik": np.nan,
            "run_time": 0.0,
        }

    # Hyperparameter selection and final refit.
    X_train = X[train_idx]
    y_train = y[train_idx]
    basis_name = force_basis if force_basis is not None else correlates

    start_time = time.perf_counter()
    best_model, cv_loglik = _select_model(
        X_train,
        y_train,
        bounds,
        basis_name,
        n_jobs,
    )
    run_time = time.perf_counter() - start_time

    result = evaluate_glm(
        best_model,
        X,
        y,
        test_idx,
        null_mean=float(y_train.mean()),
    )
    result.update(
        cv_loglik=cv_loglik,
        run_time=run_time,
    )

    # Spatial summaries.
    if "P_x" in correlates:
        result["n_fields"], result["field_size"] = count_fields(
            best_model,
            bounds,
            resolution_cm=4,
        )

        if force_basis is None or "P" in force_basis:
            result["com_x"], result["com_y"] = compute_com(
                best_model,
                bounds,
                resolution_cm=4,
            )

        elif force_basis in ("grid", "grid_gauss"):
            fitted_basis = best_model.named_steps["basis"]
            fields = ["orientation", "field_spacing"]

            if force_basis == "grid_gauss":
                sigma = fitted_basis.width_ratio * fitted_basis.field_spacing
                result["field_size"] = float(2 * np.sqrt(2 * np.log(2)) * sigma)
                fields.extend(["width_ratio", "phase_u", "phase_v"])

            for field in fields:
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
            cluster,
            np.stack([session["P_x"], session["P_y"]], axis=1),
            bins=edges,
        )
        tc = gaussian_filter_nan(tc, (1, 1), keep=True)

        _, axs = plt.subplots(
            1,
            2,
            figsize=(6, 3),
            constrained_layout=True,
        )
        plot_glm_fit(axs, tc, session, bin_size_sec, best_model)
        axs[0].set_title("Observed")
        axs[1].set_title("Fitted")
        plt.show()

    return result
