import time

import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike
from scipy.ndimage import correlate1d
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.pipeline import Pipeline

from pynts.bounds import get_variable_bounds
from pynts.glms.grid_glm import (
    make_shuffled_grid_models,
    search_gaussian_grid,
)
from pynts.glms.metrics import evaluate_glm, poisson_loglik_score
from pynts.glms.util import (
    _as_array,
    compute_com,
    count_fields_from_rate,
    get_basis,
    make_feature,
    plot_glm_fit,
)
from pynts.smoothing import gaussian_filter_nan
from pynts.util import wrap_list


def _select_model(
    X: np.ndarray,
    y: np.ndarray,
    true_bounds: np.ndarray,
    projection_bounds: np.ndarray,
    basis_name: str | list[str],
    n_jobs: int,
    random_state: int | None = 0,
) -> tuple[Pipeline, float]:
    """Select hyperparameters using a projection-independent search space."""
    ranges = true_bounds[:, 1] - true_bounds[:, 0]
    basis, hyperparams = get_basis(basis_name, bounds=true_bounds, ranges=ranges)
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
            random_state=random_state,
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
    true_bounds: ArrayLike | None = None,
    projection_bounds: ArrayLike | None = None,
    force_basis: str | None = None,
    null_model: str = "mean",
    n_jobs: int = 1,
    plot: bool = False,
    n_null_shuffles: int = 20,
    random_state: int | None = 0,
    restrict_max_time: float | None = None,
) -> dict:
    """Fit a Poisson model using CV and evaluate held-out epochs.

    Compare epoch likelihoods against a training-mean null or the mean
    likelihood of phase-shuffled Gaussian grids. Report McFadden
    pseudo-R² against each evaluated epoch's mean.

    run_time includes model selection and fitted/null model refitting.
    """

    if null_model not in ("mean", "shuffled_grid"):
        raise ValueError("null_model must be 'mean' or 'shuffled_grid'.")

    if null_model == "shuffled_grid":
        if force_basis != "grid_gauss":
            raise ValueError(
                "null_model='shuffled_grid' requires force_basis='grid_gauss'."
            )
        if n_null_shuffles < 1:
            raise ValueError("n_null_shuffles must be positive.")

    correlates = list(wrap_list(correlates))

    if not correlates:
        raise ValueError("Provide at least one correlate.")

    if epoch is None:
        epoch = cluster.time_support.intersect(session[correlates[0]].time_support)

    if restrict_max_time is not None:
        restriction = nap.IntervalSet(
            epoch.start[0], epoch.start[0] + restrict_max_time
        )
        epoch = epoch.intersect(restriction)

    for correlate in correlates:
        epoch = epoch.intersect(session[correlate].time_support)

    # Resolve bounds
    true_bounds = np.asarray(
        get_variable_bounds(session, correlates, true_bounds),
        dtype=float,
    )
    bounds = (
        true_bounds.copy()
        if projection_bounds is None
        else np.asarray(projection_bounds, dtype=float)
    )
    expected_shape = (len(correlates), 2)
    if true_bounds.shape != expected_shape or bounds.shape != expected_shape:
        raise ValueError(f"Bounds must have shape {expected_shape}.")

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
            "test_score": np.full(len(test_idx), np.nan),
            "null_score": np.full(len(test_idx), np.nan),
            "test_n_bins": test_n_bins,
            "median_test_pr2": np.nan,
            "paired_p_value": np.nan,
            "cv_score": np.nan,
            "run_time": 0.0,
        }

    # Hyperparameter selection and final refit.
    X_train = X[train_idx]
    y_train = y[train_idx]
    basis_name = force_basis if force_basis is not None else correlates

    start_time = time.perf_counter()
    best_model, cv_score = _select_model(
        X_train, y_train, true_bounds, bounds, basis_name, n_jobs, random_state
    )

    null_models = None
    if null_model == "shuffled_grid":
        null_models = make_shuffled_grid_models(
            best_model,
            n_shuffles=n_null_shuffles,
            random_state=random_state,
        )

    run_time = time.perf_counter() - start_time

    result = evaluate_glm(
        best_model,
        X,
        y,
        test_idx,
        null_mean=float(y_train.mean()),
        null_models=null_models,
    )
    result.update(
        cv_score=cv_score,
        run_time=run_time,
    )

    # Spatial summaries
    if "P_x" in correlates and "P_y" in correlates:
        # Fields
        resolution_cm = 2.5
        bounds = np.asarray(bounds, dtype=float)
        xs = np.arange(bounds[0, 0], bounds[0, 1] + resolution_cm, resolution_cm)
        ys = np.arange(bounds[1, 0], bounds[1, 1] + resolution_cm, resolution_cm)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        rate = best_model.predict(positions).reshape(xx.shape)
        result["n_fields"], result["field_size"] = count_fields_from_rate(
            rate,
            resolution_cm**2,
        )

        # COM
        if force_basis is None or "P" in force_basis:
            result["com_x"], result["com_y"] = compute_com(
                best_model,
                bounds,
                resolution_cm=4,
            )

        # Grid params
        if force_basis in ("grid", "grid_gauss"):
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

        resolution = 2.5
        edges = [
            np.linspace(
                lower,
                upper,
                max(1, int(np.ceil((upper - lower) / resolution))) + 1,
            )
            for lower, upper in bounds
        ]

        tc = nap.compute_tuning_curves(
            cluster,
            np.stack([session[v] for v in correlates], axis=1),
            bins=edges,
        )
        tc = gaussian_filter_nan(tc, [2] * len(correlates), keep=True)

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
