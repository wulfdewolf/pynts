import time
import warnings
from typing import Callable, Optional

import jax
import nemos as nmo
import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from pynts.glms.util import count_fields, get_basis, make_feature, wilcoxon_nan
from pynts.util import wrap_list

jax.config.update("jax_enable_x64", True)


def fit_glm(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    correlates: str | ArrayLike,
    epoch: Optional[nap.IntervalSet] = None,
    bin_size_sec: float = 0.05,
    bounds: Optional[ArrayLike] = None,
    force_basis=None,
    n_iter: int = 100,
):
    if epoch is None:
        epoch = cluster.time_support.intersect(
            session[list(session.keys())[0]].time_support
        )

    # Extract bounds and range if not given
    bounds = (
        [(np.nanmin(session[v]), np.nanmax(session[v])) for v in wrap_list(correlates)]
        if bounds is None
        else np.array(bounds)
    )

    # Prepare input/output
    y = cluster.count(bin_size_sec)[:, 0].restrict(epoch)
    X = np.concatenate(
        [
            make_feature(v, session[v], bounds[i], y, epoch)
            for i, v in enumerate(wrap_list(correlates))
        ],
        axis=1,
    )

    # Define data splits
    splits = (
        epoch.split((epoch.tot_length() - 0.01) / 20)
        if session["trials"] is None
        else session["trials"]
    )
    train_idx = ~np.isnan(splits[::2].intersect(session["moving"]).in_interval(y))
    test_idx = [
        ~np.isnan(test_epoch.intersect(session["moving"]).in_interval(y))
        for test_epoch in splits[1::2]
    ]

    # Fit GLM
    metric = nmo.observation_models.PoissonObservations().pseudo_r2
    basis, hyperparams = get_basis(force_basis or correlates, bounds)
    model = Pipeline(
        [
            ("basis", basis),
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("glm", PoissonRegressor()),
        ]
    )
    search_space = {
        **{
            f"basis__{hyperparam}": search_space
            for hyperparam, search_space in hyperparams.items()
        },
        "glm__alpha": np.logspace(-5, 0, 10),
    }

    cv = RandomizedSearchCV(
        model,
        search_space,
        cv=KFold(n_splits=2, shuffle=True, random_state=42),
        scoring=make_scorer(metric),
        n_iter=n_iter,
        n_jobs=12,
    )

    start_time = time.time()
    with np.errstate(divide="ignore"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The total space of parameters .* is smaller than n_iter=.*",
                category=UserWarning,
                module=r"sklearn\.model_selection\._search",
            )
            warnings.filterwarnings(
                "ignore",
                message=".*divide by zero encountered in log.*",
                category=RuntimeWarning,
            )
            cv.fit(X.values[train_idx], y.values[train_idx])
            run_time = time.time() - start_time

            scores = [
                np.nan
                if idx.sum() == 0
                else cv.best_estimator_.score(X.values[idx], y.values[idx])
                for idx in test_idx
            ]

    # Test
    null_model = DummyRegressor().fit(X.values[train_idx], y.values[train_idx])
    null_scores = [
        metric(y.values[idx], null_model.predict(X[idx])) for idx in test_idx
    ]
    p_val = wilcoxon_nan(scores, null_scores)

    result = {
        "median_score": np.nan if np.all(np.isnan(scores)) else np.nanmedian(scores),
        "p_val": p_val,
        "run_time": run_time,
        # "null_scores": null_scores,
        # "model": cv.best_estimator_,
    }

    if force_basis == "grid" or force_basis == "grid_sim":
        result["n_fields"] = count_fields(
            cv.best_estimator_,
            bounds,
            resolution_cm=4,
        )
        result["orientation"] = cv.best_estimator_.named_steps["basis"].orientation
        result["spacing"] = cv.best_estimator_.named_steps["basis"].spacing

        if result["n_fields"] < 3:
            result["p_val"] = 1.0
            result["p_val_fdr"] = 1.0

    # import matplotlib.pyplot as plt

    # from pynts.glms.util import plot_glm_fit
    # from pynts.smoothing import gaussian_filter_nan
    # from pynts.wrappers import compute_travel_projected

    # position = np.stack([session["P_x"], session["P_y"]], axis=1)
    # tc = nap.compute_tuning_curves(
    #    cluster, position, bins=40, epochs=session["moving"], feature_names=["0", "1"]
    # )
    # tc = gaussian_filter_nan(tc, (2, 2), keep=False, mode="fill")

    # fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(2, 1))
    # plot_glm_fit(axs, tc, session, bin_size_sec, cv.best_estimator_)
    # plt.savefig(f"fit_{cluster.index[0]}.png")
    # plt.show()
    # print(result)
    # quit()
    return result
