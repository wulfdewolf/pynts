from typing import Callable, Optional

import nemos as nmo
import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike
from scipy.stats import wilcoxon
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from pynts.glms.util import get_basis, make_feature
from pynts.util import wrap_list


def fit_glm(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    correlates: str | ArrayLike,
    epoch: Optional[nap.IntervalSet] = None,
    bin_size_sec: float = 0.02,
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

    # Prepare bases
    basis, hyperparams = get_basis(force_basis or correlates, bounds)

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
    splits = epoch.split((epoch.tot_length() - 0.01) / 20)
    train_idx = ~np.isnan(splits[::2].intersect(session["moving"]).in_interval(y))
    test_idx = [
        ~np.isnan(test_epoch.intersect(session["moving"]).in_interval(y))
        for test_epoch in splits[1::2]
    ]

    # Fit GLM
    metric = nmo.observation_models.PoissonObservations().pseudo_r2
    cv = RandomizedSearchCV(
        Pipeline([("basis", basis), ("glm", nmo.glm.GLM(regularizer="Ridge"))]),
        {
            **{
                f"basis__{hyperparam}": search_space
                for hyperparam, search_space in hyperparams.items()
            },
            "glm__regularizer_strength": np.logspace(-5, 0, 10),
        },
        cv=KFold(n_splits=2, shuffle=True, random_state=42),
        scoring=make_scorer(metric),
        n_iter=n_iter,
    )
    with np.errstate(divide="ignore"):
        cv.fit(X.values[train_idx], y.values[train_idx])

    scores = [
        cv.best_estimator_.score(X.values[idx], y.values[idx]) for idx in test_idx
    ]

    # Test
    null_model = DummyRegressor().fit(X.values[train_idx], y.values[train_idx])
    null_scores = np.array(
        [metric(y.values[idx], null_model.predict(X[idx])) for idx in test_idx]
    )
    _, p_val = wilcoxon(
        scores, null_scores, alternative="greater", zero_method="zsplit"
    )

    # from pynts.glms.util import plot_grid_fit

    # plot_grid_fit(cluster, session, bin_size_sec, cv.best_estimator_)

    return {
        "scores": scores,
        "null_scores": null_scores,
        "p_val": p_val,
        "model": cv.best_estimator_,
    }
