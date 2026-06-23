from typing import Optional

import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
from nemos.basis import BSplineEval
from numpy.typing import ArrayLike
from scipy.stats import wilcoxon
from sklearn.dummy import DummyRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import HalvingRandomSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from pynts import wrappers


def fit_predictive_glm(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    epoch: Optional[nap.IntervalSet] = None,
    bin_size_sec: float = 0.02,
    projection_range: ArrayLike = np.arange(-30, 31, 2),
    shift_type: str = "travel",
    range: Optional[ArrayLike] = None,
):
    if epoch is None:
        epoch = cluster.time_support.intersect(session["S"].time_support)

    range = (
        [
            (np.nanmin(session["P_x"]), np.nanmax(session["P_x"])),
            (np.nanmin(session["P_y"]), np.nanmax(session["P_y"])),
        ]
        if range is None
        else range
    )
    env_size = max(_range[1] - _range[0] for _range in range)

    results = []

    y = cluster.count(bin_size_sec)[:, 0]
    shifted_position = {
        shift: getattr(wrappers, f"compute_{shift_type}_projected")(
            session_type,
            session,
            ("P_x", "P_y"),
            shift,
        )
        .interpolate(y)
        .restrict(epoch)
        for shift in projection_range
    }
    y = y.restrict(epoch)

    splits = epoch.split((epoch.tot_length() - 0.01) / 20)
    train_idx = ~np.isnan(splits[::2].intersect(session["moving"]).in_interval(y))
    test_idx = [
        ~np.isnan(test_epoch.intersect(session["moving"]).in_interval(y))
        for test_epoch in splits[1::2]
    ]

    # Fit GLMs
    metric = nmo.observation_models.PoissonObservations().pseudo_r2
    basis = BSplineEval(n_basis_funcs=10, label="P_x", bounds=range[0]) * BSplineEval(
        n_basis_funcs=10, label="P_y", bounds=range[1]
    )
    hyperparams = {
        "basis__P_x__n_basis_funcs": np.arange(5, int(0.5 * env_size), 1),
        "basis__P_y__n_basis_funcs": np.arange(5, int(0.5 * env_size), 1),
    }
    for shift, shifted in tqdm(shifted_position.items()):
        cv = RandomizedSearchCV(
            Pipeline(
                [
                    ("basis", basis.to_transformer()),
                    ("glm", TweedieRegressor(solver="newton-cholesky")),
                ]
            ),
            {
                **hyperparams,
                "glm__alpha": np.logspace(-5, 0, 10),
                "glm__power": np.linspace(0.0, 1.0, 10),
            },
            cv=KFold(n_splits=2, shuffle=True, random_state=42),
            scoring=make_scorer(metric),
            n_iter=40,
            verbose=3,
        )
        with np.errstate(divide="ignore"):
            cv.fit(shifted.values[train_idx], y.values[train_idx])

        results.append(
            {
                "shift": shift,
                "score": [
                    cv.best_estimator_.score(shifted.values[idx], y.values[idx])
                    for idx in test_idx
                ],
            }
        )

    # Test
    null_model = DummyRegressor().fit(shifted.values[train_idx], y.values[train_idx])
    null_scores = np.array(
        [metric(y.values[idx], null_model.predict(shifted[idx])) for idx in test_idx]
    )
    for result in results:
        _, p = wilcoxon(
            result["score"], null_scores, alternative="greater", zero_method="zsplit"
        )
        result["p_val"] = p

    # Correct
    pvals = [r["p_val"] for r in results]
    _, pvals_fdr, _, _ = multipletests(
        pvals,
        method="fdr_bh",
    )
    for r, p in zip(results, pvals_fdr):
        r["p_val_fdr"] = p

    return results
