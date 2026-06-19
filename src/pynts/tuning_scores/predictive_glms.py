from typing import Optional

import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
from nemos.basis import BSplineEval
from numpy.typing import ArrayLike
from scipy.stats import wilcoxon
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, KFold
from sklearn.pipeline import Pipeline
from statsmodels.stats.multitest import multipletests

from pynts import wrappers


def fit_predictive_glm(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    epoch: Optional[nap.IntervalSet] = None,
    bin_size_sec: float = 0.05,
    projection_range: ArrayLike = np.arange(-30, 31, 10),
    shift_type: str = "travel",
):
    if epoch is None:
        epoch = cluster.time_support.intersect(session["S"].time_support)

    results = []

    basis = BSplineEval(
        n_basis_funcs=5,
        order=4,
        label="P_x",
    ) * BSplineEval(
        n_basis_funcs=5,
        order=4,
        label="P_y",
    )
    hyperparams = {
        "basis__P_x__n_basis_funcs": np.arange(5, 21, 1),
        "basis__P_y__n_basis_funcs": np.arange(5, 21, 1),
    }
    min_time_sec = 60
    min_resources = int(min_time_sec / bin_size_sec)

    y = cluster.count(bin_size_sec, ep=epoch)[:, 0]
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

    splits = epoch.split((epoch.tot_length() - 0.01) / 10)
    train_idx = ~np.isnan(splits[::2].in_interval(y))
    test_idx = [~np.isnan(test_epoch.in_interval(y)) for test_epoch in splits[1::2]]

    # Fit GLMs
    for shift, shifted in shifted_position.items():
        cv = HalvingRandomSearchCV(
            Pipeline(
                [
                    ("basis", basis.to_transformer()),
                    (
                        "glm",
                        nmo.glm.GLM(regularizer="Ridge", solver_name="Newton"),
                    ),
                ]
            ),
            {**hyperparams, "glm__regularizer_strength": np.logspace(-3, 0, 20)},
            cv=KFold(n_splits=4, shuffle=True, random_state=42),
            n_candidates=40,
            min_resources=min_resources,
            max_resources=train_idx.sum(),
            verbose=1,
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
    metric = nmo.observation_models.PoissonObservations().pseudo_r2
    null_scores = np.array(
        [
            metric(y.values[idx], np.full(idx.sum(), np.nanmean(y[train_idx])))
            for idx in test_idx
        ]
    )
    for result in results:
        diff = np.asarray(result["score"]) - np.asarray(null_scores)
        _, p = wilcoxon(diff, alternative="greater", zero_method="zsplit")
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
