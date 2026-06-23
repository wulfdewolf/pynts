from typing import Optional

import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
from numpy.typing import ArrayLike
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from pynts import wrappers


class GridBasis(BaseEstimator, TransformerMixin):
    def __init__(self, spacing=40.0, orientation=0.0):
        self.spacing = spacing
        self.orientation = orientation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)

        x = X[:, 0]
        y = X[:, 1]

        k = 2 * np.pi / self.spacing

        directions = [
            self.orientation,
            self.orientation + np.pi / 3,
            self.orientation + 2 * np.pi / 3,
        ]

        features = []

        for theta in directions:
            proj = x * np.cos(theta) + y * np.sin(theta)
            phase = k * proj

            features.append(np.cos(phase))
            features.append(np.sin(phase))

        return np.column_stack(features)

    @property
    def n_features_out_(self):
        return 6


def fit_predictive_grid_glm(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    epoch: Optional[nap.IntervalSet] = None,
    bin_size_sec: float = 0.02,
    projection_range: ArrayLike = [5],  # np.arange(-30, 31, 2),
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
    basis = GridBasis()
    hyperparams = {
        "basis__spacing": np.arange(0.2 * env_size, 0.5 * env_size, 2),
        "basis__orientation": np.linspace(
            0,
            np.pi / 3,
            30,
            endpoint=False,
        ),
    }
    for shift, shifted in tqdm(shifted_position.items()):
        cv = RandomizedSearchCV(
            Pipeline(
                [
                    ("basis", basis),
                    ("glm", TweedieRegressor()),
                ]
            ),
            {
                **hyperparams,
                "glm__alpha": np.logspace(-5, 0, 10),
                "glm__power": np.linspace(0.0, 1.0, 10),
            },
            cv=KFold(n_splits=2, shuffle=True, random_state=42),
            scoring=make_scorer(metric),
            n_iter=1000,
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
                "model": cv.best_estimator_,
            }
        )

    # import matplotlib.pyplot as plt
    # from pynts.smoothing import gaussian_filter_nan
    # from pynts.tuning_scores.grid_score import autocorr2d

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # tc = nap.compute_tuning_curves(data=cluster, features=shifted, bins=40)[0]
    # tc = gaussian_filter_nan(tc, (2, 2), keep=False, mode="fill")
    ##tc = autocorr2d(tc.values)
    # ax1.imshow(tc.values, interpolation="none")

    # basis = cv.best_estimator_.named_steps["basis"]
    # xs = np.linspace(range[0][0], range[0][1], 100)
    # ys = np.linspace(range[1][0], range[1][1], 100)
    # X, Y = np.meshgrid(xs, ys)
    # grid = np.column_stack([X.ravel(), Y.ravel()])
    # h = cv.best_estimator_.predict(grid).reshape(100, 100)
    # ax2.imshow(h, interpolation="none")
    # plt.show()

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
