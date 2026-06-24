from typing import Callable, Optional

import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
from nemos.basis import BSplineEval, CyclicBSplineEval
from numpy.typing import ArrayLike
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from pynts import wrappers
from pynts.util import wrap_list
from pynts.wrappers import compute_travel_projected


def interpolate(var, y, other):
    if var == "H":
        return type(y)(
            d=np.arctan2(
                np.sin(y).restrict(other.time_support).interpolate(other).values,
                np.cos(y).restrict(other.time_support).interpolate(other).values,
            ),
            t=other.times(),
            time_support=other.time_support,
        )
    else:
        y.interpolate(other)


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


def get_basis(var, bounds):
    range = max(b[1] - b[0] for b in bounds)
    print(range)

    if var == ("P_x", "P_y"):
        basis = (
            BSplineEval(n_basis_funcs=10, label="P_x", bounds=bounds[0])
            * BSplineEval(n_basis_funcs=10, label="P_y", bounds=bounds[1])
        ).to_transformer()
        hyperparams = {
            "basis__P_x__n_basis_funcs": np.arange(5, int(0.5 * range), 1),
            "basis__P_y__n_basis_funcs": np.arange(5, int(0.5 * range), 1),
        }
    elif var == "S":
        basis = BSplineEval(
            n_basis_funcs=10, label="S", bounds=bounds[0]
        ).to_transformer()
        hyperparams = {
            "basis__n_basis_funcs": np.arange(5, int(0.5 * range), 1),
        }
    elif var == "H":
        basis = CyclicBSplineEval(
            n_basis_funcs=10, label="H", bounds=bounds[0]
        ).to_transformer()
        hyperparams = {
            "basis__n_basis_funcs": np.arange(5, int(0.5 * np.degrees(range)), 1),
        }
    elif var == "grid":
        basis = GridBasis()
        hyperparams = {
            "basis__spacing": np.arange(0.2 * range, 0.5 * range, 2),
            "basis__orientation": np.linspace(
                0,
                np.pi / 3,
                30,
                endpoint=False,
            ),
        }
    else:
        raise ValueError(f"Unknown variable to fit GLM for {var}.")

    return basis, hyperparams


def fit_glm(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    var: str | ArrayLike,
    epoch: Optional[nap.IntervalSet] = None,
    bin_size_sec: float = 0.02,
    bounds: Optional[ArrayLike] = None,
    projection: Callable = compute_travel_projected,
    projection_range: Optional[ArrayLike] = None,
):
    if epoch is None:
        epoch = cluster.time_support.intersect(session["S"].time_support)

    if projection_range is None:
        projection_range = [0]

    # Extract bounds and range if not given
    bounds = (
        [(np.nanmin(session[v]), np.nanmax(session[v])) for v in wrap_list(var)]
        if bounds is None
        else bounds
    )

    # Prepare bases
    basis, hyperparams = get_basis(var, bounds)
    if var == "grid":
        var = ("P_x", "P_y")

    # Prepare input/output
    y = cluster.count(bin_size_sec)[:, 0]
    shifted_behaviour = {
        shift: interpolate(
            var,
            projection(
                session_type,
                session,
                var,
                shift,
            ),
            y,
        ).restrict(epoch)
        for shift in projection_range
    }
    y = y.restrict(epoch)

    # Define data splits
    splits = epoch.split((epoch.tot_length() - 0.01) / 20)
    train_idx = ~np.isnan(splits[::2].intersect(session["moving"]).in_interval(y))
    test_idx = [
        ~np.isnan(test_epoch.intersect(session["moving"]).in_interval(y))
        for test_epoch in splits[1::2]
    ]

    # Fit GLMs
    results = []
    metric = nmo.observation_models.PoissonObservations().pseudo_r2
    for shift, shifted in tqdm(shifted_behaviour.items()):
        cv = RandomizedSearchCV(
            Pipeline(
                [
                    ("basis", basis),
                    ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
                    ("glm", PoissonRegressor()),
                ]
            ),
            {**hyperparams, "glm__alpha": np.logspace(-5, 0, 10)},
            cv=KFold(n_splits=2, shuffle=True, random_state=42),
            scoring=make_scorer(metric),
            n_iter=40,
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
