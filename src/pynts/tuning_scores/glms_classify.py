from itertools import combinations
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


def get_basis_config(session_type):
    if "OF" in session_type:
        return {
            "P": {
                "basis": BSplineEval(n_basis_funcs=5, label="P_x")
                * BSplineEval(n_basis_funcs=5, label="P_y"),
                "hyperparams": {
                    "P_x__n_basis_funcs": np.arange(5, 21, 1),
                    "P_y__n_basis_funcs": np.arange(5, 21, 1),
                },
            },
            "S": {
                "basis": BSplineEval(
                    n_basis_funcs=5,
                    bounds=(3.0, 80.0),
                    label="S",
                ),
                "hyperparams": {
                    "n_basis_funcs": np.arange(5, 31, 5),
                },
            },
            "H": {
                "basis": CyclicBSplineEval(
                    n_basis_funcs=5, bounds=(-np.pi, np.pi), label="H"
                ),
                "hyperparams": {
                    "n_basis_funcs": np.arange(5, 31, 5),
                },
            },
            "T": {
                "basis": CyclicBSplineEval(
                    n_basis_funcs=5, bounds=(-np.pi, np.pi), label="T"
                ),
                "hyperparams": {
                    "n_basis_funcs": np.arange(5, 31, 5),
                },
            },
        }
    else:
        raise ValueError(f"Unknown session type: {session_type}. Only OF is supported.")


def interpolate(var, y, other):
    if var == "H" or var == "T":
        return nap.TsdFrame(
            d=np.arctan2(
                np.sin(y).restrict(other.time_support).interpolate(other).values,
                np.cos(y).restrict(other.time_support).interpolate(other).values,
            ),
            t=other.times(),
            time_support=other.time_support,
        )
    else:
        return nap.TsdFrame(
            t=other.times(),
            d=y.interpolate(other)[:, None].values,
            time_support=other.time_support,
        )


def fit_glm_classify(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    epoch: Optional[nap.IntervalSet] = None,
    bin_size_sec: float = 0.02,
    alpha: float = 0.05,
):
    if epoch is None:
        epoch = cluster.time_support.intersect(session["S"].time_support)

    # Prepare output
    y = cluster.count(bin_size_sec)[:, 0]
    y = y.restrict(epoch)

    # Prepare input
    basis_config = get_basis_config(session_type)

    def _make_feature(v):
        key = v.split("_")[0]
        config = basis_config[key]
        bounds = config["basis"][v].bounds
        return (
            interpolate(v, session[v], y)
            .restrict(epoch)
            .clip(*((None, None) if bounds is None else bounds))
        )

    features = {
        var[0].split("_")[0]: np.concatenate([_make_feature(v) for v in var], axis=1)
        for var in (
            [("P"), ("S")] if session_type == "VR" else [("P_x", "P_y"), ("S"), ("H")]
        )
    }
    if "theta" in session:
        theta_channel = next(
            theta_channel
            for theta_channel in session["theta"]["channel_name"]
            if cluster["extremum_channel"].item() in theta_channel
        )
        theta = session["theta"]
        features["T"] = interpolate(
            "T", theta[:, theta["channel_name"] == theta_channel], y
        )

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
    for spec in [
        list(c)
        for r in range(1, len(features) + 1)
        for c in combinations(features.keys(), r)
    ]:
        X = np.concatenate([features[v] for v in spec], axis=1).values
        basis = sum(
            (basis_config[v]["basis"] for v in spec[1:]),
            basis_config[spec[0]]["basis"],
        )
        basis_search_space = {
            f"basis__{v + '__' if isinstance(basis, nmo.basis.AdditiveBasis) and v != 'P' else ''}{hyperparam}": values
            for v in spec
            for hyperparam, values in basis_config[v]["hyperparams"].items()
        }
        cv = RandomizedSearchCV(
            Pipeline(
                [
                    ("basis", basis.to_transformer()),
                    ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
                    ("glm", PoissonRegressor()),
                ]
            ),
            {**basis_search_space, "glm__alpha": np.logspace(-5, 0, 10)},
            cv=KFold(n_splits=2, shuffle=True, random_state=42),
            scoring=make_scorer(metric),
            n_iter=50,
        )
        with np.errstate(divide="ignore"):
            cv.fit(X[train_idx], y.values[train_idx])

        results.append(
            {
                "spec": spec,
                "score": [
                    cv.best_estimator_.score(X[idx], y.values[idx]) for idx in test_idx
                ],
                "model": cv.best_estimator_,
            }
        )

    # Test
    null_model = DummyRegressor().fit(X[train_idx], y.values[train_idx])
    null_scores = np.array(
        [metric(y.values[idx], null_model.predict(X[idx])) for idx in test_idx]
    )
    results.append({"spec": ["null"], "score": null_scores, "model": null_model})
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

    # -----------------------------
    # Classify
    # -----------------------------
    spec_to_scores = {
        tuple(
            sorted([v for v in r["spec"] if v is not None and v != "null"])
        ): np.array(r["score"])
        for r in results
    }
    single_vars = [s for s in spec_to_scores.keys() if len(s) == 1]
    best_spec = max(single_vars, key=lambda s: spec_to_scores[s].mean())

    for k in range(2, 5):
        candidates = [
            s
            for s in spec_to_scores.keys()
            if len(s) == k and set(best_spec).issubset(s)
        ]

        if not candidates:
            break

        best_candidate = max(candidates, key=lambda s: spec_to_scores[s].mean())

        pval = wilcoxon(
            spec_to_scores[best_candidate],
            spec_to_scores[best_spec],
            alternative="greater",
            zero_method="zsplit",
        )[1]

        if pval < alpha:
            best_spec = best_candidate

    for r in results:
        r["best_spec"] = best_spec

    return results
