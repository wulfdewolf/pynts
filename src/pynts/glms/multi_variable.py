from functools import reduce
from itertools import combinations
from typing import Optional

import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from pynts.bounds import get_variable_bounds
from pynts.glms.metrics import (
    evaluate_glm,
    paired_loglik_pvalue,
    poisson_loglik_score,
)
from pynts.glms.util import FANCY_LABELS, get_basis, make_feature


def fit_glm_classify(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    bounds: dict[str, ArrayLike | None] | None = None,
    epoch: Optional[nap.IntervalSet] = None,
    bin_size_sec: float = 0.02,
    alpha: float = 0.05,
    n_iter: int = 50,
    n_jobs: int = 1,
) -> list[dict]:
    if epoch is None:
        epoch = cluster.time_support.intersect(session["S"].time_support)

    # Prepare output
    y = cluster.count(bin_size_sec)[:, 0].restrict(epoch)
    y_values = np.ascontiguousarray(y.values, dtype=np.float64)

    # Resolve bounds
    supplied_bounds = {} if bounds is None else bounds
    variables = ["P", "S"] if session_type == "VR" else ["P_x", "P_y", "S", "H"]
    bounds_session = session

    if "theta" in session:
        variables.append("T")
        bounds_session = {**session, "T": session["theta"]}

    bounds = {}
    for variable in variables:
        supplied = supplied_bounds.get(variable)
        bounds[variable] = get_variable_bounds(
            bounds_session,
            [variable],
            None if supplied is None else [supplied],
        )[0]

    # Prepare input
    features = {
        var: np.concatenate(
            [make_feature(v, session[v], bounds[v], y, epoch) for v in var], axis=1
        )
        for var in (
            [("P"), ("S")] if session_type == "VR" else [("P_x", "P_y"), ("S"), ("H")]
        )
    }
    if "theta" in session:
        theta = session["theta"]
        if theta.values.ndim == 2:
            theta_channel = next(
                theta_channel
                for theta_channel in session["theta"]["channel_name"]
                if cluster["extremum_channel"].item() in theta_channel
            )
            theta = theta[:, theta["channel_name"] == theta_channel]
        features["T"] = make_feature("T", theta, bounds["T"], y, epoch)

    # Define data splits
    splits = epoch.split((epoch.tot_length() - 0.01) / 20)
    train_idx = ~np.isnan(splits[:10].intersect(session["moving"]).in_interval(y))
    test_idx = [
        ~np.isnan(test_epoch.intersect(session["moving"]).in_interval(y))
        for test_epoch in splits[10:]
    ]

    # Fit GLMs
    results = {}
    scorer = make_scorer(poisson_loglik_score)
    null_mean = float(y_values[train_idx].mean())

    for spec in tqdm(
        [
            list(c)
            for r in range(1, len(features) + 1)
            for c in combinations(features.keys(), r)
        ],
        unit="spec",
    ):
        X = np.ascontiguousarray(
            np.concatenate([features[v] for v in spec], axis=1), dtype=np.float64
        )

        bases, hyperparams = zip(
            *[get_basis(v, [bounds[_v] for _v in v]) for v in spec]
        )
        basis = reduce(lambda a, b: a + b, bases)
        basis_search_space = {
            f"{v + '__' if len(spec) > 1 and len(v) == 1 else ''}{hyperparam}": search_space
            for v, d in zip(spec, hyperparams)
            for hyperparam, search_space in d.items()
        }

        cv = GridSearchCV(
            Pipeline(
                [
                    ("basis", basis.to_transformer()),
                    ("glm", PoissonRegressor(solver="newton-cholesky")),
                ]
            ),
            {**basis_search_space, "glm__alpha": [0.00001, 0.0001, 0.001, 0.01]},
            cv=2,
            scoring=scorer,
            n_jobs=n_jobs,
        )
        with np.errstate(divide="ignore"):
            cv.fit(X[train_idx], y_values[train_idx])

        spec_label = tuple(FANCY_LABELS[v] for v in spec)
        evaluation = evaluate_glm(
            cv.best_estimator_,
            X,
            y_values,
            test_idx,
            null_mean,
        )

        results[spec_label] = {
            **evaluation,
            "spec": spec_label,
            "cv_loglik": float(cv.best_score_),
            "model": cv.best_estimator_,
        }

    # -----------------------------
    # Classify
    # -----------------------------
    single_vars = [spec for spec in results if len(spec) == 1]
    best_spec = max(single_vars, key=lambda s: results[s]["median_test_pr2"])

    for k in range(2, 5):
        candidates = [
            s for s in results.keys() if len(s) == k and set(best_spec).issubset(s)
        ]

        if not candidates:
            break

        best_candidate = max(candidates, key=lambda s: results[s]["median_test_pr2"])

        pval = paired_loglik_pvalue(
            results[best_candidate]["test_loglik"],
            results[best_spec]["test_loglik"],
        )

        if pval < alpha:
            best_spec = best_candidate

    best_label = "null" if best_spec == "null" else " + ".join(best_spec)

    return [
        {
            "spec": "".join(spec),
            "best_spec": best_label,
            "test_pr2": result["median_test_pr2"],
            "test_loglik": np.nanmedian(result["test_loglik"]),
            "p_val": result["paired_p_value"],
            "cv_loglik": result["cv_loglik"],
        }
        for spec, result in results.items()
    ]
