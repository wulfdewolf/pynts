from functools import reduce
from itertools import combinations
from typing import Optional

import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from pynts.bounds import get_variable_bounds
from pynts.glms.metrics import evaluate_glm, paired_score_pvalue, poisson_loglik_score
from pynts.glms.util import FANCY_LABELS, get_basis, make_feature


def fit_glm_classify(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    bounds: dict[str, ArrayLike | None] | None = None,
    epoch: Optional[nap.IntervalSet] = None,
    bin_size_sec: float | None = 0.02,
    alpha: float = 0.05,
    n_iter: int = 50,
    n_jobs: int = 1,
) -> list[dict]:
    if epoch is None:
        epoch = cluster.time_support.intersect(session["S"].time_support)

    # Prepare output
    if bin_size_sec is None:
        if not isinstance(cluster, (nap.Tsd, nap.TsdFrame)):
            raise TypeError("Continuous responses require a Tsd or TsdFrame.")

        if isinstance(cluster, nap.TsdFrame):
            if cluster.shape[1] != 1:
                raise ValueError("Expected a single response column.")
            cluster = cluster[:, 0]

        family = "gaussian"
        y = cluster.restrict(epoch)
        regressor = Ridge(solver="svd")
        scorer = "neg_mean_squared_error"
        alpha_grid = np.logspace(-4, 4, 9)
    else:
        family = "poisson"
        y = cluster.count(bin_size_sec)[:, 0].restrict(epoch)
        regressor = PoissonRegressor(solver="newton-cholesky")
        scorer = make_scorer(poisson_loglik_score)
        alpha_grid = [0.00001, 0.0001, 0.001, 0.01]
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
            *[
                get_basis(
                    v,
                    bounds=[bounds[_v] for _v in v],
                    ranges=[bounds[_v][1] - bounds[_v][0] for _v in v],
                )
                for v in spec
            ]
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
                    ("imputer", SimpleImputer(strategy="median")),
                    ("basis", basis.to_transformer()),
                    ("glm", regressor),
                ]
            ),
            {**basis_search_space, "glm__alpha": alpha_grid},
            cv=2,
            scoring=scorer,
            n_jobs=n_jobs,
        )
        with np.errstate(divide="ignore"):
            cv.fit(X[train_idx], y_values[train_idx])

        spec_label = tuple(FANCY_LABELS[v] for v in spec)
        evaluation = evaluate_glm(
            cv.best_estimator_, X, y_values, test_idx, null_mean, family=family
        )

        results[spec_label] = {
            **evaluation,
            "spec": spec_label,
            "cv_score": float(cv.best_score_),
            "model": cv.best_estimator_,
        }

    # -----------------------------
    # Classify
    # -----------------------------
    # start with the simplest model that beats the null
    specs = list(results)

    def performance(spec: tuple[str, ...]) -> float:
        scores = np.asarray(results[spec]["test_score"], dtype=float)
        scores = scores[np.isfinite(scores)]
        return float(np.median(scores)) if scores.size else -np.inf

    def beats_null(spec: tuple[str, ...]) -> bool:
        return results[spec]["p_value"] < alpha and results[spec]["median_test_pr2"] > 0

    # Test each model against the null
    for spec in specs:
        p_value = float(results[spec]["mean_null_p_value"])
        results[spec]["p_value"] = p_value if np.isfinite(p_value) else 1.0

    # Compare every nested pair differing by one variable
    edges = [
        (parent, child)
        for parent in specs
        for child in specs
        if len(child) == len(parent) + 1 and set(parent) < set(child)
    ]

    edge_p_values = {}
    for parent, child in edges:
        p_value = paired_score_pvalue(
            results[child]["test_score"],
            results[parent]["test_score"],
        )
        edge_p_values[parent, child] = p_value if np.isfinite(p_value) else 1.0

    # Try singles, then pairs, etc., until a model beats the null
    best_spec = None
    for size in sorted({len(spec) for spec in specs}):
        candidates = [
            spec
            for spec in specs
            if len(spec) == size and beats_null(spec) and np.isfinite(performance(spec))
        ]
        if candidates:
            best_spec = max(candidates, key=performance)
            break

    # Add variables only when they improve on the current model
    best_label = "null"
    if best_spec is not None:
        while True:
            candidates = [
                child
                for parent, child in edges
                if parent == best_spec
                and beats_null(child)
                and edge_p_values[parent, child] < alpha
                and performance(child) > performance(parent)
            ]
            if not candidates:
                break

            best_spec = max(candidates, key=performance)

        best_label = "".join(best_spec)

    return [
        {
            "spec": "".join(spec),
            "best_spec": best_label,
            "test_pr2": result["median_test_pr2"],
            "test_score": float(np.nanmedian(result["test_score"])),
            "model": result["model"],
            "p_val": result["mean_null_p_value"],
            "cv_score": result["cv_score"],
        }
        for spec, result in results.items()
    ]
