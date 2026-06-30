from functools import reduce
from itertools import combinations
from typing import Callable, Optional

import nemos as nmo
import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike
from scipy.stats import wilcoxon
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from pynts.glms.util import FANCY_LABELS, get_basis, interpolate, make_feature
from pynts.util import wrap_list


def fit_glm_classify(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    bounds: dict,
    epoch: Optional[nap.IntervalSet] = None,
    bin_size_sec: float = 0.02,
    alpha: float = 0.05,
):
    if epoch is None:
        epoch = cluster.time_support.intersect(session["S"].time_support)

    # Prepare output
    y = cluster.count(bin_size_sec)[:, 0].restrict(epoch)

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
        theta_channel = next(
            theta_channel
            for theta_channel in session["theta"]["channel_name"]
            if cluster["extremum_channel"].item() in theta_channel
        )
        theta = session["theta"]
        features["T"] = make_feature(
            "T", theta[:, theta["channel_name"] == theta_channel], bounds["T"], y, epoch
        )

    # Define data splits
    splits = epoch.split((epoch.tot_length() - 0.01) / 20)
    train_idx = ~np.isnan(splits[::2].intersect(session["moving"]).in_interval(y))
    test_idx = [
        ~np.isnan(test_epoch.intersect(session["moving"]).in_interval(y))
        for test_epoch in splits[1::2]
    ]

    # Fit GLMs
    results = {}
    metric = nmo.observation_models.PoissonObservations().pseudo_r2
    for spec in tqdm(
        [
            list(c)
            for r in range(1, len(features) + 1)
            for c in combinations(features.keys(), r)
        ],
        unit="spec",
    ):
        X = np.concatenate([features[v] for v in spec], axis=1).values

        bases, hyperparams = zip(
            *[get_basis(v, [bounds[_v] for _v in v]) for v in spec]
        )
        basis = reduce(lambda a, b: a + b, bases)
        basis_search_space = {
            f"basis__{v + '__' if len(spec) > 1 and len(v) == 1 else ''}{hyperparam}": search_space
            for v, d in zip(spec, hyperparams)
            for hyperparam, search_space in d.items()
        }

        cv = RandomizedSearchCV(
            Pipeline(
                [
                    ("basis", basis.to_transformer()),
                    ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
                    ("glm", PoissonRegressor(solver="newton-cholesky")),
                ]
            ),
            {**basis_search_space, "glm__alpha": np.logspace(-5, 0, 10)},
            cv=KFold(n_splits=2, shuffle=True, random_state=42),
            scoring=make_scorer(metric),
            n_iter=100,
        )
        with np.errstate(divide="ignore"):
            cv.fit(X[train_idx], y.values[train_idx])

        spec_label = tuple(FANCY_LABELS[v] for v in spec)
        results[spec_label] = {
            "spec": spec_label,
            "scores": [
                cv.best_estimator_.score(X[idx], y.values[idx]) for idx in test_idx
            ],
            "model": cv.best_estimator_,
        }

    # Test
    null_model = DummyRegressor().fit(X[train_idx], y.values[train_idx])
    null_scores = np.array(
        [metric(y.values[idx], null_model.predict(X[idx])) for idx in test_idx]
    )
    results["null"] = {"spec": "null", "scores": null_scores, "model": null_model}
    for result in results.values():
        _, p = wilcoxon(
            result["scores"], null_scores, alternative="greater", zero_method="zsplit"
        )
        result["p_val"] = p

    # -----------------------------
    # Classify
    # -----------------------------
    single_vars = [s for s in results.keys() if len(s) == 1 and s != "null"]
    best_spec = max(single_vars, key=lambda s: np.nanmean(results[s]["scores"]))

    for k in range(2, 5):
        candidates = [
            s for s in results.keys() if len(s) == k and set(best_spec).issubset(s)
        ]

        if not candidates:
            break

        best_candidate = max(candidates, key=lambda s: np.nanmean(results[s]["scores"]))

        pval = wilcoxon(
            results[best_candidate]["scores"],
            results[best_spec]["scores"],
            alternative="greater",
            zero_method="zsplit",
        )[1]

        if pval < alpha:
            best_spec = best_candidate

    best_spec = "null" if results[best_spec]["p_val"] > alpha else best_spec
    for r in results.values():
        r["best_spec"] = best_spec

    return list(results.values())
