from jax.experimental.pallas.tpu import GridDimensionSemantics
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

from pynts import wrappers


from sklearn.base import BaseEstimator, TransformerMixin


class GridBasis(BaseEstimator, TransformerMixin):
    """
    Hexagonal Fourier basis with harmonics.

    Parameters
    ----------
    spacing : float
        Grid spacing in same units as position.
    orientation : float
        Grid orientation in radians.
    n_harmonics : int
        Number of harmonics to include.
    """

    def __init__(
        self,
        spacing=40.0,
        orientation=0.0,
        n_harmonics=3,
    ):
        self.spacing = spacing
        self.orientation = orientation
        self.n_harmonics = n_harmonics

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)

        x = X[:, 0]
        y = X[:, 1]

        k = 2 * np.pi / self.spacing

        features = []

        for h in range(1, self.n_harmonics + 1):
            kh = h * k

            for theta in (
                self.orientation,
                self.orientation + np.pi / 3,
                self.orientation + 2 * np.pi / 3,
            ):
                phase = kh * (x * np.cos(theta) + y * np.sin(theta))

                features.append(np.cos(phase))
                features.append(np.sin(phase))

        return np.column_stack(features)

    @property
    def n_features_out_(self):
        return 6 * self.n_harmonics


def fit_predictive_glm(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    epoch: Optional[nap.IntervalSet] = None,
    bin_size_sec: float = 0.02,
    projection_range: ArrayLike = [0],  # np.arange(-30, 31, 1),
    shift_type: str = "travel",
):
    if epoch is None:
        epoch = cluster.time_support.intersect(session["S"].time_support)

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
    basis = BSplineEval(n_basis_funcs=10, label="P_x", bounds=(0, 100)) * BSplineEval(
        n_basis_funcs=10, label="P_y", bounds=(0, 100)
    )
    grid_basis = GridBasis(30, 60)
    hyperparams = {
        "basis__spacing": np.arange(20, 80, 2),
        "basis__orientation": np.linspace(
            0,
            np.pi / 3,
            30,
            endpoint=False,
        ),
        "basis__n_harmonics": [1, 2, 3, 4],
    }
    # hyperparams = {
    #    "basis__P_x__n_basis_funcs": np.arange(5, 21, 1),
    #    "basis__P_y__n_basis_funcs": np.arange(5, 21, 1),
    # }
    for shift, shifted in shifted_position.items():
        cv = RandomizedSearchCV(
            Pipeline(
                [
                    ("basis", grid_basis),
                    (
                        "glm",
                        # nmo.glm.GLM(regularizer="Ridge", solver_name="Newton"),
                        TweedieRegressor(),
                    ),
                ]
            ),
            {
                **hyperparams,
                "glm__alpha": np.logspace(-5, 0, 20),
                "glm__power": np.linspace(0.0, 1.0, 20),
            },
            cv=KFold(n_splits=4, shuffle=True, random_state=42),
            scoring=make_scorer(metric),
            n_iter=10,
            verbose=1,
            n_jobs=10,
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
        print(results[-1])

    from pynts.smoothing import gaussian_filter_nan
    import matplotlib.pyplot as plt

    tcs = nap.compute_tuning_curves(data=cluster, features=shifted, bins=40)
    tcs = gaussian_filter_nan(tcs, (2, 2), mode="fill", keep=False)
    tcs.plot()
    plt.show()

    xs = np.linspace(0, 100, 200)
    ys = np.linspace(0, 100, 200)

    XX, YY = np.meshgrid(xs, ys)

    pos = np.column_stack(
        [
            XX.ravel(),
            YY.ravel(),
        ]
    )

    Phi = grid_basis.transform(pos)

    field = Phi @ cv.best_estimator_.named_steps["glm"].coef_
    field = field.reshape(XX.shape)
    import matplotlib.pyplot as plt

    plt.imshow(
        field,
        origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
    )
    plt.colorbar()
    plt.show()
    quit()

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
