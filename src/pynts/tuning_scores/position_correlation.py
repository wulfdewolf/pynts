from typing import Callable

import pandas as pd
import pynapple as nap
from numpy.typing import ArrayLike


def compute_position_correlation(
    session: dict,
    session_type: str,
    clusters: nap.TsGroup,
    projection: Callable,
    projection_range: ArrayLike,
    is_shuffle: bool = False,
):

    results = []
    for shift in projection_range:
        shifted = projection(session_type, session, ("P_x", "P_y"), shift)

        time_support = session["P_x"].time_support.intersect(shifted.time_support)
        shifted = shifted.restrict(time_support)

        x = session["P_x"].restrict(time_support).as_series()
        y = session["P_y"].restrict(time_support).as_series()
        shifted_x = shifted["P_x"].as_series()
        shifted_y = shifted["P_y"].as_series()

        results.append(
            {"shift": shift, "corr_x": x.corr(shifted_x), "corr_y": y.corr(shifted_y)}
        )

    return pd.DataFrame(results)
