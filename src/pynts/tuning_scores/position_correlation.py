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

        results.append(
            {
                "shift": shift,
                "corr_x": shifted["P_x"].as_series().corr(session["P_x"].as_series()),
                "corr_y": shifted["P_y"].as_series().corr(session["P_y"].as_series()),
            }
        )

    return pd.DataFrame(results)
