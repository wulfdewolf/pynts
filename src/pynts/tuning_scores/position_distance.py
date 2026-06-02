from typing import Callable

import numpy as np
import pandas as pd
import pynapple as nap
from numpy.typing import ArrayLike


def compute_position_distance(
    session: dict,
    session_type: str,
    clusters: nap.TsGroup,
    projection: Callable,
    projection_range: ArrayLike,
    is_shuffle: bool = False,
):

    results = []
    for shift in projection_range:
        shifted = projection(
            session_type,
            session,
            ("P_x", "P_y"),
            shift,
        )

        time_support = session["P_x"].time_support.intersect(shifted.time_support)
        shifted = shifted.restrict(time_support)

        x = session["P_x"].restrict(time_support).values
        y = session["P_y"].restrict(time_support).values
        shifted_x = shifted["P_x"].values
        shifted_y = shifted["P_y"].values

        # Euclidean distance at each sample
        distance = np.sqrt((shifted_x - x) ** 2 + (shifted_y - y) ** 2)

        results.append(
            {
                "shift": shift,
                "dist": np.nanmean(distance),
            }
        )

    return pd.DataFrame(results)
