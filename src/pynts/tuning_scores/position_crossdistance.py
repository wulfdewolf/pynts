import numpy as np
import pandas as pd
import pynapple as nap
from numpy.typing import ArrayLike

from pynts.wrappers import compute_time_projected, compute_travel_projected


def compute_position_crossdistance(
    session: dict,
    session_type: str,
    clusters: nap.TsGroup,
    projection_range_travel: ArrayLike,
    projection_range_time: ArrayLike,
    is_shuffle: bool = False,
):

    # ----------------------------------------------------------
    # Precompute all projected positions
    # ----------------------------------------------------------
    travel_positions = {}
    time_positions = {}

    for shift in projection_range_travel:
        shifted = compute_travel_projected(
            session_type,
            session,
            ("P_x", "P_y"),
            shift,
        )

        travel_positions[shift] = (shifted["P_x"], shifted["P_y"])

    for shift in projection_range_time:
        shifted = compute_time_projected(
            session_type,
            session,
            ("P_x", "P_y"),
            shift,
        )

        time_positions[shift] = (shifted["P_x"], shifted["P_y"])

    # ----------------------------------------------------------
    # Compute full pairwise matrix
    # ----------------------------------------------------------
    results = []

    for travel_shift in projection_range_travel:
        x_travel, y_travel = travel_positions[travel_shift]

        for time_shift in projection_range_time:
            x_time, y_time = time_positions[time_shift]

            time_support = x_travel.time_support.intersect(x_time.time_support)
            _x_travel = x_travel.restrict(time_support).values
            _y_travel = y_travel.restrict(time_support).values
            x_time = x_time.restrict(time_support).values
            y_time = y_time.restrict(time_support).values

            distance = np.hypot(
                _x_travel - x_time,
                _y_travel - y_time,
            )

            results.append(
                {
                    "travel_shift": travel_shift,
                    "time_shift": time_shift,
                    "dist": np.nanmean(distance),
                }
            )

    return pd.DataFrame(results)
