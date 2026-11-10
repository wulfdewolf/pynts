import numpy as np
import pandas as pd
import pynapple as nap


def compute_straightness_index(
    session: dict,
    session_type: str,
    clusters: nap.TsGroup,
    window_sec: float = 2.0,
    *,
    window_distance: float | None = None,
    distance_step: float | None = None,
) -> pd.DataFrame:
    """Average straightness over time or fixed-path-length windows."""
    del clusters, session_type

    spatial = window_distance is not None
    window = window_distance if spatial else window_sec

    if not np.isfinite(window) or window <= 0:
        raise ValueError("Window size must be finite and positive.")

    if distance_step is not None and not spatial:
        raise ValueError("distance_step requires window_distance.")

    step = window if distance_step is None else distance_step
    if not np.isfinite(step) or step <= 0:
        raise ValueError("distance_step must be finite and positive.")

    positions = (
        pd.concat(
            [
                session["P_x"].as_series().rename("x"),
                session["P_y"].as_series().rename("y"),
            ],
            axis=1,
        )
        .dropna()
        .sort_index()
    )

    times = positions.index.to_numpy(dtype=float)
    xy = positions.to_numpy(dtype=float)
    straightness = np.array([], dtype=float)

    if len(times) >= 2:
        distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(distances)))

        if spatial:
            # Remove stationary duplicates for spatial interpolation.
            coordinate, indices = np.unique(cumulative, return_index=True)
            coordinates_xy = xy[indices]

            last_start = cumulative[-1] - window
            starts = (
                np.arange(0.0, last_start + step, step)
                if last_start >= 0
                else np.array([], dtype=float)
            )
            starts = starts[starts <= last_start]
        else:
            coordinate = times
            coordinates_xy = xy
            starts = times[times + window <= times[-1]]

        ends = starts + window

        start_positions = np.column_stack(
            [
                np.interp(starts, coordinate, coordinates_xy[:, axis])
                for axis in range(2)
            ]
        )
        end_positions = np.column_stack(
            [np.interp(ends, coordinate, coordinates_xy[:, axis]) for axis in range(2)]
        )

        path_lengths = (
            np.full(starts.size, window)
            if spatial
            else (
                np.interp(ends, times, cumulative)
                - np.interp(starts, times, cumulative)
            )
        )
        displacements = np.linalg.norm(
            end_positions - start_positions,
            axis=1,
        )

        valid = path_lengths > 0
        straightness = np.clip(
            displacements[valid] / path_lengths[valid],
            0.0,
            1.0,
        )

    return pd.DataFrame(
        [
            {
                "window_type": "distance" if spatial else "time",
                "window_sec": np.nan if spatial else float(window),
                "window_distance": float(window) if spatial else np.nan,
                "distance_step": float(step) if spatial else np.nan,
                "straightness_index": (
                    float(straightness.mean()) if straightness.size else np.nan
                ),
                "n_samples": int(straightness.size),
            }
        ]
    )
