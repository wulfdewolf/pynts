import numpy as np
import pandas as pd
import pynapple as nap


def compute_straightness_index(
    session: dict,
    session_type: str,
    clusters: nap.TsGroup,
    window_sec: float = 2.0,
) -> pd.DataFrame:
    """Average straightness over overlapping fixed-duration windows."""
    del clusters, session_type

    if not np.isfinite(window_sec) or window_sec <= 0:
        raise ValueError("window_sec must be finite and positive.")

    positions = pd.concat(
        [
            session["P_x"].as_series().rename("x"),
            session["P_y"].as_series().rename("y"),
        ],
        axis=1,
    ).dropna()

    times = positions.index.to_numpy(dtype=float)
    xy = positions.to_numpy(dtype=float)
    straightness = np.array([], dtype=float)

    if len(times) >= 2:
        distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumulative_distance = np.concatenate(([0.0], np.cumsum(distances)))

        # Keep complete windows and interpolate their exact endpoints.
        starts = np.flatnonzero(times + window_sec <= times[-1])
        end_times = times[starts] + window_sec
        end_positions = np.column_stack(
            [np.interp(end_times, times, xy[:, axis]) for axis in range(2)]
        )

        path_lengths = (
            np.interp(end_times, times, cumulative_distance)
            - cumulative_distance[starts]
        )
        displacements = np.linalg.norm(end_positions - xy[starts], axis=1)

        valid = path_lengths > 0
        straightness = np.clip(
            displacements[valid] / path_lengths[valid],
            0.0,
            1.0,
        )

    return pd.DataFrame(
        [
            {
                "window_sec": float(window_sec),
                "straightness_index": (
                    float(straightness.mean()) if straightness.size else np.nan
                ),
                "n_samples": int(straightness.size),
            }
        ]
    )
