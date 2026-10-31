import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike

from pynts.bounds import ANGULAR_VARIABLES, apply_variable_bounds, get_variable_bounds
from pynts.util import wrap_list


def compute_travel_projected(
    session_type: str,
    session: dict,
    var_label: str | list[str],
    travel: float,
    bounds: ArrayLike | None = None,
) -> nap.TsdFrame:
    """Project variables along cumulative distance traveled."""
    variables = list(wrap_list(var_label))
    reference = session[variables[0]]
    values = np.column_stack(
        [session[variable].values for variable in variables]
    ).astype(float)

    # Unwrap before filling gaps or interpolating along distance.
    for dim, variable in enumerate(variables):
        if variable in ANGULAR_VARIABLES:
            valid = np.isfinite(values[:, dim])
            values[valid, dim] = np.unwrap(values[valid, dim])

    frame = nap.TsdFrame(
        t=reference.times(),
        d=values,
        columns=variables,
        time_support=reference.time_support,
    )
    frame = frame.as_dataframe().interpolate().dropna()

    var = nap.TsdFrame(
        t=frame.index.to_numpy(),
        d=frame.to_numpy(),
        columns=variables,
        time_support=reference.time_support,
    )

    if "VR" in session_type:
        position = session["travel"]
    else:
        position = nap.TsdFrame(
            t=session["P_x"].times(),
            d=np.column_stack(
                [
                    session["P_x"].values,
                    session["P_y"].values,
                ]
            ),
            columns=["P_x", "P_y"],
            time_support=session["P_x"].time_support,
        )

    position_frame = position.as_dataframe().interpolate().dropna()
    position = nap.TsdFrame(
        t=position_frame.index.to_numpy(),
        d=position_frame.to_numpy(),
        time_support=position.time_support,
    )
    position = position.restrict(var.time_support).interpolate(var)

    positions = np.asarray(position.values, dtype=float)
    if positions.ndim == 1:
        positions = positions[:, None]

    deltas = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative_distance = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    target_distance = np.clip(
        cumulative_distance + travel,
        cumulative_distance[0],
        cumulative_distance[-1],
    )

    # Repeated distances occur during pauses. Use their last sample.
    last_at_distance = np.concatenate((np.diff(cumulative_distance) > 0, [True]))
    source_distance = cumulative_distance[last_at_distance]
    source_values = var.values[last_at_distance]

    projected = np.empty_like(var.values, dtype=float)

    for dim in range(len(variables)):
        projected[:, dim] = np.interp(
            target_distance,
            source_distance,
            source_values[:, dim],
        )

    # Preserve the original samples when no projection is requested.
    if travel == 0:
        projected = var.values.copy()

    bounds = get_variable_bounds(session, variables, bounds)
    projected = apply_variable_bounds(projected, variables, bounds)

    return nap.TsdFrame(
        t=var.times(),
        d=projected,
        columns=variables,
        time_support=var.time_support,
    )


def compute_time_projected(
    session_type: str,
    session: dict,
    var_label: str | list[str],
    shift: float,
    bounds: ArrayLike | None = None,
) -> nap.TsdFrame:
    """Shift regularly sampled variables in time."""
    variables = list(wrap_list(var_label))
    reference = session[variables[0]]
    times = reference.times()
    values = np.column_stack([session[variable].values for variable in variables])

    shift_bins = int(shift * reference.rate)

    if shift_bins > 0:
        times = times[:-shift_bins]
        values = values[shift_bins:]
    elif shift_bins < 0:
        times = times[-shift_bins:]
        values = values[:shift_bins]

    bounds = get_variable_bounds(session, variables, bounds)
    values = apply_variable_bounds(values, variables, bounds)

    # Infer support from the retained timestamps, not the full recording.
    return nap.TsdFrame(
        t=times,
        d=values,
        columns=variables,
    )


def compute_direction_projected(
    session_type: str,
    session: dict,
    var_label: str | list[str],
    travel: float,
    bounds: ArrayLike | None = None,
) -> nap.TsdFrame:
    """Project 2D position along the current movement direction."""
    variables = list(wrap_list(var_label))

    if variables != ["P_x", "P_y"]:
        raise ValueError("Direction projection requires ['P_x', 'P_y'].")

    x = np.asarray(session["P_x"].values, dtype=float)
    y = np.asarray(session["P_y"].values, dtype=float)

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    displacement = np.hypot(dx, dy)

    direction = np.zeros((len(x), 2), dtype=float)
    valid = np.isfinite(displacement) & (displacement > 0)

    direction[valid, 0] = dx[valid] / displacement[valid]
    direction[valid, 1] = dy[valid] / displacement[valid]

    # Retain the last movement direction during pauses.
    last_valid = np.maximum.accumulate(np.where(valid, np.arange(len(x)), -1))
    has_direction = last_valid >= 0
    direction[has_direction] = direction[last_valid[has_direction]]

    projected = np.column_stack([x, y]) + travel * direction
    bounds = get_variable_bounds(session, variables, bounds)
    projected = apply_variable_bounds(projected, variables, bounds)

    return nap.TsdFrame(
        t=session["P_x"].times(),
        d=projected,
        columns=variables,
        time_support=session["P_x"].time_support,
    )
