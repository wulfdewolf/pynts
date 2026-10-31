import numpy as np
from numpy.typing import ArrayLike

from pynts.util import wrap_list

ANGULAR_VARIABLES = {"H", "T"}

BOUND_STEPS = {
    "P_x": 5.0,
    "P_y": 5.0,
    "P": 5.0,
    "S": 1.0,
}


def get_variable_bounds(
    session: dict,
    variables: str | list[str],
    bounds: ArrayLike | None = None,
    rounding_steps: dict[str, float] | None = None,
) -> np.ndarray:
    """Infer angular domains or outward-rounded linear bounds."""
    variables = list(wrap_list(variables))

    if bounds is not None:
        return np.asarray(bounds, dtype=float).reshape(len(variables), 2)

    steps = {**BOUND_STEPS, **(rounding_steps or {})}
    inferred = []

    for variable in variables:
        values = np.asarray(session[variable].values, dtype=float)
        values = values[np.isfinite(values)]
        lower, upper = values.min(), values.max()

        if variable in ANGULAR_VARIABLES:
            tolerance = 1e-8

            if lower < -np.pi - tolerance or upper > 2 * np.pi + tolerance:
                raise ValueError(f"{variable} is outside supported radian domains.")

            if lower < -tolerance:
                if upper > np.pi + tolerance:
                    raise ValueError(f"{variable} mixes angular conventions.")
                inferred.append((-np.pi, np.pi))
            else:
                # Observations within [0, pi] are ambiguous.
                inferred.append((0.0, 2 * np.pi))
        else:
            step = steps.get(variable, 1.0)
            inferred.append(
                (
                    np.floor(lower / step) * step,
                    np.ceil(upper / step) * step,
                )
            )

    return np.asarray(inferred, dtype=float)


def apply_variable_bounds(
    values: ArrayLike,
    variables: str | list[str],
    bounds: ArrayLike,
) -> np.ndarray:
    """Return 2D values with linear clipping and angular wrapping."""
    variables = list(wrap_list(variables))
    values = np.array(values, dtype=float, copy=True)

    if values.ndim == 1:
        values = values[:, None]

    bounds = np.asarray(bounds, dtype=float).reshape(len(variables), 2)

    for dim, variable in enumerate(variables):
        lower, upper = bounds[dim]

        if variable in ANGULAR_VARIABLES:
            values[:, dim] = (values[:, dim] - lower) % (upper - lower) + lower
        else:
            values[:, dim] = np.clip(values[:, dim], lower, upper)

    return values
