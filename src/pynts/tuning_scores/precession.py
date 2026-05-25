from typing import Optional, Tuple

import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike
from pycircstat2.correlation import circ_corrcl
from pycircstat2.regression import CLRegression
from scipy import ndimage as ndi
from scipy.stats import circmean
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from pynts.smoothing import apply_smoothing
from pynts.wrappers import compute_travel_projected


def compute_precession(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    range: Optional[ArrayLike] = None,
    num_bins: int = 60,
    smooth_sigma: float | ArrayLike = 2,
    epoch: Optional[nap.IntervalSet] = None,
    min_spikes: int = 100,
    direction: str | int = "movement",
    precession_range: Tuple[int, int] = (-50, 50),
    is_shuffle: bool = False,
):
    """
    Global phase precession analysis using continuous
    signed in-field position.

    In-field position is computed by:
        1. Finding nearest field center
        2. Computing vector from animal -> center
        3. Projecting onto directional vector
    """
    results = {
        "slope": np.nan,
        "direction": direction,
        "spike_pos": [],
        "spike_phase": [],
    }

    if "theta" not in session:
        return results

    if epoch is None:
        epoch = cluster.time_support

    # ------------------------------------------------------------
    # Position

    if range is None:
        range = [
            (np.nanmin(session["P_x"]), np.nanmax(session["P_x"])),
            (np.nanmin(session["P_y"]), np.nanmax(session["P_y"])),
        ]

    P = np.stack([session["P_x"], session["P_y"]], axis=1)

    # ------------------------------------------------------------
    # Tuning curve

    def compute_tuning_curve(ep):
        return nap.compute_tuning_curves(
            cluster,
            P,
            bins=num_bins,
            range=range,
            epochs=ep.intersect(session["moving"]),
        )[0]

    tc, _ = apply_smoothing(
        compute_tuning_curve,
        epoch=epoch,
        dim=2,
        smooth_sigma=smooth_sigma,
        sigma_range=np.linspace(1, 20, 20),
        mode="fill",
        keep=False,
    )

    # ------------------------------------------------------------
    # Detect field centers

    peaks = peak_local_max(
        tc.values,
        min_distance=4,
        threshold_rel=0.3,
    )

    if len(peaks) == 0:
        return results

    x_edges = np.linspace(range[0][0], range[0][1], tc.shape[0] + 1)
    y_edges = np.linspace(range[1][0], range[1][1], tc.shape[1] + 1)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    field_centers = np.column_stack(
        (
            x_centers[peaks[:, 0]],
            y_centers[peaks[:, 1]],
        )
    )

    # ------------------------------------------------------------
    # Direction vectors at ALL time points

    if direction == "movement":
        vel = np.zeros_like(P)
        vel[1:] = np.diff(P, axis=0) / np.diff(P.times())[:, None]

        with np.errstate(invalid="ignore", divide="ignore"):
            D = vel.values / np.linalg.norm(vel.values, axis=1, keepdims=True)

    elif direction == "hd":
        hd = session["H"].values

        D = np.column_stack(
            (
                np.cos(hd),
                np.sin(hd),
            )
        )

    elif isinstance(direction, int):
        shifted = compute_travel_projected(
            session_type,
            session,
            ("P_x", "P_y"),
            direction,
        )

        future_vec = shifted.values - P.values

        with np.errstate(invalid="ignore", divide="ignore"):
            D = future_vec / np.linalg.norm(
                future_vec,
                axis=1,
                keepdims=True,
            )

    elif direction == "stops":
        stops = (
            session["S"]
            .threshold(3.0, method="below")
            .time_support.drop_short_intervals(0.4)
        )
        if len(stops) <10:
            return results
        else:
            results["n_stops"] = len(stops)

        mask = P.in_interval(stops).values

        P_next_stop_mean = np.full_like(P, np.nan, dtype=float)
        changes = np.diff(mask.astype(int))
        stop_starts = np.where(changes == 1)[0] + 1
        stop_ends = np.where(changes == -1)[0] + 1

        # handle edge cases
        if mask[0]:
            stop_starts = np.r_[0, stop_starts]

        if mask[-1]:
            stop_ends = np.r_[stop_ends, len(mask)]

        # previous boundary
        prev_end = 0

        for start, end in zip(stop_starts, stop_ends):
            # mean position during this stop block
            mean_pos = P[start:end].mean(axis=0)

            # fill preceding movement block
            P_next_stop_mean[prev_end:start] = mean_pos

            # fill stop block itself
            P_next_stop_mean[start:end] = mean_pos

            prev_end = end

        future_vec = P_next_stop_mean.values - P.values

        with np.errstate(invalid="ignore", divide="ignore"):
            D = future_vec / np.linalg.norm(
                future_vec,
                axis=1,
                keepdims=True,
            )
    else:
        raise ValueError("direction must be 'movement', 'hd', or int")

    # ------------------------------------------------------------
    # Nearest field center for EVERY position sample

    pos = P.values

    dist_to_fields = np.linalg.norm(
        pos[:, None, :] - field_centers[None, :, :],
        axis=2,
    )

    nearest_idx = np.argmin(
        dist_to_fields,
        axis=1,
    )

    nearest_centers = field_centers[nearest_idx]

    # ------------------------------------------------------------
    # Vector from animal -> nearest field center
    vec_from_center = pos - nearest_centers

    # Project onto direction of movement/heading
    in_field_pos = np.sum(
        vec_from_center * D,
        axis=1,
    )

    # ------------------------------------------------------------
    # Build Tsd for spike sampling

    in_field_tsd = nap.Tsd(
        t=P.times(),
        d=in_field_pos,
    )

    # ------------------------------------------------------------
    # Spike variables

    spike_train = cluster[cluster.index[0]]

    spike_phase = spike_train.value_from(
        session["theta"],
        ep=session["moving"],
    )

    spike_pos = spike_train.value_from(
        in_field_tsd,
        ep=session["moving"],
    )

    # ------------------------------------------------------------
    # Clean

    valid = (
        np.isfinite(spike_phase.values)
        & np.isfinite(spike_pos.values)
        & (spike_pos.values >= precession_range[0])
        & (spike_pos.values <= precession_range[1])
    )

    spike_phase = spike_phase[valid]
    spike_pos = spike_pos[valid]

    if len(spike_pos) < min_spikes:
        return results

    if np.std(spike_pos) == 0:
        return results

    # ------------------------------------------------------------
    # Circular-linear correlation+regression

    try:
        corr = circ_corrcl(
            x=spike_pos.values,
            a=spike_phase.values,
        )
        cl = CLRegression(
            formula="θ ~ x",
            theta=spike_phase.values,
            X=spike_pos.values,
            model_type="mean",
        )
        slope = cl.result["beta"][0]

        results.update(
            {
                "pval": corr.p_value,
                "corr": corr.r,
                "slope": slope,
                "direction": direction,
                "spike_pos": spike_pos.values,
                "spike_phase": spike_phase.values,
            }
        )
        return results

    except ValueError:
        return results
