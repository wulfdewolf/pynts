from typing import Optional

import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike
from pycircstat2.correlation import circ_corrcl
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from pynts.smoothing import apply_smoothing
from pynts.wrappers import compute_travel_projected


def classify_precession(score, null_distribution, alpha=0.05):
    n_fields = len(score["circ_corr"])

    results = {
        "sig": [],
        "pval": [],
    }

    null_corr = np.concatenate(
        [
            np.asarray(corr_list)
            for corr_list in null_distribution["circ_corr"]
            if len(corr_list) > 0
        ]
    )

    for i in range(n_fields):
        corr = score["circ_corr"][i]

        pval = (np.nansum(np.abs(null_corr) >= np.abs(corr)) + 1) / (len(null_corr) + 1)

        results["sig"].append(pval < alpha)
        results["pval"].append(pval)

    return results


def compute_precession(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    range: Optional[ArrayLike] = None,
    num_bins: Optional[int] = 60,
    bin_size: Optional[float] = None,
    smooth_sigma: float | ArrayLike = 2,
    epoch: Optional[nap.IntervalSet] = None,
    is_shuffle: bool = False,
    min_spikes_per_field: int = 100,
    direction: str | int = "movement",
):
    """
    Compute phase precession relative to nearest place-field peak.

    Each spike is assigned to the closest detected firing-field center,
    rather than requiring spikes to fall inside segmented watershed masks.

    Parameters
    ----------
    direction : {"movement", "hd"} or int
        Defines the directional vector used for projection:
        - "movement": instantaneous movement direction
        - "hd": head direction
        - int: future travel projection shift
    """

    if "theta" not in session:
        return {}

    if epoch is None:
        epoch = cluster.time_support

    moving_ep = epoch.intersect(session["moving"])

    # ------------------------------------------------------------------
    # Spatial range

    if range is None:
        range = [
            (np.nanmin(session["P_x"]), np.nanmax(session["P_x"])),
            (np.nanmin(session["P_y"]), np.nanmax(session["P_y"])),
        ]

    P = np.stack([session["P_x"], session["P_y"]], axis=1)

    # ------------------------------------------------------------------
    # Binning

    if num_bins is None:
        bins = [int((dim_range[1] - dim_range[0]) // bin_size) for dim_range in range]
    else:
        bins = num_bins

    # ------------------------------------------------------------------
    # Tuning curve

    def compute_tuning_curve(epochs):
        return nap.compute_tuning_curves(
            cluster,
            P,
            bins=bins,
            range=range,
            epochs=epochs.intersect(session["moving"]),
        )[0]

    tc, smooth_sigma = apply_smoothing(
        compute_tuning_curve,
        epoch=epoch,
        dim=2,
        smooth_sigma=smooth_sigma,
        sigma_range=np.linspace(1, 4, 20),
        mode="fill",
        keep=False,
    )

    # ------------------------------------------------------------------
    # Theta channel selection

    theta = session["theta"]

    if "extremum_channel" in cluster.metadata_columns:
        theta_channel = next(
            theta_channel
            for theta_channel in theta["channel_name"]
            if cluster["extremum_channel"].item() in theta_channel
        )

        theta = theta[:, theta["channel_name"] == theta_channel]

    # ------------------------------------------------------------------
    # Spike-aligned variables

    spike_train = cluster[cluster.index[0]]

    spike_phases = spike_train.value_from(
        theta,
        ep=moving_ep,
    )

    spike_positions = spike_train.value_from(
        P,
        ep=moving_ep,
    )

    # ------------------------------------------------------------------
    # Direction vectors

    if direction == "movement":
        vel = np.zeros_like(P)

        vel[1:] = np.diff(P, axis=0) / np.diff(P.times())[:, None]

        spike_direction = spike_train.value_from(
            vel,
            ep=moving_ep,
        ).values

        with np.errstate(invalid="ignore", divide="ignore"):
            spike_direction = spike_direction / np.linalg.norm(
                spike_direction,
                axis=1,
                keepdims=True,
            )

    elif direction == "hd":
        spike_hd = spike_train.value_from(
            session["H"],
            ep=moving_ep,
        ).values

        spike_direction = np.column_stack(
            (
                np.cos(spike_hd),
                np.sin(spike_hd),
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
            future_vec = future_vec / np.linalg.norm(
                future_vec,
                axis=1,
                keepdims=True,
            )

        spike_direction = spike_train.value_from(
            nap.TsdFrame(
                d=future_vec,
                t=shifted.times(),
            ),
            ep=moving_ep,
        ).values

    else:
        raise ValueError("direction must be 'movement', 'hd', or an integer")

    # ------------------------------------------------------------------
    # Detect field peaks

    peaks = peak_local_max(
        tc.values,
        min_distance=3,
        threshold_rel=0.2,
    )

    if len(peaks) == 0:
        return {
            "corr": [],
            "pval": [],
            "direction": str(direction),
            "spike_dist": [],
            "spike_phases": [],
            "field_centers": [],
        }

    # ------------------------------------------------------------------
    # Convert peak indices -> physical coordinates

    x_edges = np.linspace(
        range[0][0],
        range[0][1],
        tc.shape[0] + 1,
    )

    y_edges = np.linspace(
        range[1][0],
        range[1][1],
        tc.shape[1] + 1,
    )

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    field_centers = np.column_stack(
        (
            x_centers[peaks[:, 0]],
            y_centers[peaks[:, 1]],
        )
    )

    # ------------------------------------------------------------------
    # Assign each spike to nearest field center

    spike_xy = spike_positions.values

    dist_to_fields = np.linalg.norm(
        spike_xy[:, None, :] - field_centers[None, :, :],
        axis=2,
    )

    closest_field = np.argmin(
        dist_to_fields,
        axis=1,
    )

    # ------------------------------------------------------------------
    # Results

    results = {
        "corr": [],
        "pval": [],
        "direction": str(direction),
        "spike_dist": [],
        "spike_phases": [],
        "field_centers": field_centers,
    }

    # ------------------------------------------------------------------
    # Per-field analysis

    for field_id in range(len(field_centers)):
        spike_idx = np.where(closest_field == field_id)[0]

        if len(spike_idx) < min_spikes_per_field:
            continue

        center = field_centers[field_id]

        # --------------------------------------------------------------
        # Field-specific variables

        sp_positions = spike_positions.values[spike_idx]
        sp_phases = spike_phases.values[spike_idx]
        sp_direction = spike_direction[spike_idx]

        # --------------------------------------------------------------
        # Remove invalid directions

        valid = np.all(
            np.isfinite(sp_direction),
            axis=1,
        )

        if np.sum(valid) < min_spikes_per_field:
            continue

        sp_positions = sp_positions[valid]
        sp_phases = sp_phases[valid]
        sp_direction = sp_direction[valid]

        # --------------------------------------------------------------
        # Projection onto directional vector

        vec_to_center = sp_positions - center

        proj_cm = np.sum(
            vec_to_center * sp_direction,
            axis=1,
        )

        valid = np.isfinite(proj_cm) & np.isfinite(sp_phases)

        if np.sum(valid) < min_spikes_per_field:
            continue

        proj_cm = proj_cm[valid]
        sp_phases = sp_phases[valid]

        if np.std(proj_cm) == 0:
            continue

        # --------------------------------------------------------------
        # Circular-linear correlation

        try:
            result = circ_corrcl(
                sp_phases,
                proj_cm,
            )

            corr = result.r
            pval = result.p_value

        except ValueError:
            continue

        # --------------------------------------------------------------
        # Store

        results["corr"].append(corr)
        results["pval"].append(pval)
        results["spike_dist"].append(proj_cm)
        results["spike_phases"].append(sp_phases)

    return results
