from typing import Optional

import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike
from pycircstat2.correlation import circ_corrcc
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
    min_spikes_per_field: int = 10,
    direction: str = "movement",
):
    """
    Compute phase precession within segmented place fields.

    Parameters
    ----------
    direction : {"movement", "hd"}
        Defines the directional vector used for projection:
        - "movement": instantaneous movement direction
        - "hd": head direction
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

    spike_phases = cluster[cluster.index[0]].value_from(
        theta,
        ep=moving_ep,
    )

    spike_positions = cluster[cluster.index[0]].value_from(
        P,
        ep=moving_ep,
    )

    # ------------------------------------------------------------------
    # Direction vectors

    if direction == "movement":
        vel = np.zeros_like(P)

        vel[1:] = np.diff(P, axis=0) / np.diff(P.times())[:, None]

        spike_direction = (
            cluster[cluster.index[0]]
            .value_from(
                vel,
                ep=moving_ep,
            )
            .values
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            spike_direction = spike_direction / np.linalg.norm(
                spike_direction,
                axis=1,
                keepdims=True,
            )

    elif direction == "hd":
        spike_hd = (
            cluster[cluster.index[0]]
            .value_from(
                session["H"],
                ep=moving_ep,
            )
            .values
        )

        spike_direction = np.column_stack(
            (
                np.cos(spike_hd),
                np.sin(spike_hd),
            )
        )
    elif isinstance(direction, int):
        shifted = compute_travel_projected(
            session_type, session, ("P_x", "P_y"), direction
        )
        future_vec = shifted.values - P.values

        with np.errstate(invalid="ignore", divide="ignore"):
            future_vec = future_vec / np.linalg.norm(
                future_vec,
                axis=1,
                keepdims=True,
            )

        spike_direction = (
            cluster[cluster.index[0]]
            .value_from(
                nap.TsdFrame(d=future_vec, t=shifted.times()),
                ep=moving_ep,
            )
            .values
        )

    else:
        raise ValueError("direction must be 'movement', 'hd', or an integer")

    # ------------------------------------------------------------------
    # Field segmentation

    mask = tc > 0.2 * np.nanmax(tc)

    distance = ndi.distance_transform_edt(mask)

    peaks = peak_local_max(
        tc.values,
        min_distance=3,
        threshold_rel=0.2,
    )

    markers = np.zeros_like(tc, dtype=int)

    for i, (y, x) in enumerate(peaks):
        if mask[y, x]:
            markers[y, x] = i + 1

    labels = watershed(
        -distance,
        markers,
        mask=mask,
    )

    n_fields = labels.max()

    # ------------------------------------------------------------------
    # Results

    results = {
        "circ_corr": [],
        "direction": direction,
    }

    # ------------------------------------------------------------------
    # Per-field analysis

    for field_id in np.arange(1, n_fields + 1).astype(int):
        field_mask = labels == field_id

        coords = np.argwhere(field_mask)

        if coords.shape[0] < 5:
            continue

        center = coords.mean(axis=0)

        # --------------------------------------------------------------
        # Find spikes inside field

        spike_idx = []

        for i, (y, x) in enumerate(spike_positions.values):
            y_idx = int(
                np.clip(
                    np.round(y),
                    0,
                    tc.shape[0] - 1,
                )
            )

            x_idx = int(
                np.clip(
                    np.round(x),
                    0,
                    tc.shape[1] - 1,
                )
            )

            if labels[y_idx, x_idx] == field_id:
                spike_idx.append(i)

        spike_idx = np.asarray(spike_idx)

        if len(spike_idx) < min_spikes_per_field:
            continue

        # --------------------------------------------------------------
        # Field-specific variables

        sp_positions = spike_positions.values[spike_idx]
        sp_phases = spike_phases.values[spike_idx]
        sp_direction = spike_direction[spike_idx]

        # --------------------------------------------------------------
        # Projection onto direction vector

        vec_to_center = sp_positions - center

        proj_cm = np.sum(
            vec_to_center * sp_direction,
            axis=1,
        )

        # --------------------------------------------------------------
        # Circular-linear correlation

        phase_unwrapped = np.unwrap(sp_phases)

        try:
            circ_corr = circ_corrcc(
                phase_unwrapped,
                proj_cm,
            )
        except ValueError as _:
            continue

        # --------------------------------------------------------------
        # Store

        results["circ_corr"].append(circ_corr)

    return results
