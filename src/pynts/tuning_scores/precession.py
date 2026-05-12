from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from attr import field
from numpy.typing import ArrayLike
from pycircstat2.correlation import circ_corrcc
from scipy import ndimage as ndi
from scipy.ndimage import label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from pynts.smoothing import apply_smoothing


def classify_precession(score, null_distribution, alpha=0.05):
    n_fields = len(score["circ_corr_movement"])
    results = {
        "sig_corr_movement": [],
        "pval_corr_movement": [],
        "sig_corr_hd": [],
        "pval_corr_hd": [],
    }

    null_corr_movement = np.concatenate(
        [
            np.array(corr_list)
            for corr_list in null_distribution["circ_corr_movement"]
            if len(corr_list) > 0
        ]
    )
    null_corr_hd = np.concatenate(
        [
            np.array(corr_list)
            for corr_list in null_distribution["circ_corr_hd"]
            if len(corr_list) > 0
        ]
    )

    for i in range(n_fields):
        obs_corr_movement = score["circ_corr_movement"][i]
        obs_corr_hd = score["circ_corr_hd"][i]

        # Two-sided p-value
        p_corr_movement = (
            np.nansum(np.abs(null_corr_movement) >= np.abs(obs_corr_movement)) + 1
        ) / (len(null_corr_movement) + 1)
        p_corr_hd = (np.nansum(np.abs(null_corr_hd) >= np.abs(obs_corr_hd)) + 1) / (
            len(null_corr_hd) + 1
        )

        # Append per field
        results["sig_corr_movement"].append(p_corr_movement < alpha)
        results["pval_corr_movement"].append(p_corr_movement)
        results["sig_corr_hd"].append(p_corr_hd < alpha)
        results["pval_corr_hd"].append(p_corr_hd)

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
    is_shuffle=False,
    min_spikes_per_field: int = 10,
):
    if "theta" not in session:
        return {}

    if epoch is None:
        epoch = cluster.time_support

    range = (
        [
            (np.nanmin(session["P_x"]), np.nanmax(session["P_x"])),
            (np.nanmin(session["P_y"]), np.nanmax(session["P_y"])),
        ]
        if range is None
        else range
    )

    P = np.stack([session["P_x"], session["P_y"]], axis=1)

    if num_bins is None:
        bins = [int((dim_range[1] - dim_range[0]) // bin_size) for dim_range in range]
    else:
        bins = num_bins

    # Compute tuning curve
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

    # Select theta channel
    theta = session["theta"]
    if "extremum_channel" in cluster.metadata_columns:
        theta_channel = next(
            theta_channel
            for theta_channel in session["theta"]["channel_name"]
            if cluster["extremum_channel"].item() in theta_channel
        )
        theta = theta[:, theta["channel_name"] == theta_channel]

    # Extract spikes
    spike_phases = cluster[cluster.index[0]].value_from(
        theta, ep=epoch.intersect(session["moving"])
    )
    spike_positions = cluster[cluster.index[0]].value_from(
        P, ep=epoch.intersect(session["moving"])
    )
    spike_hd = cluster[cluster.index[0]].value_from(
        session["H"], ep=epoch.intersect(session["moving"])
    )

    vel = np.zeros_like(P)
    vel[1:] = np.diff(P, axis=0) / np.diff(P.times())[:, None]
    spike_vel = cluster[cluster.index[0]].value_from(vel, ep=epoch)

    # -----------------------------
    # 1. Create field mask and distance
    mask = tc > 0.2 * np.nanmax(tc)
    distance = ndi.distance_transform_edt(mask)

    # 2. Detect peaks for watershed
    peaks = peak_local_max(tc.values, min_distance=3, threshold_rel=0.2)
    markers = np.zeros_like(tc, dtype=int)
    for i, (y, x) in enumerate(peaks):
        if mask[y, x]:
            markers[y, x] = i + 1

    # 3. Watershed to segment fields
    labels = watershed(-distance, markers, mask=mask)
    n_fields = labels.max()

    # -----------------------------
    results = {
        "centers": [],
        "spike_idx": [],
        "proj_movement_cm": [],
        "proj_hd_cm": [],
        "spike_phases": [],
        "slope_movement_deg_per_cm": [],
        "slope_hd_deg_per_cm": [],
        "circ_corr_movement": [],
        "circ_corr_hd": [],
    }

    for field_id in np.arange(1, n_fields + 1).astype(int):
        field_mask = labels == field_id
        coords = np.argwhere(field_mask)
        if coords.shape[0] < 5:
            continue
        center = coords.mean(axis=0)

        # -----------------------------
        # Find spikes in field
        spike_idx = []
        for i, (y, x) in enumerate(spike_positions.values):
            y_idx = int(np.clip(np.round(y), 0, tc.shape[0] - 1))
            x_idx = int(np.clip(np.round(x), 0, tc.shape[1] - 1))
            if labels[y_idx, x_idx] == field_id:
                spike_idx.append(i)

        spike_idx = np.array(spike_idx)
        if len(spike_idx) < min_spikes_per_field:
            continue

        sp_positions_field = spike_positions.values[spike_idx]
        sp_phases_field = spike_phases.values[spike_idx]
        sp_hd_field = spike_hd.values[spike_idx]
        sp_vel_field = spike_vel.values[spike_idx]

        # -----------------------------
        # Compute projections in cm
        vec_to_center = sp_positions_field - center  # (N,2)

        # Movement direction (signed distance along vector to center)
        with np.errstate(invalid="ignore", divide="ignore"):
            movement_dir_unit = (
                sp_vel_field / np.linalg.norm(sp_vel_field, axis=1)[:, None]
            )
        proj_movement_cm = np.sum(vec_to_center * movement_dir_unit, axis=1)

        # Head direction (signed distance along HD vector)
        sp_hd_field_unit = np.column_stack((np.cos(sp_hd_field), np.sin(sp_hd_field)))
        proj_hd_cm = np.sum(vec_to_center * sp_hd_field_unit, axis=1)

        # -----------------------------
        # Circular-linear correlation
        phase_unwrapped = np.unwrap(sp_phases_field)
        circ_corr_movement = circ_corrcc(phase_unwrapped, proj_movement_cm)
        circ_corr_hd = circ_corrcc(phase_unwrapped, proj_hd_cm)

        # -----------------------------
        # Debug plot
        # fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        # ax[0].scatter(proj_movement_cm, phase_unwrapped, c="blue")
        # fit_line = np.poly1d(np.polyfit(proj_movement_cm, phase_unwrapped, 1))
        # ax[0].plot(
        #    proj_movement_cm,
        #    fit_line(proj_movement_cm),
        #    "r--",
        #    label=f"slope={slope_movement:.1f}°/cm",
        # )
        # ax[0].set_xlabel("Distance from center (cm, movement)")
        # ax[0].set_ylabel("Spike phase (rad)")
        # ax[0].set_title(f"Field {field_id} | Corr={circ_corr_movement:.2f}")
        # ax[0].legend()

        # ax[1].scatter(proj_hd_cm, phase_unwrapped, c="green")
        # fit_line_hd = np.poly1d(np.polyfit(proj_hd_cm, phase_unwrapped, 1))
        # ax[1].plot(
        #    proj_hd_cm,
        #    fit_line_hd(proj_hd_cm),
        #    "r--",
        #    label=f"slope={slope_hd:.1f}°/cm",
        # )
        # ax[1].set_xlabel("Distance from center (cm, head direction)")
        # ax[1].set_ylabel("Spike phase (rad)")
        # ax[1].set_title(f"Field {field_id} | Corr={circ_corr_hd:.2f}")
        # ax[1].legend()

        # plt.tight_layout()
        # plt.show()

        # -----------------------------
        # Store results per field
        results["centers"].append(center)
        results["spike_idx"].append(spike_idx)
        results["proj_movement_cm"].append(proj_movement_cm)
        results["proj_hd_cm"].append(proj_hd_cm)
        results["spike_phases"].append(sp_phases_field)
        results["circ_corr_movement"].append(circ_corr_movement)
        results["circ_corr_hd"].append(circ_corr_hd)

    return results
