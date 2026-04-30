from typing import Optional

import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike

from pynts.smoothing import apply_smoothing


def compute_theta_index(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    range: Optional[ArrayLike] = None,
    num_bins: Optional[int] = 60,
    bin_size: Optional[float] = None,
    smooth_sigma: float | ArrayLike = 2,
    epoch: Optional[nap.IntervalSet] = None,
    is_shuffle=False,
):
    """
    Theta index as defined in https://elifesciences.org/articles/35949#s4
    """
    if epoch is None:
        epoch = cluster.time_support

    # Estimate firing rate
    fr = cluster.count(0.002).smooth(0.005, windowsize=1, norm=False)

    # Compute PSD
    psd = nap.compute_power_spectral_density(fr, fs=fr.rate)

    # Compute band powers
    theta_power = np.nanmean(psd[(psd.index >= 6) & (psd.index <= 10)])
    left_power = np.nanmean(psd[(psd.index >= 3) & (psd.index <= 5)])
    right_power = np.nanmean(psd[(psd.index >= 11) & (psd.index <= 13)])

    # Compute theta index
    baseline = (left_power + right_power) / 2
    x = theta_power - baseline
    y = theta_power + baseline
    with np.errstate(invalid="ignore", divide="ignore"):
        theta_index = x / y

    result = {
        "theta_index": theta_index,
        "sig": theta_index > 0.07,
    }

    if "theta" in session:
        theta = session["theta"]
        if "extremum_channel" in cluster.metadata_columns:
            theta_channel = next(
                theta_channel
                for theta_channel in session["theta"]["channel_name"]
                if cluster["extremum_channel"].item() in theta_channel
            )
            theta = theta[:, theta["channel_name"] == theta_channel]
        else:
            theta = theta % (2 * np.pi)

        range = (
            (np.nanmin(session["H"]), np.nanmax(session["H"]))
            if range is None
            else range
        )
        if num_bins is None:
            bins = int((range[1] - range[0]) // bin_size)
        else:
            bins = num_bins

        # Compute theta tuning curves
        def compute_tuning_curve(epochs):
            return nap.compute_tuning_curves(
                cluster,
                theta,
                bins=bins,
                range=range,
                epochs=epochs.intersect(session["moving"]),
            )[0]

        tc, smooth_sigma = apply_smoothing(
            compute_tuning_curve,
            epoch=epoch,
            dim=1,
            smooth_sigma=smooth_sigma,
            sigma_range=np.linspace(1, 6, 20),
            mode="wrap",
            keep=False,
        )
        result["_smooth_sigma"] = smooth_sigma

        # Get preferred
        angles = tc.coords[tc.dims[0]].values
        weights = tc.values
        mask = ~np.isnan(weights)
        result["preferred"] = np.arctan2(
            np.sum(weights[mask] * np.sin(angles[mask])),
            np.sum(weights[mask] * np.cos(angles[mask])),
        )

    return result
