import numpy as np
import pynapple as nap

from pynts.util import gaussian_filter_nan
from pynts.wrappers import find_optimal_smoothing


def compute_theta_index(
    session,
    session_type,
    cluster,
    smooth_sigma=6,
    epoch=None,
    num_bins=61,
    bounds=(-np.pi, np.pi),
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

        # Compute theta tuning curves
        def compute_tuning_curve(epochs):
            return nap.compute_tuning_curves(
                cluster,
                theta,
                bins=num_bins,
                range=bounds,
                epochs=epoch.intersect(session["moving"].intersect(epochs)),
            )

        tc = compute_tuning_curve(epoch)

        with np.errstate(invalid="ignore", divide="ignore"):
            if smooth_sigma == "cv":
                smooth_sigma = [0] + [
                    find_optimal_smoothing(
                        compute_tuning_curve,
                        epoch,
                        np.arange(
                            int(num_bins // 4),
                        ),
                        mode="wrap",
                    )
                ]
            elif type(smooth_sigma) is int:
                smooth_sigma = [0] + [smooth_sigma]
            if smooth_sigma:
                tc = gaussian_filter_nan(tc, smooth_sigma, mode="wrap", keep=False)
        result["_smooth_sigma"] = smooth_sigma

        # Get preferred
        angles = tc.coords[tc.dims[1]].values
        weights = tc[0].values
        mask = ~np.isnan(weights)
        result["preferred"] = np.arctan2(
            np.sum(weights[mask] * np.sin(angles[mask])),
            np.sum(weights[mask] * np.cos(angles[mask])),
        )

    return result
