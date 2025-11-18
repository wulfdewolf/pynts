import numpy as np
import pynapple as nap

from pynts.wrappers import find_optimal_smoothing
from pynts.util import gaussian_filter_nan


def compute_theta_index(
    session,
    session_type,
    cluster_spikes,
    smooth_sigma=True,
    epoch=None,
    num_bins=60,
    bounds=(-np.pi, np.pi),
):
    """
    Theta index as defined in https://elifesciences.org/articles/35949#s4
    """
    if epoch is None:
        epoch = cluster_spikes.time_support

    # Estimate firing rate
    fr = cluster_spikes.count(0.002).smooth(0.005, windowsize=1, norm=False)

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

    if "extremum_channel" in cluster_spikes.metadata_columns:
        # Compute theta tuning curves
        def compute_tuning_curve(epochs):
            return nap.compute_tuning_curves(
                cluster_spikes,
                session["theta"][
                    :,
                    int(cluster_spikes["extremum_channel"].item().replace("CH", ""))
                    - 1,
                ],
                bins=num_bins,
                range=bounds,
                epochs=epoch.intersect(session["moving"].intersect(epochs)),
            )

        with np.errstate(invalid="ignore", divide="ignore"):
            if isinstance(smooth_sigma, bool) and smooth_sigma:
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
        tc = compute_tuning_curve(epoch)
        if smooth_sigma:
            tc = gaussian_filter_nan(tc, smooth_sigma, mode="wrap")
        result["preferred"] = tc.coords["0"].values[tc.argmax()]
        result["_smooth_sigma"] = smooth_sigma

    return result
