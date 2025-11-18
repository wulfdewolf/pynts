import numpy as np
import pynapple as nap

from pynts.wrappers import find_optimal_smoothing
from pynts.util import gaussian_filter_nan, wrap_list


def classify_spatial_information(score, null_distribution, alpha=0.001):
    return {
        "sig": score["spatial_information"]
        > np.nanpercentile(null_distribution["spatial_information"], 100 * (1 - alpha)),
        "pval": (
            np.sum(
                null_distribution["spatial_information"] >= score["spatial_information"]
            )
            + 1
        )
        / (len(null_distribution["spatial_information"]) + 1),
    }


def compute_spatial_information(
    session,
    session_type,
    cluster_spikes,
    num_bins,
    bounds,
    smooth_sigma=True,
    epoch=None,
):
    if epoch is None:
        epoch = cluster_spikes.time_support
    key = "P" if "VR" in session_type else ("P_x", "P_y")
    dim = 1 if "VR" in session_type else 2
    mode = "wrap" if "VR" in session_type else "reflect"
    data = np.stack([session[k] for k in wrap_list(key)], axis=1)

    def compute_tuning_curve(epochs):
        return nap.compute_tuning_curves(
            cluster_spikes,
            data,
            bins=num_bins,
            range=bounds,
            epochs=epochs.intersect(session["moving"]),
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
                    mode=mode,
                )
            ] * dim
        tc = compute_tuning_curve(epoch)
        if smooth_sigma:
            tc = gaussian_filter_nan(tc, smooth_sigma, mode="wrap")

        return {
            "spatial_information": nap.compute_mutual_information(tc)[
                "bits/spike"
            ].item(),
            "_smooth_sigma": smooth_sigma,
        }
