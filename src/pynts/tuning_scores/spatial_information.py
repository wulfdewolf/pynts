import numpy as np
import pynapple as nap

from pynts.util import gaussian_filter_nan, wrap_list
from pynts.wrappers import find_optimal_smoothing


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
    num_bins=None,
    bin_size=2.5,
    smooth_sigma=2,
    epoch=None,
    is_shuffle=False,
):
    if epoch is None:
        epoch = cluster_spikes.time_support

    if "VR" in session_type:
        dim = 1
        mode = "wrap"
        key = "P"
        range = [(np.nanmin(session["P"]), np.nanmax(session["P"]))]
    else:
        mode = "reflect"
        key = ("P_x", "P_y")
        range = [
            (np.nanmin(session["P_x"]), np.nanmax(session["P_x"])),
            (np.nanmin(session["P_y"]), np.nanmax(session["P_y"])),
        ]
    P = np.stack([session[k] for k in wrap_list(key)], axis=1)
    if num_bins is None:
        bins = [(int(dim[0] // bin_size), int(dim[1] // bin_size)) for dim in range]
    else:
        bins = num_bins

    def compute_tuning_curve(epochs):
        return nap.compute_tuning_curves(
            cluster_spikes,
            P,
            bins=bins,
            range=range,
            epochs=epochs.intersect(session["moving"]),
        )

    tc = compute_tuning_curve(epoch)

    with np.errstate(invalid="ignore", divide="ignore"):
        if isinstance(smooth_sigma, bool) and smooth_sigma:
            smooth_sigma = [0] + [
                find_optimal_smoothing(
                    compute_tuning_curve,
                    epoch,
                    np.arange(
                        int(num_bins // 6),
                    ),
                    mode=mode,
                )
            ] * dim
        elif type(smooth_sigma) is int:
            smooth_sigma = [0] + [smooth_sigma] * dim

        if smooth_sigma:
            tc = gaussian_filter_nan(tc, smooth_sigma, mode=mode, keep=False)

        return {
            "spatial_information": nap.compute_mutual_information(tc)[
                "bits/spike"
            ].item(),
            "_smooth_sigma": smooth_sigma,
        }
