import numpy as np
import pynapple as nap

from pynts.util import gaussian_filter_nan
from pynts.wrappers import find_optimal_smoothing


def classify_hd_information(score, null_distribution, alpha=0.01):
    return {
        "sig": score["hd_information"]
        > np.nanpercentile(null_distribution["hd_information"], 100 * (1 - alpha)),
        "pval": (
            np.sum(null_distribution["hd_information"] >= score["hd_information"]) + 1
        )
        / (len(null_distribution["hd_information"]) + 1),
    }


def compute_hd_information(
    session,
    session_type,
    cluster_spikes,
    bounds,
    num_bins=60,
    smooth_sigma=(0, 3),
    epoch=None,
):
    if epoch is None:
        epoch = cluster_spikes.time_support

    def compute_tuning_curve(epochs):
        return nap.compute_tuning_curves(
            cluster_spikes,
            session["H"],
            bins=num_bins,
            range=bounds,
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
                        int(num_bins // 4),
                    ),
                    mode="wrap",
                )
            ]
        elif isinstance(smooth_sigma, int):
            smooth_sigma = (0, smooth_sigma)
        if smooth_sigma:
            tc = gaussian_filter_nan(tc, smooth_sigma, mode="wrap")
        return {
            "hd_information": nap.compute_mutual_information(tc)["bits/spike"].item(),
            "preferred": tc[0].coords["0"].values[tc[0].argmax()],
            "_smooth_sigma": smooth_sigma,
        }
