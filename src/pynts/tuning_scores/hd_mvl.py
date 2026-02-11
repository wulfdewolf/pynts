import numpy as np
import pynapple as nap

from pynts.util import gaussian_filter_nan
from pynts.wrappers import find_optimal_smoothing


def classify_hd_mvl(score, null_distribution, alpha=0.01):
    return {
        "sig": score["hd_mvl"]
        > np.nanpercentile(null_distribution["hd_mvl"], 100 * (1 - alpha)),
        "pval": (np.nansum(null_distribution["hd_mvl"] >= score["hd_mvl"]) + 1)
        / (len(null_distribution["hd_mvl"]) + 1),
    }


def compute_hd_mvl(
    session,
    session_type,
    cluster_spikes,
    bounds,
    num_bins=60,
    smooth_sigma=3,
    epoch=None,
    is_shuffle=False,
):
    if epoch is None:
        epoch = cluster_spikes.time_support

    def compute_tuning_curve(epochs):
        return nap.compute_tuning_curves(
            cluster_spikes,
            session["H"],
            bins=num_bins,
            range=bounds,
            epochs=epoch.intersect(session["moving"].intersect(epochs)),
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
        angles = tc.coords["0"].values
        dx = np.cos(angles)
        dy = np.sin(angles)
        totx = np.nansum(dx * tc.values) / np.nansum(tc.values)
        toty = np.nansum(dy * tc.values) / np.nansum(tc.values)
        return {
            "hd_mvl": np.sqrt(totx**2 + toty**2),
            "preferred": tc.coords["0"].values[tc.argmax()],
            "_smooth_sigma": smooth_sigma,
        }
