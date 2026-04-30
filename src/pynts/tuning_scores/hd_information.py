from typing import Optional

import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike

from pynts.smoothing import apply_smoothing


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
    if epoch is None:
        epoch = cluster.time_support

    range = (
        (np.nanmin(session["H"]), np.nanmax(session["H"])) if range is None else range
    )
    if num_bins is None:
        bins = int((range[1] - range[0]) // bin_size)
    else:
        bins = num_bins

    def compute_tuning_curve(epochs):
        return nap.compute_tuning_curves(
            cluster,
            session["H"],
            bins=bins,
            range=range,
            epochs=epochs.intersect(session["moving"]),
        )

    tc, smooth_sigma = apply_smoothing(
        compute_tuning_curve,
        epoch=epoch,
        dim=1,
        smooth_sigma=smooth_sigma,
        sigma_range=np.linspace(1, 6, 20),
        mode="wrap",
        keep=False,
    )

    # Compute preferred
    angles = tc[0].coords[tc.dims[0]].values
    weights = tc[0].values
    mask = ~np.isnan(weights)
    preferred = np.arctan2(
        np.sum(weights[mask] * np.sin(angles[mask])),
        np.sum(weights[mask] * np.cos(angles[mask])),
    )
    return {
        "hd_information": nap.compute_mutual_information(tc)["bits/spike"].item(),
        "preferred": preferred,
        "_smooth_sigma": smooth_sigma,
    }
