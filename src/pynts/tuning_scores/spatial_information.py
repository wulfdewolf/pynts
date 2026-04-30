from typing import Optional

import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike

from pynts.smoothing import apply_smoothing
from pynts.util import wrap_list


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
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    num_bins: Optional[int] = None,
    bin_size: float = 2.5,
    range: Optional[ArrayLike] = None,
    smooth_sigma: float | ArrayLike = 2,
    epoch: Optional[nap.IntervalSet] = None,
    is_shuffle: bool = False,
):
    if epoch is None:
        epoch = cluster.time_support

    if "VR" in session_type:
        dim = 1
        mode = "wrap"
        key = "P"
        range = (
            [(np.nanmin(session["P"]), np.nanmax(session["P"]))]
            if range is None
            else range
        )
    else:
        dim = 2
        mode = "fill"
        key = ("P_x", "P_y")
        range = (
            [
                (np.nanmin(session["P_x"]), np.nanmax(session["P_x"])),
                (np.nanmin(session["P_y"]), np.nanmax(session["P_y"])),
            ]
            if range is None
            else range
        )
    P = np.stack([session[k] for k in wrap_list(key)], axis=1)
    if num_bins is None:
        bins = [int((dim_range[1] - dim_range[0]) // bin_size) for dim_range in range]
    else:
        bins = num_bins

    def compute_tuning_curve(epochs):
        return nap.compute_tuning_curves(
            cluster,
            P,
            bins=bins,
            range=range,
            epochs=epochs.intersect(session["moving"]),
        )

    tc, smooth_sigma = apply_smoothing(
        compute_tuning_curve,
        epoch=epoch,
        dim=dim,
        smooth_sigma=smooth_sigma,
        sigma_range=np.linspace(1, 4, 20),
        mode=mode,
        keep=True,
    )

    return {
        "spatial_information": nap.compute_mutual_information(tc)["bits/spike"].item(),
        "_smooth_sigma": smooth_sigma,
    }
