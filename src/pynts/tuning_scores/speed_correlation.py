from typing import Optional

import numpy as np
import pynapple as nap
from imageio.typing import ArrayLike

from pynts.util import interpolate_nans


def classify_speed_correlation(score, null_distribution, alpha=0.01):
    return {
        "sig": np.abs(score["speed_correlation"])
        > np.nanpercentile(null_distribution["speed_correlation"], 100 * (1 - alpha)),
        "pval": (
            np.sum(null_distribution["speed_correlation"] >= score["speed_correlation"])
            + 1
        )
        / (len(null_distribution["speed_correlation"]) + 1),
    }


def compute_speed_correlation(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    context: Optional[str] = None,
    trial_types: Optional[ArrayLike] = None,
    epoch: Optional[nap.IntervalSet] = None,
    is_shuffle: bool = False,
):
    if epoch is None:
        epoch = cluster.time_support
    if isinstance(cluster, nap.TsGroup):
        cluster = cluster[cluster.index[0]]
    if isinstance(cluster, nap.Tsd):
        fr = interpolate_nans(cluster.bin_average(0.02)).smooth(0.3, windowsize=2)
    else:
        fr = cluster.count(0.02).smooth(0.3, windowsize=2)

    # Check if speed is regularly sampled, apply different computation if so
    time_diffs = session["S"].time_diff().values
    bin_size = time_diffs[0]
    relative_variation = np.abs(time_diffs - bin_size) / bin_size
    speed_regularly_sampled = np.all(relative_variation < 0.01)

    if not speed_regularly_sampled:
        speed = session["S"]
        fr = session["S"].value_from(fr)
    else:
        speed = interpolate_nans(session["S"].interpolate(fr))

    restriction = epoch.intersect(session["moving"])
    if context is not None:
        restriction = restriction.intersect(
            session["trials"][session["trials"]["context"] == context]
        )
    if trial_types is not None:
        restriction = restriction.intersect(
            session["trials"][session["trials"]["type"].isin(trial_types)]
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "speed_correlation": (
                fr.restrict(restriction)
                .as_series()
                .corr(speed.restrict(restriction).as_series())
            )
        }
