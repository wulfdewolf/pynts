import numpy as np
import pynapple as nap

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
    session,
    session_type,
    cluster,
    context=None,
    trial_types=None,
    smooth_sigma=False,
    epoch=None,
):
    if epoch is None:
        epoch = cluster.time_support
    if isinstance(cluster, nap.TsdFrame):
        fr = interpolate_nans(cluster[:, 0].bin_average(0.02)).smooth(
            0.30, windowsize=3, norm=False
        )
    else:
        fr = (
            cluster[cluster.index[0]].count(0.02).smooth(0.30, windowsize=3, norm=False)
        )
    speed = interpolate_nans(session["S"].interpolate(fr)).smooth(
        0.30, windowsize=3, norm=False
    )
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
