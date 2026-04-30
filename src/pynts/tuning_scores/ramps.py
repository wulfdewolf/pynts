from typing import List, Optional

import numpy as np
import pynapple as nap
import statsmodels.api as sm
from numpy.typing import ArrayLike
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection

from pynts.smoothing import apply_smoothing


def classify_ramps(score, null_distribution, alpha=0.01):
    result = {}

    for region in ["outbound", "homebound"]:
        # Test significance
        result[f"{region}_sig"] = fdrcorrection(
            [score[f"{region}_pval"]] + null_distribution[f"{region}_pval"].tolist(),
            alpha,
        )[0][0] and (
            score[f"{region}_slope"]
            > norm.ppf(
                1 - alpha / 2,
                loc=np.nanmean(null_distribution[f"{region}_slope"]),
                scale=np.nanstd(null_distribution[f"{region}_slope"]),
            )
            or score[f"{region}_slope"]
            < norm.ppf(
                alpha / 2,
                loc=np.nanmean(null_distribution[f"{region}_slope"]),
                scale=np.nanstd(null_distribution[f"{region}_slope"]),
            )
        )
        result[f"{region}_sign"] = (
            "/"
            if not result[f"{region}_sig"]
            else "+"
            if score[f"{region}_slope"] > 0
            else "-"
        )
    return result


def compute_ramps(
    session: dict,
    session_type: str,
    cluster: nap.TsGroup,
    num_bins: Optional[int] = None,
    bin_size: Optional[int] = 1,
    range: ArrayLike = (0, 200),
    context: List[str] = ["rz1"],
    trial_types: List[str] = ["b", "nb"],
    outbound: ArrayLike = (30, 90),
    homebound: ArrayLike = (110, 170),
    smooth_sigma="cv",
    epoch=None,
    is_shuffle=False,
):
    if epoch is None:
        epoch = cluster.time_support
    trials = session["trials"][session["trials"]["type"].isin(trial_types)]

    range = (
        [(np.nanmin(session["P"]), np.nanmax(session["P"]))] if range is None else range
    )
    if num_bins is None:
        bins = int((range[1] - range[0]) // bin_size)
    else:
        bins = num_bins

    def compute_tuning_curve(epochs):
        return nap.compute_tuning_curves(
            cluster,
            session["P"],
            bins=bins,
            range=range,
            epochs=epochs.intersect(session["moving"]).intersect(trials),
        )[0]

    tc, smooth_sigma = apply_smoothing(
        compute_tuning_curve,
        epoch=epoch,
        dim=1,
        smooth_sigma=smooth_sigma,
        sigma_range=np.linspace(1, 3, 10),
        mode="wrap",
        keep=True,
    )
    positions = tc.coords["0"].values

    # Compute ramp fits
    results = {"_smooth_sigma": smooth_sigma}
    for region, ramp_bounds in [("outbound", outbound), ("homebound", homebound)]:
        mask = (positions > ramp_bounds[0]) & (positions < ramp_bounds[1])
        model = sm.OLS(tc.values[mask], sm.add_constant(positions[mask])).fit()
        results[f"{region}_slope"] = model.params[1]
        results[f"{region}_intercept"] = model.params[0]
        results[f"{region}_pval"] = model.pvalues[1]
        results[f"{region}_region"] = region
    return results
