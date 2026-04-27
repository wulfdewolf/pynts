import numpy as np
import pynapple as nap
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import fdrcorrection

from pynts.wrappers import find_optimal_smoothing
from pynts.util import gaussian_filter_nan


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
    session,
    session_type,
    cluster_spikes,
    num_bins,
    bounds=[(0, 200)],
    context=["rz1"],
    trial_types=["b", "nb"],
    outbound=(30, 90),
    homebound=(110, 170),
    smooth_sigma="cv",
    epoch=None,
    is_shuffle=False,
):
    if epoch is None:
        epoch = cluster_spikes.time_support
    trials = session["trials"][session["trials"]["type"].isin(trial_types)]

    def compute_tuning_curve(epochs):
        return nap.compute_tuning_curves(
            nap.TsGroup([cluster_spikes]),
            session["P"],
            bins=num_bins,
            range=bounds,
            epochs=session["moving"].intersect(trials).intersect(epochs),
        )[0]

    tc = compute_tuning_curve(epoch)
    with np.errstate(invalid="ignore", divide="ignore"):
        if smooth_sigma == "cv":
            smooth_sigma = find_optimal_smoothing(
                compute_tuning_curve,
                cluster_spikes.time_support,
                np.arange(
                    int(num_bins // 4),
                ),
                mode="wrap",
            )
        elif type(smooth_sigma) is int:
            smooth_sigma = (0, smooth_sigma)
        if smooth_sigma:
            tc = gaussian_filter_nan(tc, smooth_sigma, mode="wrap")
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
