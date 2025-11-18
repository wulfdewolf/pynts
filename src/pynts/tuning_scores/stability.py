import warnings

import numpy as np
import pynapple as nap
from scipy.stats import norm

from pynts.util import gaussian_filter_nan, wrap_list


def classify_stability(score, null_distribution, alpha=0.01):
    return {
        "sig": score["stability"]
        > norm.ppf(
            1 - alpha,
            loc=np.nanmean(null_distribution["stability"]),
            scale=np.nanstd(null_distribution["stability"]),
        ),
    }


def compute_time_based_stability(session_type, data_type):
    def wrapper(
        session,
        session_type,
        cluster_spikes,
        bounds,
        num_bins,
        smooth_sigma=False,
        smooth_mode="reflect",
        epoch=None,
    ):
        with np.errstate(invalid="ignore", divide="ignore"):
            tcs_splits = [
                gaussian_filter_nan(
                    nap.compute_tuning_curves(
                        nap.TsGroup([cluster_spikes]),
                        np.stack([session[k] for k in wrap_list(data_type)], axis=1),
                        bins=num_bins,
                        range=bounds,
                        epochs=session["moving"].intersect(subset),
                    ),
                    sigma=smooth_sigma,
                    mode=smooth_mode,
                    keep=False,
                )[0]
                for subset in cluster_spikes.time_support.split(
                    (cluster_spikes.time_support.tot_length() / 4) - 0.5
                )
            ]

        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(
                np.array([tcs_split.values.flatten() for tcs_split in tcs_splits])
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="Mean of empty slice",
            )
            return {"stability": np.nanmean(corr[np.triu_indices(4, k=1)])}

    return wrapper


def compute_trial_based_stability(
    session_type,
    data_type,
):
    def wrapper(
        session,
        session_type,
        cluster_spikes,
        context,
        trial_types,
        bounds,
        num_bins,
        smooth_sigma,
    ):
        trials = session["trials"][session["trials"]["type"].isin(trial_types)]

        tcs_splits = []
        for trial_subset in np.array_split(np.arange(len(trials)), 4):
            with np.errstate(invalid="ignore", divide="ignore"):
                tcs_splits.append(
                    gaussian_filter_nan(
                        nap.compute_1d_tuning_curves(
                            nap.TsGroup([cluster_spikes]),
                            session[data_type],
                            nb_bins=num_bins,
                            minmax=bounds,
                            ep=session["moving"].intersect(trials[trial_subset]),
                        ),
                        sigma=(smooth_sigma, 0),
                        mode="wrap",
                        keep=False,
                    ).T
                )
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(
                np.array([tcs_split[0].flatten() for tcs_split in tcs_splits])
            )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="Mean of empty slice",
            )
            return {"stability": np.nanmean(corr[np.triu_indices(4, k=1)])}

    return wrapper
