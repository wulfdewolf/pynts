from sympy.parsing.sympy_parser import null
import multiprocessing as mp
import warnings
from functools import reduce
from itertools import product

import numpy as np
import pandas as pd
import pynapple as nap
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

from pynts.util import (
    gaussian_filter_nan,
    shift_circularly,
    wrap_list,
)


def find_optimal_smoothing(
    tuning_curve_fn,
    time_support,
    smoothing_range,
    mode: str,
    n_splits: int = 5,
) -> float:
    """
    Find the optimal smoothing parameter via cross-validation.

    Splits the time support into *n_splits* folds, computes tuning curves on
    each fold and its complement, then selects the sigma minimising the mean
    squared error between held-out and smoothed complement curves.

    Falls back to the smallest sigma when all scores are NaN (e.g. too few
    spikes for any fold to contain data).
    """
    splits = time_support.split(time_support.tot_length() / n_splits - 0.01)

    split_curves = [tuning_curve_fn(split) for split in splits]
    rest_curves = [
        tuning_curve_fn(
            reduce(lambda a, b: a.union(b), [s for j, s in enumerate(splits) if j != i])
        )
        for i in range(len(splits))
    ]

    def mse_for_sigma(sigma: float) -> float:
        per_fold = [
            (
                split_curve.values
                - gaussian_filter_nan(
                    rest_curve,
                    mode=mode,
                    sigma=[0] + [sigma] * (len(split_curve.shape) - 1),
                )
            )
            ** 2
            for split_curve, rest_curve in zip(split_curves, rest_curves)
        ]
        return np.nanmean(per_fold)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="Mean of empty slice"
        )
        scores = [mse_for_sigma(sigma) for sigma in smoothing_range]

    if np.all(np.isnan(scores)):
        # No fold had enough data to score any sigma; return minimum smoothing
        return smoothing_range[0]

    return smoothing_range[np.nanargmin(scores)]


def with_null_distribution(tuning_score_fn, classification_fn, n_shuffles, cv_smooth):
    """
    Decorator to compute the null distribution of a tuning score.
    """

    def wrapper(session, session_type, cluster, epoch=None, **kwargs):
        if cv_smooth:
            kwargs["smooth_sigma"] = True
        score = tuning_score_fn(
            session,
            session_type,
            cluster,
            epoch=epoch,
            **kwargs,
        )
        if np.isnan(list(score.values())[0]):
            return {**score, "sig": False, "null": pd.DataFrame([])}
        null_distribution = _compute_null_distribution(
            cluster,
            session,
            session_type,
            score,
            cv_smooth,
            tuning_score_fn,
            n_shuffles,
            epoch,
        )
        return {
            **score,
            **classification_fn(score, null_distribution),
            "null": null_distribution,
        }

    return wrapper


def for_cluster(args):
    cluster_id, session, session_type, clusters, tuning_score_fn, cluster_attributes = (
        args
    )

    if isinstance(clusters[cluster_id], nap.Tsd):
        cluster = clusters[cluster_id]
    else:
        cluster = clusters[[cluster_id]]
    tuning_results = wrap_list(tuning_score_fn(session, session_type, cluster))
    results = []
    for tuning_result in tuning_results:
        results.append(
            {
                "cluster_id": int(cluster_id),
                **{
                    cluster_attribute: clusters[cluster_attribute][cluster_id]
                    for cluster_attribute in cluster_attributes
                },
                **(
                    tuning_result.pop("null").add_prefix("null_").to_dict(orient="list")
                    if "null" in tuning_result
                    else {}
                ),
                **tuning_result,
            }
        )
    return results


def for_all_clusters(tuning_score_fn, n_workers, cluster_attributes=[]):
    def wrapper(session, session_type, clusters):
        cluster_ids = list(clusters.index)

        # Sequential path
        if n_workers == 1:
            all_results = []
            for cluster_id in tqdm(cluster_ids, unit="cluster", total=len(cluster_ids)):
                all_results.extend(
                    for_cluster(
                        (
                            cluster_id,
                            session,
                            session_type,
                            clusters,
                            tuning_score_fn,
                            cluster_attributes,
                        )
                    )
                )

        # Parallel path
        else:
            args_list = [
                (
                    cluster_id,
                    session,
                    session_type,
                    clusters,
                    tuning_score_fn,
                    cluster_attributes,
                )
                for cluster_id in cluster_ids
            ]
            # if negative use max
            _n_workers = (
                max(1, mp.cpu_count() + n_workers) if n_workers < 0 else n_workers
            )

            with Pool(nodes=_n_workers) as pool:
                results_iter = pool.imap(for_cluster, args_list)
                all_results = []
                for result in tqdm(
                    results_iter, total=len(cluster_ids), unit="cluster"
                ):
                    all_results.extend(result)

        return pd.DataFrame(all_results)

    return wrapper


def for_all_groups(tuning_score_fn, session_type, groupers):
    """
    Decorator factory that computes the tuning score for all group combinations.

    Parameters
    ----------
    tuning_score_fn : callable
        The function to compute a tuning score.
    session_type : str
        Type of session (passed through).
    groupers : dict[str, callable]
        Mapping from group name to a function that takes `session` and
        returns the iterable of values to group over.

        Example:
        {
            "context": get_bin_config(session_type)["P"]["regions"],
            "trial_types": all_unique_combinations(session["trials"]["type"].unique()),
            "performance": session["trials"]["performance"].unique()
        }
    """

    def wrapper(session, session_type, cluster):
        results = []

        for combo in product(*groupers.values()):
            group_kwargs = dict(zip(groupers.keys(), combo))
            for result in wrap_list(
                tuning_score_fn(
                    session,
                    session_type,
                    cluster,
                    **group_kwargs,
                )
            ):
                results.append({**group_kwargs, **result})
        return results

    return wrapper


def for_epochs(tuning_score_fn, session, epochs: int | dict):
    """
    Decorator to compute over given epochs.
    """
    if isinstance(epochs, int):
        all = session["S"].time_support
        epochs = {
            "all": all,
            **(
                {
                    f"epoch_{i}": ep
                    for i, ep in enumerate(
                        all.split((all.tot_length() - 0.01) / epochs)
                    )
                }
                if epochs > 1
                else {}
            ),
        }

    def wrapper(session, session_type, clusters):
        results = []
        for epoch_name, epoch in epochs.items():
            for result in wrap_list(
                tuning_score_fn(
                    session,
                    session_type,
                    clusters,
                    epoch=epoch,
                )
            ):
                results.append({"epoch": epoch_name, **result})
        return results

    return wrapper


def _compute_null_distribution(
    cluster,
    session,
    session_type,
    result,
    cv_smooth,
    tuning_score_fn,
    n_shuffles,
    epoch=None,
):
    """
    Function to compute the null distribution of a tuning score by shuffling the spikes.
    """
    pass_on = {k[1:]: result[k] for k in result if k.startswith("_")}
    pass_on["is_shuffle"] = True
    return pd.DataFrame(
        [
            tuning_score_fn(
                session,
                session_type,
                (
                    nap.shift_timestamps(
                        cluster,
                        min_shift=20.0,
                        max_shift=cluster.time_support.end[-1] - 20.0,
                    )
                    if isinstance(cluster, nap.Ts | nap.TsGroup)
                    else nap.Tsd(
                        d=shift_circularly(
                            cluster.values.flatten(),
                            min_shift=20.0,
                            max_shift=cluster.time_support.end[-1] - 20.0,
                        ),
                        t=cluster.times(),
                    )
                ),
                epoch=epoch,
                **pass_on,
            )
            for _ in range(n_shuffles)
        ]
    )


def with_shifts(
    tuning_score_fn,
    classification_fn,
    session,
    session_type,
    var,
    n_shuffles,
    cv_smooth,
    projection,
    projection_range,
):
    """
    Decorator to compute the tuning score for all projections of a given variable.
    """
    shifted_behaviour = {
        shift: projection(
            session_type,
            session,
            var,
            shift,
        )
        for shift in projection_range
    }
    shifted_behaviour = {
        shift: {sub_var: projected[sub_var] for sub_var in wrap_list(var)}
        for shift, projected in shifted_behaviour.items()
    }

    def wrapper(session, session_type, cluster, epoch=nap.IntervalSet(-np.inf, np.inf)):
        results = [
            {
                **tuning_score_fn(
                    {
                        **projected,
                        "moving": session["moving"],
                        "trials": session["trials"] if "VR" in session_type else None,
                    },
                    session_type,
                    cluster,
                    smooth_sigma=cv_smooth,
                    epoch=epoch.intersect(list(projected.values())[0].time_support),
                ),
                "shift": shift,
            }
            for shift, projected in shifted_behaviour.items()
        ]

        if not all(np.isnan(list(r.values())[0]) for r in results):
            # Compute null distribution for no travel
            zero_lag = results[list(shifted_behaviour.keys()).index(0.0)]
            zero_lag["null"] = _compute_null_distribution(
                cluster,
                {
                    **shifted_behaviour[0],
                    "moving": session["moving"],
                    "trials": session["trials"] if "VR" in session_type else None,
                },
                session_type,
                zero_lag,
                None,
                tuning_score_fn,
                n_shuffles,
                epoch,
            )
            # Classify w.r.t. best travel
            results = [
                {
                    **r,
                    **classification_fn(r, zero_lag["null"]),
                }
                for r in results
            ]

        return results

    return wrapper


def compute_direction_projected(session_type, session, var_label, shift):

    # Compute velocity (forward difference)
    dx = np.diff(session["P_x"], append=session["P_x"].values[-1])
    dy = np.diff(session["P_y"], append=session["P_y"].values[-1])

    # Compute norm
    disp = np.sqrt(dx**2 + dy**2)

    # Avoid division by zero
    vhat_x = np.zeros_like(dx)
    vhat_y = np.zeros_like(dy)

    mask = disp > 0
    vhat_x[mask] = dx[mask] / disp[mask]
    vhat_y[mask] = dy[mask] / disp[mask]

    # Project forward
    X_proj = session["P_x"].values + shift * vhat_x
    Y_proj = session["P_y"].values + shift * vhat_y

    return nap.TsdFrame(
        d=np.stack([X_proj, Y_proj], axis=1),
        t=session["P_x"].times(),
        time_support=session["P_x"].time_support,
        columns=["P_x", "P_y"],
    )


def compute_time_projected(session_type, session, var_label, shift):
    # Wrap var_label to list
    if isinstance(var_label, str):
        var_label = [var_label]

    var = (
        np.stack([session[label] for label in var_label], axis=1)
        if len(var_label) > 1
        else session[var_label[0]][:, None]
    )

    shift_bins = int(shift * var.rate)

    if shift_bins > 0:
        t = var.times()[:-shift_bins]
        d = var.values[shift_bins:]
    elif shift_bins < 0:
        t = var.times()[-shift_bins:]
        d = var.values[:shift_bins]
    else:
        t = var.times()
        d = var.values

    return nap.TsdFrame(
        t=t,
        d=d,
        columns=var_label,
    )


def compute_travel_projected(session_type, session, var_label, travel):
    # Wrap var_label to list
    if isinstance(var_label, str):
        var_label = [var_label]

    # Extract variables
    var_values = (
        np.stack([session[label] for label in var_label], axis=1)
        if len(var_label) > 1
        else session[var_label[0]][:, None]
    ).values

    # Get positions
    if "VR" in session_type:
        P = session["travel"]  # shape (T, D)
    else:
        P = np.stack([session["P_x"], session["P_y"]], axis=1)  # (T, 2)

    times = P.times() if hasattr(P, "times") else np.arange(len(P))

    # Compute cumulative distances
    deltas = np.diff(P, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1) if P.ndim > 1 else np.abs(deltas)
    cum_distances = np.insert(np.cumsum(segment_lengths), 0, 0)

    # Target distances for projection
    target_distances = cum_distances + travel  # vector of length T

    # Clip to bounds
    target_distances = np.clip(target_distances, cum_distances[0], cum_distances[-1])

    # Interpolate each dimension
    projected_values = np.empty_like(var_values)
    for dim in range(var_values.shape[1]):
        projected_values[:, dim] = np.interp(
            target_distances, cum_distances, var_values[:, dim]
        )

    return nap.TsdFrame(t=times, d=projected_values, columns=var_label)
