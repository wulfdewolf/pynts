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


def find_optimal_smoothing(tuning_curve_fn, time_support, smoothing_range, mode):
    """
    Function to find the optimal smoothing parameter for a given tuning curve function.
    """

    splits = time_support.split(time_support.tot_length() / 4 - 0.01)

    split_curves = [tuning_curve_fn(split) for split in splits]
    rest_curves = [
        tuning_curve_fn(
            reduce(lambda a, b: a.union(b), [s for j, s in enumerate(splits) if j != i])
        )
        for i in range(len(splits))
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="Mean of empty slice"
        )
        scores = [
            np.nanmean(
                [
                    (
                        split_curve
                        - gaussian_filter_nan(
                            rest_curve,
                            mode=mode,
                            sigma=[0] + [sigma] * (len(split_curve.shape) - 1),
                        )
                    )
                    ** 2
                    for split_curve, rest_curve in zip(split_curves, rest_curves)
                ]
            )
            for sigma in smoothing_range
        ]
        if np.all(np.isnan(scores)):
            return smoothing_range[0]
        else:
            return smoothing_range[np.nanargmin(scores)]


def with_null_distribution(tuning_score_fn, classification_fn, n_shuffles, cv_smooth):
    """
    Decorator to compute the null distribution of a tuning score.
    """

    def wrapper(session, session_type, cluster_spikes, epoch=None, **kwargs):
        if cv_smooth:
            kwargs["smooth_sigma"] = True
        score = tuning_score_fn(
            session,
            session_type,
            cluster_spikes,
            epoch=epoch,
            **kwargs,
        )
        if np.isnan(list(score.values())[0]):
            return {**score, "sig": False, "null": pd.DataFrame([])}
        null_distribution = _compute_null_distribution(
            cluster_spikes,
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

    tuning_results = wrap_list(
        tuning_score_fn(session, session_type, clusters[cluster_id])
    )
    results = []
    for tuning_result in tuning_results:
        results.append(
            {
                "cluster_id": cluster_id,
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

    def wrapper(session, session_type, cluster_spikes):
        results = []

        for combo in product(*groupers.values()):
            group_kwargs = dict(zip(groupers.keys(), combo))
            for result in wrap_list(
                tuning_score_fn(
                    session,
                    session_type,
                    cluster_spikes,
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
    behaviour,
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
    return pd.DataFrame(
        [
            tuning_score_fn(
                behaviour,
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

    def wrapper(session, session_type, cluster_spikes, epoch=None):
        results = [
            {
                **tuning_score_fn(
                    {
                        **projected,
                        "moving": session["moving"],
                        "trials": session["trials"] if "VR" in session_type else None,
                    },
                    session_type,
                    cluster_spikes,
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
                cluster_spikes,
                {
                    **shifted_behaviour[0],
                    "moving": session["moving"],
                    "trials": session["trials"] if "VR" in session_type else None,
                },
                session_type,
                zero_lag,
                cv_smooth,
                tuning_score_fn,
                n_shuffles,
                epoch,
            )
            # Compute a bootstrapped max null distribution, store it in best lag
            best = results[np.nanargmax([list(r.values())[0] for r in results])]
            bootstrap = np.random.randint(0, n_shuffles, size=(n_shuffles, n_shuffles))
            best["null"] = pd.DataFrame(
                {
                    list(zero_lag.keys())[0]: np.asarray(
                        zero_lag["null"][list(zero_lag.keys())[0]]
                    )[bootstrap].max(axis=1)
                }
            )
            # Classify w.r.t. best travel
            results = [
                {
                    **r,
                    **classification_fn(r, best["null"]),
                }
                for r in results
            ]

        return results

    return wrapper


def compute_travel_projected(session_type, session, var_label, travel):
    var_values = (
        session[var_label]
        if len(var_label) == 1
        else np.stack([session[sub_var] for sub_var in wrap_list(var_label)], axis=1)
    )
    P = (
        session["travel"]
        if "VR" in session_type
        else np.stack([session["P_x"], session["P_y"]], axis=1)
    )
    n = len(P)

    # Compute cumulative distances based on dimensionality
    if P.ndim == 1:
        segment_lengths = np.abs(np.diff(P))
    else:
        deltas = np.diff(P, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)

    cum_distances = np.insert(np.cumsum(segment_lengths), 0, 0)

    projected_values = []
    valid_times = []
    j = 0
    times = P.times()

    for i in range(n):
        target_distance = cum_distances[i] + travel

        # Advance j until we find the segment that contains the projected distance
        while j < n and cum_distances[j] < target_distance:
            j += 1

        if j >= n:
            break  # Stop if out of bounds

        d1 = cum_distances[j - 1]
        d2 = cum_distances[j]
        t = (target_distance - d1) / (d2 - d1)

        interp_val = var_values[j - 1] + t * (var_values[j] - var_values[j - 1])

        projected_values.append(interp_val)
        valid_times.append(times[i])

    return nap.TsdFrame(t=valid_times, d=projected_values, columns=wrap_list(var_label))
