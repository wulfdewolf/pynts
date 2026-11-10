import multiprocessing as mp
import time
import warnings
from collections.abc import Callable, Iterable
from functools import reduce
from itertools import combinations, product
from typing import Any, Optional

import numpy as np
import pandas as pd
import pynapple as nap
from numpy.typing import ArrayLike
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

from pynts.bounds import apply_variable_bounds, get_variable_bounds
from pynts.util import shift_circularly, wrap_list


def with_null_distribution(
    tuning_score_fn, classification_fn, n_shuffles, *args, **kwargs
):
    """
    Decorator to compute the null distribution of a tuning score.
    """

    def wrapper(session, session_type, cluster, epoch=None, *args, **kwargs):
        score = tuning_score_fn(
            session,
            session_type,
            cluster,
            epoch=epoch,
            *args,
            **kwargs,
        )
        val = list(score.values())[0]
        if np.isnan(val).all():
            return {
                **score,
                "sig": [False] if isinstance(val, list) else False,
                "null": pd.DataFrame([]),
            }
        if "_smooth_sigma" in score:
            kwargs["smooth_sigma"] = score["_smooth_sigma"]
        null_distribution = _compute_null_distribution(
            cluster,
            session,
            session_type,
            score,
            tuning_score_fn,
            n_shuffles,
            epoch,
            *args,
            **kwargs,
        )
        return {
            **score,
            **classification_fn(score, null_distribution),
            "null": null_distribution,
        }

    return wrapper


def for_cluster(args):
    (
        cluster_id,
        session,
        session_type,
        clusters,
        tuning_score_fn,
        region_selection,
        cluster_attributes,
        args,
        kwargs,
    ) = args

    if isinstance(clusters[cluster_id], nap.Tsd):
        cluster = clusters[cluster_id]
    else:
        cluster = clusters[[cluster_id]]

    # Check region selection
    if (
        region_selection is not None
        and clusters["brain_region"][cluster_id] not in region_selection
    ):
        print("Skipping because cluster not in region selection...")
        return []

    tuning_results = wrap_list(
        tuning_score_fn(session, session_type, cluster, *args, **kwargs)
    )
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


def for_all_clusters(
    tuning_score_fn,
    n_workers,
    cluster_attributes=[],
    region_selection=None,
    *args,
    **kwargs,
):
    def wrapper(session, session_type, clusters):
        if region_selection is not None:
            cluster_ids = clusters[clusters["brain_region"].isin(region_selection)]
        else:
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
                            region_selection,
                            cluster_attributes,
                            args,
                            kwargs,
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
                    region_selection,
                    cluster_attributes,
                    args,
                    kwargs,
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


def for_all_groups(tuning_score_fn, session_type, groupers, *args, **kwargs):
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
                    session, session_type, cluster, *args, **group_kwargs, **kwargs
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

    def wrapper(session, session_type, clusters, *args, **kwargs):
        results = []
        for epoch_name, epoch in epochs.items():
            for result in wrap_list(
                tuning_score_fn(
                    session,
                    session_type,
                    clusters,
                    epoch=epoch,
                    skip_null=epoch_name != "all",
                    *args,
                    **kwargs,
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
    tuning_score_fn,
    n_shuffles,
    epoch=None,
    *args,
    **kwargs,
):
    """
    Function to compute the null distribution of a tuning score by shuffling the spikes.
    """
    kwargs["is_shuffle"] = True
    for k in result:
        if k.startswith("_"):
            kwargs[k[1:]] = result[k]
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
                *args,
                **kwargs,
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
    projection,
    projection_range,
    skip_null=False,
    *args,
    bounds: ArrayLike | None = None,
    **kwargs,
) -> Callable[..., list[dict[str, Any]]]:
    """Compute tuning scores across bounded projections."""
    variables = list(wrap_list(var))
    bounds = get_variable_bounds(
        session,
        variables,
        bounds if bounds is not None else kwargs.get("true_bounds"),
    )

    shifted_behaviour = {}
    for shift in projection_range:
        projected = projection(session_type, session, var, shift)
        values = np.column_stack([projected[variable].values for variable in variables])
        projected = nap.TsdFrame(
            t=projected.times(),
            d=apply_variable_bounds(values, variables, bounds),
            columns=variables,
            time_support=projected.time_support,
        )
        shifted_behaviour[shift] = {
            variable: projected[variable] for variable in variables
        }

    def wrapper(
        session,
        session_type,
        cluster,
        epoch=nap.IntervalSet(-np.inf, np.inf),
        skip_null=skip_null,
    ):
        results = [
            {
                **tuning_score_fn(
                    {
                        **projected,
                        "moving": session["moving"],
                        "trials": session["trials"] if "trials" in session else None,
                    },
                    session_type,
                    cluster,
                    epoch=epoch.intersect(cluster.time_support),
                    *args,
                    **kwargs,
                ),
                "shift": shift,
            }
            for shift, projected in shifted_behaviour.items()
        ]

        if not skip_null and not all(np.isnan(list(r.values())[0]) for r in results):
            # Compute null distribution for no travel
            zero_lag = results[list(shifted_behaviour.keys()).index(0.0)]
            zero_lag["null"] = _compute_null_distribution(
                cluster,
                {
                    **shifted_behaviour[0],
                    "moving": session["moving"],
                    "trials": session["trials"] if "trials" in session else None,
                },
                session_type,
                zero_lag,
                tuning_score_fn,
                n_shuffles,
                epoch,
                *args,
                **kwargs,
            )
            # Classify w.r.t. best travel
            results = [
                {
                    **r,
                    **classification_fn(r, zero_lag["null"]),
                }
                for r in results
            ]

        # Correct p values
        if "p_val" in results[0]:
            pvals = [r["p_val"] for r in results]
            _, pvals_fdr, _, _ = multipletests(
                pvals,
                method="fdr_bh",
            )
            for r, p in zip(results, pvals_fdr):
                r["p_val_fdr"] = p

        return results

    return wrapper


def with_projection(
    tuning_score_fn: Callable[..., dict[str, Any]],
    session: dict,
    session_type: str,
    var: str | Iterable[str],
    projections: dict[str, tuple[Callable[..., Any], Iterable[float]]],
    **fit_kwargs: Any,
) -> Callable[..., list[dict[str, Any]]]:
    """Select projection shifts by CV and compare held-out likelihoods."""
    variables = list(wrap_list(var))
    fit_kwargs = dict(fit_kwargs)
    bounds = get_variable_bounds(session, variables, fit_kwargs.get("true_bounds"))
    fit_kwargs["true_bounds"] = bounds

    projected_sessions = {}
    for name, (projection, shifts) in projections.items():
        candidates = []
        for shift in shifts:
            projected = projection(session_type, session, variables, shift)
            values = np.column_stack(
                [projected[variable].values for variable in variables]
            )
            projected = nap.TsdFrame(
                t=projected.times(),
                d=apply_variable_bounds(values, variables, bounds),
                columns=variables,
                time_support=projected.time_support,
            )
            candidates.append(
                (
                    shift,
                    {
                        **session,
                        **{variable: projected[variable] for variable in variables},
                    },
                )
            )
        projected_sessions[name] = candidates

    common_epoch = session[variables[0]].time_support
    for candidates in projected_sessions.values():
        for _, projected_session in candidates:
            for variable in variables:
                common_epoch = common_epoch.intersect(
                    projected_session[variable].time_support
                )

    def wrapper(
        session: dict,
        session_type: str,
        cluster: nap.TsGroup,
        epoch: Optional[nap.IntervalSet] = None,
    ) -> list[dict[str, Any]]:
        shared_epoch = common_epoch.intersect(cluster.time_support)
        if epoch is not None:
            shared_epoch = shared_epoch.intersect(epoch)

        results = []
        total_fits = sum(len(candidates) for candidates in projected_sessions.values())
        split_fields = (
            "test_pr2",
            "test_score",
            "null_score",
            "test_n_bins",
        )

        with tqdm(
            total=total_fits,
            desc="Fitting projections",
            unit="fit",
        ) as progress:
            for name, candidates in projected_sessions.items():
                projection_start = time.perf_counter()

                for shift, projected_session in candidates:
                    progress.set_description(f"Fitting {name}, shift={shift:g}")

                    candidate_kwargs = {
                        **fit_kwargs,
                        "projection_bounds": np.array(
                            [
                                [
                                    np.nanmin(projected_session[variable].values),
                                    np.nanmax(projected_session[variable].values),
                                ]
                                for variable in variables
                            ],
                            dtype=float,
                        ),
                    }
                    result = tuning_score_fn(
                        projected_session,
                        session_type,
                        cluster,
                        correlates=variables,
                        epoch=shared_epoch,
                        **candidate_kwargs,
                    )

                    results.append(
                        {
                            **result,
                            **{
                                field: np.asarray(result[field]).tolist()
                                for field in split_fields
                            },
                            "projection": name,
                            "shift": shift,
                        }
                    )
                    progress.update()

                progress.write(
                    f"{name}: {len(candidates)} fits in "
                    f"{time.perf_counter() - projection_start:.1f}s"
                )

        return results

    return wrapper
