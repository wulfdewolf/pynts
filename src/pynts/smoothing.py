import warnings
from typing import Callable, Optional

import numpy as np
import pynapple as nap
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve
from numpy.typing import ArrayLike


def gaussian_filter_nan(X, sigma, mode="reflect", keep=True):
    # Detect xarray
    is_xarray = hasattr(X, "values") and hasattr(X, "dims") and hasattr(X, "coords")
    data = X.values if is_xarray else np.asarray(X)

    # Ensure sigma is iterable
    if np.isscalar(sigma):
        sigma = [sigma] * (data.ndim if data.ndim <= 2 else data.ndim - 1)

    # Case 1: pure 1D or 2D
    if data.ndim == 1:
        kernel = Gaussian1DKernel(stddev=sigma[0])
        Y = convolve(
            data,
            kernel,
            boundary=mode,
            nan_treatment="interpolate",
            preserve_nan=keep,
            normalize_kernel=True,
        )

    elif data.ndim == 2:
        kernel = Gaussian2DKernel(x_stddev=sigma[0], y_stddev=sigma[1])
        Y = convolve(
            data,
            kernel,
            boundary=mode,
            nan_treatment="interpolate",
            preserve_nan=keep,
            normalize_kernel=True,
        )

    # Case 2: leading "unit" dimension → apply per slice
    elif data.ndim == 3:
        kernel = Gaussian2DKernel(x_stddev=sigma[0], y_stddev=sigma[1])

        Y = np.empty_like(data)
        for i in range(data.shape[0]):
            Y[i] = convolve(
                data[i],
                kernel,
                boundary=mode,
                nan_treatment="interpolate",
                preserve_nan=keep,
                normalize_kernel=True,
            )
    else:
        raise ValueError("Only supports up to 3D (with leading non-spatial axis)")

    # Restore xarray
    if is_xarray:
        import xarray as xr

        return xr.DataArray(Y, dims=X.dims, coords=X.coords, attrs=X.attrs)
    else:
        return Y


def find_optimal_smoothing(
    tuning_curve_fn,
    epoch,
    dim,
    sigma_range,
    *smooth_args,
    **smooth_kwargs,
) -> float:
    """
    Optimal smoothing via 2-fold temporal split (half-half CV).

    - Splits time support into exactly 2 equal halves
    - Computes tuning curves on each half
    - Evaluates smoothing by correlation between halves
    - Selects sigma maximizing mean symmetric correlation
    """

    # --- enforce exactly two equal splits ---
    half_t = epoch.tot_length() / 2 - 0.01
    splits = epoch.split(half_t)

    if len(splits) != 2:
        return sigma_range[0]

    tc1 = tuning_curve_fn(splits[0]).values
    tc2 = tuning_curve_fn(splits[1]).values

    def corr(a, b):
        a = a.ravel()
        b = b.ravel()
        mask = ~np.isnan(a) & ~np.isnan(b)
        if np.sum(mask) < 10:
            return np.nan
        return np.corrcoef(a[mask], b[mask])[0, 1]

    def score_sigma(sigma):
        # smooth both halves independently
        s1 = gaussian_filter_nan(
            tc1, sigma=[sigma] * dim, *smooth_args, **smooth_kwargs
        )
        s2 = gaussian_filter_nan(
            tc2, sigma=[sigma] * dim, *smooth_args, **smooth_kwargs
        )

        # symmetric comparison
        return np.nanmean(
            [
                corr(s1, tc2),
                corr(s2, tc1),
            ]
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="Mean of empty slice",
        )
        scores = np.array([score_sigma(s) for s in sigma_range])

    if np.all(np.isnan(scores)):
        return sigma_range[0]

    return sigma_range[np.nanargmax(scores)]


def apply_smoothing(
    tuning_curve_fn: Callable,
    dim: int,
    epoch: nap.IntervalSet,
    sigma_range: Optional[ArrayLike] = None,
    smooth_sigma: Optional[float | ArrayLike | str] = None,
    *smooth_args,
    **smooth_kwargs,
):
    tc = tuning_curve_fn(epoch)

    # Early exit for no smoothing
    if smooth_sigma is None or smooth_sigma is False:
        return tc, False

    with np.errstate(invalid="ignore", divide="ignore"):
        # Case 1: cross-validated sigma
        if isinstance(smooth_sigma, str):
            if smooth_sigma != "cv":
                raise ValueError(f"Unknown smooth_sigma string: {smooth_sigma}")
            if sigma_range is None:
                raise ValueError(
                    "Cross-validated smoothing request, but no range provided"
                )
            val = find_optimal_smoothing(
                tuning_curve_fn=tuning_curve_fn,
                epoch=epoch,
                dim=dim,
                sigma_range=sigma_range,
                *smooth_args,
                **smooth_kwargs,
            )
            smooth_sigma = (val,) * dim

        # Case 2: scalar → expand to all dims
        elif np.isscalar(smooth_sigma):
            smooth_sigma = (smooth_sigma,) * dim

        # Case 3: array-like
        else:
            sigma = np.atleast_1d(smooth_sigma)

            if sigma.size == 1:
                smooth_sigma = (sigma.item(),) * dim
            elif sigma.size == dim:
                smooth_sigma = tuple(sigma.tolist())
            else:
                raise ValueError(
                    f"smooth_sigma must be scalar or length {dim}, got {sigma.size}"
                )

        tc = gaussian_filter_nan(tc, smooth_sigma, *smooth_args, **smooth_kwargs)

    return tc, smooth_sigma
