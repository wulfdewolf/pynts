import numpy as np
from scipy.ndimage import gaussian_filter
import pynapple as nap
from scipy.interpolate import interp1d


def wrap_list(obj):
    return obj if isinstance(obj, list | tuple) else [obj]


def shift_circularly(arr, min_shift, max_shift):
    shift = np.random.randint(low=min_shift, high=max_shift, size=1)[0]
    shifted_arr = np.concatenate([arr[-shift:], arr[:-shift]])
    return shifted_arr


def gaussian_filter_nan(X, sigma, mode="reflect", keep=True):
    # Check if input is xarray DataArray or Dataset (duck typing)
    is_xarray = hasattr(X, "values") and hasattr(X, "dims") and hasattr(X, "coords")

    # Extract raw numpy array
    data = X.values if is_xarray else X

    V = data.copy()
    V[np.isnan(data)] = 0
    VV = gaussian_filter(V, sigma=sigma, mode=mode, truncate=6)

    W = np.ones_like(data)
    W[np.isnan(data)] = 0
    WW = gaussian_filter(W, sigma=sigma, mode=mode, truncate=6)

    Y = VV / WW
    if keep:
        Y[np.isnan(data)] = np.nan

    if is_xarray:
        # Rebuild xarray with same dims and coords
        import xarray as xr

        return xr.DataArray(Y, dims=X.dims, coords=X.coords, attrs=X.attrs)
    else:
        return Y


def interpolate_nans(tsd, pkind="cubic"):
    times = tsd.times()
    arr = tsd.values
    """
     Interpolates data to fill nan values

     Parameters:
         padata : nd array
             source data with np.NaN values

     Returns:
         nd array
             resulting data with interpolated values instead of nans
     """
    aindexes = np.arange(arr.shape[0])
    (agood_indexes,) = np.where(np.isfinite(arr))
    f = interp1d(
        agood_indexes,
        arr[agood_indexes],
        bounds_error=False,
        copy=False,
        fill_value="extrapolate",
        kind=pkind,
    )
    return nap.Tsd(d=f(aindexes), t=times)
