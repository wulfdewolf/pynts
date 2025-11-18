import numpy as np
from scipy.ndimage import gaussian_filter


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
