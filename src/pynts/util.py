import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


def wrap_list(obj):
    return obj if isinstance(obj, list | tuple) else [obj]


def shift_circularly(arr, min_shift, max_shift):
    shift = np.random.randint(low=min_shift, high=max_shift, size=1)[0]
    shifted_arr = np.concatenate([arr[-shift:], arr[:-shift]])
    return shifted_arr


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
