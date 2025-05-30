import numpy as np
from scipy.interpolate import interp1d

def cal_skew(window):
    window_no_nan = window[~np.isnan(window)]
    n = len(window_no_nan)
    if n < 3:
        return np.nan
    mean = np.mean(window_no_nan)
    std = np.std(window_no_nan, ddof=1)
    if std == 0:
        return 0.0
    skew = np.mean((window_no_nan - mean) ** 3) / (std ** 3)
    return skew

def cal_kurt(window):
    window_no_nan = window[~np.isnan(window)]
    n = len(window_no_nan)
    if n < 4:
        return np.nan
    mean = np.mean(window_no_nan)
    std = np.std(window_no_nan, ddof=1)
    if std == 0:
        return 0.0
    m4 = np.mean((window_no_nan - mean) ** 4)
    kurt = (n * (n + 1) * m4) / ((n - 1) * (n - 2) * (n - 3) * (std ** 4)) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return kurt

def sliding_window(arr, window, axis=0):
    if arr.shape[axis] < window:
        shape = list(arr.shape)
        shape[axis] = 1
        return np.full(shape, np.nan)
    shape = list(arr.shape)
    shape[axis] = arr.shape[axis] - window + 1
    shape.insert(axis + 1, window)
    strides = list(arr.strides)
    strides.insert(axis + 1, strides[axis])
    return np.lib.stride_tricks.as_strided(arr, shape=tuple(shape), strides=tuple(strides))

def fill_inf(arr):
    if np.all(np.isnan(arr)):
        return arr
    arr_filled = arr.copy()
    pos_inf = np.isposinf(arr)
    neg_inf = np.isneginf(arr)
    valid = arr[~(pos_inf | neg_inf | np.isnan(arr))]
    if valid.size > 0:
        arr_filled[pos_inf] = np.nanmax(valid)
        arr_filled[neg_inf] = np.nanmin(valid)
    else:
        arr_filled[:] = np.nan
    return arr_filled

def fill_inf_with_max_min(array, axis):
    mask = ~np.all(np.isnan(array), axis=axis)
    if np.sum(mask) <= 1:
        return array
    if axis == 0:
        res_unnan = np.apply_along_axis(fill_inf, axis, array[:, mask])
        res = np.full(array.shape, np.nan)
        res[:, mask] = res_unnan
    else:
        res_unnan = np.apply_along_axis(fill_inf, axis, array[mask, :])
        res = np.full(array.shape, np.nan)
        res[mask, :] = res_unnan
    return res

def interpolator_arr(arr):
    x = np.arange(len(arr))
    if np.all(np.isnan(arr)):
        return np.full_like(arr, np.nan)
    valid = ~np.isnan(arr)
    if np.sum(valid) < 2:
        return np.full_like(arr, np.nan)
    interpolator = interp1d(x[valid], arr[valid], bounds_error=False, fill_value="extrapolate")
    return interpolator(x)
