import torch
import numpy as np
from scipy.interpolate import interp1d

def fill_inf_with_max_min(arr: np.ndarray, axis: int = 0, backend: str = "numpy", device: str = "cpu") -> np.ndarray:
    """Replace inf/-inf with max/min non-inf values along axis, in-place"""
    if backend == "torch":
        arr_torch = torch.from_numpy(arr).float().to(device) if isinstance(arr, np.ndarray) else arr
        mask_inf_pos = torch.isinf(arr_torch) & (arr_torch > 0)
        mask_inf_neg = torch.isinf(arr_torch) & (arr_torch < 0)
        if mask_inf_pos.any() or mask_inf_neg.any():
            valid = arr_torch[~torch.isinf(arr_torch)]
            if valid.numel() > 0:
                max_val = torch.nanmax(valid, dim=axis, keepdim=True).values
                min_val = torch.nanmin(valid, dim=axis, keepdim=True).values
                arr_torch[mask_inf_pos] = max_val
                arr_torch[mask_inf_neg] = min_val
        return arr_torch.cpu().numpy() if isinstance(arr, np.ndarray) else arr_torch
    else:
        mask_inf_pos = np.isinf(arr) & (arr > 0)
        mask_inf_neg = np.isinf(arr) & (arr < 0)
        if mask_inf_pos.any() or mask_inf_neg.any():
            valid = arr[~np.isinf(arr)]
            if valid.size > 0:
                max_val = np.nanmax(arr, axis=axis, keepdims=True)
                min_val = np.nanmin(arr, axis=axis, keepdims=True)
                arr[mask_inf_pos] = max_val
                arr[mask_inf_neg] = min_val
        return arr

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
    kurt = np.mean((window_no_nan - mean) ** 4) / (std ** 4) - 3
    return kurt

def interpolator_arr(arr):
    x = np.arange(len(arr))
    if np.all(np.isnan(arr)):
        return np.full_like(arr, np.nan)
    valid = ~np.isnan(arr)
    if np.sum(valid) < 2:
        return np.full_like(arr, np.nan)
    interpolator = interp1d(x[valid], arr[valid], bounds_error=False, fill_value="extrapolate")
    return interpolator(x)