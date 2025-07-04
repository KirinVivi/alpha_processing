import torch
import numpy as np
from functools import wraps
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

def fill_inf_with_max_min_section_decorator(process_input=True, process_output=True):
    """Decorator to handle inf values along time and section axes"""
    def decorator(func):
        @wraps(func)
        @torch.no_grad()
        def wrapper(self, *args, **kwargs):
            backend = kwargs.get("backend", self.backend)
            device = kwargs.get("device", self.device)
            axis = self.config.get("processor_params", {}).get("level2", {}).get("inf_handling", {}).get("axis", 1)

            # Handle input
            x = args[0]
            if process_input and isinstance(x, (np.ndarray, torch.Tensor)):
                if backend == "torch" and isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).float().to(device)
                x = fill_inf_with_max_min(x, axis=axis, backend=backend, device=device)
                if backend == "torch" and isinstance(args[0], np.ndarray):
                    x = x.cpu().numpy()

            # Execute function
            result = func(self, x, *args[1:], backend=backend, device=device, **kwargs)

            # Handle output
            if process_output and isinstance(result, (np.ndarray, torch.Tensor)):
                if backend == "torch" and isinstance(result, np.ndarray):
                    result = torch.from_numpy(result).float().to(device)
                result = fill_inf_with_max_min(result, axis=axis, backend=backend, device=device)
                if backend == "torch" and isinstance(args[0], np.ndarray):
                    result = result.cpu().numpy()

            return result
        return wrapper
    return decorator

def fill_inf_with_max_min_ts_decorator(process_input=True, process_output=True):
        def decorator(func):
            @wraps(func)
            @torch.no_grad()
            def wrapper(self, *args, **kwargs):
                backend = kwargs.get("backend", self.backend)
                device = kwargs.get("device", self.device)
                axis = self.config.get("processor_params", {}).get("level1", {}).get("inf_handling", {}).get("axis", 0)
                arr = args[0]
                if process_input and isinstance(arr, (np.ndarray, torch.Tensor)):
                    if backend == "torch" and isinstance(arr, np.ndarray):
                        arr = torch.from_numpy(arr).float().to(device)
                    arr = fill_inf_with_max_min(arr, axis=axis, backend=backend, device=device)
                    if backend == "torch" and isinstance(args[0], np.ndarray):
                        arr = arr.cpu().numpy()
                result = func(self, arr, *args[1:], **kwargs)
                if process_output and isinstance(result, (np.ndarray, torch.Tensor)):
                    if backend == "torch" and isinstance(result, np.ndarray):
                        result = torch.from_numpy(result).float().to(device)
                    result = fill_inf_with_max_min(result, axis=axis, backend=backend, device=device)
                    if backend == "torch" and isinstance(args[0], np.ndarray):
                        result = result.cpu().numpy()
                return result
            return wrapper
        return decorator

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