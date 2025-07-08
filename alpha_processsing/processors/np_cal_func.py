import torch
import pandas as pd
import numpy as np
from functools import wraps
from scipy.interpolate import interp1d

def fill_inf_with_max_min(arr: np.ndarray, axis: int = 0, backend: str = "numpy", device: str = "cpu") -> np.ndarray:
    """Replace inf/-inf with max/min non-inf values along axis, in-place"""
    if backend == "torch":
        arr_torch = torch.from_numpy(arr).double().to(device) if isinstance(arr, np.ndarray) else arr
        mask_inf_pos = torch.isinf(arr_torch) & (arr_torch > 0)
        mask_inf_neg = torch.isinf(arr_torch) & (arr_torch < 0)
        if mask_inf_pos.any() or mask_inf_neg.any():
            mask_nan = torch.isnan(arr_torch)
            arr_torch[mask_inf_pos | mask_inf_neg] = float('nan')  # Replace inf with NaN
            mean_val = torch.nanmean(arr_torch, dim=axis, keepdim=True)
            mean_val_broadcast = torch.broadcast_to(mean_val, arr_torch.shape)
            mask_invalid = mask_nan | mask_inf_neg | mask_inf_pos
            arr_torch[mask_invalid] = mean_val_broadcast[mask_invalid] 
            valid = arr_torch[~(mask_nan|mask_inf_neg|mask_inf_pos)]
            if valid.numel() > 0:
                max_val = torch.max(arr_torch, dim=axis, keepdim=True).values
                min_val = torch.min(arr_torch, dim=axis, keepdim=True).values
                max_val_broadcast = torch.broadcast_to(max_val, arr_torch.shape)
                min_val_broadcast = torch.broadcast_to(min_val, arr_torch.shape)
                arr_torch[mask_inf_pos] = max_val_broadcast[mask_inf_pos]
                arr_torch[mask_inf_neg] = min_val_broadcast[mask_inf_neg]
                arr_torch[mask_nan] = float('nan')  # Ensure NaN remains NaN
            else:
                arr_torch[mask_inf_pos] = float('nan')
                arr_torch[mask_inf_neg] = float('nan')
        return arr_torch.cpu().numpy() if isinstance(arr, np.ndarray) else arr_torch
    else:
        mask_inf_pos = np.isinf(arr) & (arr > 0)
        mask_inf_neg = np.isinf(arr) & (arr < 0)
        arr[np.isinf(arr)] = np.nan
        if mask_inf_pos.any() or mask_inf_neg.any():
            valid = arr[~np.isnan(arr)]
            if valid.size > 0:
                max_val = np.nanmax(arr, axis=axis, keepdims=True)
                min_val = np.nanmin(arr, axis=axis, keepdims=True)
                max_val_broadcast = np.broadcast_to(max_val, arr.shape)
                min_val_broadcast = np.broadcast_to(min_val, arr.shape)
                arr[mask_inf_pos] = max_val_broadcast[mask_inf_pos]
                arr[mask_inf_neg] = min_val_broadcast[mask_inf_neg]
            else:
                arr[mask_inf_pos] = float('nan')
                arr[mask_inf_neg] = float('nan')
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

           
            x = args[0]
            is_df = isinstance(x, pd.DataFrame)
            columns = x.columns if is_df else None
            # keep track of date index if DataFrame
            if is_df:
                if isinstance(x.index, pd.DatetimeIndex):
                    date_index = x.index[0].date() if len(x.index) > 0 else None
                else:
                    date_index = x.index[0] if len(x.index) > 0 else None
                x_np = x.values
            else:
                date_index = None
                x_np = x
            # Handle input
            if process_input and isinstance(x_np, (np.ndarray, torch.Tensor)):
                x_np = fill_inf_with_max_min(x_np, axis=axis, backend=backend, device=device)

            # Execute function
            result = func(self, x_np, *args[1:], backend=backend, device=device, **kwargs)

            # Handle output
            if process_output and isinstance(result, (np.ndarray, torch.Tensor)):
                result = fill_inf_with_max_min(result, axis=axis, backend=backend, device=device)
                
            # Convert back to DataFrame if needed
            if is_df and isinstance(result, np.ndarray):
                if result.shape[0] == 1:
                    result = pd.DataFrame(result, index=[date_index], columns=columns)
                else:
                    raise ValueError("Unexpected result shape")
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

            # keep track of DataFrame properties
            is_df = isinstance(arr, pd.DataFrame)
            index, columns = None, None
            if is_df:
                index, columns = arr.index, arr.columns
                arr_np = arr.values
            else:
                arr_np = arr

            # handle input
            if process_input and isinstance(arr_np, (np.ndarray, torch.Tensor)):
                arr_np = fill_inf_with_max_min(arr_np, axis=axis, backend=backend, device=device)
            # call original function
            result = func(self, arr_np, *args[1:], **kwargs)

            # handle output
            if process_output and isinstance(result, (np.ndarray, torch.Tensor)):
                result = fill_inf_with_max_min(result, axis=axis, backend=backend, device=device)

            # Convert back to DataFrame if needed
            if is_df and isinstance(result, np.ndarray):
                result = pd.DataFrame(result, index=index, columns=columns)
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


def _bin_per_column(col: torch.Tensor, n_bins: int):
    # 统一把 inf 和 nan 替换为 nan，并保留 mask
    col = torch.nan_to_num(col, nan=float('nan'), posinf=float('nan'), neginf=float('nan'))
    finite = torch.isfinite(col)
    valid = col[finite].double()
    xmin, xmax = valid.min(), valid.max()
    edges = torch.linspace(xmin, xmax, n_bins+1, device=col.device, dtype=torch.float64)
    idx = torch.bucketize(col.double(), edges, right=False)
    idx = torch.clamp(idx, 0, n_bins)
    idx = torch.where(finite, idx, torch.zeros_like(idx))
    return idx
