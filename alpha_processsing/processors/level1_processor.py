import numpy as np
import pandas as pd
import torch
from functorch import vmap
from functools import wraps
from joblib import Parallel, delayed
from .np_cal_func import fill_inf_with_max_min_ts_decorator, _bin_per_column
from .filter_func import (
    dwt_ca_fixed, dwt_da_fixed, dwt_ca, dwt_da, savgol_filter, median_filter,
    lowpass_filter, highpass_filter, bandpass_filter,wavelet_denoise, ewma_filter,
    kalman_filter, hp_filter, robust_zscore_filter, interpolator_arr, rolling_rank_filter, 
    kaufman_adaptive_moving_average, mesa_adaptive_moving_average, hilbert_transform_instantaneous_phase, 
    fft_amp_interp1d, fft_angle_interp1d, fft_real_interp1d
)



class ProcessCalculator:
    def cal_functions(self, result: np.ndarray, func_str: tuple, backend="torch", device="cpu") -> np.ndarray:
        from ..utils.data_utils import extract_values_from_string
        if not isinstance(func_str, tuple):
            para_check = extract_values_from_string(func_str)
            if para_check:
                func_name, *para = para_check
                func = getattr(self, func_name)
                result = func(result, *para, backend=backend, device=device)
            else:
                func = getattr(self, func_str)
                result = func(result, backend=backend, device=device)
        else:
            for func_name_para in func_str:
                para_check = extract_values_from_string(func_name_para)
                if para_check:
                    func_name, *para = para_check
                    func = getattr(self, func_name)
                    result = func(result, *para, backend=backend, device=device)
                else:
                    func = getattr(self, func_name_para)
                    result = func(result, backend=backend, device=device)
        result[np.isinf(result)] = np.nan
        return result

    def specialized_output_shape(self, input_arr: np.ndarray, output_arr: np.ndarray) -> np.ndarray:
        padding = input_arr.shape[0] - output_arr.shape[0]
        if padding > 0:
            output_arr = np.vstack([output_arr, np.full((padding, output_arr.shape[1]), np.nan)])
        return output_arr

    def check_valid(self, arr: np.ndarray) -> np.ndarray:
        if np.all(np.isnan(arr)):
            return None
        non_nan_values = arr[~np.isnan(arr)]
        return arr if not np.all(non_nan_values == 0) else None

class ProcessCalculatorL1(ProcessCalculator):
    def __init__(self, config):
        self.config = config
        self.backend = config.get("backend", "torch")
        self.device = config.get("device", "cpu")
        self.n_jobs = config.get("n_jobs", -1)
        self.shrink_factor = config.get("processor_params", {}).get("level1", {}).get("shrink_factor", 1.0)
        torch.set_num_threads(8)

    def _safe_normalize(self, array: np.ndarray, backend: str = "numpy", device: str = "cpu") -> np.ndarray:
        max_abs = np.log(np.finfo(np.float64).max) / 2
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            array = torch.clamp(array / self.shrink_factor, -max_abs, max_abs)
            return array
        else:
            array = np.clip(array / self.shrink_factor, -max_abs, max_abs)
            return array
        
    def _normalize(self, array: np.ndarray, shrink_value: float=1, backend: str = "numpy", device: str = "cpu") -> np.ndarray: 
        # Vectorized version for speed
        # Handle all-NaN columns to avoid warnings
        all_nan_mask = np.all(np.isnan(array), axis=0, keepdims=True)
        with np.errstate(all='ignore'):
            max_vals = np.nanmax(array, axis=0, keepdims=True)
            min_vals = np.nanmin(array, axis=0, keepdims=True)
        max_vals = np.where(all_nan_mask, 0, max_vals)
        min_vals = np.where(all_nan_mask, 0, min_vals)
        ranges = max_vals - min_vals
        # Avoid division by zero and NaN propagation
        with np.errstate(invalid='ignore', divide='ignore'):
            norm_arr = (array - min_vals) / ranges
            norm_arr = np.where(np.abs(ranges) < 1e-9, 0.5, norm_arr)
        mapped_arr = norm_arr * 2 * shrink_value - shrink_value
        mapped_arr[np.isnan(array)] = np.nan
        mapped_arr[:, all_nan_mask[0]] = np.nan
        return mapped_arr

    def _is_all_nan(self, array: np.ndarray, axis: int = None) -> np.ndarray:
        return np.all(np.isnan(array), axis=axis)

    @fill_inf_with_max_min_ts_decorator()
    def abs_v(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            result = torch.abs(array).cpu().numpy()
        else:
            result = np.abs(array)
        return result

    @fill_inf_with_max_min_ts_decorator()
    def reciprocal(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).double().to(device)
            result = torch.where(array != 0, 1 / array, torch.tensor(float('nan'), device=device)).cpu().numpy()
        else:
            result = np.where(array != 0, 1 / array, np.nan)
        return result

    @fill_inf_with_max_min_ts_decorator()
    def sigmoid(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            result = torch.sigmoid(array).cpu().numpy()
        else:
            result = 1 / (1 + np.exp(-array))
        return result

    @fill_inf_with_max_min_ts_decorator()
    def exp(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            result = torch.exp(array).cpu().numpy()
        else:
            result = np.exp(array)
        return result
    
    @fill_inf_with_max_min_ts_decorator()
    def sec_zsc(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        """
        Compute the z-score of each column in the array, handling NaN values.
        """
        if backend == "torch":
            array = torch.from_numpy(array).double().to(device)
            mean = torch.nanmean(array, dim=1, keepdim=True)
            mask = ~torch.isnan(array)
            count = mask.sum(dim=1, keepdim=True)
            diff = array - mean
            diff[~mask] = 0  # 忽略 NaN 的差值
            var = (diff ** 2).sum(dim=1, keepdim=True) / (count - 1)
            std = torch.sqrt(var)
            result = (array - mean) / std
        else:
            mean = np.nanmean(array, axis=1, keepdims=True)
            std = np.nanstd(array, axis=1, keepdims=True)
            result = (array - mean) / std
        return result

    @fill_inf_with_max_min_ts_decorator()
    def tan(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            result = torch.tan(array).cpu().numpy()
        else:
            result = np.tan(array)
        return result

    @fill_inf_with_max_min_ts_decorator()
    def sin(self, array: np.ndarray, backend="numpy", device="cpu") ->np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).double().to(device)
            result = torch.sin(array).cpu().numpy()
        else:
            result = np.sin(array)
        return result

    @fill_inf_with_max_min_ts_decorator()
    def cos(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).double().to(device)
            result = torch.cos(array).cpu().numpy()
        else:
            result = np.cos(array)
        return result

    @fill_inf_with_max_min_ts_decorator()
    def asin(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        array = self._safe_normalize(array, backend=backend, device=device)
        if backend == "torch":
            result = torch.asin(array.double()).cpu().numpy()
        else:
            result = np.arcsin(array)
        return result
    
    @fill_inf_with_max_min_ts_decorator()
    def acos(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        array = self._safe_normalize(array, backend=backend, device=device)
        if backend == "torch":
            result = torch.acos(array.double()).cpu().numpy()
        else:
            result = np.arccos(array)
        return result
    
    @fill_inf_with_max_min_ts_decorator()
    def atan(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        array = self._safe_normalize(array, backend=backend, device=device)
        if backend == "torch":
            result = torch.atan(array.double()).cpu().numpy()
        else:
            result = np.arctan(array)
        return result
    
    @fill_inf_with_max_min_ts_decorator()
    def lg1p(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        """
        Compute the natural logarithm of one plus the input array element-wise.
        """
        if backend == "torch":
            array = torch.from_numpy(array).double().to(device)
            result = torch.log1p(array).cpu().numpy()       
        else:
            result = np.log1p(array)        
        return result
    
    
    @fill_inf_with_max_min_ts_decorator()
    def sinh(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        array = self._safe_normalize(array, backend=backend, device=device)
        if backend == "torch":
            result = torch.sinh(array.double()).cpu().numpy()
        else:
            result = np.sinh(array)
        return result

    @fill_inf_with_max_min_ts_decorator()
    def cosh(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        array = self._safe_normalize(array, backend=backend, device=device)
        if backend == "torch":
            result = torch.cosh(array.double()).cpu().numpy()
        else:
            result = np.cosh(array)
        return result

    @fill_inf_with_max_min_ts_decorator()
    def tanh(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).double().to(device)
            result = torch.tanh(array).cpu().numpy()
        else:
            result = np.tanh(array)
        return result
    @fill_inf_with_max_min_ts_decorator()
    def pct_rank(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            rank = torch.argsort(torch.argsort(array, dim=1), dim=1).float()
            valid_count = torch.sum(~torch.isnan(array), dim=1, keepdim=True)
            result = torch.where(valid_count != 1, rank / valid_count, torch.tensor(float('nan'), device=device)).cpu().numpy()
        else:
            result = np.apply_along_axis(
                lambda x: np.argsort(np.argsort(x)) / np.sum(~np.isnan(x)), 1, array
            )
            result = np.where(np.isnan(array), np.nan, result)
        return result

    @fill_inf_with_max_min_ts_decorator()
    def bin(self, array: np.ndarray, bins=10, backend="numpy", device="cpu") -> np.ndarray:
        """Bin the values of the array into specified number of bins.
        Args:
            array (np.ndarray): Input array to be binned.
            bins (int): Number of bins to create.
            backend (str): Backend to use ('numpy' or 'torch').
            device (str): Device to use for torch backend ('cpu' or 'cuda').
        """
        finite = np.isfinite(array)
        valid = array[:, finite.all(axis=0)]
        if not finite.any():
            return np.full_like(array, np.nan)
        if backend == "torch":
           x = torch.from_numpy(valid).float().to(device)
           bin_fn = lambda col: _bin_per_column(col, bins)
           idxs = vmap(bin_fn)(x).cpu().numpy()
           array[:,finite.all(axis=0)] = idxs
        else: # numpy backend
            
            mins = np.nanmin(valid, axis=0)
            maxs = np.nanmax(valid, axis=0)
        
            # generate edges for bins
            edges = np.linspace(0, 1, bins+1)[:, None]  # normalized
            edges = mins[None, :] + edges * (maxs - mins)[None, :]  # [sig, bins+1]

            # broadcast array to match edges
            x3 = valid[:, :, None]  # [s, t, 1]
            edges3 = edges.T[None, :, :]  # [s, 1, b+1]

            # judge x3 locates whether in edges3
            inds = np.sum(x3 >= edges3, axis=2)  # result in 0..n_bins
            inds = np.minimum(inds, bins)
            array[:,finite.all(axis=0)] = inds

        return array

    @fill_inf_with_max_min_ts_decorator()
    def diff(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            result = torch.diff(array, dim=0, prepend=torch.full((1, array.shape[1]), float('nan'), device=device)).cpu().numpy()
        else:
            result = np.diff(array, axis=0, prepend=np.full((1, array.shape[1]), np.nan))
        return self.specialized_output_shape(array, result)
    # Todo:check the dim problem
    @fill_inf_with_max_min_ts_decorator()
    def apo(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            result = torch.abs(array - torch.mean(array, dim=0, keepdim=True)).cpu().numpy()
        else:
            result = np.abs(array - np.nanmean(array, axis=0, keepdims=True))
        return result
    
    # Todo: shape wrong
    @fill_inf_with_max_min_ts_decorator()
    def aroonosc(self, array: np.ndarray, period: int = 4, backend="numpy", device="cpu") -> np.ndarray:
        """
        Calculate Aroon Oscillator for a given period
        calculation formula: 
        ((Aroon Up - Aroon Down) / (Aroon Up + Aroon Down)) * 100
        Aroon Up = (period - index of last highest high) / period * 100
        Aroon Down = (period - index of last lowest low) / period * 100
        Args:
            array (np.ndarray): Input array with shape (n_samples, n_features)
            period (int): Period for Aroon calculation
            backend (str): Backend to use ('numpy' or 'torch')
            device (str): Device to use for torch backend ('cpu' or 'cuda') 
        """
        valid_mask = np.isfinite(array).any(axis=0)
        valid_arr = array[:, valid_mask]
        if backend == "torch":
            x = torch.from_numpy(valid_arr).double().to(device)  # [T, F]
            T, F = x.shape
            if T < period:
                return np.full_like(array, np.nan, dtype=np.float64)

            xw = x.unfold(0, period, 1)  # [T-period+1, F, period]
            idx_high = xw.argmax(dim=2)  # [T-period+1, F]
            idx_low = xw.argmin(dim=2)

            since_high = (period - 1) - idx_high
            since_low = (period - 1) - idx_low

            up = (period - since_high).double() / period * 100
            down = (period - since_low).double() / period * 100
            out = up - down  # [T-period+1, F]

            pad = torch.full((period-1, F), float('nan'), device=device)
            res = torch.cat([pad, out], dim=0).cpu().numpy()  # [T, F]
            result = np.full_like(array, np.nan, dtype=np.float64)
            result[:, valid_mask] = res
            return result
        else:
            T, F = valid_arr.shape
            result = np.full_like(array, np.nan, dtype=np.float64)
            if T < period:
                return result

            sw = np.lib.stride_tricks.sliding_window_view(valid_arr, period, axis=0)
            idx_high = sw.argmax(axis=2)  # [T-period+1, F]
            idx_low = sw.argmin(axis=2)

            since_high = (period - 1) - idx_high
            since_low = (period - 1) - idx_low

            up = (period - since_high).astype(np.float64) / period * 100
            down = (period - since_low).astype(np.float64) / period * 100
            out = up - down  # [T-period+1, F]

            result[period-1:, valid_mask] = out
            return result
    
    

    @fill_inf_with_max_min_ts_decorator()
    def rollmean(self, array: np.ndarray, window: int, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).double().to(device)
            kernel = torch.ones(1, 1, window, device=device) / window
            # array shape: (time, features) -> (features, time)
            array_t = array.T.unsqueeze(1)  # (features, 1, time)
            result = torch.conv1d(
                array_t,
                kernel,
                padding=0  # mode='valid' equivalent: no padding
            ).squeeze(1).T.cpu().numpy()  # back to (time, features)
            
        else:
            result = np.apply_along_axis(
                lambda x: np.convolve(x, np.ones(window)/window, mode='valid'), 0, array
            )
        result = np.pad(result, pad_width=((window - 1, 0), (0, 0)), mode='constant', constant_values=np.nan)
        return result

    @fill_inf_with_max_min_ts_decorator()
    def rollmax(self, array: np.ndarray, window: int, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            x = torch.from_numpy(array).float().to(device)  # shape [T, F]
            T, F = x.shape
            x_t = x.T.unsqueeze(1)  # [F,1,T]
            # no padding
            out = torch.nn.functional.max_pool1d(x_t, kernel_size=window, stride=1, padding=0)
            out = out.squeeze(1).T  # shape [T-window+1, F]
            # pad head with NaN to align windows to the end
            pad_len = window - 1
            pad = torch.full((pad_len, F), float('nan'), dtype=out.dtype, device=device)
            result = torch.cat([pad, out], dim=0)
            return result.cpu().numpy()
        else:
            arr = array
            out = np.full_like(arr, fill_value=np.nan, dtype=float)
            vals = np.apply_along_axis(
                lambda x: np.lib.stride_tricks.sliding_window_view(x, window).max(axis=1),
                axis=0, arr=arr
            )
            out[window-1:] = vals
        return out
    
    #Todo: implement entropy
    @fill_inf_with_max_min_ts_decorator()
    def entropy(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            result = -torch.sum(array * torch.log(array + 1e-10), dim=0).cpu().numpy()
        else:
            result = -np.nansum(array * np.log(array + 1e-10), axis=0)
        return result
    
    @fill_inf_with_max_min_ts_decorator()
    def rollmin(self, array: np.ndarray, window: int, backend="numpy", device="cpu") -> np.ndarray:
    
        if backend == "torch":
            x = torch.from_numpy(array).float().to(device)  # shape [T, F]
            T, F = x.shape
            x_neg = -x  # 将 min 转换为 -max
            x_t = x_neg.T.unsqueeze(1)  # [F,1,T]  # [F,1,T]
            # no padding
            out = torch.nn.functional.max_pool1d(x_t, kernel_size=window, stride=1, padding=0)
            out = out.squeeze(1).T  # shape [T-window+1, F]
            out = -out
            # pad head with NaN to align windows to the end
            pad_len = window - 1
            pad = torch.full((pad_len, F), float('nan'), dtype=out.dtype, device=device)
            result = torch.cat([pad, out], dim=0)
            return result.cpu().numpy()
        else:
            arr = array
            out = np.full_like(arr, fill_value=np.nan, dtype=float)
            vals = np.apply_along_axis(
                lambda x: np.lib.stride_tricks.sliding_window_view(x, window).min(axis=1),
                axis=0, arr=arr
            )
            out[window-1:] = vals
            return out
    

    @fill_inf_with_max_min_ts_decorator()
    def rollzscore(self, array: np.ndarray, window: int, backend="numpy", device="cpu") -> np.ndarray:
        mean = self.rolling_mean(array, window, backend=backend, device=device)
        std = self.rolling_std(array, window, backend=backend, device=device)
        return np.where(std != 0, (array - mean) / std, np.nan)

    @fill_inf_with_max_min_ts_decorator()
    def rollstd(self, array: np.ndarray, window: int, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            x = torch.from_numpy(array).double().to(device)
            T, F = x.shape
            # 使用滑动窗口视图在 PyTorch 中无法实现，改用 conv1d 累积法
            # sum_x 和 sum_x2
            weight = torch.ones(1, 1, window, device=device)
            x2 = x * x

            sx = torch.conv1d(x.T.unsqueeze(1), weight, stride=1, padding=0).squeeze(1)  # [F, T-window+1]
            sx2 = torch.conv1d(x2.T.unsqueeze(1), weight, stride=1, padding=0).squeeze(1)

            # 基于公式 var = (sum(x^2) - sum(x)^2/N) / (N-1)
            N = window
            var = (sx2 - sx * sx / N) / (N - 1)
            std = torch.sqrt(var)

            # 填充前 window-1 行 NaN
            pad = window - 1
            nanpad = torch.full((pad, F), float('nan'), dtype=std.dtype, device=device)
            result = torch.cat([nanpad, std.T], dim=0)
            return result.cpu().numpy()
        else:
            arr = array.astype(float)
            T, F = arr.shape
            out = np.full((T, F), np.nan, dtype=float)
            # 滑动窗口视图
            sw = np.lib.stride_tricks.sliding_window_view(arr, window, axis=0)
            # 计算每个窗口的 std
            stds = sw.std(axis=2, ddof=1)
            out[window-1:] = stds
            return out
    
    @fill_inf_with_max_min_ts_decorator()
    def rollsum(self, array: np.ndarray, window: int, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            x = torch.from_numpy(array).double().to(device)
            T, F = x.shape
            # 使用滑动窗口视图在 PyTorch 中无法实现，改用 conv1d 累积法
            weight = torch.ones(1, 1, window, device=device)
            out = torch.conv1d(x.T.unsqueeze(1), weight, stride=1, padding=0).squeeze(1)   
            # 填充前 window-1 行 NaN
            pad = window - 1
            nanpad = torch.full((pad, F), float('nan'), dtype=out.dtype, device=device)
            result = torch.cat([nanpad, out.T], dim=0)  
            return result.cpu().numpy()
        else:
            arr = array.astype(float)
            T, F = arr.shape
            out = np.full((T, F), np.nan, dtype=float)
            # 滑动窗口视图
            sw = np.lib.stride_tricks.sliding_window_view(arr, window, axis=0)
            # 计算每个窗口的 sum
            sums = sw.sum(axis=2)
            out[window-1:] = sums
            return out
        
    @fill_inf_with_max_min_ts_decorator()
    def rolling_momentum(self, array, window, moment: int, backend="numpy", device="cpu"):
        """
        rolling_skew (moment=3) 或 rolling_kurt (moment=4) 模糊实现。
        符合 pandas default(bias=False), 窗口末端对齐，前 window-1 填 NaN。
        """
        assert moment in (3, 4), "moment must be 3 (skew) or 4 (kurt)"
        ddof = 1
        if backend == "numpy":
            arr = array.astype(float)
            T, F = arr.shape
            out = np.full((T, F), np.nan, dtype=float)
            sw = np.lib.stride_tricks.sliding_window_view(arr, window, axis=0)  # shape [T-w+1, F, w]
            m = np.nanmean(sw, axis=2)
            dev = sw - m[..., None]
            m2 = np.nanmean(dev**2, axis=2)
            n = np.sum(~np.isnan(sw), axis=2)
            if moment == 3:
                m3 = np.nanmean(dev**3, axis=2)
                # unbiased skew: sqrt(n(n-1))/(n-2) * m3/m2^1.5
                coef = np.sqrt(n*(n-1)) / (n-2)
                out[window-1:] = coef * m3 / (m2 ** 1.5)
            else:
                m4 = np.nanmean(dev**4, axis=2)
                # unbiased kurtosis using scipy's k statistic approach
                # here use fisher (excess)
                coef2 = (n*(n+1)/((n-1)*(n-2)*(n-3)))
                coef3 = 3*(n-1)**2 / ((n-2)*(n-3))
                out[window-1:] = coef2 * m4 / (m2 ** 2) - coef3
            return out

        else:
            x = torch.from_numpy(array).double().to(device)  # shape [T, F]
            F = x.shape[1]
            pad = window - 1
            x_win = x.unfold(0, window, 1)  # shape [T-window+1, F, window]
            m = torch.nanmean(x_win, dim=2)
            dev = x_win - m[..., None]
            m2 = torch.nanmean(dev * dev, dim=2)
            n = torch.sum(~torch.isnan(x_win), dim=2).double()
            if moment == 3:
                m3 = torch.nanmean(dev**3, dim=2)
                g2 = m4 / (m2**2) - 3
                # adjusted Fisher–Pearson G2
                vals = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1)*g2 + 6)
            else:
                m4 = torch.nanmean(dev**4, dim=2)
                g2 = m4 / (m2**2) - 3
                vals = ((n - 1) / ((n - 2)*(n - 3))) * ((n + 1)*g2 + 6)

            # mask invalid (n <= moment)
            vals = torch.where(n <= moment, torch.full_like(vals, float('nan')), vals)
            pad = torch.full((window-1, F), float('nan'), dtype=vals.dtype, device=device)
            return torch.cat([pad, vals], dim=0).cpu().numpy()

    def rollskew(self, array, window, backend="numpy", device="cpu"):
        return self.rolling_momentum(array, window, moment=3, backend=backend, device=device)

    def rollkurt(self, array, window, backend="numpy", device="cpu"):
        return self.rolling_momentum(array, window, moment=4, backend=backend, device=device)
   
    @fill_inf_with_max_min_ts_decorator()
    def expandmean(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        nan_mask = np.isnan(array)
        array = np.where(nan_mask, 0, array)  # 将 NaN 替换为 0
        if backend == "torch":
            x = torch.from_numpy(array).double().to(device)
            result = torch.cumsum(x, dim=0) / torch.arange(1, x.shape[0] + 1, device=device).unsqueeze(1)
            nan_mask_tensor = torch.from_numpy(nan_mask).to(device)
            result[nan_mask_tensor] = float('nan')  # 恢复 NaN  # 恢复 NaN
            return result.cpu().numpy()
        else:
            res = np.cumsum(array, axis=0) / np.arange(1, array.shape[0] + 1)[:, None]
            res[nan_mask] = np.nan  # 恢复 NaN
            return res
        
    @fill_inf_with_max_min_ts_decorator()
    def expandstd(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        nan_mask = np.isnan(array)
        array = np.where(nan_mask, 0, array)  # 将 NaN 替换为 0
        if backend == "torch":
            x = torch.from_numpy(array).double().to(device)
            mean = torch.cumsum(x, dim=0) / torch.arange(1, x.shape[0] + 1, device=device).unsqueeze(1)
            cumsum_sq = torch.cumsum(x * x, dim=0)
            result = torch.sqrt((cumsum_sq - mean * mean * torch.arange(1, x.shape[0] + 1, device=device).unsqueeze(1)) / 
                                (torch.arange(1, x.shape[0] + 1, device=device) - 1).unsqueeze(1))
            result[nan_mask] = float('nan')
            return result.cpu().numpy()
        else:
            mean = np.cumsum(array, axis=0) / np.arange(1, array.shape[0] + 1)[:, None]
            cumsum_sq = np.cumsum(array * array, axis=0)
            res = np.sqrt((cumsum_sq - mean * mean * np.arange(1, array.shape[0] + 1)[:, None]) /
                          (np.arange(1, array.shape[0] + 1)[:, None] - 1))
            res[nan_mask] = np.nan
            return res
        
    @fill_inf_with_max_min_ts_decorator()
    def expandmax(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        nan_mask = np.isnan(array)
        if backend == "torch":
            x = torch.from_numpy(array).double().to(device)
            x = torch.nan_to_num(x, nan=float('-inf'))
            result = torch.cummax(x, axis=0)[0]
            result[nan_mask] = float('nan')
            return result.cpu().numpy()
        else:
            result = np.maximum.accumulate(array, axis=0)
            result[nan_mask] = np.nan  # 恢复 NaN
            return result
        
    @fill_inf_with_max_min_ts_decorator()
    def expandmin(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        nan_mask = np.isnan(array)
        if backend == "torch":
            x = torch.from_numpy(array).double().to(device)
            x = torch.nan_to_num(x, nan=float('inf'))
            result = torch.cummin(x, axis=0)[0]
            result[nan_mask] = float('nan')
            return result.cpu().numpy()
        else:
            result = np.minimum.accumulate(array, axis=0)
            result[nan_mask] = np.nan  # 恢复 NaN
            return result
    
    @fill_inf_with_max_min_ts_decorator()
    def expandsum(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        nan_mask = np.isnan(array)
        array = np.where(nan_mask, 0, array)  # 将 NaN 替换为 0
        if backend == "torch":
            x = torch.from_numpy(array).double().to(device)
            result = torch.cumsum(x, dim=0)
            return result.cpu().numpy()
        else:
            return np.cumsum(array, axis=0)
        
    @fill_inf_with_max_min_ts_decorator()
    def expandzscore(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":  
            mean = self.expanding_mean(array, backend=backend, device=device) 
            std =  self.expanding_std(array, backend=backend, device=device)
            return np.where(std != 0, (array - mean) / std, np.nan)
        else:
            mean = self.expanding_mean(array, backend=backend, device=device)
            std = self.expanding_std(array, backend=backend, device=device)
            return np.where(std != 0, (array - mean) / std, np.nan)
        
    @fill_inf_with_max_min_ts_decorator()
    def expandskew(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        return pd.DataFrame(array).expanding().skew().values
    

    @fill_inf_with_max_min_ts_decorator()
    def expandkurt(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
       return pd.DataFrame(array).expanding().kurt().values
    
    @fill_inf_with_max_min_ts_decorator()
    def apply_filter(self, array: np.ndarray, filter_name: str, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        if self._is_all_nan(array, axis=0).any():
            result = np.full_like(array, np.nan)
            col_mask = ~self._is_all_nan(array, axis=0)
            if not np.any(col_mask):
                return result
            valid_array = array[:, col_mask]
        else:
            valid_array = array

        filter_func = {
            "fft_A_interp": fft_amp_interp1d,
            "fft_angle_interp": fft_angle_interp1d,
            "fft_real_interp": fft_real_interp1d,
            "dwt_ca_fixed": dwt_ca_fixed,
            "dwt_da_fixed": dwt_da_fixed,
            "dwt_ca": dwt_ca,
            "dwt_da": dwt_da,
            "savgol_filter": savgol_filter,
            "median_filter": median_filter,
            "lowpass_filter": lowpass_filter,
            "highpass_filter": highpass_filter,
            "bandpass_filter": bandpass_filter,
            "wavelet_denoise": wavelet_denoise,
            "ewma_filter": ewma_filter,
            "kalman_filter": kalman_filter,
            "hp_filter": hp_filter,
            "robust_zscore_filter": robust_zscore_filter,
            "rolling_rank_filter": rolling_rank_filter,
            "kaufman_adaptive_moving_average": kaufman_adaptive_moving_average,
            "mesa_adaptive_moving_average": mesa_adaptive_moving_average,
            "hilbert_transform_instantaneous_phase": hilbert_transform_instantaneous_phase, 
        }.get(filter_name)
        if not callable(filter_func):
            raise ValueError(f"Filter {filter_name} not found")

        filter_kwargs = self.config.get("filter_params", {}).get("level1", {}).get("filters", {}).get(filter_name, {})
        filter_kwargs.update(kwargs)
        result_valid = filter_func(valid_array, backend=backend, device=device, **filter_kwargs)

        if filter_name in ["dwt_ca_fixed", "dwt_da_fixed", "dwt_ca", "dwt_da"]:
            result_valid = interpolator_arr(result_valid)

        if self._is_all_nan(array, axis=0).any():
            result[:, col_mask] = result_valid
        else:
            result = result_valid

        return self.specialized_output_shape(array, result)
    
    def fft_amp(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "fft_A_interp", backend=backend, device=device, **kwargs)
    
    def fft_angle(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "fft_angle_interp", backend=backend, device=device, **kwargs)
    
    def fft_r(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:      
        return self.apply_filter(array, "fft_real_interp", backend=backend, device=device, **kwargs)

    def dwt_ca(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "dwt_ca", backend=backend, device=device, **kwargs) 
    
    def dwt_da(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "dwt_da", backend=backend, device=device, **kwargs)

    def dwt_ca_fver(self, array: np.ndarray, backend="torch", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "dwt_ca_fixed", backend=backend, device=device, **kwargs)

    def dwt_da_fver(self, array: np.ndarray, backend="torch", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "dwt_da_fixed", backend=backend, device=device, **kwargs)

    def savgolfilt(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "savgol_filter", backend=backend, device=device, **kwargs)  
    
    def medfilt(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "median_filter", backend=backend, device=device, **kwargs)  
    
    def lowpfilt(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "lowpass_filter", backend=backend, device=device, **kwargs) 
    
    def highpfilt(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "highpass_filter", backend=backend, device=device, **kwargs)    
    
    def bandpfilt(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "bandpass_filter", backend=backend, device=device, **kwargs)
    
    def wdfilt(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "wavelet_denoise", backend=backend, device=device, **kwargs)
    
    def ewma(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "ewma_filter", backend=backend, device=device, **kwargs)    
    
    def kalfilt(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "kalman_filter", backend=backend, device=device, **kwargs)
    
    def hpfilt(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "hp_filter", backend=backend, device=device, **kwargs)  
    
    def rob_zsfilt(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "robust_zscore_filter", backend=backend, device=device, **kwargs)
    
    def rollrankfilt(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "rolling_rank_filter", backend=backend, device=device, **kwargs)

    def kauama(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "kaufman_adaptive_moving_average", backend=backend, device=device, **kwargs)
    
    def mesama(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "mesa_adaptive_moving_average", backend=backend, device=device, **kwargs)
    
    def hb_insphase(self, array: np.ndarray, backend="numpy", device="cpu", **kwargs) -> np.ndarray:
        return self.apply_filter(array, "hilbert_transform_instantaneous_phase", backend=backend, device=device, **kwargs)
    
    @fill_inf_with_max_min_ts_decorator()
    def trewgt(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        """
        Calculate weighted mean across time axis
        """
        if backend == "torch":
            array = torch.from_numpy(array).double().to(device)
            arr_diff = torch.diff(array, dim=0, prepend=torch.full((1, array.shape[1]), float('nan'), device=device))
            trend = torch.where(arr_diff > 0, 1, torch.where(arr_diff < 0, -1, 0))
            result = array * trend
            return result.cpu().numpy()

        else:
           arr_diff = np.diff(array, axis=0, prepend=np.full((1, array.shape[1]), np.nan))
           trend = np.where(arr_diff > 0, 1, np.where(arr_diff < 0, -1, 0))
           res = array * trend
           return res

    @fill_inf_with_max_min_ts_decorator()
    def revwtg(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        """
        Calculate reversal weighted mean across time axis
        """
        if backend == "torch":
            array = torch.from_numpy(array).double().to(device)
            arr_diff = torch.diff(array, dim=0, prepend=torch.full((1, array.shape[1]), float('nan'), device=device))
            mask = ~torch.isnan(array)
            count = mask.sum(dim=1, keepdim=True)
            diff = array - torch.nanmean(arr_diff, dim=0, keepdim=True)
            diff[~mask] = 0  # 忽略 NaN 的差值
            var = (diff ** 2).sum(dim=1, keepdim=True) / (count - 1)
            std = torch.sqrt(var)
            dev = abs(arr_diff - torch.nanmean(arr_diff, dim=0, keepdim=True)) / std
            trend = torch.where(dev > 1.5, -1, torch.where(dev < -1.5, 1, 0))
            result = array * trend
            return result.cpu().numpy()
           
        else:
            arr_diff = np.diff(array, axis=0, prepend=np.full((1, array.shape[1]), np.nan))
            dev = abs(arr_diff - np.nanmean(arr_diff, axis=0, keepdims=True)) / np.nanstd(arr_diff, axis=0, keepdims=True)
            trend = np.where(dev > 1.5, -1, np.where(dev < -1.5, 1, 0))
            res = array * trend
            return res
    
    @fill_inf_with_max_min_ts_decorator()
    def trerevwtg(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).double().to(device)
            arr_diff = torch.diff(array, dim=0, prepend=torch.full((1, array.shape[1]), float('nan'), device=device))
            mask = ~torch.isnan(array)
            count = mask.sum(dim=1, keepdim=True)
            diff = array - torch.nanmean(arr_diff, dim=0, keepdim=True)
            diff[~mask] = 0  # 忽略 NaN 的差值
            var = (diff ** 2).sum(dim=1, keepdim=True) / (count - 1)
            std = torch.sqrt(var)
            dev = abs(arr_diff - torch.nanmean(arr_diff, dim=0, keepdim=True)) / std
            reverse = torch.where(dev > 1.5, -1, torch.where(dev < -1.5, 1, 0))
            trend = torch.where(arr_diff > 0, 1, torch.where(arr_diff < 0, -1, 0))
            result = array* (reverse + trend)
            return result.cpu().numpy()
        else:
            arr_diff = np.diff(array, axis=0, prepend=np.full((1, array.shape[1]), np.nan))
            dev = abs(arr_diff - np.nanmean(arr_diff, axis=0, keepdims=True)) / np.nanstd(arr_diff, axis=0, keepdims=True)
            reverse = np.where(dev > 1.5, -1, np.where(dev < -1.5, 1, 0))
            trend = np.where(arr_diff > 0, 1, np.where(arr_diff < 0, -1, 0))
            res = array * (reverse + trend)
            return res
        
    
    def process_in_chunks(self, data: np.ndarray, chunk_size=1000) -> np.ndarray:
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("Input must be 2D array")
        n_cols = data.shape[1]
        result = []
        for i in range(0, n_cols, chunk_size):
            chunk = data[:, i:i+chunk_size]
            result.append(self.process(chunk))
        return np.hstack(result)
