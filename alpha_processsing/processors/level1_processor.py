import numpy as np
import torch

from functools import wraps
from joblib import Parallel, delayed
from .np_cal_func import fill_inf_with_max_min_ts_decorator
from .filter_func import (
    dwt_ca_fixed, dwt_da_fixed, dwt_ca, dwt_da, savgol_filter, median_filter,
    lowpass_filter, highpass_filter, bandpass_filter,wavelet_denoise, 
    ewma_filter, kalman_filter, hp_filter, robust_zscore_filter, interpolator_arr,
    rolling_rank_filter, kaufman_adaptive_moving_average, mesa_adaptive_moving_average,
    hilbert_transform_instantaneous_phase
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
            array = torch.from_numpy(array).float().to(device)
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
    def tan(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            result = torch.tan(array).cpu().numpy()
        else:
            result = np.tan(array)
        return result

    @fill_inf_with_max_min_ts_decorator()
    def sin(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            result = torch.sin(array).cpu().numpy()
        else:
            result = np.sin(array)
        return result

    @fill_inf_with_max_min_ts_decorator()
    def cos(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            result = torch.cos(array).cpu().numpy()
        else:
            result = np.cos(array)
        return result

    @fill_inf_with_max_min_ts_decorator()
    def sinh(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        array = self._safe_normalize(array, backend=backend, device=device)
        if backend == "torch":
            result = torch.sinh(array).cpu().numpy()
        else:
            result = np.sinh(array)
        return result

    @fill_inf_with_max_min_ts_decorator()
    def cosh(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        array = self._safe_normalize(array, backend=backend, device=device)
        if backend == "torch":
            result = torch.cosh(array).cpu().numpy()
        else:
            result = np.cosh(array)
        return result

    @fill_inf_with_max_min_ts_decorator()
    def tanh(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            result = torch.tanh(array).cpu().numpy()
        else:
            result = np.tanh(array)
        return result

    @fill_inf_with_max_min_ts_decorator()
    def diff(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            result = torch.diff(array, dim=0).cpu().numpy()
            result = np.vstack([np.full((1, array.shape[1]), np.nan), result])
        else:
            result = np.diff(array, axis=0)
            result = np.vstack([np.full((1, array.shape[1]), np.nan), result])
        return self.specialized_output_shape(array, result)

    @fill_inf_with_max_min_ts_decorator()
    def apo(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            result = torch.abs(array - torch.mean(array, dim=0, keepdim=True)).cpu().numpy()
        else:
            result = np.abs(array - np.nanmean(array, axis=0, keepdims=True))
        return result
    

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
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            high = torch.max(array[-period:], dim=0).values
            low = torch.min(array[-period:], dim=0).values
            result = ((high - low) / (high + low)) * 100
            return result.cpu().numpy()
        else:
            high = np.max(array[-period:], axis=0)
            low = np.min(array[-period:], axis=0)
            result = ((high - low) / (high + low)) * 100
        return result
    
    

    @fill_inf_with_max_min_ts_decorator()
    def rolling_mean(self, array: np.ndarray, window: int, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            kernel = torch.ones(1, 1, window, device=device) / window
            result = torch.conv1d(
                array.unsqueeze(0).unsqueeze(1),
                kernel,
                padding=window//2
            ).squeeze(0).squeeze(1).cpu().numpy()
        else:
            result = np.apply_along_axis(
                lambda x: np.convolve(x, np.ones(window)/window, mode='same'), 0, array
            )
        return result

    @fill_inf_with_max_min_ts_decorator()
    def rolling_max(self, array: np.ndarray, window: int, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            result = torch.nn.functional.max_pool1d(
                array.unsqueeze(1),
                kernel_size=window,
                stride=1,
                padding=window//2
            ).squeeze(1).cpu().numpy()
        else:
            result = np.apply_along_axis(
                lambda x: np.lib.stride_tricks.sliding_window_view(x, window).max(axis=1), 0, array
            )
            pad = (array.shape[0] - result.shape[0]) // 2
            result = np.pad(result, ((pad, array.shape[0] - result.shape[0] - pad), (0, 0)), mode='edge')
        return result
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
    def rolling_min(self, array: np.ndarray, window: int, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            result = torch.nn.functional.max_pool1d(
                (-array).unsqueeze(1),
                kernel_size=window,
                stride=1,
                padding=window//2
            ).squeeze(1).cpu().numpy() * -1
        else:
            result = np.apply_along_axis(
                lambda x: np.lib.stride_tricks.sliding_window_view(x, window).min(axis=1), 0, array
            )
            pad = (array.shape[0] - result.shape[0]) // 2
            result = np.pad(result, ((pad, array.shape[0] - result.shape[0] - pad), (0, 0)), mode='edge')
        return result

    @fill_inf_with_max_min_ts_decorator()
    def rolling_zscore(self, array: np.ndarray, window: int, backend="numpy", device="cpu") -> np.ndarray:
        mean = self.rolling_mean(array, window, backend=backend, device=device)
        std = self.rolling_std(array, window, backend=backend, device=device)
        return np.where(std != 0, (array - mean) / std, np.nan)

    @fill_inf_with_max_min_ts_decorator()
    def rolling_std(self, array: np.ndarray, window: int, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            mean = self.rolling_mean(array.cpu().numpy(), window, backend=backend, device=device)
            mean = torch.from_numpy(mean).float().to(device)
            squared_diff = (array - mean) ** 2
            kernel = torch.ones(1, 1, window, device=device) / window
            result = torch.sqrt(
                torch.conv1d(squared_diff.unsqueeze(0).unsqueeze(1), kernel, padding=window//2)
            ).squeeze(0).squeeze(1).cpu().numpy()
        else:
            result = np.apply_along_axis(
                lambda x: np.lib.stride_tricks.sliding_window_view(x, window).std(axis=1), 0, array
            )
            pad = (array.shape[0] - result.shape[0]) // 2
            result = np.pad(result, ((pad, array.shape[0] - result.shape[0] - pad), (0, 0)), mode='edge')
        return result

    @fill_inf_with_max_min_ts_decorator()
    def rolling_skew(self, array: np.ndarray, window: int, backend="numpy", device="cpu") -> np.ndarray:
        mean = self.rolling_mean(array, window, backend=backend, device=device)
        std = self.rolling_std(array, window, backend=backend, device=device)
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            mean = torch.from_numpy(mean).float().to(device)
            std = torch.from_numpy(std).float().to(device)
            centered = array - mean
            m3 = torch.conv1d(
                (centered ** 3).unsqueeze(0).unsqueeze(1),
                torch.ones(1, 1, window, device=device) / window,
                padding=window//2
            ).squeeze(0).squeeze(1)
            result = torch.where(std != 0, m3 / (std ** 3), torch.tensor(float('nan'), device=device)).cpu().numpy()
        else:
            centered = array - mean
            m3 = np.apply_along_axis(
                lambda x: np.lib.stride_tricks.sliding_window_view(x, window).mean(axis=1), 0, centered
            )
            result = np.where(std != 0, m3 / (std ** 3), np.nan)
            pad = (array.shape[0] - result.shape[0]) // 2
            result = np.pad(result, ((pad, array.shape[0] - result.shape[0] - pad), (0, 0)), mode='edge')
        return result

    @fill_inf_with_max_min_ts_decorator()
    def rolling_kurt(self, array: np.ndarray, window: int, backend="numpy", device="cpu") -> np.ndarray:
        mean = self.rolling_mean(array, window, backend=backend, device=device)
        std = self.rolling_std(array, window, backend=backend, device=device)
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            mean = torch.from_numpy(mean).float().to(device)
            std = torch.from_numpy(std).float().to(device)
            centered = array - mean
            m4 = torch.conv1d(
                (centered ** 4).unsqueeze(0).unsqueeze(1),
                torch.ones(1, 1, window, device=device) / window,
                padding=window//2
            ).squeeze(0).squeeze(1)
            result = torch.where(std != 0, m4 / (std ** 4) - 3, torch.tensor(float('nan'), device=device)).cpu().numpy()
        else:
            centered = array - mean
            m4 = np.apply_along_axis(
                lambda x: np.lib.stride_tricks.sliding_window_view(x ** 4, window).mean(axis=1), 0, centered
            )
            result = np.where(std != 0, m4 / (std ** 4) - 3, np.nan)
            pad = (array.shape[0] - result.shape[0]) // 2
            result = np.pad(result, ((pad, array.shape[0] - result.shape[0] - pad), (0, 0)), mode='edge')
        return result
   
    
    @fill_inf_with_max_min_ts_decorator()
    def pct_rank(self, array: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            rank = torch.argsort(torch.argsort(array, dim=0), dim=0).float()
            valid_count = torch.sum(~torch.isnan(array), dim=0, keepdim=True)
            result = torch.where(valid_count != 0, rank / valid_count, torch.tensor(float('nan'), device=device)).cpu().numpy()
        else:
            result = np.apply_along_axis(
                lambda x: np.argsort(np.argsort(x)) / np.sum(~np.isnan(x)), 0, array
            )
            result = np.where(np.isnan(array), np.nan, result)
        return result

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
    
    @fill_inf_with_max_min_ts_decorator(process_input=True, process_output=False)
    def meanwgt(self, array: np.ndarray, weights: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        """
        Calculate weighted mean across time axis
        """
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            weights = torch.from_numpy(weights).float().to(device)
            weights = weights / torch.nansum(weights, dim=0, keepdim=True)
            result = torch.nansum(array * weights, dim=0, keepdim=True).cpu().numpy()
        else:
            weights = weights / np.nansum(weights, axis=0, keepdims=True)
            result = np.nansum(array * weights, axis=0, keepdims=True)
        return result.T

    @fill_inf_with_max_min_ts_decorator(process_input=True, process_output=False)
    def revwtg(self, array: np.ndarray, weights: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        """
        Calculate reverse weighted mean across time axis
        """
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            weights = torch.from_numpy(weights).float().to(device)
            weights = weights / torch.nansum(weights, dim=0, keepdim=True)
            result = torch.nansum(array * (1 - weights), dim=0, keepdim=True).cpu().numpy()
        else:
            weights = weights / np.nansum(weights, axis=0, keepdims=True)
            result = np.nansum(array * (1 - weights), axis=0, keepdims=True)
        return result.T
    
    
    @fill_inf_with_max_min_ts_decorator(process_input=False, process_output=True)
    def expwgt(self, array: np.ndarray, span: int, backend="numpy", device="cpu") -> np.ndarray:
        n = array.shape[0]
        alpha = 2 / (span + 1)
        weights = np.exp(-alpha * np.arange(n)[::-1])
        weights = weights / np.sum(weights)
        weights = np.tile(weights, (array.shape[1], 1)).T
        if backend == "torch":
            array = torch.from_numpy(array).float().to(device)
            weights = torch.from_numpy(weights).float().to(device)
            result = torch.nansum(array * weights, dim=0, keepdim=True).cpu().numpy()
        else:
            result = np.nansum(array * weights, axis=0, keepdims=True)
        return result.T

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
