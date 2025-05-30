import numpy as np
import torch
from functools import wraps
from joblib import Parallel, delayed
from .np_cal_func import fill_inf_with_max_min

class ProcessCalculator:
    def cal_functions(self, array, func_str: tuple, backend="torch", device="cpu") -> np.ndarray:
        """Apply specified functions to the input array"""
        from ..utils.data_utils import extract_values_from_string
        if not isinstance(func_str, tuple):
            para_check = extract_values_from_string(func_str)
            if para_check:
                func_name, *para = para_check
                func = getattr(self, func_name)
                array = func(array, *para, backend=backend, device=device)
            else:
                func = getattr(self, func_str)
                array = func(array, backend=backend, device=device)
        else:
            for func_name_para in func_str:
                para_check = extract_values_from_string(func_name_para)
                if para_check:
                    func_name, *para = para_check
                    func = getattr(self, func_name)
                    array = func(array, *para, backend=backend, device=device)
                else:
                    func = getattr(self, func_name_para)
                    array = func(array, backend=backend, device=device)
        array[np.isinf(array)] = np.nan
        return array

    def specialized_output_shape(self, input_arr: np.ndarray, output_arr: np.ndarray) -> np.ndarray:
        """Adjust output shape"""
        padding = input_arr.shape[0] - output_arr.shape[0]
        if padding > 0:
            output_arr = np.vstack([output_arr, np.full((padding, input_arr.shape[1]), np.nan)])
        return output_arr

    def check_valid(self, arr: np.ndarray) -> np.ndarray:
        """Check if array is valid"""
        if np.all(np.isnan(arr)):
            return None
        non_nan_values = arr[~np.isnan(arr)]
        return arr if not np.all(non_nan_values == 0) else None

class ProcessCalculatorL2(ProcessCalculator):
    def __init__(self, config):
        """Initialize processor"""
        self.config = config
        self.backend = config.get("backend", "torch")
        self.device = config.get("device", "cpu")
        self.n_jobs = config.get("n_jobs", -1)
        torch.set_num_threads(8)

    def fill_inf_with_max_min_section_decorator(self, process_input=True, process_output=True):
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

    @fill_inf_with_max_min_section_decorator()
    def sum(self, x: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        """Calculate sum across time axis"""
        if backend == "torch":
            x = torch.from_numpy(x).float().to(device)
            result = torch.nansum(x, dim=0, keepdim=True).cpu().numpy()
            return result.transpose()
        else:
            result = np.nansum(x, axis=0, keepdims=True)
        return result.transpose()

    @fill_inf_with_max_min_section_decorator()
    def mean(self, x: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        """Calculate mean across time axis"""
        if backend == "torch":
            x = torch.from_numpy(x).float().to(device)
            result = torch.nanmean(x, dim=0, keepdim=True).cpu().numpy()
        else:
            result = np.nanmean(x, axis=0, keepdims=True)
        return result.transpose()

    @fill_inf_with_max_min_section_decorator()
    def stddev(self, x: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        """Calculate standard deviation across time axis"""
        if backend == "torch":
            x = torch.from_numpy(x).float().to(device)
            mean = torch.nanmean(x, dim=0, keepdim=True)
            result = torch.sqrt(
                torch.nanmean((x - mean) ** 2), dim=0, keepdim=True).cpu().numpy()
        else:
            result = np.nanstd(x, axis=0, keepdims=True)
        return result.transpose()

    @fill_inf_with_max_min_section_decorator()
    def quantile_25(self, x: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        """Calculate 25th percentile across time axis"""
        if backend == "torch":
            x = torch.from_numpy(x).float().to(device)
            result = torch.nanquantile(x, 0.25, dim=0, keepdim=True).cpu().numpy()
        else:
            result = np.nanpercentile(x, 25, axis=0, keepdims=True)
        return result.transpose()

    @fill_inf_with_max_min_section_decorator()
    def quantile_75(self, x: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        """Calculate 75th percentile across time axis"""
        if backend == "torch":
            x = torch.from_numpy(x).float().to(device)
            result = torch.nanquantile(x, 0.75, dim=0, keepdim=True).cpu().numpy()
        else:
            result = np.nanpercentile(x, 75, axis=0, keepdims=True)
        return result.transpose()

    @fill_inf_with_max_min_section_decorator()
    def skew(self, x: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        """Calculate skewness across time axis"""
        if backend == "torch":
            x = torch.from_numpy(x).float().to(device)
            mean = torch.nanmean(x, dim=0, keepdim=True)
            std = torch.sqrt(
                torch.nanmean((x - mean) ** 2, dim=0, keepdim=True)
            )
            centered = x - mean
            m3 = torch.nanmean(centered ** 3, dim=0, keepdim=True)
            result = torch.where(
                std != 0, 
                m3 / (std ** 3),
                torch.tensor(0.0, device=device)
            ).cpu().numpy()
        else:
            mean = np.nanmean(x, axis=0, keepdims=True)
            std = np.nanstd(x, axis=0, keepdims=True)
            centered = x - mean
            m3 = np.nanmean(centered ** 3, axis=0, keepdims=True)
            result = np.where(std != 0, m3 / (std ** 3), np.nan)
        return result.transpose()

    @fill_inf_with_max_min_section_decorator()
    def kurt(self, x: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        """Calculate kurtosis across time axis"""
        if backend == "torch":
            x = torch.from_numpy(x).float().to(device)
            mean = torch.nanmean(x, dim=0, keepdim=True)
            std = torch.sqrt(
                torch.nanmean((x - mean) ** 2, dim=0, keepdim=True)
            )
            centered = x - mean
            m4 = torch.nanmean(centered ** 4, dim=0, keepdim=True)
            result = torch.where(
                std != 0, 
                m4 / (std ** 4) - 3,
                torch.tensor(0.0, device=device)
            ).cpu().numpy()
        else:
            mean = np.nanmean(x, axis=0, keepdims=True)
            std = np.nanstd(x, axis=0, keepdims=True)
            centered = x - mean
            m4 = np.nanmean(centered ** 4, axis=0, keepdims=True)
            result = np.where(std != 0, m4 / (std ** 4) - 3, np.nan)
        return result.transpose()

    @fill_inf_with_max_min_section_decorator()
    def cv(self, x: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        """Calculate coefficient of variation across time axis"""
        if backend == "torch":
            x = torch.from_numpy(x).float().to(device)
            mean = torch.nanmean(x, dim=0, keepdim=True)
            std = torch.sqrt(
                torch.nanmean((x - mean) ** 2, dim=0, keepdim=True))
            result = torch.where(
                mean != 0,
                std / mean,
                torch.tensor(float('nan'), device=device)
            ).cpu().numpy()
        else:
            mean = np.nanmean(x, axis=0, keepdims=True)
            std = np.nanstd(x, axis=0, keepdims=True)
            result = np.where(mean != 0, std / mean, np.nan)
        return result.transpose()

    @fill_inf_with_max_min_section_decorator()
    def ptp(self, x: np.ndarray, backend="numpy", device="cpu") -> np.ndarray:
        """Calculate peak-to-peak across time axis"""
        if backend == "torch":
            x = torch.from_numpy(x).float().to(device)
            max_val = torch.nanmax(x, dim=0, keepdim=True).values
            min_val = torch.nanmin(x, dim=0, keepdim=True).values
            result = (max_val - min_val).cpu().numpy()
        else:
            result = np.nanmax(x, axis=0, keepdims=True) - np.nanmin(x, axis=0, keepdims=True)
        return result.transpose()

    def process_parallel(self, x: np.ndarray) -> np.ndarray:
        """Parallel processing for large datasets"""
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError("Input must be 2D array")
        
        def process_column(col, func_name):
            func = getattr(self, func_name)
            return func(col, backend=self.backend, device=self.device)
        
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(process_column)(x[:, i], func_name)
            for i in range(x.shape[1])
            for func_name in ["sum", "mean", "stddev", "quantile_25", "quantile_75", "skew", "kurt", "cv", "ptp"]
        )
        return np.column_stack(result).transpose()

    def process_in_chunks(self, x: np.ndarray, chunk_size: int=1000) -> np.ndarray:
        """Process data in chunks for memory efficiency"""
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError("Input must be 2D array")
        n_cols = x.shape[1]
        result = []
        for i in range(0, n_cols, chunk_size):
            chunk = x[:, i:i+chunk_size]
            result.append(self.process(chunk))
        return np.hstack(result).transpose()