import numpy as np
import pandas as pd
from functools import wraps
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from np_cal_func import fill_inf_with_max_min
# wrapper
def fill_inf_with_max_min_section_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        arr = args[0]
        if isinstance(arr, np.ndarray):
            arr = fill_inf_with_max_min(arr, axis=0)
        result = func(arr, *args[1:], **kwargs)
        if isinstance(result, np.ndarray):
            result = fill_inf_with_max_min(result, axis=1)
        return result
    return wrapper

class ProcessCalculator12:
    @fill_inf_with_max_min_section_decorator
    def sum(self, array):
        return np.nansum(array, axis=0, keepdims=True)
    @fill_inf_with_max_min_section_decorator
    def mean(self, array):
        return np.nanmean(array, axis=0, keepdims=True)
    @fill_inf_with_max_min_section_decorator
    def stddev(self, array):
        return np.nanstd(array, axis=0, keepdims=True)
    @fill_inf_with_max_min_section_decorator
    def mean_median_diff(self, array):
        return (np.nanmean(array, axis=0) - np.nanmedian(array, axis=0)).reshape(1, -1)
    @fill_inf_with_max_min_section_decorator
    def median(self, array):
        return np.nanmedian(array, axis=0).reshape(1, -1)
    @fill_inf_with_max_min_section_decorator
    def skew(self, array):
        arr_df = pd.DataFrame(array)
        return arr_df.skew().values.reshape(1, -1)

    @fill_inf_with_max_min_section_decorator
    def kurt(self, array):
        arr_df = pd.DataFrame(array)
        return arr_df.kurt().values.reshape(1, -1)
    @fill_inf_with_max_min_section_decorator
    def cv(self, array):
        means = np.nanmean(array, axis=0)
        stds = np.nanstd(array, axis=0)
        near_zero_mask = np.abs(means) < 1e-4
        # Use a small epsilon to avoid division by zero
        epsilon = np.percentile(np.abs(means[~near_zero_mask]), 10) if any(~near_zero_mask) else 1e-4
        adjusted_means = np.where(near_zero_mask, np.sign(means) * epsilon + epsilon, means)
        cv_results = stds / adjusted_means
        return cv_results.reshape(1, -1)

    @fill_inf_with_max_min_section_decorator
    def ptp(self, array):
        return (np.nanmax(array, axis=0) - np.nanmin(array, axis=0)).reshape(1, -1)

    @fill_inf_with_max_min_section_decorator
    def quantile(self, array, para):
        return np.nanquantile(array, para / 100, 0).reshape(1, -1)

    @fill_inf_with_max_min_section_decorator
    def acf_v(self, array, nlags):
        nlags = int(nlags)

        def interp_inside_missing(x, inplace: bool = False):
            """
            Parameters:
            x: array_like; inplace : bool, default = False, whether to modify in place
            Returns:
            y: ndarray, linear interpolation
            """
            if not inplace:
                x = x.copy()
            inner_nan = np.isnan(x)
            inner_nan[0] = False
            idx_nan = np.where(inner_nan)[0]
            idx_valid = np.where(~inner_nan)[0]
            if len(idx_nan) > 0 and len(idx_valid) > 0:
                x[idx_nan] = np.interp(idx_nan, idx_valid, x[idx_valid])
            return x

        def cal_acf(x, nlags):
            x = np.array(x)
            n = x.shape[0]
            x -= np.nanmean(x)
            if c_0 == 0:  # Handle zero variance case
                return 0
                
            # Use vectorized operations for better performance
            x_shifted = x[nlags:]
            x_base = x[:-nlags]
            
            # Calculate autocorrelation directly
            c_0 = np.nanvar(x)
            c_k = np.nanmean(x_shifted * x_base)
            r_k = c_k / c_0
            
            return r_k

        def run_acf(x):
            if np.all(np.isnan(x)):  # Handle all-NaN case
                return np.nan
            try:
                x_clean = interp_inside_missing(x)
                return cal_acf(x_clean, nlags)
            except:
                return np.nan

        # Vectorized calculation across columns
        return np.apply_along_axis(run_acf, 0, array).reshape(1, -1)