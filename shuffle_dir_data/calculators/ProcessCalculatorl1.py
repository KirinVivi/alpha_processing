import pandas as pd
import numpy as np
from functools import wraps
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from filter_func import *
from np_cal_func import *

def fill_inf_with_max_min_ts_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        arr = args[0]
        if isinstance(arr, np.ndarray):
            arr = fill_inf_with_max_min(arr, axis=0)
        result = func(arr, *args[1:], **kwargs)
        if isinstance(result, np.ndarray):
            result = fill_inf_with_max_min(result, axis=0)
        return result
    return wrapper

class Processcalculatorl1:
    # Cross-sectional processors
    @fill_inf_with_max_min_ts_decorator
    def abs_v(self, array: np.ndarray) -> np.ndarray:
        return np.abs(array)

    @fill_inf_with_max_min_ts_decorator
    def reciprocal(self, array: np.ndarray) -> np.ndarray:
        return np.divide(1, array, where=(array != 0))

    @fill_inf_with_max_min_ts_decorator
    def pct_rank(self, array: np.ndarray) -> np.ndarray:
        return pd.DataFrame(array).rank(pct=True, axis=1).values

    @fill_inf_with_max_min_ts_decorator
    def sigmoid(self, array: np.ndarray) -> np.ndarray:
        return np.where(~np.isnan(array), 1/(1 + np.exp(-array)), np.nan)

    @fill_inf_with_max_min_ts_decorator
    def exptran(self, array: np.ndarray) -> np.ndarray:
        return np.where(~np.isnan(array), np.exp(array), np.nan)

    @fill_inf_with_max_min_ts_decorator
    def logtran(self, array: np.ndarray) -> np.ndarray:
        return np.where(~np.isnan(array), np.log1p(array), np.nan)

    @fill_inf_with_max_min_ts_decorator
    def tanh(self, array: np.ndarray) -> np.ndarray:
        return np.where(~np.isnan(array), np.tanh(array), np.nan)

    @fill_inf_with_max_min_ts_decorator
    def sinh(self, array: np.ndarray) -> np.ndarray:
        valid = array[~np.isnan(array)]
        if len(valid) == 0:
            return array
        x_scale = np.percentile(np.abs(valid), 85)
        if x_scale > 10:
            return np.where(~np.isnan(array), np.sinh(array * 0.1) / 0.1, np.nan)
        elif x_scale < 0.1:
            return np.where(~np.isnan(array), np.sinh(array * 10) / 10, np.nan)
        else:
            return np.where(~np.isnan(array), np.sinh(array), np.nan)

    @fill_inf_with_max_min_ts_decorator
    def cosh(self, array: np.ndarray) -> np.ndarray:
        valid = array[~np.isnan(array)]
        if len(valid) == 0:
            return array
        x_scale = np.percentile(np.abs(valid), 85)
        if x_scale > 10:
            return np.where(~np.isnan(array), np.cosh(array * 0.1), np.nan)
        elif x_scale < 0.1:
            return np.where(~np.isnan(array), np.cosh(array * 10), np.nan)
        else:
            return np.where(~np.isnan(array), np.cosh(array), np.nan)

    @fill_inf_with_max_min_ts_decorator
    def tan(self, array: np.ndarray) -> np.ndarray:
        return np.where(~np.isnan(array), np.tan(array), np.nan)

    @fill_inf_with_max_min_ts_decorator
    def sin(self, array: np.ndarray) -> np.ndarray:
        return np.where(~np.isnan(array), np.sin(array), np.nan)

    @fill_inf_with_max_min_ts_decorator
    def cos(self, array: np.ndarray) -> np.ndarray:
        return np.where(~np.isnan(array), np.cos(array), np.nan)

    @fill_inf_with_max_min_ts_decorator
    def sectional_zscore(self, array: np.ndarray) -> np.ndarray:
        mean = np.nanmean(array, axis=1, keepdims=True)
        std = np.nanstd(array, axis=1, keepdims=True)
        return (array - mean) / std

    # Time-series filter processors
    @fill_inf_with_max_min_ts_decorator
    def diff(self, array: np.ndarray) -> np.ndarray:
        return np.diff(array, axis=0)

    @fill_inf_with_max_min_ts_decorator
    def ftt_A_interpld(self, array: np.ndarray) -> np.ndarray:
        if np.all(np.isnan(array)) or np.all(array == 0):
            return np.full_like(array, np.nan)
        data_filled = np.apply_along_axis(interpolator_arr, 0, array)
        return np.apply_along_axis(lambda x: ftt(x)[1], 0, data_filled)

    @fill_inf_with_max_min_ts_decorator
    def ftt_angle_interpid(self, array: np.ndarray) -> np.ndarray:
        data_filled = np.apply_along_axis(interpolator_arr, 0, array)
        return np.apply_along_axis(lambda x: ftt(x)[1], 0, data_filled)

    @fill_inf_with_max_min_ts_decorator
    def ftt_x_interp1d(self, array: np.ndarray) -> np.ndarray:
        data_filled = np.apply_along_axis(interpolator_arr, 0, array)
        return np.apply_along_axis(lambda x: ftt(x)[2], 0, data_filled)

    @fill_inf_with_max_min_ts_decorator
    def dwt_ca(self, array: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda x: dwt(x)[0], 0, array)

    @fill_inf_with_max_min_ts_decorator
    def dwt_da(self, array: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(lambda x: dwt(x)[1], 0, array)

    @fill_inf_with_max_min_ts_decorator
    def np_sgf_onstandstd(self, array: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(np_savitzky_golay, 0, array)

    @fill_inf_with_max_min_ts_decorator
    def ewma50(self, array: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(ewma_filter, 0, array)

    @fill_inf_with_max_min_ts_decorator
    def butter_hpf(self, array: np.ndarray) -> np.ndarray:
        n = int(len(array) // 2)
        if n == 0:
            return np.full_like(array, np.nan)
        return np.apply_along_axis(lambda x: butter_highpass_filter(x, 0.5, n, 2), 0, array)

    # Time-series rolling processors (统一 rolling 处理)
    def _rolling_func(self, array: np.ndarray, window_size: int, func: str) -> np.ndarray:
        df = pd.DataFrame(array)
        return getattr(df.rolling(window=window_size), func)().values

    @fill_inf_with_max_min_ts_decorator
    def rolling_zscore(self, array: np.ndarray, window_size: int) -> np.ndarray:
        df = pd.DataFrame(array)
        return ((df - df.rolling(window=window_size).mean()) / df.rolling(window=window_size).std()).values

    @fill_inf_with_max_min_ts_decorator
    def rolling_mean(self, array: np.ndarray, window_size: int) -> np.ndarray:
        return self._rolling_func(array, window_size, 'mean')

    @fill_inf_with_max_min_ts_decorator
    def rolling_std(self, array: np.ndarray, window_size: int) -> np.ndarray:
        return self._rolling_func(array, window_size, 'std')

    @fill_inf_with_max_min_ts_decorator
    def rolling_max(self, array: np.ndarray, window_size: int) -> np.ndarray:
        return self._rolling_func(array, window_size, 'max')

    @fill_inf_with_max_min_ts_decorator
    def rolling_min(self, array: np.ndarray, window_size: int) -> np.ndarray:
        return self._rolling_func(array, window_size, 'min')

    @fill_inf_with_max_min_ts_decorator
    def rolling_median(self, array: np.ndarray, window_size: int) -> np.ndarray:
        return self._rolling_func(array, window_size, 'median')

    @fill_inf_with_max_min_ts_decorator
    def rolling_skew(self, array: np.ndarray, window_size: int) -> np.ndarray:
        return self._rolling_func(array, window_size, 'skew')

    @fill_inf_with_max_min_ts_decorator
    def rolling_kurt(self, array: np.ndarray, window_size: int) -> np.ndarray:
        return self._rolling_func(array, window_size, 'kurt')

    @fill_inf_with_max_min_ts_decorator
    def trend_sign(self, array: np.ndarray) -> np.ndarray:
        arr_diff = np.diff(array, axis=0)
        trend = np.where(arr_diff > 0, 1, -1)
        return array[1:, :] * trend

    @fill_inf_with_max_min_ts_decorator
    def reversal_sign(self, array: np.ndarray) -> np.ndarray:
        arr_diff = np.diff(array, axis=0)
        stand_dev = np.abs((arr_diff - np.nanmean(arr_diff, axis=0)) / np.nanstd(arr_diff, axis=0))
        trend = np.where(stand_dev > 1.5, -1, 1)
        return array[1:, :] * trend

    @fill_inf_with_max_min_ts_decorator
    def trerev_sign(self, array: np.ndarray) -> np.ndarray:
        arr_diff = np.diff(array, axis=0)
        stand_dev = np.abs((arr_diff - np.nanmean(arr_diff, axis=0)) / np.nanstd(arr_diff, axis=0))
        trend = np.where(arr_diff > 0, 1, -1)
        reverse = np.where(stand_dev > 1.5, -1, 1)
        return array[1:, :] * (trend + reverse)
