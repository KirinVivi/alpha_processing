import ray
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import permutations
from ..utils.data_utils import fill_data

@ray.remote
class BaseCalculator:
    def __init__(self, data, cal_func_class, fill_method=None, window=None):
        self.data = data
        self.cal_class = cal_func_class
        self.fill_method = fill_method
        self.window = window
        self.grouped_data = None

    def shuffle_data(self):
        if isinstance(self.data, pd.DataFrame):
            self.data.index = pd.to_datetime(self.data.index)
            self.data = self.data.sort_index()
            self.grouped_data = [
                fill_data(group.sort_index().values, self.fill_method)
                for _, group in self.data.groupby(self.data.index.date)
            ]
        elif isinstance(self.data, dict):
            self.grouped_data = list(self.data.values())
        else:
            raise TypeError("data has to be DataFrame or dict")

    def apply_function(self, func_tuple):
        self.shuffle_data()
        
        result = self.calculate_daily(func_tuple)
        if isinstance(self.data, pd.DataFrame):
            ind = sorted(self.data.index)
            return pd.DataFrame(result, index=ind, columns=self.data.columns)
        return pd.DataFrame(result, index=list(self.data.keys())[self.window - 1:])
       

    def calculate_daily(self, func_tuple):
        daily_results = [
            self.cal_class.cal_functions(group[:(group.shape[0] // 2), :].copy(), func_tuple)
            for group in self.grouped_data
        ]
        return np.vstack([
            self.cal_class.specialized_output_shape(group, res)
            for group, res in zip(self.grouped_data, daily_results)
        ])

    # def calculate_rolling(self, func_tuple):
    #     grouped_values = self.grouped_data.copy()
    #     rolling_results = []
    #     # for i in range(len(grouped_values) - self.window + 1):
    #     #     window_data = np.vstack(grouped_values[i:i + self.window])
    #     #     rolling_result = window_data[:(-1 * grouped_values[0].shape[0] // 2), :].copy()
    #     # if self.window is None:
    #     #         raise ValueError("window must be specified for method='window'")
    #     grouped_data = group_data_by_window(self.data, self.window)
    #     rolling_results.append(self.cal_class.cal_functions(grouped_data, func_tuple))
    #     return [
    #         self.cal_class.specialized_output_shape(np.vstack(grouped_values[i:i + self.window]), res)
    #         for i, res in enumerate(rolling_results)
    #     ]

    def run_main(self, func_list):
        res_d = {}
        func_tuple_list = [
            perm for i in range(1, min(3, len(func_list) + 1))
            for perm in permutations(func_list, i)
        ]
        for func_tuple in tqdm(func_tuple_list, desc=f"processing {self.fill_method}"):
            result = self.apply_function(func_tuple)
            if not isinstance(result, dict) and self.cal_class.check_valid(result.values):
                res_d[f"{func_tuple}"] = result
        return res_d

    def check_valid(self, arr: np.ndarray) -> np.ndarray:
        """Check if array is valid (not all NaN or all zero)"""
        if np.all(np.isnan(arr)):
            return None
        non_nan_values = arr[~np.isnan(arr)]
        return arr if not np.all(non_nan_values == 0) else None