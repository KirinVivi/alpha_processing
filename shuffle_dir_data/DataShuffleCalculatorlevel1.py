import ray
import utils
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import permutations

import os

@ray.remote(max_concurrency=int(os.getenv("MAX_CONCURRENCY", 30)))
class DatashuffleCalculatorlevel1:
	def __init__(self, data, cal_func_class, window=None, fill_method=None):
		"""
		Initialize the Datashufflecalculator class.
		Parameters:
		data(pd.DataFrame): The input data.
		window(int):The rolling window size(in days)for calculations.    
		"""

		self.data = data
		self.grouped_data = None
		self.cal_class = cal_func_class
		self.window = window
		self.fill_method = fill_method

	def shuffle_data(self):
		"""
		Shuffle the data within each day.
		"""
		if not isinstance(self.data.index, pd.DatetimeIndex):
			self.data.index = pd.to_datetime(self.data.index)
		self.data = self.data.sort_index()
		# self.data = self.data.groupby(self.data.index.date).apply(lambda x: x.sample(frac=1)).reset_index(level=0, drop=True)  # initial grouped data
		if not self.fill_method:
			self.grouped_data = [group.sort_index().values for _, group in self.data.groupby(self.data.index.date)]
		else:
			fill_func = getattr(utils, self.fill_method)
			self.grouped_data = [fill_func(group.sort_index().values) for _, group in self.data.groupby(self.data.index.date)]

	def specialized_output_shape(self, input_arr, output_arr):
		padding = input_arr.shape[0] - output_arr.shape[0]
		if padding > 0:
			output_arr = np.vstack([output_arr, np.full((padding, input_arr.shape[1]), np.nan)])
		return output_arr

	def cal_functions(self, result, fun_class, func_str):
		# Judge the input data if the n dims of it is 2
		# Apply functions in the given permutation order
		# the processing order makes sense
		# Apply factors over 2, it has to meet the requirements that each permutation has the element of processor multi
		if not isinstance(func_str, tuple):
			para_check = utils.extract_values_from_string(func_str)
			if para_check:
				func_name, para = para_check
				func = getattr(fun_class, func_name)
				result = func(result, para)
			else:
				func_name = func_str
				func = getattr(fun_class, func_name)
				result = func(result)
		else:
			for func_name_para in func_str:
				para_check = utils.extract_values_from_string(func_name_para)
				if para_check:
					func_name, para = para_check
					func = getattr(fun_class, func_name)
					result = func(result, para)
				else:
					func = getattr(fun_class, func_name_para)
					result = func(result)
		result[np.isinf(result)] = np.nan
		return result

	def permute_functions(self, shuffle):
		# Generate all permutations of the shuffle
		res = []
		for i in shuffle:
			result = utils.extract_values_from_string(i)
			if result:
				if len(result) == 3:
					x, a, b = result
					num_divisions = 2  # You can make this configurable
					random_para = [random.uniform(a + j * (b - a) / num_divisions, a + (j + 1) * (b - a) / num_divisions) for j in range(num_divisions)]
					shuffle_paras = [f'{x}_{int(para) if para > 1 else int(para * 100)}' for para in random_para]
				else:
					x, a = result
					shuffle_paras = [f'{x}_{a}']
			else:
				shuffle_paras = [i]
			res.extend(shuffle_paras)
		perm = []
		for i in range(1, min(3, len(res) + 1)):
			perm.extend(list(permutations(res, i)))
		# TODO:
		res_lis = list(set(perm))
		# new_funcs = ['exptran', 'logtran', 'sinh', 'cosh', "tanh", "sectional zscore", 'np_sgf_onstandstd', 'ewma50', 'butter hpf', 'trend sign', "reversal sign", "trerev sign"]
		# new_res = []  # for i in res_lis: if len(set(i) & set(new_funcs)) != 0: new_res.append(i) # return new_res
		return res_lis

	def check_valid(self, arr):
		non_nan_values = arr[~np.isnan(arr)]
		# 选择所有非 NaN 的值 #如果非 NaN 的值全是零
		if np.all(non_nan_values == 0):
			return None
		return arr

	def calculate_daily(self, func_tuple):
		"""
		Calculate the result using the provided function on daily data.
		Parameters: func (callable): The function to apply to the data.        
		Returns:
			np.ndarray: The result of the calculation
		"""
		daily_data_lis = self.grouped_data.copy()
		daily_results = []
		for group in daily_data_lis:
			daily_result = group[:(group.shape[0] // 2), :].copy()
			if len(func_tuple) == 1:
				permutation = func_tuple[0]
				daily_results.append(self.cal_functions(daily_result, self.cal_class, permutation))
			else:
				daily_results.append(self.cal_functions(daily_result, self.cal_class, func_tuple))
		daily_results = [self.specialized_output_shape(group, i) for group, i in zip(daily_data_lis, daily_results)]
		return np.vstack(daily_results)

	def calculate_rolling(self, func_tuple):
		"""    
		Calculate the result using the provided function on daily data.
		Parameters :
		func(callable):The function to apply to the data,ZWW@INNOAN
		Returns:
		np.ndarray: The result of the calculation.
		"""
		grouped_values = self.grouped_data.copy()
		rolling_results = []
		for i in range(len(grouped_values) - self.window + 1):
			window_data = np.vstack(grouped_values[i:i + self.window])
			rolling_result = window_data[:(-1 * grouped_values[0].shape[0] // 2), :].copy()
			if len(func_tuple) == 1:
				permutation = func_tuple[0]
				rolling_results.append(self.cal_functions(rolling_result, self.cal_class, permutation))
			else:
				rolling_results.append(self.cal_functions(rolling_result, self.cal_class, func_tuple))
			rolling_results = [self.specialized_output_shape(window_data, i) for i in rolling_results]
		return rolling_results

	def apply_function(self, func_tuple, method='daily'):
		"""
		Apply the function using the specified method.
		Parameters:
		func(callable):The function to apply to the data.
		method(str):The method to use('daily' or 'rolling').
		Returns:
		np.ndarray: The result of the calculation.
		"""
		self.shuffle_data()
		if method == 'daily':
			result = self.calculate_daily(func_tuple)
			ind = list(self.data.index)
			ind.sort()
			# res = result
			res = pd.DataFrame(result, index=ind, columns=self.data.columns)
		elif method == 'rolling':
			result = self.calculate_rolling(func_tuple)
			date_index = list(set(self.data.index.date))
			date_index.sort()
			res = dict(zip(pd.to_datetime(date_index[self.window - 1:]), result))
		else:
			raise ValueError("Method should be either 'daily' or 'rolling'")
		return res

	def run_main(self, func_list, method='daily'):
		res_d = {}
		func_tuple_list = self.permute_functions(func_list)
		for func_tuple in tqdm(func_tuple_list, desc=f'{self.fill_method}'):
			res = self.apply_function(func_tuple, method)
			if not isinstance(res, dict):
				result = self.check_valid(res)
			else:
				result = res.copy()
			res_d[f"daily_{func_tuple}"] = result
		return res_d