import ray
# Initialize Ray
import utils
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import permutations

@ray.remote(max_concurrency=20)
class Datashufflecalculatorlevel2:
	def __init__(self, period_data, cal_func_class, window=None):
		"""
		Initialize the DatashuffleCalculator class.
		Parameters:
		data(pd.DataFrame): The input data.
		window(int):The rolling window size(in days) for calculations.
		"""
		self.data = period_data
		self.grouped_data = None
		self.cal_class = cal_func_class
		self.window = window

	def shuffle_data(self):
		"""
		Shuffle the data within each day.
		"""
		if isinstance(self.data, pd.DataFrame):
			self.data.index = pd.to_datetime(self.data.index)
			self.data = self.data.sort_index()
			# self.data = self.data.groupby(self.data.index.date).apply(lambda x: x.sample(frac=1)).reset_index(level=0, drop=True)
			self.grouped_data = [group.sort_index().values for _, group in self.data.groupby(self.data.index.date)]
		elif isinstance(self.data, dict):
			self.grouped_data = [group for _, group in self.data.items()]
		else:
			raise TypeError

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

	def check_nan_rows(self, df):
		# 计算每一行是否全是 NaN
		nan_rows = df.isna().all(axis=1)
		# 统计全是 NaN 的行数
		nan_row_count = nan_rows.sum()
		zero_row_count = (df == 0).all(axis=1).sum()
		return (nan_row_count < (len(df) / 3)) and (zero_row_count < (len(df) / 3))

	def calculate_daily(self, func_tuple):
		"""
		Calculate the result using the provided function on daily data.
		Parameters:
		func(callable): The function to apply to the data.
		Returns:
		np.ndarray: The result of the calculation.
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

		return np.vstack(daily_results)

	def calculate_rolling(self, func_tuple):
		"""
		Calculate the result using the provided function on rolling window data.
		Parameters:
		func(callable):The function to apply to the data.
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
		return rolling_results

	def apply_function(self, func_tuple, method='daily'):
		"""
		Apply the function using the specified method.
		Parameters:
		func(callable):The function to apply to the data.
		method (str):The method to use('daily' or 'rolling ).
		Returns:
		np.ndarray: The result of the calculation.
		"""
		self.shuffle_data()
		if method == 'daily':
			result = self.calculate_daily(func_tuple)
		elif method == 'rolling':
			result = self.calculate_rolling(func_tuple)
		else:
			raise ValueError("Method should be either 'daily' or 'rolling'")
		if not isinstance(self.data, dict):
			date_index = list(set(self.data.index.date))
			date_index.sort()
			res = pd.DataFrame(result, index=pd.to_datetime(date_index))
		else:
			print(np.vstack(result).shape)
			res = pd.DataFrame(np.vstack(result), index=list(self.data.keys())[self.window - 1:])
		return res

	def run_main(self, smooth_methods, columns=None, method='daily'):
		"""
		input 的data是dict的时候,columns 是需要的
		"""
		if isinstance(self.data, pd.DataFrame):
			cols = self.data.columns
		# elif isinstance(self.data,dict):
		# if columns is None:
		#     raise ValueError('Need columns of data')
		# else:
		#     cols = columns[self.window-1:1
		smooth_tuple_list = [(f'{i}',) for i in smooth_methods]
		dic = {}
		for smooth_tuple in tqdm(smooth_tuple_list):
			result = self.apply_function(smooth_tuple, method)
			if not isinstance(self.data, dict):
				result.columns = cols
			if self.check_nan_rows(result):
				# ic_check = utils.cal_ic(result)
				# if ic_check[e]:
				dic[smooth_tuple] = result
			# else:
			#     dic[smooth_tuple] = None
		return dic