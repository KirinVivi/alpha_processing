from math import nan
import ray
import numpy as np
import pandas as pd

from itertools import permutations
from utils.data_utils import extract_value_from_string


class BaseCalculator:
	def __init__(self, groupdata, cal_func_class):
		self.data_lis = groupdata
		self.cal_class = cal_func_class

	def cal_functions(self, result, fun_class, func_str):
		if not isinstance(func_str, tuple):
			para_check = extract_value_from_string(func_str)
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
				para_check = extract_value_from_string(func_name_para)
				if para_check:
					func_name, para = para_check
					func = getattr(fun_class, func_name)
					result = func(result, para)
				else:
					func = getattr(fun_class, func_name_para)
					result = func(result)
		return self.check_valid(result)

	def apply_function(self, func_tuple):
		"""
		Returns:
			daily_results:
			list of pd.DataFrames
		"""
		daily_data_lis = self.data_lis.copy()
		daily_results = []
		for group in daily_data_lis:
			if len(func_tuple) == 1:	
				daily_results.append(self.cal_functions(group, self.cal_class, func_tuple[0]))
			else:
				daily_results.append(self.cal_functions(group, self.cal_class, func_tuple))	
		return daily_results




	def check_valid(self, arr: pd.DataFrame) -> pd.DataFrame:
		"""Check if array is valid (not all zero)"""
		if np.nansum(abs(arr)) == 0 or np.isnan(arr.values).all():
			return pd.DataFrame(np.full_like(arr.values, np.nan), index=arr.index, columns=arr.columns)
