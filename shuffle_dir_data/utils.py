import os
import re
import ray
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def interpolator_arr(arr):
	x = np.arange(len(arr))
	if np.all(np.isnan(arr)):
		return x * np.nan
	valid = ~np.isnan(arr)
	if len(x[valid]) < 2:
		return x * np.nan
	interpolator = interp1d(x[valid], arr[valid], bounds_error=False, fill_value="extrapolate")
	data_filled = interpolator(x)
	return data_filled
def ffill(array):
	mask = np.isnan(array)
	idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
	np.maximum.accumulate(idx, axis=0, out=idx)
	return array[idx, np.arange(idx.shape[1])[None, :]]

def bfill(array):
	flipped_arr = np.flip(array, axis=0)
	mask = np.isnan(flipped_arr)
	idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
	np.maximum.accumulate(idx, axis=0, out=idx)
	out = flipped_arr[idx, np.arange(idx.shape[1])[None, :]]
	return np.flip(out, axis=0)

def fill_cross_sectional_mean(arr):
	row_means = np.nanmean(arr, axis=1, keepdims=True)
	col_nan_mask = np.isnan(arr).all(axis=0)
	arr_fill_nan = arr[:, ~col_nan_mask]
	nan_mask = np.isnan(arr_fill_nan)
	arr_fill_nan[nan_mask] = np.take(row_means, np.nonzero(nan_mask)[0])
	arr[:, ~col_nan_mask] = arr_fill_nan
	return arr

def fill_cross_sectional_median(arr):
	row_median = np.nanmedian(arr, axis=1, keepdims=True)
	col_nan_mask = np.isnan(arr).all(axis=0)
	arr_fill_nan = arr[:, ~col_nan_mask]
	nan_mask = np.isnan(arr_fill_nan)
	arr_fill_nan[nan_mask] = np.take(row_median, np.nonzero(nan_mask)[0])
	arr[:, ~col_nan_mask] = arr_fill_nan
	return arr

def get_data(data_name, data_class, freq, col_type):
	"""
	input:
		data_name, data_class: tick 5min 15min, moneyflow 5min 15min 30min 60min, DailyBase daily
		frequency:5min 15min 30min 60min
		column type: price,volume,amount,transaction,ratio
	return:
		sample data: 90 days 20160101-20160401
		frequency, column type @INNOAM
	"""
	all_data_path = '/dat5/all/AlphaDigData/EquityAhare_complete 20231229'
	data_path_dic = {}
	if freq == 'daily':
		data_path_dic = {
			"moneyflow": f'{all_data_path}/Moneyflow',
			"DailyBase": f'{all_data_path}/DailyBase'
		}
	elif freq in ['5', '15', '30', '60']:
		data_path_dic = {
			"tick": f'{all_data_path}/{freq}MinuteBase',
			"moneyflow": f'{all_data_path}/moneyflow_{freq}min'
		}
	else:
		raise KeyError
	data_path = data_path_dic[data_class]
	res_dic = {}
	dat = pd.read_hdf(f'{data_path}/{data_name}.hdf5')
	data = dat.loc['2018-01':'2018-04'].dropna(how='all', axis=1).copy()
	del dat
	data.columns = [int(i[:6]) for i in data.columns]
	res_dic['data'] = data
	res_dic['data_file'] = os.path.basename(data_path)
	return res_dic

def extract_value_from_string(input_string):
	# Define the pattern using regular expressions
	pattern1 = re.compile(r'(\w+)\((\d+),(\d+)\)')
	pattern2 = re.compile(r'(\w+)(\d+)')
	# Use the pattern to match the input_string
	match1 = pattern1.match(input_string)
	match2 = pattern2.match(input_string)
	# Check if there is a match
	if match1:
		# Extract the values from the matched groups
		x = match1.group(1)
		a = int(match1.group(2))
		b = int(match1.group(3))
		return x, a, b
	elif match2:
		x = match2.group(1)
		a = int(match2.group(2))
		return x, a
	else:
		return None  # if no match is found

def cal_ic(ret, dat):
	# cal ic
	data = dat.copy()
	# common cols and days
	data.columns = [int(i) for i in data.columns]
	cols = list(set(ret.columns) & set(data.columns))
	data.index = [int(i.strftime("%m%d")) for i in data.index]
	ind = list(set(ret.index) & set(data.index))
	cols.sort()
	ind.sort()
	ret_train = ret.loc[ind][cols]
	data_train = data.loc[ind][cols]
	cover = ((data_train * ret_train).count(axis=1) / ret_train.count(axis=1)).mean()
	del data
	if cover > 0.7:
		ic = ret_train.corrwith(data_train, axis=1, method='spearman').mean()
		del ret
		if abs(ic) > 0.015:
			print('ic:', ic)
			if ic < 0:
				return True, '-1*alpha'
			else:
				return True, 'alpha'
		else:
			return False, "low ic"
	else:
		return False, 'low cover'

def cal_dot_retfactot(ret, dat):
	# cal ic
	data = dat.copy()
	# common cols and days
	data.columns = [int(i) for i in data.columns]
	cols = list(set(ret.columns) & set(data.columns))
	data.index = [int(i.strftime("%m%d")) for i in data.index]
	ind = list(set(ret.index) & set(data.index))
	cols.sort()
	ind.sort()
	ret_train = ret.loc[ind][cols]
	data_train = data.loc[ind][cols]
	cover = ((data_train * ret_train).count(axis=1) / ret_train.count(axis=1)).mean()
	del data
	if cover > 0.7:
		ic = ret_train.corrwith(data_train, axis=1, method='spearman').mean()
		del ret
		if abs(ic) > 0.015:
			print('ic:', ic)
			if ic < 0:
				return True, "-1*alpha"
			else:
				return True, 'alpha'
		else:
			return False, 'low ic'
	else:
		return False, 'low cover'

@ray.remote
def compute_corr_batch(main_df, batch_data):
	"""
	并行计算 main df 和 batch data 的rank correlation.
	"""
	results = {}
	for fill_func, groups in batch_data.items():
		results[fill_func] = {}
		for cal_func, subgroups in groups.items():
			results[fill_func][cal_func] = {}
			for smooth_func, df in subgroups.items():
				rank_corr_ana = cal_ic(main_df, df)
				if rank_corr_ana[0]:
					results[fill_func][cal_func][smooth_func] = rank_corr_ana[1]
				else:
					results[fill_func][cal_func][smooth_func] = None
	return results

def clean_empty_nodes(d):
	"""
	递归清理嵌套字典中的空节点。如果一个字典的值是空字典，则将其删除
	"""
	if not isinstance(d, dict):
		return d #如果不是字典，直接返回
	# 遍历当前字典的键值对
	keys_to_delete = []
	for key, value in d.items():
		if isinstance(value, dict):
			# 递归清理子字典
			cleaned = clean_empty_nodes(value)
			if not cleaned:
				#如果子字典为空，标记为待删除
				keys_to_delete.append(key)
			else:
				d[key] = cleaned
		elif value is None:
			#如果值为 None 或空字符串，标记为待删除!
			keys_to_delete.append(key)
		elif isinstance(value, str) and (value == ""):
			#如果值为 None 或空字符串，标记为待删除:
			keys_to_delete.append(key)
	for key in keys_to_delete:
		del d[key]
	return d

def save_nested_dict(data, save_dir):
	os.makedirs(save_dir, exist_ok=True)
	for fill_func, subgroups in data.items():
		n = 0
		for cal_func, subsubgroups in subgroups.items():
			for smooth_func, df in subsubgroups.items():
				# 保存 DataFrame 文件，使用 group/subgroup 作为路径
				n += 1
				file_path = os.path.join(save_dir, f"{fill_func}_{cal_func}_{smooth_func}.csv")
				df.to_csv(file_path, index=True)
		print(f'saved {fill_func} {n} files')