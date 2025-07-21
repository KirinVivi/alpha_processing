from math import nan
from typing import Dict, Any, List
import ray
import numpy as np
import pandas as pd

from itertools import permutations
from utils.data_utils import extract_value_from_string
from processors.level1_processor import ProcessCalculatorL1
from processors.level2_processor import ProcessCalculatorL2
import logging
logger = logging.getLogger(__name__)


class AlphaCalculatorProcessor:
    def __init__(self, initial_data_list: List[pd.DataFrame], config: Dict[str, Any]):
        """
        在每個 Ray Worker 上只會被初始化一次。
        接收不會改變的初始數據和配置。
        """
        self.initial_data = initial_data_list
        self.config = config
        self.cal_func_class_l1 = ProcessCalculatorL1(config['filter_params'])
        self.cal_func_class_l2 = ProcessCalculatorL2({})
        logger.info(f"Processor initialized on worker {ray.get_runtime_context().get_node_id()}")

    def __call__(self, task_batch: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        這個方法是核心，它會並行地處理一批任務。
        `task_batch` 是一個字典，包含了多行任務，例如 {'l1_tuple': [...], 'l2_func': [...] }
        """
        output_records = []
        
        # 將字典批次轉換為記錄列表以便遍歷
        tasks = pd.DataFrame(task_batch).to_dict('records')

        for task in tasks:
            l1_func_tuple = task['l1_tuple']
            l2_func_list = task['l2_func']
            
            # --- 執行 Level 1 計算 ---
            # _apply_function_mock 是您原來的邏輯
            l1_result = _apply_function_mock(l1_func_tuple, self.initial_data, self.cal_func_class_l1)
            
            # --- L1 結果過濾 ---
            none_count = sum(1 for x in l1_result if np.all(np.isnan(x)))
            if len(l1_result) == 0 or (none_count / len(l1_result) > 0.3):
                continue
			for i in range(len(l1_result)):
				if np.all(np.isnan(l1_result[i])) and i > 0:
					l1_result[i] = l1_result[i - 1].copy()

            # --- 執行 Level 2 計算 ---
            for l2_func in l2_func_list:	
				final_df = _apply_function_mock(l2_func, l1_result, self.cal_func_class_l2)
				for i in range(len(final_df)):
					if np.all(np.isnan(final_df[i])) and i > 0:
						final_df[i] = final_df[i - 1]
				resdf = pd.concat(final_df).sort_index()
				if check_nan_rows(resdf):
					output_records.append({
						"full_key": (l1_func_tuple, l2_func),
						"final_df": resdf
					})
		return output_records


def cal_functions(result, fun_class, func_str):
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
	return check_valid(result)


def _apply_function_mock(func_tuple, daily_data_lis, cal_class):
	"""
	Returns:
		daily_results:
		list of pd.DataFrames
	"""
	daily_results = []
	for group in daily_data_lis:
		if len(func_tuple) == 1:	
			daily_results.append(cal_functions(group, cal_class, func_tuple[0]))
		else:
			daily_results.append(cal_functions(group, cal_class, func_tuple))	
	return daily_results




def check_valid(arr: pd.DataFrame) -> pd.DataFrame:
	"""Check if array is valid (not all zero)"""
	if np.nansum(abs(arr)) == 0 or np.isnan(arr.values).all():
		return pd.DataFrame(np.full_like(arr.values, np.nan), index=arr.index, columns=arr.columns)
	
def check_nan_rows(self, df):
	nan_rows = df.isna().all(axis=1)
	zero_rows = (df == 0).all(axis=1)
	return (nan_rows.sum() < len(df) / 3) and (zero_rows.sum() < len(df) / 3)

# def run_l1_tasks(self, func_list, data_lis=None):
# 	res_dict = {}
# 	func_tuple_list = [
# 		perm for i in range(1, min(4, len(func_list) + 1))
# 		for perm in permutations(func_list, i)
# 	]
# 	if data_lis is None:
# 		data_lis = self.data_lis.copy()
# 	for func_tuple in tqdm(func_tuple_list, desc=f"processing {getattr(self, 'fill_method', '')}"):
# 		result = self.apply_function(func_tuple, data_lis)
# 		none_count = sum(1 for x in result if np.all(np.isnan(x)))
# 		if len(result) == 0 or (none_count / len(result) >0.3):
# 			continue
# 		for i in range(len(result)):
# 			if np.all(np.isnan(result[i])) and i > 0:
# 				result[i] = result[i - 1].copy()
# 		res_dict[f"{func_tuple}"] = result
# 	return res_dict

# @ray.remote
# def run_l1_single_task(initial_data_ref: ray.ObjectRef, func_tuple: Tuple, cal_class: object) -> tuple[str, ray.ObjectRef]:
# 	"""
# 	執行單一的 Level 1 計算。
# 	接收初始數據的引用和一個函數組合。
# 	返回 func_tuple 和指向結果的引用。
# 	"""
# 	data_lis = ray.get(initial_data_ref)
	
# 	# --- 這是您 L1 迴圈內的核心邏輯 ---
# 	result = _apply_function_mock(func_tuple, data_lis, cal_class)
	
# 	none_count = sum(1 for x in result if np.all(np.isnan(x)))
# 	if len(result) == 0 or (none_count / len(result) > 0.3):
# 		return ray.put(None) # 返回 None 表示該結果被過濾

# 	for i in range(len(result)):
# 		if np.all(np.isnan(result[i])) and i > 0:
# 			result[i] = result[i - 1].copy()
	
# 	# 將結果放入共享記憶體，只返回引用
# 	return  ray.put(result)

# @ray.remote
# def run_l2_single_tasks(l1_result_ref: ray.ObjectRef, func_tuple: Tuple, cal_class: object):
# 	result_list = ray.get(l1_result_ref)
# 	if result_list is None:
# 		return None
# 	result = _apply_function_mock(func_tuple, result_list, cal_class)
# 	for i in range(len(result)):
# 		if np.all(np.isnan(result[i])) and i > 0:
# 			result[i] = result[i - 1]
# 	resdf = pd.concat(result).sort_index()
# 	if check_nan_rows(resdf):
# 		return resdf
# 	return None

