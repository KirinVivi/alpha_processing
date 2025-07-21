
from venv import logger
import ray
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from typing import List
from path_set import PathSet
from utils import fill_utils as fu
from utils import data_utils as du

import logging
logger = logging.getLogger(__name__)

def load_data(config: dict) -> pd.DataFrame:
    """Load or generate data."""
    table_name = config.get('table_name')
    table_freq = config.get('table_freq', '15')  # Default to 15 minutes if not specified
    table_params = config.get('table_params', '') # Default to 15 minutes if not specified
    suffix = f"{table_freq}_{table_params}" if table_params else table_freq
    if not table_name:
        raise ValueError("Table name must be specified in the config.")
    cal_data_prefix = f'{table_name}'
    cal_data_path = Path(PathSet.original_alpha_data_path) / f'{cal_data_prefix}_{suffix}.parquet'

    if not cal_data_path.exists():
        print(f'Generating {cal_data_prefix}_{table_params} data...')
        script_path = Path(PathSet.code_recorded_path) / 'data_calculator' / f'{cal_data_prefix}' / 'cal_data.py'
        if not script_path.exists():
            raise FileNotFoundError(f"Data generation script not found: {script_path}")
        try:
            # Use -m to run as a module if the script is part of a package
            subprocess.run(
                ["python", "-m", f"{script_path.parent.name}.{script_path.stem}"],
                cwd="C:/Users/admin/Desktop/zww/ssh_alpha", # Adjust this path as needed
                check=True
            ) 
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to generate data: {e}")  
    try:
        return pd.read_parquet(cal_data_path)
    except Exception as e:
        raise ValueError(f"Failed to load data from {cal_data_path}: {e}")



def group_data_by_params(data: pd.DataFrame, delay_period: int=0, window: int=0) -> list:
    """Group data by specified window days (non-overlapping)."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex")
    grouped_daily_data = [
            group.sort_index() for _, group in data.groupby(data.index.date)
        ]
    n = len(grouped_daily_data)
    if delay_period > 0 and window > 0:
       grouped_data = [pd.concat(grouped_daily_data[i:i+window]).sort_index() for i in range(n - window + 1)]
    elif delay_period > 0:
        grouped_data = grouped_daily_data.copy()
    elif window > 0:
        grouped_data = [pd.concat(grouped_daily_data[i:i+window]).sort_index() for i in range(n - window + 1)]
        grouped_data = [data.iloc[: -(grouped_daily_data[0].shape[0]//2), :] for data in grouped_data]
    else:
        grouped_data = [data.iloc[:(grouped_daily_data[0].shape[0]//2), :] for data in grouped_daily_data]
    return grouped_data


def pick_filling_method(data: pd.DataFrame, table_params:str='', window:int=0) -> list:
    fill_methods = [
        "none", # 0.27
        "ffill", # 2.66 s
        "bfill", # 2.84 s
        "fill_cross_sectional_mean", # 2.81 S
        "linear_interp", # 105,18 s
        "time_weighted_mean", # 1.42 S
        "knn_imputation" # 5.97S
    ]
    # 1.group data by window days
    if table_params:
        lookback_days = int(table_params.split('_')[0][1:])  #eg 'b10' -> 10
        delay_days = int(table_params.split('_')[1][1:])  #eg 'd1' -> 1
        grouped_data = group_data_by_params(data, delay_period=delay_days, window=window)
    else:
        grouped_data = group_data_by_params(data)
    # 2. 计算每种填充方法与原始数据的相关性等指标
    corr_results, filled_data_dic = fu.analyze_fill_correlation(
        grouped_data,
        fill_methods=fill_methods,
        corr_type="pearson",  # 可选 "pearson", "spearman", "kendall"
        axis=0,               # 按列相关性
        save_dir=None         # 如需保存填充结果可指定目录
    )

    # 3. 自动筛选出最稳健的填充方法
    robust_methods = fu.select_robust_fill_methods(
        corr_results,
        corr_threshold=0.9,   # 相关性阈值
        std_threshold=0.1,    # 相关性标准差阈值
        min_valid_ratio=0.8,  # 有效相关性比例
        max_methods=3         # 最多返回几个方法
    )
    return robust_methods , [filled_data_dic.get(method) for method in robust_methods]
    
    


# def process_level1(l1_actor_pool: list, func_list: List[str]) -> dict:
#     """Process Level 1 data, optionally by window."""
#     if not l1_actor_pool:
#         logger.error("L1 Actor pool is empty!")
#         return {}
    
#     result = {}
#     try:
#         processor = l1_actor_pool[0]
#         result = ray.get(processor.run_main.remote(func_list)) # class: Dict
#     except Exception as e:
#         logger.debug(f"fail to process")
#     return du.clean_empty_nodes(result)

# def process_level2(l2_actor_pool: list, l1_res_dict: dict, smoothing_method: List[str], batch_size: int, prefix: str) -> dict:
#     """Process Level 2 data with smoothing."""
    
#     dic = {}
#     # l2_res_dict:
#     # {func_tuple: [pd.Df, pd.Df...]}
#     func_tuple_lis, total_size = list(l1_res_dict.keys()), len(l1_res_dict.keys())
#     all_results_flat = []
#     # for func_tuple, group_dats in tqdm(l1_res_dict.items(), desc=f"run smoothing: {prefix}"):     
#     try:
#         for i in range(0, total_size, batch_size):
#             current_batch_size = min(batch_size, total_size - i)
#             batch_tuples = func_tuple_lis[i : i + current_batch_size]
#             ray_results = []
#             for idx, func_tuple in enumerate(batch_tuples):
#                 actor_to_use = l2_actor_pool[idx % len(l2_actor_pool)]
#                 data_for_actor = l1_res_dict.get(func_tuple)
#                 ray_results.append(actor_to_use.run_main.remote(smoothing_method, data_for_actor))
            
            
#             batch_results = ray.get(ray_results)
#             all_results_flat.extend(batch_results)
#             logger.info(f"{prefix} completed batch {i // batch_size + 1}, {len(all_results_flat)} tasks")
#         dic = dict(zip(func_tuple_lis, all_results_flat)) # {call_func: {smooth_func: pd.Df}}
        
#     except Exception as e:
#         logger.debug(f"Level 2 processing failed for {e}")
#     return dic