import glob
from math import e
from operator import le
import os
import gc
from turtle import delay
from venv import logger

from matplotlib import table
import ray
import yaml
import pandas as pd
import numpy as np
from os.path import join, exists
from pathlib import Path
import subprocess
from tqdm import tqdm
from typing import Dict, List
import ast
from shuffle_dir_data import run_single
from utils import data_utils as utils
from utils import fill_utils as fu
from utils import pooling_corr as pc
from . import path_set
from ..calculators.level1_calculator import Level1Calculator
from ..processors.level1_processor import ProcessCalculatorL1
from alpha_processsing.calculators import level1_calculator

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise ValueError(f"Failed to load config: {e}")
        
def init_ray(config: dict):
    """Initialize Ray with dynamic CPU allocation."""
    num_cpus = config.get('n_jobs', os.cpu_count() or 8)
    temp_dir = config.get('ray_temp_dir', '/tmp/ray')
    try:
        ray.init(num_cpus=num_cpus, _temp_dir=temp_dir, ignore_reinit_error=True)
    except Exception as e:
        raise RuntimeError(f"Ray initialization failed: {e}")

    

def load_data(config: dict) -> pd.DataFrame:
    """Load or generate data."""
    table_name = config.get('table_name')
    table_freq = config.get('table_freq', '15')  # Default to 15 minutes if not specified
    table_params = config.get('table_params', '') # Default to 15 minutes if not specified
    suffix = f"{table_freq}_{table_params}" if table_params else table_freq
    if not table_name:
        raise ValueError("Table name must be specified in the config.")
    cal_data_prefix = f'{table_name}'
    cal_data_path = Path(path_set.original_alpha_data_path) / f'{cal_data_prefix}_{suffix}.parquet'

    if not cal_data_path.exists():
        print(f'Generating {cal_data_prefix}_{table_params} data...')
        script_path = Path(path_set.code_recorded_path) / 'data_calculator' / f'{cal_data_prefix}' / 'cal_data.py'
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

def pick_filling_method(data: pd.DataFrame, table_params:str='', window:int=0) -> list:
    fill_methods = [
        "ffill", "bfill", "fill_cross_sectional_mean", 
        "linear_interpolation", "time_weighted_mean", "knn_imputation",
        "moving_average", "factor_model"
    ]
    # 1.group data by window days
    if table_params:
        lookback_days = int(table_params.split('_')[0][1:])  #eg 'b10' -> 10
        delay_days = int(table_params.split('_')[1][1:])  #eg 'd1' -> 1
        grouped_data = group_data_by_params(data, delay_period=delay_days, window=window)
    else:
        grouped_data = group_data_by_params(data)
    # 2. 计算每种填充方法与原始数据的相关性等指标
    corr_results, _ = fu.analyze_fill_correlation(
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
    return robust_methods if robust_methods else None

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

def generate_alpha_code(cal_data_prefix: str, fill_func: str, cal_funcs: str, smooth_funcs: str, alpha_txt: str, config: dict, template_type: str = 'single') -> None:
    """Generate alpha_cls.py and alpha_config.py.

    Args:
        template_type: 'single' for single table, 'combine' for combined tables.
    """
    alpha_name = f"{cal_data_prefix}_{fill_func or 'none'}_{cal_funcs}_{smooth_funcs}".replace('__', '_')
    output_path = Path(path_set.backtest_path) / 'signals_106' / alpha_name
    output_path.mkdir(parents=True, exist_ok=True)

    fill_func_txt = f"alpha = dp.utils2.{fill_func}(data)\n" if fill_func else "alpha = data.copy()\n"

    cal_func_txt = ""
    cal_funcs_list = ast.literal_eval(f"[{cal_funcs}]")
    for func in cal_funcs_list:
        para_check = utils.extract_values_from_string(func)
        if para_check:
            func_name, para = para_check
            cal_func_txt += f"        alpha = dp.utils2.{func_name}(alpha, {para})\n"
        else:
            cal_func_txt += f"        alpha = dp.utils2.{func}(alpha)\n"

    smooth_func_txt = ""
    smooth_funcs_list = ast.literal_eval(f"[{smooth_funcs}]")
    for func in smooth_funcs_list:
        para_check = utils.extract_values_from_string(func)
        if para_check:
            func_name, para = para_check
            smooth_func_txt += f"        alpha = dp.utils2.{func_name}(alpha, {para})\n"
        else:
            smooth_func_txt += f"        alpha = dp.utils2.{func}(alpha)\n"

    work_func_txt = fill_func_txt + cal_func_txt + smooth_func_txt

    if template_type == 'combine':
        template_filename = 'combine_window.txt'
    else:
        template_filename = f'{cal_data_prefix}.txt'
    template_path = Path(path_set.code_recorded_path) / 'alpha_cls_txt' / template_type / template_filename
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Template not found: {template_path}")

    template = template.replace('{handle_func}', work_func_txt).replace('{alpha_part}', alpha_txt)

    with open(output_path / 'alpha_cls.py', 'w', encoding='utf-8') as f:
        f.write(template)

    config_template_path = Path(path_set.demo_path) / 'alpha_config.txt'
    try:
        with open(config_template_path, 'r', encoding='utf-8') as f:
            config_template = f.read()
        with open(output_path / 'alpha_config.py', 'w', encoding='utf-8') as f:
            f.write(config_template)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config template not found: {config_template_path}")

    print(f"Generated alpha code for {alpha_name}")

def process_level1(group_data_list: list, func_list: List[str], config: dict) -> dict:
    """Process Level 1 data, optionally by window."""
    cal_func_cls1 = ProcessCalculatorL1(config)
    result = {}
    try:
        cal_processor = Level1Calculator.remote(
            group_data_list, cal_func_class=cal_func_cls1)
        result = ray.get(cal_processor.run_main.remote(func_list)) # class: Dict
        ray.kill(cal_processor)
    except Exception as e:
        logger.debug(f"fail to process")
    return utils.clean_empty_nodes(result)

def process_level2(l1_res_dict: dict, smoothing_method: List[str], config: dict, batch_size: int, prefix: str) -> dict:
    """Process Level 2 data with smoothing."""
    from ..calculators.level2_calculator import Level2Calculator
    from ..processors.level2_processor import ProcessCalculatorL2
    dic = {}
    cal_func_cls2 = ProcessCalculatorL2(config)
    # l2_res_dict:
    # {func_tuple: [pd.Df, pd.Df...]}
    func_tuple_lis, total_size = list(l1_res_dict.keys()), len(l1_res_dict.keys())
    all_results = []
    # for func_tuple, group_dats in tqdm(l1_res_dict.items(), desc=f"run smoothing: {prefix}"):     
    try:
        for i in range(0, total_size, batch_size):
            current_batch_size = min(batch_size, total_size - i)
            cal_processorsl2 = [
                Level2Calculator.remote(l1_res_dict.get(func_tuple_lis[i + j]), cal_func_class=cal_func_cls2)
                for j in range(current_batch_size)
            ]
            ray_results2 = [cal_processor.run_main.remote(smoothing_method, method="daily") for cal_processor in cal_processorsl2]
            results2 = ray.get(ray_results2) # class: list [dict, dict, dict,...]
            all_results.extend(results2)
            for cal_processor in cal_processorsl2:
                ray.kill(cal_processor)
            logger.info(f"{prefix} completed batch {i // batch_size + 1}, {len(results2)} tasks")
        dic = dict(zip(func_tuple_lis, all_results)) # {call_func: {smooth_func: pd.Df}}
        
    except Exception as e:
        logger.debug(f"Level 2 processing failed for {e}")
    return dic
    


def cleanup():
    """Clean up Ray and memory."""
    ray.shutdown()
    gc.collect()

def run_single_experiment(config: dict):
    """
    Args:
        configs:
            exp_id: name eg.demo
            filling: fill_func
            data: grouped_Data_list
            level1_funcs_list:
            level2_funcs_list:
            filter_params: initial the filterprocessors , dict
    """
    data = config.get("data", None)
    prefix = config.get('exp_id', 'unknown exp_id')
    fill_func = config.get('filling', "")
    level1_funcs_list = config.get("level1_funcs_list",[])
    level2_funcs_list = config.get("level2_funcs_list",[])
    filter_initial_config = config.get("filter_params")
    if data is None:
        raise ValueError("data is None")
    # step1 : level1 and level2 process
    clean_mid_dic = process_level1(data, level1_funcs_list, filter_initial_config)
    res_dict = process_level2(clean_mid_dic, level2_funcs_list, {}, 20, f"{prefix}_{fill_func}")
    del clean_mid_dic
    gc.collect()
    # step2: save mid data
    save_dir = join(path_set.mid_data_path, f"{prefix}")
    try:
        utils.save_nested_dict(res_dict, save_dir=save_dir, filename=fill_func)
        del res_dict
        gc.collect()
        logger.info(f'{prefix}_{fill_func} parquets stored successfully!')
    except Exception as e:
        logger.error(f"Failed to save mid data: {e}")
        raise
    # step3: compute ic
    ret_df = pd.read_hdf(path_set.PathSet.ret_path)
    ic_series = [utils.compute_corr_batch.remote(ret_df, batch) for batch in utils.load_nested_dict(save_dir, 25)]
    ic_results = ray.get(ic_series)
    remain_dfs, remain_ic = {}, {}

    # step4: pooling corr filter
    for fill_func, groups in ic_results.items():
        for cal_func, subgroups in groups.items():
            for smooth_func, df in subgroups.items():
                remain_dfs[f"{fill_func}__{cal_func}__{smooth_func}"] = df[0]
                remain_ic[f"{fill_func}__{cal_func}__{smooth_func}"] = df[1]
    remain_ic_series = pd.Series(remain_ic)
    remain_pcor_alphas = pc.filter_corr_features_pooled(remain_dfs, remain_ic_series, 0.9)

    # Parse the selected alpha names back into components
    selected_alphas = []
    for alpha_name in remain_pcor_alphas:
        parts = alpha_name.split("__")
        if len(parts) >= 3:
            fill_func = parts[0]
            cal_func = "__".join(parts[1:-1])  # Handle cases where cal_func contains "__"
            smooth_func = parts[-1]
            selected_alphas.append((fill_func, cal_func, smooth_func))
        else:
            logger.warning(f"Invalid alpha name format: {alpha_name}")

    logger.info(f"Selected {len(selected_alphas)} alphas after pooling correlation filter")

    # Generate alpha code for each selected alpha combination
    logger.info(f"Would generate alpha code for: {prefix}")
    for fill_func, cal_func, smooth_func in selected_alphas:
        # Note: cal_data_prefix and alpha_txt need to be defined in the calling context
        # generate_alpha_code(cal_data_prefix, fill_func, cal_func, smooth_func, alpha_txt, config, template_type='single')
        
        pass  # Placeholder until cal_data_prefix and alpha_txt are available

    pass


def loop_experiments(config_yaml: dict):
    """
    遍历 config.yaml 中的 experiments，依次运行每个实验。
    每个实验的 func_list 可以从 function_settings 或 experiment 自己的 func_params 读取。
    """
    experiments = config_yaml.get("experiments", [])
    global_func_list = config_yaml.get("function_settings", {})
    for exp in experiments:
        if not exp.get("active", True):
            continue
        exp_id = exp.get("id") 
        data_params = exp.get("data_params", {})
        func_params = exp.get("func_params", {})
        # 优先用每个实验自己的 func_list，否则用全局 func_list
        func_list = (
            func_params.get("level1", {}).get("func_list")
            or global_func_list
        )
        level1_nopara_func_list = global_func_list.get("level1_functions", {}).get("section_funcs", [])
        level1_nopara_func_list.extend(global_func_list.get("level1_functions", {}).get("ts_funcs", [])).get("filt_funcs", [])
        level1_nopara_func_list.extend(global_func_list.get("level1_functions", {}).get("ts_funcs", [])).get("expand_funcs", [])
        level1_para_funcs_list = global_func_list.get("level1_functions", {}).get("ts_funcs", []).get("roll_funcs")
        level2_funcs_list = global_func_list.get("level2_functions", {})
        logger.info(f"Running experiment: {exp_id}")
        if func_list == global_func_list:
            logger.info("Using global function list.")
        else:
            logger.info(f"Using experiment-specific function list: {func_list}")
        for dp in data_params:
            logger.info(f"Processing data parameters: {exp_id}_{dp}")
            if isinstance(dp, str):
                data_dict = {
                    "table_name": exp_id,
                    "table_freq": dp.split('_')[0], 
                    "table_params":str.join('_', dp.split('_')[1:])
                }
            else:
                data_dict = {
                    "table_name": exp_id,
                    "table_freq": dp
                }
            cal_data = load_data(data_dict)
            if cal_data.empty:
                logger.warning(f"No data found for {exp_id}_{dp}. Skipping experiment.")
                continue
            logger.info(f"Loaded data for {exp_id}_{dp}. Starting processing...")
            # step1 : pick filling method
            filling_methods = pick_filling_method(cal_data, data_dict.get("table_params", ""))
            logger.info(f"robust filling method:{filling_methods}")
            for f_method in filling_methods:
                logger.info(f"Processing filling method: {f_method}")
                filled_array =[fu.apply_fillingdata(cal_data, filling_methods)]
            # step2: find the rolling parameters
                shape0 = filled_array[1].shape[0]
                para0, para1 = round(np.sqrt(shape0)), 2*round(np.sqrt(shape0))
                lev1_func_list = level1_nopara_func_list.copy()
                lev1_func_list.extend([f"{para_func}_{para0}" for para_func in level1_para_funcs_list])
                lev1_func_list.extend([f"{para_func}_{para1}" for para_func in level1_para_funcs_list])
                experiment_config ={
                    'exp_id': f"{exp_id}_{dp}",
                    'filling': f"{f_method}", 
                    'data': filled_array,
                    'level1_func_list':lev1_func_list,
                    'level2_func_list':level2_funcs_list,
                    'filter_params':  global_func_list.get("processorfilter_params", {}).get('level1',{})
                }
                result = run_single_experiment(experiment_config)
            


            
        # 这里可以调用你的主流程，比如
        # result = run_single_experiment(data_params, func_list, ...)
        # save_result(result, exp_id)
        # 或者直接调用 ProcessCalculatorL1 等

        # 示例：假设有一个 process_experiment 函数
        # process_experiment(data_params, func_list, ...)

    print("All experiments finished.")
