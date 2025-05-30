import os
import gc
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
from ..utils import data_utils as utils
from . import path_set
from ..calculators.level1_calculator import Level1Calculator
from ..processors.level1_processor import ProcessCalculatorL1

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

def load_data(table_name: str, table_type: str, table_freq: str, config: dict) -> pd.DataFrame:
    """Load or generate data."""
    cal_data_prefix = f'{table_name}_{table_type[0]}_{table_freq}'
    cal_data_path = Path(path_set.original_alpha_data_path) / f'{cal_data_prefix}.hdf5'

    if not cal_data_path.exists():
        print(f'Generating {cal_data_prefix}.hdf5')
        script_path = Path(path_set.code_recorded_path) / 'data_calculator' / f'{cal_data_prefix}.py'
        if not script_path.exists():
            raise FileNotFoundError(f"Data generation script not found: {script_path}")
        try:
            subprocess.run(["python", str(script_path)], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to generate data: {e}")  
    try:
        return pd.read_hdf(cal_data_path)
    except Exception as e:
        raise ValueError(f"Failed to load data from {cal_data_path}: {e}")

def group_data_by_window(data: pd.DataFrame, window_days: int) -> list:
    """Group data by specified window days (non-overlapping)."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex")
    
    freq = f'{window_days}D'
    return [group for _, group in data.groupby(pd.Grouper(freq=freq))]

def generate_alpha_code(cal_data_prefix: str, fill_func: str, cal_funcs: str, smooth_funcs: str, alpha_txt: str, config: dict, template_type: str = 'single') -> None:
    """Generate alpha_cls.py and alpha_config.py.

    Args:
        template_type: 'single' for single table, 'combine' for combined tables.
    """
    alpha_name = f"{cal_data_prefix}_{fill_func or 'none'}_{cal_funcs}_{smooth_funcs}".replace('__', '_')
    output_path = Path(path_set.backtest_path) / 'signals_106' / alpha_name
    output_path.mkdir(parents=True, exist_ok=True)

    fill_func_txt = f"alpha = dp.utils.{fill_func}(data)\n" if fill_func else "alpha = data.copy()\n"

    cal_func_txt = ""
    cal_funcs_list = ast.literal_eval(f"[{cal_funcs}]")
    for func in cal_funcs_list:
        para_check = utils.extract_values_from_string(func)
        if para_check:
            func_name, para = para_check
            cal_func_txt += f"        alpha = dp.utils.{func_name}(alpha, {para})\n"
        else:
            cal_func_txt += f"        alpha = dp.utils.{func}(alpha)\n"

    smooth_func_txt = ""
    smooth_funcs_list = ast.literal_eval(f"[{smooth_funcs}]")
    for func in smooth_funcs_list:
        para_check = utils.extract_values_from_string(func)
        if para_check:
            func_name, para = para_check
            smooth_func_txt += f"        alpha = dp.utils.{func_name}(alpha, {para})\n"
        else:
            smooth_func_txt += f"        alpha = dp.utils.{func}(alpha)\n"

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

def process_level1(data: pd.DataFrame, func_list: List[str], fill_method_list: List[str], config: dict, window_days: int = None) -> dict:
    """Process Level 1 data, optionally by window."""
    cal_func_cls1 = ProcessCalculatorL1(config)
    res_d = {}
    
    if window_days:
        grouped_data = group_data_by_window(data, window_days)
        for fill_m in fill_method_list:
            window_results = []
            for group in grouped_data:
                try:
                    cal_processor = Level1Calculator.remote(
                        group, cal_func_class=cal_func_cls1, fill_method=fill_m, window=window_days
                    )
                    result = ray.get(cal_processor.run_main.remote(func_list, method="rolling"))
                    window_results.append(result)
                    ray.kill(cal_processor)
                except Exception as e:
                    print(f"Failed to process window for fill_method {fill_m}: {e}")
                    window_results.append({})
            
            aggregated_result = {}
            for result in window_results:
                for key, value in result.items():
                    if key not in aggregated_result:
                        aggregated_result[key] = []
                    aggregated_result[key].append(value)
            for key in aggregated_result:
                aggregated_result[key] = np.vstack(aggregated_result[key]) if aggregated_result[key] else np.array([])
            res_d[fill_m] = aggregated_result
    else:
        for fill_m in fill_method_list:
            try:
                cal_processor = Level1Calculator.remote(
                    data, cal_func_class=cal_func_cls1, fill_method=fill_m
                )
                res_d[fill_m] = ray.get(cal_processor.run_main.remote(func_list, method="daily"))
                ray.kill(cal_processor)
            except Exception as e:
                print(f"Failed to process fill_method {fill_m}: {e}")
                res_d[fill_m] = {}
    return utils.clean_empty_nodes(res_d)

def process_level2(res_d: dict, smoothing_method: List[str], config: dict, batch_size: int, prefix: str) -> dict:
    """Process Level 2 data with smoothing."""
    from ..calculators.level2_calculator import Level2Calculator
    from ..processors.level2_processor import ProcessCalculatorL2
    
    cal_func_cls2 = ProcessCalculatorL2(config)
    dic = {}
    
    for key, window_res in tqdm(res_d.items(), desc=f"run smoothing: {prefix}"):
        for fill_m, func_list_values in window_res.items():
            total_size = len(func_list_values)
            func_names_lis = list(func_list_values.keys())
            all_results = []
            try:
                for i in range(0, total_size, batch_size):
                    current_batch_size = min(batch_size, total_size - i)
                    cal_processorsl2 = [
                        Level2Calculator.remote(func_list_values.get(func_names_lis[i + j]), cal_func_class=cal_func_cls2)
                        for j in range(current_batch_size)
                    ]
                    ray_results2 = [cal_processor.run_main.remote(smoothing_method, method="daily") for cal_processor in cal_processorsl2]
                    results2 = ray.get(ray_results2)
                    all_results.extend(results2)
                    for cal_processor in cal_processorsl2:
                        ray.kill(cal_processor)
                    print(f"{key} {fill_m} completed batch {i // batch_size + 1}, {len(results2)} tasks")
                dic[f"{key}_{fill_m}"] = dict(zip(func_names_lis, all_results))
            except Exception as e:
                print(f"Level 2 processing failed for {key} {fill_m}: {e}")
                continue
    
    return dic

def compute_ic(save_dir: str, batch_size: int, ret_path: str = '/home/zww/data/full ret data/fut_ret.hdf5') -> dict:
    """Compute IC for processed data."""
    try:
        ret = pd.read_hdf(ret_path)
        futures = [utils.compute_corr_batch.remote(ret, batch) for batch in utils.load_nested_dict(save_dir, batch_size)]
        alpha_results = ray.get(futures)
        final_results = {}
        for batch_result in alpha_results:
            for fill_func, group in batch_result.items():
                if fill_func not in final_results:
                    final_results[fill_func] = {}
                for cal_func, subgroup in group.items():
                    if cal_func not in final_results[fill_func]:
                        final_results[fill_func][cal_func] = {}
                    final_results[fill_func][cal_func].update(subgroup)
        return final_results
    except Exception as e:
        print(f"IC calculation failed: {e}")
        raise

def cleanup():
    """Clean up Ray and memory."""
    ray.shutdown()
    gc.collect()