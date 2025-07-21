import glob
from math import e
from operator import le
import os
import gc
from turtle import delay
from venv import logger

from pathlib import Path
from typing import List
from matplotlib import table
import ray
import yaml
import pandas as pd
import numpy as np
from os.path import join, exists
from pathlib import Path

from tqdm import tqdm
from typing import Dict, List

from utils import data_utils as du
from utils import fill_utils as fu
from utils import pooling_corr as pc
from . import path_set




    





# def generate_alpha_code(cal_data_prefix: str, fill_func: str, cal_funcs: str, smooth_funcs: str, alpha_txt:str) -> None:
#     """Generate alpha_cls.py and alpha_config.py.

#     Args:
#         template_type: 'single' for single table, 'combine' for combined tables.
#     """
#     alpha_name = f"{cal_data_prefix}_{fill_func}_{cal_funcs}_{smooth_funcs}".replace('__', '_')
#     output_path = Path(path_set.backtest_path) / 'signals_106' / alpha_name
#     output_path.mkdir(parents=True, exist_ok=True)

#     if fill_func == 'None':
#         fill_func = ''  
#     cal_func = "(" + cal_funcs.split('_(')[1]
#     cal_funcs_list = ast.literal_eval(cal_func)
#     smooth_funcs_list = ast.literal_eval(smooth_funcs)

#     fill_func_txt = f"alpha = dp.utils2.{fill_func}(data)\n" if fill_func else "alpha = data.copy()\n"
    
#     cal_func_txt = ""
#     for func in cal_funcs_list:
#         para_check = utils.extract_values_from_string(func)
#         if para_check:
#             func_name, para = para_check
#             cal_func_txt += f"        alpha = dp.utils2.{func_name}(alpha, {para})\n"
#         else:
#             cal_func_txt += f"        alpha = dp.utils2.{func}(alpha)\n"

#     smooth_func_txt = ""
#     smooth_funcs_list = ast.literal_eval(f"[{smooth_funcs}]")
#     for func in smooth_funcs_list:
#         para_check = utils.extract_values_from_string(func)
#         if para_check:
#             func_name, para = para_check
#             smooth_func_txt += f"       alpha = dp.utils2.{func_name}(alpha, {para})\n"
#         else:
#             smooth_func_txt += f"       alpha = dp.utils2.{func}(alpha)\n"

#     work_func_txt = fill_func_txt + cal_func_txt + smooth_func_txt
#     template_filename = f'{cal_data_prefix}.txt'
#     template_path = Path(path_set.code_recorded_path) / 'alpha_cls_txt' / template_filename
#     try:
#         with open(template_path, 'r', encoding='utf-8') as f:
#             template = f.read()
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Template not found: {template_path}")
#     # write alpha_cls.py
#     template = template.replace('{handle_func}', work_func_txt).replace('{alpha_part}', alpha_txt)
    
#     with open(output_path / 'alpha_cls.py', 'w', encoding='utf-8') as f:
#         f.write(template)
#     # write alpha_config.py
#     config_template_path = Path(path_set.demo_path) / 'alpha_config.txt'
#     try:
#         with open(config_template_path, 'r', encoding='utf-8') as f:
#             config_template = f.read()
#         with open(output_path / 'alpha_config.py', 'w', encoding='utf-8') as f:
#             f.write(config_template)
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Config template not found: {config_template_path}")
#     # write __init__.py
#     with open(output_path / '__init__.py', 'w', encoding='utf-8') as f:
#             pass
#     logger.info(f"Generated alpha code for {alpha_name}")


def cleanup():
    """Clean up Ray and memory."""
    ray.shutdown()
    gc.collect()

# def run_single_experiment(config: dict):
#     """
#     Args:
#         configs:
#             exp_id: name eg.demo
#             filling: fill_func
#             data: grouped_Data_list
#             level1_funcs_list:
#             level2_funcs_list:
#             filter_params: initial the filterprocessors , dict
#     """
#     data = config.get("data", None)
#     prefix = config.get('exp_id', 'unknown exp_id')
#     fill_func = config.get('filling', "")
#     level1_funcs_list = config.get("level1_funcs_list",[])
#     level2_funcs_list = config.get("level2_funcs_list",[])
#     filter_initial_config = config.get("filter_params")
#     if data is None:
#         raise ValueError("data is None")
#     # step1 : level1 and level2 process
#     clean_mid_dic = process_level1(data, level1_funcs_list, filter_initial_config)
#     res_dict = process_level2(clean_mid_dic, level2_funcs_list, {}, 20, f"{prefix}_{fill_func}")
#     del clean_mid_dic
#     gc.collect()
#     # step2: save mid data
#     save_dir = join(path_set.mid_data_path, f"{prefix}")
#     try:
#         utils.save_nested_dict(res_dict, save_dir=save_dir, filename=fill_func)
#         del res_dict
#         gc.collect()
#         logger.info(f'{prefix}_{fill_func} parquets stored successfully!')
#     except Exception as e:
#         logger.error(f"Failed to save mid data: {e}")
#         raise
#     # step3: compute ic
#     ret_df = pd.read_hdf(path_set.PathSet.ret_path)
#     ic_series = [utils.compute_corr_batch.remote(ret_df, batch) for batch in utils.load_nested_dict(save_dir, 25)]
#     ic_results = ray.get(ic_series)
#     remain_dfs, remain_ic = {}, {}

#     # step4: pooling corr filter
#     for fill_func, groups in ic_results.items():
#         for cal_func, subgroups in groups.items():
#             for smooth_func, df in subgroups.items():
#                 remain_dfs[f"{fill_func}__{cal_func}__{smooth_func}"] = df[0]
#                 remain_ic[f"{fill_func}__{cal_func}__{smooth_func}"] = df[1]
#     remain_ic_series = pd.Series(remain_ic)
#     remain_pcor_alphas, remain_pcor_ic = pc.filter_corr_features_pooled(remain_dfs, remain_ic_series, 0.9)
#     # step5 delete the drop alphas
#     drop_alphas = list(set(remain_dfs.keys()) - set(remain_pcor_alphas))
#     utils.delete_drop_mid_data(save_dir, drop_alphas)
#     # step6: generate code
#     # Parse the selected alpha names back into components
#     selected_alphas = []
#     for alpha_name in remain_pcor_alphas:
#         parts = alpha_name.split("__")
#         if len(parts) >= 3:
#             fill_func = parts[0]
#             cal_func = "__".join(parts[1:-1])  # Handle cases where cal_func contains "__"
#             smooth_func = parts[-1]
#             alpha_txt = "-1*alpha" if remain_pcor_ic[alpha_name] < 0 else "alpha"
#             selected_alphas.append((fill_func, cal_func, smooth_func, alpha_txt))
#         else:
#             logger.warning(f"Invalid alpha name format: {alpha_name}")
#     logger.info(f"Selected {len(selected_alphas)} alphas after pooling correlation filter")

#     # Generate alpha code for each selected alpha combination
#     logger.info(f"Would generate alpha code for: {prefix}")
#     for fill_func, cal_func, smooth_func, alpha_txt in selected_alphas:
#         # Note: cal_data_prefix and alpha_txt need to be defined in the calling context
#         generate_alpha_code(prefix, fill_func, cal_func, smooth_func, alpha_txt)
        
#     logger.info(f"Generated alpha code for {prefix}")


# def loop_experiments(config_yaml: dict):
#     """
#     遍历 config.yaml 中的 experiments,依次运行每个实验。
#     每个实验的 func_list 可以从 function_settings 或 experiment 自己的 func_params 读取。
#     """
#     experiments = config_yaml.get("experiments", [])
#     global_func_list = config_yaml.get("function_settings", {})
#     for exp in experiments:
#         if not exp.get("active", True):
#             continue
#         exp_id = exp.get("id") 
#         data_params = exp.get("data_params", {})
#         func_params = exp.get("func_params", {})
#         # 优先用每个实验自己的 func_list，否则用全局 func_list
#         func_list = (
#             func_params.get("level1", {}).get("func_list")
#             or global_func_list
#         )
#         level1_nopara_func_list = global_func_list.get("level1_functions", {}).get("section_funcs", [])
#         level1_nopara_func_list.extend(global_func_list.get("level1_functions", {}).get("ts_funcs", [])).get("filt_funcs", [])
#         level1_nopara_func_list.extend(global_func_list.get("level1_functions", {}).get("ts_funcs", [])).get("expand_funcs", [])
#         level1_para_funcs_list = global_func_list.get("level1_functions", {}).get("ts_funcs", []).get("roll_funcs")
#         level2_funcs_list = global_func_list.get("level2_functions", {})
#         logger.info(f"Running experiment: {exp_id}")
#         if func_list == global_func_list:
#             logger.info("Using global function list.")
#         else:
#             logger.info(f"Using experiment-specific function list: {func_list}")
#         for dp in data_params:
#             logger.info(f"Processing data parameters: {exp_id}_{dp}")
#             if isinstance(dp, str):
#                 data_dict = {
#                     "table_name": exp_id,
#                     "table_freq": dp.split('_')[0], 
#                     "table_params":str.join('_', dp.split('_')[1:])
#                 }
#             else:
#                 data_dict = {
#                     "table_name": exp_id,
#                     "table_freq": dp
#                 }
#             cal_data = load_data(data_dict)
#             if cal_data.empty:
#                 logger.warning(f"No data found for {exp_id}_{dp}. Skipping experiment.")
#                 continue
#             logger.info(f"Loaded data for {exp_id}_{dp}. Starting processing...")
#             # step1 : pick filling method
#             filling_methods = pick_filling_method(cal_data, data_dict.get("table_params", ""))
#             logger.info(f"robust filling method:{filling_methods}")
#             for f_method in filling_methods:
#                 logger.info(f"Processing filling method: {f_method}")
#                 filled_array =[fu.apply_fillingdata(cal_data, filling_methods)]
#             # step2: find the rolling parameters
#                 shape0 = filled_array[1].shape[0]
#                 para0, para1 = round(np.sqrt(shape0)), 2*round(np.sqrt(shape0))
#                 lev1_func_list = level1_nopara_func_list.copy()
#                 lev1_func_list.extend([f"{para_func}_{para0}" for para_func in level1_para_funcs_list])
#                 lev1_func_list.extend([f"{para_func}_{para1}" for para_func in level1_para_funcs_list])
#                 experiment_config ={
#                     'exp_id': f"{exp_id}_{dp}",
#                     'filling': f"{f_method}", 
#                     'data': filled_array,
#                     'level1_func_list':lev1_func_list,
#                     'level2_func_list':level2_funcs_list,
#                     'filter_params':  global_func_list.get("processorfilter_params", {}).get('level1',{})
#                 }
#                 result = run_single_experiment(experiment_config)
            


            
        # 这里可以调用你的主流程，比如
        # result = run_single_experiment(data_params, func_list, ...)
        # save_result(result, exp_id)
        # 或者直接调用 ProcessCalculatorL1 等

        # 示例：假设有一个 process_experiment 函数
        # process_experiment(data_params, func_list, ...)

    logger.info("All experiments finished.")
