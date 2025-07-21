import gc
import pandas as pd
import ray
from os.path import join
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List
from calculators.base_calculator import run_l1_single_task, run_l2_single_tasks
from processors.level1_processor import ProcessCalculatorL1
from processors.level2_processor import ProcessCalculatorL2
from tqdm import tqdm
# 假设这些是你项目中的模块
# import path_set
# import utils
# import pooling_correlation as pc
import yaml
import numpy  as np
from itertools import permutations
from path_set import PathSet
from utils import data_utils as du
from utils import fill_utils as fu
from utils import pooling_corr as pc
from runs.data_processing import load_data, pick_filling_method
from runs.code_generator import generate_alpha_code
logger = logging.getLogger(__name__)


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
    
@dataclass
class Alpha:
    """用一个数据类来清晰地表示一个Alpha，避免繁琐的字符串处理。"""
    name: str
    fill_func: str
    cal_func: str
    smooth_func: str
    dataframe: pd.DataFrame
    ic: float = 0.0
    direction: int = 1 # 1 for positive, -1 for negative  

def _prepare_alpha_data(config: Dict[str, Any]) -> str:
    """步骤 1 & 2: 数据预处理并保存中间结果。"""
    # initial the data and params
    data = config["data"]
    prefix = config['exp_id']
    fill_func = config['filling']
    logger.info(f"Processing Level 1 & 2: {prefix}_{fill_func}")
    # ray put the data list and generate the func tuple list
    initial_data_ref = ray.put(data)
    func_tuple_list_l1 = [
        perm for i in range(1, min(4, len(config["level1_funcs_list"]) + 1))
        for perm in permutations(config["level1_funcs_list"], i)
    ]
    # generate the processors instance
    cal_func_class_l1 = ProcessCalculatorL1(config['filter_params'])
    cal_func_class_l2 = ProcessCalculatorL2({})
    # streaming the calcultion
    final_results_refs_map = {}
    for l1_func_tuple in tqdm(func_tuple_list_l1, desc="Launching L1 tasks"):
        l1_ref = run_l1_single_task.remote(initial_data_ref, l1_func_tuple, cal_func_class_l1)
        for l2_func in config['level2_funcs_list']:
            l2_ref = run_l2_single_tasks.remote(l1_ref, l2_func, cal_func_class_l2)
            final_results_refs_map[(l1_func_tuple, l2_func)] = l2_ref

    # --- collecting the resuly ---
    
    logger.info("All the tasks has been launched, waiting for results...")
    final_refs_list = list(final_results_refs_map.values())
    # results_list 現在是一個混雜了 DataFrame 和 None 的列表
    results_list = ray.get(final_refs_list)

    # --- combining the results ---
    logger.info("Combining the final dict...")
    final_dict = {}
    
    # 👇 *** 簡化的邏輯 *** 👇
    # 使用 zip 將原始的 key 和計算結果按順序配對
    for full_key, result_df in zip(final_results_refs_map.keys(), results_list):
        # 只需要檢查結果是否為 None
        if result_df is not None:
            final_dict[full_key] = result_df
    res_dict = du.restructure_results(final_dict)     
    save_dir = join(PathSet.mid_data_path, prefix)
    try:
        du.save_nested_dict(res_dict, save_dir=save_dir, filename=fill_func)
        del res_dict
        gc.collect()
        logger.info(f'Mid data saved to: {save_dir}')
        return save_dir
    except Exception as e:
        logger.error(f"Failed to save mid data: {e}")
        raise

def _calculate_and_filter_alphas(save_dir: str) -> List[Alpha]:
    """步骤 3 & 4: 计算IC并进行相关性过滤。"""
    # 步骤3: 并行计算IC
    logger.info("Starting IC calculation...")
    ret_df = pd.read_hdf(PathSet.ret_path)
    # 使用`utils.load_nested_dict`分批加载数据以控制内存
    ic_series_remote = [du.compute_corr_batch.remote(ret_df, batch) for batch in du.load_nested_dict(save_dir, 25)]
    ic_results_ray = ray.get(ic_series_remote)
    ic_results = {}
    for res in ic_results_ray:
        ic_results.update(res)
    # 将嵌套的结果解析并展平为Alpha对象列表
    all_alphas = []
    for fill, groups in ic_results.items():
        for cal, subgroups in groups.items():
            for smooth, (df, ic_val) in subgroups.items():
                alpha_name = f"{fill}__{cal}__{smooth}"
                all_alphas.append(Alpha(name=alpha_name, fill_func=fill, cal_func=cal, smooth_func=smooth, dataframe=df, ic=ic_val))
    
    if not all_alphas:
        logger.warning("No alphas passed the IC filter.")
        return []

    # 步骤4: 基于IC池化相关性过滤
    logger.info(f"Starting correlation filtering for {len(all_alphas)} ..")
    alpha_dfs = {alpha.name: alpha.dataframe for alpha in all_alphas}
    alpha_ics = pd.Series({alpha.name: alpha.ic for alpha in all_alphas})
    
    # pc.filter_corr_features_pooled 返回的是保留下来的alpha名字列表和对应的IC Series
    kept_alpha_names, kept_alpha_ics = pc.filter_corr_features_pooled(alpha_dfs, alpha_ics, 0.9)
    
    # 构建最终保留的Alpha对象列表
    final_alphas = []
    kept_alphas_map = {alpha.name: alpha for alpha in all_alphas if alpha.name in kept_alpha_names}
    for name, ic in kept_alpha_ics.items():
        alpha = kept_alphas_map[name]
        alpha.ic = ic # 更新IC值
        alpha.direction = -1 if ic < 0 else 1
        final_alphas.append(alpha)

    logger.info(f"{len(final_alphas)} alphas passed the correlation filter。")
    return final_alphas

def run_single_experiment_optimized(config: dict):
    """
    运行单次量化实验的主流程。
    通过将复杂逻辑拆分到内部辅助函数中来提高清晰度。
    """
    if config.get("data") is None:
        raise ValueError("Config missing 'data'。")

    exp_prefix = config.get('exp_id', 'unknown_exp')
    logger.info(f"===== Starting running experiment: {exp_prefix} =====")

    # 1. 数据处理与保存
    save_dir = _prepare_alpha_data(config)

    # 2. IC计算与过滤
    final_alphas = _calculate_and_filter_alphas(save_dir)
    
    if not final_alphas:
        logger.warning(f"Exp {exp_prefix} did not filter out any alpha, skipping...")
        return

    # 3. 清理被丢弃的alpha文件
    all_alpha_names = {p.name for p in Path(save_dir).glob(f"{config['filling']}__*.parquet")}
    kept_alpha_names = {f"{alpha.fill_func}__{alpha.name}.parquet" for alpha in final_alphas}
    drop_alpha_files = list(all_alpha_names - kept_alpha_names)
    logger.info(f"Deleting {len(drop_alpha_files)} filtering files...")
    du.delete_drop_mid_data(save_dir, [Path(f).stem for f in drop_alpha_files])

    # 4. 为最终选定的alpha生成代码
    logger.info(f" Preparing to generate code for {len(final_alphas)} ...")
    for alpha in final_alphas:
        alpha_txt = "-1*alpha" if alpha.direction == -1 else "alpha"
        generate_alpha_code(
            cal_data_prefix=exp_prefix,
            fill_func=alpha.fill_func,
            cal_funcs=alpha.cal_func,
            smooth_funcs=alpha.smooth_func,
            alpha_txt=alpha_txt
        )
    
    logger.info(f"===== Exp {exp_prefix} successfully completed! =====")


# 假设这是你的YAML解析库
# import yaml 

def _build_experiment_configs(exp: Dict[str, Any], global_func_list: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    为一个实验配置块（exp）生成所有可能的具体实验配置。
    这个辅助函数封装了所有的嵌套循环和参数组合逻辑。
    """
    exp_id_prefix = exp["id"]
    data_params_list = exp.get("data_params", [])
    
    # 修复原始代码中的Bug并简化函数列表的构建
    level1_funcs = global_func_list.get("level1_functions", {})
    level1_nopara = []
    level1_nopara.extend(level1_funcs.get("section_funcs", []))
    level1_nopara.extend(level1_funcs.get("ts_funcs", {}).get("filt_funcs", []))
    level1_nopara.extend(level1_funcs.get("ts_funcs", {}).get("expand_funcs", []))
    level1_para_funcs = level1_funcs.get("ts_funcs", {}).get("roll_funcs", [])
    level2_funcs_list = global_func_list.get("level2_functions", {})
    filter_params = global_func_list.get("processorfilter_params", {}).get('level1', {})

    generated_configs = []

    for dp in data_params_list:
        logger.info(f"Processing data parameters: {exp_id_prefix}_{dp}")
        # --- 数据加载 ---
        table_params_str = ""
        if isinstance(dp, str):
            parts = dp.split('_', 1)
            table_freq = parts[0]
            if len(parts) > 1:
                table_params_str = parts[1]
        else: # 假设其他情况dp就是freq
             table_freq = dp
        
        data_dict = {"table_name": exp_id_prefix, "table_freq": table_freq, "table_params": table_params_str}
        cal_data = load_data(data_dict)
        
        if cal_data.empty:
            logger.warning(f"No data found for: {exp_id_prefix}_{dp}, skipping...")
            continue
        
        # --- 填充方法循环 ---
        filling_methods, filled_data_list = pick_filling_method(cal_data, data_dict.get("table_params", ""))
        for f_method, filled_data in zip(filling_methods, filled_data_list):
            logger.info(f"Applying filling method: {f_method}")

            
            # --- 动态参数计算 ---
            shape0 = filled_data.shape[0]
            para0, para1 = round(np.sqrt(shape0)), 2 * round(np.sqrt(shape0))
            
            lev1_func_list = level1_nopara.copy()
            lev1_func_list.extend([f"{func}_{para0}" for func in level1_para_funcs])
            lev1_func_list.extend([f"{func}_{para1}" for func in level1_para_funcs])

            # --- 组装最终配置 ---
            config = {
                'exp_id': f"{exp_id_prefix}_{dp}",
                'filling': f_method, 
                'data': filled_data,
                'level1_funcs_list': lev1_func_list,
                'level2_funcs_list': level2_funcs_list,
                'filter_params': filter_params,
            }
            generated_configs.append(config)
            
    return generated_configs


def loop_experiments(config_yaml_path: str):
    """
    遍历配置文件，为每个实验生成配置并运行。
    """
    config_yaml = load_config(config_yaml_path)

    experiments = config_yaml.get("experiments", [])
    global_func_list = config_yaml.get("function_settings", {})

    logger.info("Starting experiment loop...")
    for exp in experiments:
        if not exp.get("active", True):
            logger.info(f"Exp {exp.get('id')} inactive, skipping...")
            continue
        experiment_configs = _build_experiment_configs(exp, global_func_list)
        for config in experiment_configs:
            try:
                run_single_experiment_optimized(config)
            except Exception as e:
                logger.error(f"Running {config['exp_id']} (filling: {config['filling']}) failed: {e}")
                continue
    
    logger.info("All experiments completed.")
    ray.shutdown()