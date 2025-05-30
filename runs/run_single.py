import time
import numpy as np
from tqdm import tqdm
from os.path import join
from ..utils import data_utils as utils
from . import path_set
from runs.common import load_config, init_ray, load_data, process_level1, process_level2, compute_ic, generate_alpha_code, cleanup

def main():
    t1 = time.time()
    
    # Load config
    config_path = join(os.path.dirname(__file__), '..', 'config.yaml')
    config = load_config(config_path)
    
    # Initialize Ray
    init_ray(config)
    
    # Data parameters
    table_name, table_type, table_freq = "sell_volume_exlarge_order", "moneyflow", "15"
    cal_data_prefix = f'{table_name}_{table_type[0]}_{table_freq}'
    
    # Load data
    cal_data = load_data(table_name, table_type, table_freq, config)
    
    # Get window sizes
    window_sizes = config.get('filter_params', {}).get('level1', {}).get('windows', [2, 5, 7])
    
    # Define function list
    rolling_para = window_sizes
    func_list = [
        'diff', 'abs_v', 'sigmoid', 'tan', 'sin', 'cos', 'reciprocal',
        'dwt_ca_fixed', 'dwt_da_fixed', 'dwt_ca', 'dwt_da', 'ewma_filter', 'kalman_filter', 'robust_zscore_filter',
        *[f'rolling_mean_{w}' for w in rolling_para],
        *[f'rolling_max_{w}' for w in rolling_para],
        *[f'rolling_min_{w}' for w in rolling_para],
        *[f'rolling_zscore_{w}' for w in rolling_para],
        *[f'rolling_std_{w}' for w in rolling_para],
        *[f'rolling_skew_{w}' for w in rolling_para],
        *[f'rolling_kurt_{w}' for w in rolling_para],
        'pct_rank'
    ]
    
    # Step 1: Process Level 1 by window
    res_d = {}
    fill_method_list = config.get('filter_params', {}).get('level1', {}).get('fill_methods', [None, "fill_cross_sectional_mean", "ffill"])
    for window_days in window_sizes:
        print(f"Processing window size: {window_days} days")
        window_res_d = process_level1(cal_data, func_list, fill_method_list, config, window_days=window_days)
        res_d[f"window_{window_days}"] = window_res_d
    
    # Step 2: Process Level 2
    dic = process_level2(res_d, config.get('filter_params', {}).get('level2', {}).get('smoothing_methods', [
        'quantile_25', 'mean', 'quantile_75', 'skew', 'kurt', 'cv', 'ptp', 'sum', 'stddev'
    ]), config, config.get('batch_size', 20), f"{table_freq}_{table_name}")
    
    t2 = time.time()
    print(f'Used {round(t2 - t1, 2)}s')
    
    # Step 3: Save mid data
    save_dir = join(path_set.mid_data_path, f"{cal_data_prefix}_window")
    try:
        utils.save_nested_dict(dic, save_dir=save_dir)
        print(f'{cal_data_prefix}_window csvs stored successfully!')
    except Exception as e:
        print(f"Failed to save mid data: {e}")
        raise
    
    # Step 4: Compute IC
    final_results = compute_ic(save_dir, config.get('batch_size', 20))
    
    # Step 5: Generate code
    cleaned_dict = utils.clean_empty_nodes(final_results)
    for fill_func, keys_b in cleaned_dict.items():
        for cal_funcs, keys_c in keys_b.items():
            for smooth_funcs, alpha_txt in keys_c.items():
                try:
                    generate_alpha_code(cal_data_prefix, fill_func, cal_funcs, smooth_funcs, alpha_txt, config, template_type='single')
                except Exception as e:
                    print(f"Failed to generate code for {fill_func}_{cal_funcs}_{smooth_funcs}: {e}")
                    continue
    
    cleanup()

if __name__ == "__main__":
    main()