import os
import gc
import ray
import time
import ast
import utils
import subprocess
import importlib
import pandas as pd
from tqdm import tqdm
from os.path import join, exists
from path_set import demo_path, backtest_path, mid_data_path, original_alpha_data_path, code_recorded_path
from DataShuffleCalculatorlevel1 import DatashuffleCalculatorlevel1
from DataShuffleCalculatorlevel2 import DatashuffleCalculatorlevel2

t1 = time.time()
ray.init(num_cpus=40)

# init
table_name, table_type, table_freq = "sell_volume_exlarge_order", "moneyflow", "15"
rolling_para = '(2,7)'

# step1: cal data
cal_data_prefix = f'{table_name}_{table_type[0]}_{table_freq}'
cal_data_path = join(original_alpha_data_path, f'{cal_data_prefix}.hdf5')
if not exists(cal_data_path):
	print(f'run to generate {cal_data_prefix}.py')
	print(join(code_recorded_path, 'data_cal_py', f'{cal_data_prefix}.py'))
	subprocess.run(["python", join(code_recorded_path, 'data_cal_py', f'{cal_data_prefix}.py')])
cal_data = pd.read_hdf(cal_data_path)

# step2: Datashuffle level1
fill_method_list = [None, "fill_cross_sectional_mean", "ffill"]
res_d = {}

func_list = [
	# time-series processors
	'diff', 'abs_v',
	'sigmoid', 'tan', 'sin', 'cos', 'reciprocal',
	# time-series filter processor
	'dwt_ca', 'dwt_da', 'ftt_A_interpid', 'ftt_X_interpld', 'ftt_angle_interpld',
	# time-series rolling processors
	f'rolling_mean_{rolling_para}', f'rolling_max_{rolling_para}',
	f'rolling_min_{rolling_para}', f'rolling_zscore_{rolling_para}',
	f'rolling_std_{rolling_para}', f'rolling_skew_{rolling_para}',
	f'rolling_kurt_{rolling_para}',
	# cross-sectional processors
	'pct_rank'
]

module_name1 = "ProcessCalculator.ProcessCalculatorl1"
class_name1 = "ProcessCalculatorl1"
module1 = importlib.import_module(module_name1)
cal_func_cls1 = getattr(module1, class_name1)
cal_processors1 = [DatashuffleCalculatorlevel1.remote(cal_data, cal_func_class=cal_func_cls1(), fill_method=f) for f in fill_method_list]
ray_results1 = [cal_processorl1.run_main.remote(func_list) for cal_processorl1 in cal_processors1]
results = ray.get(ray_results1)
res_d = dict(zip(fill_method_list, results))
# shut down the ray Actor
for cal_processorl1 in cal_processors1:
	ray.kill(cal_processorl1)

clean_res_d_l1 = utils.clean_empty_nodes(res_d)

# step3: Datashuffle level2, smooth data
module_name2 = "ProcessCalculator.ProcessCalculatorl2"
class_name2 = "ProcessCalculatorl2"
module2 = importlib.import_module(module_name2)
cal_func_cls2 = getattr(module2, class_name2)
smoothing_method = ['quantile_25', 'mean', 'quantile_75', 'skew', 'kurt', 'cv', 'ptp', 'sum', 'stddev']
dic = {}
batch_size = 20
for fill_m, func_list_values in tqdm(clean_res_d_l1.items(), desc=f"run smoothing: {table_freq}{table_name}"):
	total_size = len(func_list_values.values())
	func_names_lis = list(func_list_values.keys())
	all_results = []
	for i in range(0, total_size, batch_size):
		current_batch_size = min(batch_size, total_size - i)
		cal_processorsl2 = [DatashuffleCalculatorlevel2.remote(func_list_values.get(func_names_lis[i + j]), cal_func_class=cal_func_cls2()) for j in range(current_batch_size)]
		ray_results2 = [cal_processorl2.run_main.remote(smoothing_method) for cal_processorl2 in cal_processorsl2]
		results2 = ray.get(ray_results2)
		all_results.extend(results2)
		for cal_processorl2 in cal_processorsl2:
			ray.kill(cal_processorl2)
		print(f"{fill_m} 完成批次 {i // batch_size + 1},共 {len(results2)}个任务")
	dic[fill_m] = dict(zip(func_names_lis, all_results))

t2 = time.time()
print(f'use {round(t2 - t1, 2)}s time')
del clean_res_d_l1
gc.collect()

# step4: save mid data
save_dir = f'{mid_data_path}/{cal_data_prefix}'
batch_size = utils.save_nested_dict(dic, save_dir=save_dir)
del dic
gc.collect()
print(f'{cal_data_prefix} csvs strored already!')

# step4: cal ic and load data by batch size
ret = pd.read_hdf('/home/zww/data/full ret data/fut_ret.hdf5')
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

# step5: generate code
cleaned_dict = utils.clean_empty_nodes(final_results)
print(table_name, table_freq, cleaned_dict)
funcs_used, funcs_used_n = [], []

for fill_func, keys_b in cleaned_dict.items():
	if fill_func != 'None':
		fill_func_txt = f"alpha = dp.utils.{fill_func}(data)\n"
	else:
		fill_func = ''
		fill_func_txt = f"alpha = data.copy()\n"
	for cal_funcs, keys_c in keys_b.items():
		cal_func = "(" + cal_funcs.split('_(')[1]
		cal_func_txt = ""
		cal_func_lis = list(ast.literal_eval(cal_func))
		for func in cal_func_lis:
			para_check = utils.extract_values_from_string(func)
			if para_check:
				func_name, para = para_check
				cal_func_txt += f"        alpha = dp.utils.{func_name}(alpha, {para})\n"
				para = None
			else:
				cal_func_txt += f"        alpha = dp.utils.{func}(alpha)\n"
		for smooth_funcs, alpha_txt in keys_c.items():
			smooth_func = ast.literal_eval(smooth_funcs)
			smooth_func_txt = ""
			for func in smooth_func:
				print(func)
				para_check = utils.extract_values_from_string(func)
				if para_check:
					func_name, para = para_check
					smooth_func_txt += f"        alpha = dp.utils.{func_name}(alpha, {para})\n"
					para = None
				else:
					smooth_func_txt += f"        alpha = dp.utils.{func}(alpha)\n"
				return_txt = f"{alpha_txt}"

			name_func = [fill_func, func]
			name_func.extend(cal_func_lis)
			work_func_txt = fill_func_txt + cal_func_txt + smooth_func_txt
			print(work_func_txt)
			funcs_used = list(set(funcs_used))
			# read demo
			alpha_recorded_path = join(code_recorded_path, 'alpha_cls_txt', 'single')
			with open(join(alpha_recorded_path, f'{cal_data_prefix}.txt'), 'r', encoding='utf-8') as f:
				file = f.read()
			# txt replace
			file = file.replace('{handle_func}', work_func_txt)
			file = file.replace('{alpha_part}', return_txt)
			# create backtest file path
			alpha_name = f"{cal_data_prefix}_{'_'.join(name_func)}"
			if '__' in alpha_name:
				alpha_name = alpha_name.replace("__", "_")
			output_path = join(backtest_path, 'signals_106', alpha_name)
			print(alpha_name)
			if not os.path.exists(output_path):
				os.makedirs(output_path)
			# generate demo
			print(f'{file.strip()}')
			alpha_recorded_path = join(code_recorded_path, 'alpha_cls_txt', 'single')
			with open(join(alpha_recorded_path, f'{table_name}_{table_type[0]}{table_freq}.txt'), 'r', encoding='utf-8') as f:
				file = f.read()
			file = file.replace('{handle_func}', work_func_txt)
			file = file.replace('{alpha_part}', return_txt)
			# 1.init.py

			# 2.alpha_cls.py
			with open(join(output_path, 'alpha_cls.py'), "w", encoding="utf-8") as py_file:
				py_file.write(f'{file.strip()}')
			# 3.alpha_config.py
			with open(join(demo_path, 'alpha_config.txt'), 'r', encoding='utf-8') as file_config:
				file_config_content = file_config.read()
			with open(join(output_path, 'alpha_config.py'), "w", encoding="utf-8") as py_file:
				py_file.write(f'{file_config_content.strip()}')

ray.shutdown()