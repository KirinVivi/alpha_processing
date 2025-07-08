import ray
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import permutations
from .base_calculator import BaseCalculator

@ray.remote(max_concurrency=15)
class Level1Calculator(BaseCalculator):
	def run_main(self, func_list):
		res_dict = {}
		func_tuple_list = [
			perm for i in range(1, min(4, len(func_list) + 1))
			for perm in permutations(func_list, i)
		]
		for func_tuple in tqdm(func_tuple_list, desc=f"processing {getattr(self, 'fill_method', '')}"):
			result = self.apply_function(func_tuple)
			none_count = sum(1 for x in result if np.all(np.isnan(x)))
			if len(result) == 0 or (none_count / len(result) >0.3):
				continue
			for i in range(len(result)):
				if np.all(np.isnan(result[i])) and i > 0:
					result[i] = result[i - 1].copy()
			res_dict[f"{func_tuple}"] = result
		return res_dict

    
