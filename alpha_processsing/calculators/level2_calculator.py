from unittest import result
import ray
import pandas as pd
import numpy as np
from tqdm import tqdm
from .base_calculator import BaseCalculator

@ray.remote(max_concurrency=20)
class Level2Calculator(BaseCalculator):
    def check_nan_rows(self, df):
        nan_rows = df.isna().all(axis=1)
        zero_rows = (df == 0).all(axis=1)
        return (nan_rows.sum() < len(df) / 3) and (zero_rows.sum() < len(df) / 3)
    
    def run_main(self, func_list):
        res_dict ={}
        for func_tuple in func_list:
            result = self.apply_function(func_tuple)
            for i in range(len(result)):
                if np.all(np.isnan(result[i])) and i > 0:
                    result[i] = result[i - 1]
            resdf = pd.concat(result).sort_index()
        
            if self.check_nan_rows(resdf):
                res_dict[f"{func_tuple}"] = resdf
        return res_dict

