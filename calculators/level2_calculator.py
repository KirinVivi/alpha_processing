import ray
from .base_calculator import BaseCalculator

@ray.remote(max_concurrency=20)
class Level2Calculator(BaseCalculator):
    def check_nan_rows(self, df):
        nan_rows = df.isna().all(axis=1)
        zero_rows = (df == 0).all(axis=1)
        return (nan_rows.sum() < len(df) / 3) and (zero_rows.sum() < len(df) / 3)