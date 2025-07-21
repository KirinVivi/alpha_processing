
import ray
import pandas as pd
import numpy as np
from tqdm import tqdm
from .base_calculator import BaseCalculator

@ray.remote(max_concurrency=20)
class Level2Calculator(BaseCalculator):
    pass
