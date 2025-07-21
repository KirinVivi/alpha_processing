import ray
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import permutations
from .base_calculator import BaseCalculator

@ray.remote(max_concurrency=15)
class Level1Calculator(BaseCalculator):
	pass
    
