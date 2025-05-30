import ray
from .base_calculator import BaseCalculator

@ray.remote(max_concurrency=30)
class Level1Calculator(BaseCalculator):
    pass