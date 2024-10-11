import time
import random
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union
import numpy as np

seed = 104
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")
    
class DensePassageRetriever:
    def __init__(self, data_path) -> NoReturn:
        self.self.data_path = data_path
        
    def get_dense_embedding(self):
        return NotImplemented