import math
import numpy as np
from gradipy.tensor import Tensor


def init_kaiming_normal(*args, nonlinearity="relu"):
    gain = {"relu": math.sqrt(2), "sigmoid": 1, "tanh": 5 / 3}[nonlinearity]
    fan = math.sqrt((args[0]))  # fan_in
    std = gain / fan
    normal = (np.random.randn(*args) * std).astype(np.float32)
    return Tensor(normal)
