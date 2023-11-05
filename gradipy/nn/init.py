import math
import numpy as np
from gradipy.tensor import Tensor

# https://pytorch.org/docs/stable/nn.init.html
gain = {"relu": math.sqrt(2), "sigmoid": 1, "tanh": 5 / 3}


# default fan_mode is fan_in for all the initialization methods
def init_kaiming_normal(*args, nonlinearity="relu"):
    fan_mode = args[0]
    std = gain[nonlinearity] / math.sqrt(fan_mode)
    dist = (np.random.randn(*args) * std).astype(np.float32)
    return Tensor(dist)


def init_kaiming_uniform(*args, nonlinearity="relu"):
    fan_mode = args[0]
    bound = gain[nonlinearity] * math.sqrt(3.0 / fan_mode)
    dist = np.random.uniform(-bound, bound, size=(args)).astype(np.float32)
    return Tensor(dist)


def init_xavier_normal(*args, nonlinearity="sigmoid"):
    fan_in, fan_out = args[0], args[-1]
    std = gain[nonlinearity] * math.sqrt(2.0 / (fan_in + fan_out))
    dist = (np.random.randn(*args) * std).astype(np.float32)
    return Tensor(dist)


def init_xavier_uniform(*args, nonlinearity="sigmoid"):
    fan_in, fan_out = args[0], args[-1]
    bound = gain[nonlinearity] * math.sqrt(6.0 / (fan_in + fan_out))
    dist = np.random.uniform(-bound, bound, size=(args)).astype(np.float32)
    return Tensor(dist)
