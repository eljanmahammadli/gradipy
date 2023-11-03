import math
import numpy as np
from gradipy.tensor import Tensor


class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.data = None

    def __call__(self, logits, target):
        self.probs = logits.softmax(target)
        logprobs = self.probs.log()
        n = logprobs.shape[0]
        self.data = -logprobs.data[range(n), target.data].mean()
        return self

    def __repr__(self):
        return f"CrossEntropyLoss({self.data})"

    def backward(self):
        return self.probs.backward()


def init_kaiming_normal(*args, nonlinearity="relu"):
    gain = {"relu": math.sqrt(2), "sigmoid": 1, "tanh": 5 / 3}[nonlinearity]
    fan = math.sqrt((args[0]))  # fan_in
    std = gain / fan
    normal = (np.random.randn(*args) * std).astype(np.float32)
    return Tensor(normal)
