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


class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data = param.data - self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.data, dtype=np.float32)
