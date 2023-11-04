import numpy as np


class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.data, dtype=np.float32)


class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            param.data = param.data - self.lr * param.grad


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = [np.zeros_like(param.data, dtype=np.float32) for param in params]
        self.v = [np.zeros_like(param.data, dtype=np.float32) for param in params]
        self.t = 0

    def step(self):
        self.t += 1  # should this be inside the loop?
        for i, p in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * p.grad
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * np.square(p.grad)
            mhat = self.m[i] / (1.0 - self.b1**self.t)
            vhat = self.v[i] / (1.0 - self.b2**self.t)
            p.data -= (self.lr * mhat) / (np.sqrt(vhat) + self.eps)
