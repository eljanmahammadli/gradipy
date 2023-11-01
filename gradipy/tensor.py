import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        # ones_like is for test purposes only
        self.grad = np.ones_like(self.data) if requires_grad else None
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor({self.data})"

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        return Tensor(self.data + other.data)

    def __mul__(self, other):
        return Tensor(self.data * other.data)

    def __sub__(self, other):
        return Tensor(self.data - other.data)

    def __truediv__(self, other):
        return Tensor(self.data / other.data)

    def __pow__(self, other):
        # TODO: add scalar support
        return Tensor(self.data**other.data)

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data)

        def _backward():
            self.grad = out.grad @ other.data.T
            other.grad = self.data.T @ out.grad
            assert self.grad.shape == self.data.shape
            assert other.grad.shape == other.data.shape

        out._backward = _backward
        return out
