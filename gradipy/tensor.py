import numpy as np


class Tensor:
    def __init__(self, data, _children=()):
        self.data = (
            data if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)
        )
        # ones_like is for test purposes only
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self._backward = lambda: None
        self._prev = set(_children)

    def __repr__(self):
        return f"Tensor({self.data})"

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=np.float32)

    @property
    def shape(self):
        return self.data.shape

    def reshape(self, *shape):
        return Tensor(self.data.reshape(*shape))

    def _broadcast(self, other):
        bx, by = np.broadcast_arrays(self.data, other.data)
        ax = tuple(
            i
            for i, (dx, dy) in enumerate(zip(bx.strides, by.strides))
            if dx == 0 and dy != 0
        )
        ay = tuple(
            i
            for i, (dx, dy) in enumerate(zip(bx.strides, by.strides))
            if dx != 0 and dy == 0
        )
        return ax, ay

    def __add__(self, other):
        out = Tensor(self.data + other.data, _children=(self, other))

        def _backward():
            ax, ay = self._broadcast(other)
            self.grad += 1.0 * np.sum(out.grad, ax).reshape(self.shape)
            other.grad += 1.0 * np.sum(out.grad, ay).reshape(other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        return Tensor(self.data * other.data)

    def __sub__(self, other):
        return Tensor(self.data - other.data)

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, _children=(self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data))
        return out

    def softmax(self, target=None):
        exps = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        out = Tensor(probs, _children=(self,))

        def _backward():
            self.grad += (
                (probs - np.eye(probs.shape[1], dtype=np.float32)[target.data])
                / probs.shape[0]
            ) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), _children=(self,))

        def _backward():
            self.grad += (out.data > 0).astype(np.float32) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data, dtype=np.float32)
        for node in reversed(topo):
            node._backward()
