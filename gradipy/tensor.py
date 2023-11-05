from typing import Union, Sequence, Tuple
import numpy as np


class Tensor:
    def __init__(
        self, data: Union[list, np.ndarray], _children: Tuple["Tensor"] = ()
    ) -> None:
        self.data = (
            data if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)
        )
        # TODO: implement requires_grad parameter
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self._backward = lambda: None
        self._prev = set(_children)

    def __repr__(self) -> str:
        return f"Tensor({self.data})"

    @property
    def shape(self) -> Tuple[int]:
        return self.data.shape

    def reshape(self, *shape: Sequence[int]) -> "Tensor":
        return Tensor(self.data.reshape(*shape))

    # https://stackoverflow.com/questions/45428696
    def _broadcast(self, other: "Tensor") -> tuple:
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

    def __add__(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data + other.data, _children=(self, other))

        def _backward() -> None:
            ax, ay = self._broadcast(other)
            self.grad += 1.0 * np.sum(out.grad, ax).reshape(self.shape)
            other.grad += 1.0 * np.sum(out.grad, ay).reshape(other.shape)

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: "Tensor") -> "Tensor":
        return self + (-other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data * other.data, _children=(self, other))

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._bacward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data @ other.data, _children=(self, other))

        def _backward() -> None:
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), _children=(self,))

        def backward() -> None:
            self.grad += 1.0 / self.data * out.grad

        out._backward = backward
        return out

    def exp(self) -> "Tensor":
        out = Tensor(np.exp(self.data), _children=(self,))

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def matmul(self, other: "Tensor") -> "Tensor":
        return self @ other

    def softmax(self) -> "Tensor":
        exps = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        out = Tensor(probs, _children=(self,))

        def _backward() -> None:
            # TODO: correct this backward pass
            self.grad += np.zeros_like(out.data, dtype=np.float32)

        out._backward = _backward
        return out

    def log_softmax(self) -> "Tensor":
        exps = np.exp(self.data - np.max(self.data, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        logprobs = np.log(probs)
        out = Tensor(logprobs, _children=(self,))

        def _backward() -> None:
            # TODO: implement scalable backward pass
            self.grad += out.grad - np.exp(out.data) * out.grad.sum(axis=1).reshape(
                -1, 1
            )

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0, self.data), _children=(self,))

        def _backward() -> None:
            self.grad += (out.data > 0).astype(np.float32) * out.grad

        out._backward = _backward
        return out

    def cross_entropy(self, target: "Tensor") -> "Tensor":
        logprobs = self.log_softmax()
        n = logprobs.shape[0]
        out = Tensor(-logprobs.data[range(n), target.data].mean(), _children=(self,))

        def _backward() -> None:
            dlogits = self.softmax().data
            dlogits[range(n), target.data] -= 1
            dlogits /= n
            self.grad += dlogits * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
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
