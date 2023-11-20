from gradipy.tensor import Tensor
from gradipy.nn import functional as F


class CrossEntropyLoss:
    def __init__(self):
        self.out = None

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        self.out = F.cross_entropy(logits, target)
        return self.out

    def __repr__(self) -> str:
        return f"CrossEntropyLoss({self.out.data})"

    def backward(self):
        self.out.backward()
