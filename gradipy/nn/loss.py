from gradipy.tensor import Tensor


class CrossEntropyLoss:
    def __init__(self):
        self.out = None

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        self.out = logits.cross_entropy(target)
        return self.out

    def __repr__(self) -> str:
        return f"CrossEntropyLoss({self.out.data})"

    def backward(self):
        self.out.backward()
