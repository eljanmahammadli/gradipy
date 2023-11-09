from abc import ABC, abstractmethod
import numpy as np
from gradipy.tensor import Tensor
from .init import init_kaiming_normal


class Module:
    def __init__(self) -> None:
        self.parameters = []

    @abstractmethod
    def forward() -> Tensor:
        pass

    @abstractmethod
    def backward() -> Tensor:
        pass

    def __call__(self, *args) -> Tensor:
        return self.forward(*args)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = init_kaiming_normal(in_features, out_features)
        self.parameters = [self.weight]
        self.y = None

    def forward(self, x: Tensor) -> Tensor:
        self.y = x.matmul(self.weight)
        return self.y

    def backward(self) -> Tensor:
        self.y.backward()


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # TODO: implement better init for conv2d. Is kaiming normal good enough?
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.parameters = [self.weight]
        self.y = None

    def forward(self, x: Tensor) -> Tensor:
        self.y = x.conv2d(self.weight, None, self.stride, self.padding)
        return self.y

    def backward(self) -> Tensor:
        self.y.backward()
