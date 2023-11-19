from abc import ABC, abstractmethod
import numpy as np
from gradipy.tensor import Tensor
from .init import init_kaiming_normal


class Module(ABC):
    def __init__(self) -> None:
        self.y = None

    @abstractmethod
    def forward(self) -> Tensor:
        pass

    @abstractmethod
    def parameters(self) -> list:
        pass

    def backward(self) -> Tensor:
        self.y.backward()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __call__(self, *args) -> Tensor:
        return self.forward(*args)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        # TODO: this is fixed for relu, condider to use pytorch's init
        self.weight = init_kaiming_normal(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        # TODO: implement bias
        return x.matmul(self.weight)

    def parameters(self) -> list:
        return [self.weight] + ([] if self.bias is False else [self.bias])


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        # TODO: implement better init for conv2d. Is kaiming normal good enough?
        # TODO: implement bias
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x: Tensor) -> Tensor:
        return x.conv2d(self.weight, None, self.stride, self.padding)

    def parameters(self) -> list:
        return [self.weight] + ([] if self.bias is False else [self.bias])


class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = np.ones(num_features, dtype=np.float32)
        self.bias = np.zeros(num_features, dtype=np.float32)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            xmean = x.data.mean(axis=0)
            xvar = x.data.var(axis=0)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        out = self.weight * ((x.data - xmean) / np.sqrt(xvar + self.eps)) + self.bias
        return Tensor(out)  # what is the children of this tensor?

    def parameters(self) -> list:
        return [Tensor(self.weight), Tensor(self.bias)]
