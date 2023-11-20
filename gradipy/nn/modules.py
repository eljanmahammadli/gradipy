from abc import ABC, abstractmethod
import numpy as np
from gradipy.tensor import Tensor
from .init import init_kaiming_normal
from gradipy.nn import functional as F


class Module(ABC):
    def __init__(self) -> None:
        self.y = None

    @abstractmethod
    def forward(self) -> Tensor:
        pass

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
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        # TODO: this is fixed for relu, condider to use pytorch's init
        self.weight = init_kaiming_normal(in_features, out_features)
        if bias is True:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        # TODO: implement bias
        return x.matmul(self.weight) + self.bias

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
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if bias is True:
            self.bias = Tensor(np.zeros((out_channels, 1), dtype=np.float32))
        # TODO: implement better init for conv2d. Is kaiming normal good enough?
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight, self.stride, self.padding) + Tensor(
            self.bias.data[np.newaxis, :, np.newaxis]
        )

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
        out = self.weight * ((x.data - xmean) / np.sqrt(xvar + self.eps)) + self.bias
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return Tensor(out)  # what is the children of this tensor?

    def parameters(self) -> list:
        return [Tensor(self.weight), Tensor(self.bias)]


class MaxPool2d(Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

    def parameters(self) -> list:
        return []


class AvgPool2d(Module):
    def __init__(self, kernel_size: int = None, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)

    def parameters(self) -> list:
        return []


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return F.adaptive_avg_pool2d(x, self.output_size)

    def parameters(self) -> list:
        return []


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def parameters(self) -> list:
        return []


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list:
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
