from typing import List
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

    def __repr__(self) -> None:
        return f"{self.__class__.__name__}()"

    def modules(self) -> List["Module"]:
        modules_list = [self]

        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, Sequential):
                for layer in attr_value.layers:
                    if isinstance(layer, Module):
                        try:
                            modules_list.extend(layer.modules())
                        except:
                            modules_list.append(layer)

            elif isinstance(attr_value, Module):
                modules_list.extend(attr_value.modules())

        return modules_list


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_ = bias
        # TODO: this is fixed for relu, condider to use pytorch's init
        self.weight = init_kaiming_normal(in_features, out_features)
        if self.bias_ is True:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        # TODO: implement bias
        if self.bias_ is True:
            return x.matmul(self.weight) + self.bias
        return x.matmul(self.weight) + self.bias

    def parameters(self) -> list:
        return [self.weight] + ([] if self.bias is False else [self.bias])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, bias_={self.bias_})"


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
        self.bias_ = bias
        if self.bias_ is True:
            self.bias = Tensor(np.zeros((out_channels, 1), dtype=np.float32))
        # TODO: implement better init for conv2d. Is kaiming normal good enough?
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x: Tensor) -> Tensor:
        if self.bias_:
            return F.conv2d(x, self.weight, self.stride, self.padding) + Tensor(
                self.bias.data[np.newaxis, :, np.newaxis]
            )
        return F.conv2d(x, self.weight, self.stride, self.padding)

    def parameters(self) -> list:
        return [self.weight] + ([] if self.bias is False else [self.bias])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias_={self.bias_})"


class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = Tensor(np.ones(num_features, dtype=np.float32))
        self.bias = Tensor(np.zeros(num_features, dtype=np.float32))
        self.running_mean = Tensor(np.zeros(num_features, dtype=np.float32))
        self.running_var = Tensor(np.ones(num_features, dtype=np.float32))
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            xmean = x.data.mean(axis=0)
            xvar = x.data.var(axis=0)
        else:
            xmean = self.running_mean.data
            xvar = self.running_var
        out = self.weight.data * ((x.data - xmean) / np.sqrt(xvar + self.eps)) + self.bias.data
        if self.training:
            self.running_mean = Tensor(
                (1 - self.momentum) * self.running_mean.data + self.momentum * xmean
            )
            self.running_var = Tensor((1 - self.momentum) * self.running_var.data + self.momentum * xvar)
        return Tensor(out)  # what is the children of this tensor?

    def parameters(self) -> list:
        return [self.weight, self.bias]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps}, momentum={self.momentum})"


class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = Tensor(np.ones(num_features, dtype=np.float32))
        self.bias = Tensor(np.zeros(num_features, dtype=np.float32))
        self.running_mean = Tensor(np.zeros(num_features, dtype=np.float32))
        self.running_var = Tensor(np.ones(num_features, dtype=np.float32))
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Calculate mean and variance along the (0, 2, 3) axes
            xmean = x.data.mean(axis=(0, 2, 3), keepdims=True)
            xvar = x.data.var(axis=(0, 2, 3), keepdims=True)
        else:
            xmean = self.running_mean.data.reshape(1, -1, 1, 1)
            xvar = self.running_var.data.reshape(1, -1, 1, 1)

        # Batch normalization
        out = self.weight.data.reshape(1, -1, 1, 1) * (
            (x.data - xmean) / np.sqrt(xvar + self.eps)
        ) + self.bias.data.reshape(1, -1, 1, 1)

        if self.training:
            # Update running mean and variance
            self.running_mean = Tensor(
                (1 - self.momentum) * self.running_mean.data + self.momentum * xmean.squeeze()
            )
            self.running_var = Tensor(
                (1 - self.momentum) * self.running_var.data + self.momentum * xvar.squeeze()
            )

        return Tensor(out)

    def parameters(self) -> list:
        return [self.weight, self.bias]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps}, momentum={self.momentum})"


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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return F.adaptive_avg_pool2d(x, self.output_size)

    def parameters(self) -> list:
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_size={self.output_size})"


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def parameters(self) -> list:
        return []

    def __repr__(self) -> None:
        return f"{self.__class__.__name__}()"


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
