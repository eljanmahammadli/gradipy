from gradipy.tensor import Tensor


def softmax(input: Tensor) -> Tensor:
    return input.softmax()


def log_softmax(input: Tensor) -> Tensor:
    return input.log_softmax()


def relu(input: Tensor) -> Tensor:
    return input.relu()


def dropout(input: Tensor, p: float = 0.5):
    return input.dropout(p)


def conv2d(input: Tensor, weight: Tensor, stride: int = 1, padding: int = 0) -> Tensor:
    return input.conv2d(weight, stride, padding)


def max_pool2d(input: Tensor, kernel_size: int = None, stride: int = 1, padding: int = 0) -> Tensor:
    return input.max_pool2d(kernel_size, stride, padding)


def avg_pool2d(input: Tensor, kernel_size: int = None, stride: int = 1, padding: int = 0) -> Tensor:
    return input.avg_pool2d(kernel_size, stride, padding)


def adaptive_avg_pool2d(input: Tensor, output_size: int) -> Tensor:
    return input.adaptive_avg_pool2d(output_size)


def cross_entropy(input: Tensor, target: Tensor) -> Tensor:
    return input.cross_entropy(target)


def flatten(input: Tensor) -> Tensor:
    return input.flatten()
