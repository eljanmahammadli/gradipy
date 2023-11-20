import numpy as np
import torch
import torch.nn.functional as ptF
import gradipy.nn.functional as F
from gradipy.tensor import Tensor


def test_conv2d():
    # convnet settings
    input_size = 128
    channel = 4
    kernel_size = 4
    kernel_output = 12
    batch_size = 64
    stride = 3
    padding = 2
    ii = np.random.randn(batch_size, channel, input_size, input_size).astype(np.float32)
    wi = np.random.randn(kernel_output, channel, kernel_size, kernel_size).astype(np.float32)

    def test_pytorch():
        i = torch.from_numpy(ii)
        w = torch.from_numpy(wi)
        o = ptF.conv2d(i, w, padding=padding, stride=stride)
        return o.numpy()

    def test_gradipy():
        i = Tensor(ii)
        w = Tensor(wi)
        o = F.conv2d(i, w, padding=padding, stride=stride)
        return o.data

    for x, y in zip(test_pytorch(), test_gradipy()):
        np.testing.assert_allclose(x, y, atol=1e-4)


def test_avgpool2d():
    input_size = 128
    channel = 4
    kernel_size = 4
    batch_size = 64
    stride = 1
    padding = 2
    ii = np.random.randn(batch_size, channel, input_size, input_size).astype(np.float32)

    def test_pytorch():
        i = torch.from_numpy(ii)
        o = ptF.avg_pool2d(i, kernel_size=kernel_size, padding=padding, stride=stride)
        return o.numpy()

    def test_gradipy():
        i = Tensor(ii)
        o = F.avg_pool2d(i, kernel_size=kernel_size, padding=padding, stride=stride)
        return o.data

    for x, y in zip(test_pytorch(), test_gradipy()):
        np.testing.assert_allclose(x, y, atol=1e-5)


def test_maxpool2d():
    input_size = 128
    channel = 4
    kernel_size = 4
    batch_size = 64
    stride = 3
    padding = 2
    ii = np.random.randn(batch_size, channel, input_size, input_size).astype(np.float32)

    def test_pytorch():
        i = torch.from_numpy(ii)
        o = ptF.max_pool2d(i, kernel_size=kernel_size, padding=padding, stride=stride)
        return o.numpy()

    def test_gradipy():
        i = Tensor(ii)
        o = F.max_pool2d(i, kernel_size=kernel_size, padding=padding, stride=stride)
        return o.data

    for x, y in zip(test_pytorch(), test_gradipy()):
        np.testing.assert_allclose(x, y)


def test_cross_entropy():
    n, c = 32, 10  # batch and class size
    xi = np.random.randn(n, c).astype(np.float32)  # logits
    yi = np.random.randint(0, c, size=(n,)).astype(np.int32)  # ground truth

    def test_pytorch():
        x = torch.tensor(xi, dtype=torch.float32, requires_grad=True)
        y = torch.tensor(yi, dtype=torch.long)
        loss = ptF.cross_entropy(x, y)
        loss.backward()
        return loss.detach().numpy(), x.grad.numpy()

    def test_gradipy():
        x = Tensor(xi)
        y = Tensor(yi)
        loss = F.cross_entropy(x, y)
        loss.backward()
        return loss.data, x.grad

    for x, y in zip(test_pytorch(), test_gradipy()):
        np.testing.assert_allclose(x, y, atol=1e-6)
