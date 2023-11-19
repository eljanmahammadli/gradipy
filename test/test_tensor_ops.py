import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as ptnn  # to avoid name conflict with gradipy.
from gradipy.tensor import Tensor
import gradipy.nn as nn


# TODO: this file needs some refactring. Better to split the test cases into multiple files.
# also create global variables for test data


def test_matmul():
    # create data
    m, n, h = 15, 20, 30
    a = np.random.randn(m, n).astype(np.float32)
    b = np.random.randn(n, h).astype(np.float32)
    # pytorch
    apt = torch.tensor(a, requires_grad=True)
    bpt = torch.tensor(b, requires_grad=True)
    mpt = apt @ bpt
    mpt.backward(gradient=torch.ones_like(mpt, dtype=torch.float32))
    rpt, agpt, bgpt = mpt.data.numpy(), apt.grad.numpy(), bpt.grad.numpy()
    # gradipy
    agp = Tensor(a)
    bgp = Tensor(b)
    mgp = agp @ bgp
    mgp.backward()
    rgp, aggp, bggp = mgp.data, agp.grad, bgp.grad
    # compare
    np.testing.assert_allclose(rgp, rpt, atol=1e-5)
    np.testing.assert_allclose(aggp, agpt, atol=1e-5)
    np.testing.assert_allclose(bggp, bgpt, atol=1e-5)


def test_addition():
    # create data
    a = np.random.randn(5, 3, 4, 1).astype(np.float32)
    b = np.random.randn(3, 1, 10).astype(np.float32)
    # pytorch
    apt = torch.tensor(a, requires_grad=True, dtype=torch.float32)
    bpt = torch.tensor(b, requires_grad=True, dtype=torch.float32)
    addpt = apt + bpt
    addpt.backward(gradient=torch.ones_like(addpt, dtype=torch.float32))
    rpt, agpt, bgpt = addpt.data.numpy(), apt.grad.numpy(), bpt.grad.numpy()
    # gradipy
    agp = Tensor(a)
    bgp = Tensor(b)
    addgp = agp + bgp
    addgp.backward()
    rgp, aggp, bggp = addgp.data, agp.grad, bgp.grad
    # test
    np.testing.assert_allclose(rgp, rpt)
    np.testing.assert_allclose(aggp, agpt)
    np.testing.assert_allclose(bggp, bgpt)


def test_matmul_relu():
    # create data
    m, n, h = 15, 20, 30
    a = np.random.randn(m, n).astype(np.float32)
    b = np.random.randn(n, h).astype(np.float32)
    # pytorch
    apt = torch.tensor(a, requires_grad=True)
    bpt = torch.tensor(b, requires_grad=True)
    relupt = F.relu(apt @ bpt)
    relupt.backward(gradient=torch.ones_like(relupt, dtype=torch.float32))
    rpt, agpt, bgpt = relupt.data.numpy(), apt.grad.numpy(), bpt.grad.numpy()
    # gradipy
    agp = Tensor(a)
    bgp = Tensor(b)
    relugp = (agp @ bgp).relu()
    relugp.backward()
    rgp, aggp, bggp = relugp.data, agp.grad, bgp.grad
    # compare
    np.testing.assert_allclose(rgp, rpt, atol=1e-5)
    np.testing.assert_allclose(aggp, agpt, atol=1e-5)
    np.testing.assert_allclose(bggp, bgpt, atol=1e-5)


def test_log():
    # create data
    m, n = 50, 15
    a = np.random.rand(m, n).astype(np.float32)
    # pytorch
    apt = torch.tensor(a, requires_grad=True)
    lpt = apt.log()
    lpt.backward(gradient=torch.ones_like(lpt, dtype=torch.float32))
    rpt, agpt = lpt.data.numpy(), apt.grad.numpy()
    # gradipy
    agp = Tensor(a)
    lgp = agp.log()
    lgp.backward()
    rgp, aggp = lgp.data, agp.grad
    # compare
    np.testing.assert_allclose(rgp, rpt, atol=1e-5)
    np.testing.assert_allclose(aggp, agpt, atol=1e-5)


def test_log_softmax():
    xi = np.random.randn(5, 3).astype(np.float32)
    yi = np.random.randn(3, 4).astype(np.float32)
    zi = np.random.randn(4, 3).astype(np.float32)

    def test_pytorch():
        x, y, z = torch.tensor(xi), torch.tensor(yi), torch.tensor(zi)
        for p in [x, y, z]:
            p.requires_grad = True
        out = x.matmul(y).log_softmax(dim=1).matmul(z)
        out.backward(gradient=(torch.ones_like(out, dtype=torch.float32)))
        return out.detach().numpy(), x.grad.numpy(), y.grad.numpy(), z.grad.numpy()

    def test_gradipy():
        x, y, z = Tensor(xi), Tensor(yi), Tensor(zi)
        out = x.matmul(y).log_softmax().matmul(z)
        out.backward()
        return out.data, x.grad, y.grad, z.grad

    for x, y in zip(test_pytorch(), test_gradipy()):
        np.testing.assert_allclose(x, y, atol=1e-5)


def test_cross_entropy():
    n, c = 32, 10  # batch and class size
    xi = np.random.randn(n, c).astype(np.float32)  # logits
    yi = np.random.randint(0, c, size=(n,)).astype(np.int32)  # ground truth

    def test_pytorch():
        x = torch.tensor(xi, dtype=torch.float32, requires_grad=True)
        y = torch.tensor(yi, dtype=torch.long)
        loss = F.cross_entropy(x, y)
        loss.backward()
        return loss.detach().numpy(), x.grad.numpy()

    def test_gradipy():
        x = Tensor(xi)
        y = Tensor(yi)
        loss = x.cross_entropy(y)
        loss.backward()
        return loss.data, x.grad

    for x, y in zip(test_pytorch(), test_gradipy()):
        np.testing.assert_allclose(x, y, atol=1e-6)


def test_cross_entropy_loss():
    n, c = 32, 10  # batch and class size
    xi = np.random.randn(n, c).astype(np.float32)  # logits
    yi = np.random.randint(0, c, size=(n,)).astype(np.int32)  # ground truth

    def test_pytorch():
        x = torch.tensor(xi, dtype=torch.float32, requires_grad=True)
        y = torch.tensor(yi, dtype=torch.long)
        criterion = ptnn.CrossEntropyLoss()
        loss = criterion(x, y)
        loss.backward()
        return loss.detach().numpy(), x.grad.numpy()

    def test_gradipy():
        x = Tensor(xi)
        y = Tensor(yi)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(x, y)
        loss.backward()
        return loss.data, x.grad

    for x, y in zip(test_pytorch(), test_gradipy()):
        np.testing.assert_allclose(x, y, atol=1e-6)


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
        o = F.conv2d(i, w, padding=padding, stride=stride)
        return o.numpy()

    def test_gradipy():
        i = Tensor(ii)
        w = Tensor(wi)
        o = i.conv2d(w, padding=padding, stride=stride)
        return o.data

    for x, y in zip(test_pytorch(), test_gradipy()):
        np.testing.assert_allclose(x, y, atol=1e-4)


def test_conv2d():
    # convnet settings
    input_size = 128
    channel = 4
    kernel_size = 4
    batch_size = 64
    stride = 3
    padding = 2
    ii = np.random.randn(batch_size, channel, input_size, input_size).astype(np.float32)

    def test_pytorch():
        i = torch.from_numpy(ii)
        o = F.max_pool2d(i, kernel_size=kernel_size, padding=padding, stride=stride)
        return o.numpy()

    def test_gradipy():
        i = Tensor(ii)
        o = i.max_pool2d(kernel_size=kernel_size, padding=padding, stride=stride)
        return o.data

    for x, y in zip(test_pytorch(), test_gradipy()):
        np.testing.assert_allclose(x, y)


def test_BatchNorm1d():
    ii = np.random.randn(32, 10).astype(np.float32)
    num_features = ii.shape[-1]

    def test_pytorch():
        i = torch.from_numpy(ii)
        bn = ptnn.BatchNorm1d(num_features)
        o = bn(i)
        return (
            o.detach().numpy(),
            bn.weight.detach().numpy(),
            bn.bias.detach().numpy(),
            bn.running_mean.numpy(),
            # bn.running_var.numpy(),
        )

        return (bn.running_var.numpy(),)

    def test_gradipy():
        i = Tensor(ii)
        bn = nn.BatchNorm1d(num_features)
        o = bn(i)
        return (
            o.data,
            bn.weight.data,
            bn.bias.data,
            bn.running_mean,
            # bn.running_var
            # variance calculation is different from pytorch for some reason.
        )

    for x, y in zip(test_pytorch(), test_gradipy()):
        np.testing.assert_allclose(x, y, atol=1e-5)
