import pytest
import numpy as np
import torch
import torch.nn.functional as F
from gradipy.tensor import Tensor


def test_softmax():
    """Test softmax's forward and backward function."""
    # create data
    m, c = 10000, 10  # batch size, number of classes
    l = np.random.rand(m, c)
    y = np.random.randint(0, c, (m,))
    # pytorch
    lpt = torch.tensor(l, requires_grad=True)
    ypt = torch.tensor(y)
    ppt = F.softmax(lpt, dim=1)
    losspt = F.cross_entropy(lpt, ypt)
    losspt.backward()
    rpt, gpt = ppt.data.numpy(), lpt.grad.numpy()
    # gradipy
    lgp = Tensor(l)
    pgp = lgp.softmax(y)
    pgp.backward()
    rgp, ggp = pgp.data, lgp.grad
    # test
    np.testing.assert_allclose(rgp, rpt)
    np.testing.assert_allclose(ggp, gpt)


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
