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
