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
