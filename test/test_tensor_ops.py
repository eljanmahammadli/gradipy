import pytest
import numpy as np
import torch
import torch.nn.functional as F
from gradipy.tensor import Tensor


def cmp(item1, item2, tol=1e-6):
    """Utility function to compare two numpy arrays."""
    return np.allclose(item1, item2, rtol=tol, atol=tol)


def test_softmax():
    """Test softmax's forward and backward function."""
    x = [[0.5, 0.5], [7, 4], [9, 3]]
    y = torch.tensor([0, 1, 0])
    # pytorch
    logits = torch.tensor(x, requires_grad=True)
    probs = F.softmax(logits, dim=1)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    ppt, gpt = probs.data.numpy(), logits.grad.numpy()
    # gradipy
    logits = Tensor(x, requires_grad=True)
    probs = logits.softmax(y)
    probs.requires_grad = True
    probs._backward()

    # test
    tol = 1e-6
    p, g = probs.data, logits.grad
    assert cmp(p, ppt, tol) & cmp(g, gpt, tol)
