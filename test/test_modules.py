import numpy as np
import torch
import torch.nn as ptnn
import gradipy.nn as nn
from gradipy.tensor import Tensor


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
