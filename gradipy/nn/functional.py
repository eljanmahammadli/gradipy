import numpy as np
from gradipy.tensor import Tensor


def dropout(input: Tensor, p: float = 0.5):
    if p < 0 or p > 1:
        raise ValueError(f"dropout probability should be between 0 and 1, but got {p}")
    # Save dropout mask for backward pass. Why?
    # divide by keep_prob for scaling at test time
    dropout_mask = (np.random.rand(*input.shape) > p) / (1 - p)
    out_data = input.data * dropout_mask
    out = Tensor(out_data.astype(np.float32), _children=(input,))

    def _backward() -> None:
        pass

    out._backward = _backward
    return out


def conv2d(input, weight: Tensor, stride: int = 1, padding: int = 0) -> Tensor:
    bsz, inpsz, inchn, ksz = input.shape[0], input.shape[-1], input.shape[1], weight.shape[-1]
    pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    inp = np.pad(input.data, pad_width, mode="constant", constant_values=0)
    bstrd, chstrd, rstrd, cstrd = inp.strides
    outsz = int((inpsz - ksz + 2 * padding) / stride)
    vshp = (bsz, inchn, outsz + 1, outsz + 1, ksz, ksz)
    vstrd = (bstrd, chstrd, stride * rstrd, stride * cstrd, rstrd, cstrd)
    inp_view = np.lib.stride_tricks.as_strided(inp, vshp, vstrd)
    out = Tensor(np.einsum("bchwkt,fckt->bfhw", inp_view, weight.data), _children=(input, weight))

    def _backward() -> None:
        pass

    out._backward = _backward
    return out


def max_pool2d(input, kernel_size: int = None, stride: int = 1, padding: int = 0) -> Tensor:
    if padding > kernel_size // 2:
        raise ValueError(
            f"padding should be at most half of kernel size, but got pad={padding} and kernel_size={kernel_size}"
        )
    bsz, inpsz, inchn = input.shape[0], input.shape[-1], input.shape[1]
    pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    inp = np.pad(input.data, pad_width, mode="constant", constant_values=np.nan)
    outshp = int(((inpsz - kernel_size + 2 * padding) // stride) + 1)
    pooled = np.zeros((bsz, inchn, outshp, outshp))

    for w in range(outshp):
        for h in range(outshp):
            start_w, start_h = w * stride, h * stride
            slice = inp[:, :, start_w : start_w + kernel_size, start_h : start_h + kernel_size]
            pooled[:, :, w, h] = np.nanmax(slice, axis=(2, 3))
    out = Tensor(pooled.astype(np.float32), _children=(input,))

    def _backward() -> None:
        pass

    out._backward = _backward
    return out


def avg_pool2d(input, kernel_size: int = None, stride: int = 1, padding: int = 0) -> Tensor:
    if padding > kernel_size // 2:
        raise ValueError(
            f"padding should be at most half of kernel size, but got pad={padding} and kernel_size={kernel_size}"
        )
    bsz, inpsz, inchn = input.shape[0], input.shape[-1], input.shape[1]
    pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    inp = np.pad(input.data, pad_width, mode="constant", constant_values=0)
    outshp = int(((inpsz - kernel_size + 2 * padding) // stride) + 1)
    pooled = np.zeros((bsz, inchn, outshp, outshp))

    for w in range(outshp):
        for h in range(outshp):
            start_w, start_h = w * stride, h * stride
            slice = inp[:, :, start_w : start_w + kernel_size, start_h : start_h + kernel_size]
            pooled[:, :, w, h] = np.mean(slice, axis=(2, 3))
    out = Tensor(pooled.astype(np.float32), _children=(input,))

    def _backward() -> None:
        pass

    out._backward = _backward
    return out


def adaptive_avg_pool2d(input, output_size: int) -> Tensor:
    # TODO: getting atol=1e-3 with pytorch.
    bsz, inpsz, inchn = input.shape[0], input.shape[-1], input.shape[1]
    stride = inpsz // output_size
    kernel_size = inpsz - (output_size - 1) * stride
    padding = 0
    pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    inp = np.pad(input.data, pad_width, mode="constant", constant_values=0)
    outshp = int(((inpsz - kernel_size + 2 * padding) // stride) + 1)
    pooled = np.zeros((bsz, inchn, outshp, outshp))

    for w in range(outshp):
        for h in range(outshp):
            start_w, start_h = w * stride, h * stride
            slice = inp[:, :, start_w : start_w + kernel_size, start_h : start_h + kernel_size]
            pooled[:, :, w, h] = np.mean(slice, axis=(2, 3))
    out = Tensor(pooled.astype(np.float32), _children=(input,))

    def _backward() -> None:
        pass

    out._backward = _backward
    return out


def cross_entropy(input, target: Tensor) -> Tensor:
    logprobs = input.log_softmax()
    n = logprobs.shape[0]
    out = Tensor(-logprobs.data[range(n), target.data].mean(), _children=(input,))

    def _backward() -> None:
        dlogits = input.softmax().data
        dlogits[range(n), target.data] -= 1
        dlogits /= n
        input.grad += dlogits * out.grad

    out._backward = _backward
    return out
