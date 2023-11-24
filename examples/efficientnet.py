import torch
import numpy as np
from gradipy.tensor import Tensor
import gradipy.nn as nn
import gradipy.nn.functional as F


def load_weights_from_pytorch(path="./weights/efficientnet-b0-355c32eb.pth"):
    state_dict = torch.load(path)
    for key, value in state_dict.items():
        print(f"Key: {key}, --> Tensor Shape: {value.shape}")


batch_norm_momentum = 0.9
batch_norm_epsilon = 1e-5
has_se = None
id_skip = None
inp = 3
oup = 1000
image_size = 224


class MBConvBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()


if __name__ == "__main__":
    load_weights_from_pytorch()
