import torch
import numpy as np
from gradipy.tensor import Tensor
import gradipy.nn as nn
import gradipy.nn.functional as F
from efficientnet_args import global_params, blocks_args


def load_weights_from_pytorch(path="./weights/efficientnet-b0-355c32eb.pth"):
    state_dict = torch.load(path)
    for key, value in state_dict.items():
        print(f"Key: {key}, --> Tensor Shape: {value.shape}")


def get_same_padding_conv2d():
    pass


class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params, image_size=None) -> None:
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params["batch_norm_momentum"]  # pytorch's difference from tensorflow
        self._bn_eps = global_params["batch_norm_epsilon"]
        self.has_se = (self._block_args["se_ratio"] is not None) and (
            0 < self._block_args["se_ratio"] <= 1
        )
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect
        inp = self._block_args["input_filters"]  # number of input channels
        oup = self._block_args["input_filters"] * self._block_args["expand_ratio"]
        if self._block_args["expand_ratio"] != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)


if __name__ == "__main__":
    load_weights_from_pytorch()
