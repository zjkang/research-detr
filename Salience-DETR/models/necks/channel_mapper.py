from functools import partial
from typing import List

from torch import nn
from models.bricks.misc import Conv2dNormActivation


# 是将输入的不同尺度的特征图转换成具有相同通道数的输出特征图。
# 这个模块通常在特征金字塔网络（FPN）或类似结构中使用，
# 来处理不同层级的特征图，使得它们能够统一进入后续的网络部分
class ChannelMapper(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        norm_layer=partial(nn.GroupNorm, 32), #归一化层，这里默认是 nn.GroupNorm，分为32组
        activation_layer: nn.Module = None,
        dilation: int = 1,
        inplace: bool = True,
        bias: bool = None,
    ):
        self.in_channels = in_channels
        super().__init__()
        self.convs = nn.ModuleList()
        self.num_channels = [out_channels] * num_outs
        for in_channel in in_channels:
            self.convs.append(
                Conv2dNormActivation(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=bias,
                    groups=groups,
                    dilation=dilation,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    inplace=inplace,
                )
            )
        for _ in range(num_outs - len(in_channels)):
            self.convs.append(
                Conv2dNormActivation(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=bias,
                    groups=groups,
                    dilation=dilation,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    inplace=inplace,
                )
            )
            in_channel = out_channels

        self.init_weights()

    def init_weights(self):
        # initialize modules
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight, gain=1)
                if layer.bias:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, inputs):
        inputs = list(inputs.values())
        assert len(inputs) == len(self.in_channels)
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        for i in range(len(inputs), len(self.convs)):
            if i == len(inputs):
                outs.append(self.convs[i](inputs[-1]))
            else:
                outs.append(self.convs[i](outs[-1]))
        return outs
