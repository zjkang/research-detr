# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from src.lib.models.networks.hough_module import Hough

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample 
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def fill_fc_weights(layers):
    # layers.modules() 返回该模块中的所有子模块（如卷积层、全连接层、激活函数等）
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# opt.arch, opt.heads, opt.head_conv, opt.region_num, opt.vote_field_siz
# e.g.,
# arch: res_101
# num_layers=101, arch=res, get_model=get_houghnet_net
# heads={hm: # of class * # of region = 80 * 17,
#        wh: 2 * 80,
#        reg: 2 (center),
#        voting_head: {hm},},
# head_conv=64 for resnet,
# region_num=17,vote_field_size=65
class HoughNetResNet(nn.Module):

    def __init__(self, block, layers, heads, region_num, vote_field_size, model_v1, head_conv, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads
        self.region_num = region_num
        self.vote_field_size = vote_field_size

        super(HoughNetResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        # 特征图上采样并恢复到较高的分辨率
        # 也就是从512 channels to 256 -> 126 -> 64
        # 关键点：
        # 1. 每次循环中的 fc (3x3 卷积) 保持空间维度不变。
        # 2. 每次循环中的 up (转置卷积) 将空间维度增加一倍。
        # 3. 总共进行了 3 次上采样，每次将尺寸翻倍。
        # 这种设计允许网络逐步增加特征图的空间分辨率，同时逐步减少通道数（从 256 到 64）。
        # 这是目标检测和语义分割等任务中常见的上采样策略，旨在恢复高分辨率的特征图，
        # 以便进行精确的空间预测
        self.deconv_layers = self._make_deconv_layer2(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )
        # self.final_layer = []

        # 这段代码的主要目的是：
        # 为每个头部创建适当的卷积层。
        # 对于需要 Hough 投票的头部，额外创建一个 Hough 投票实例。
        # 动态地为模型添加这些组件作为属性
        self.voting_heads = list(heads['voting_heads'])
        del heads['voting_heads']
        voting = False
        self.heads = heads
        for head in sorted(self.heads):
          num_output = self.heads[head]
          if head_conv > 0:
            fc = nn.Sequential(
                nn.Conv2d(64, head_conv,
                  kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output,
                  kernel_size=1, stride=1, padding=0))

            for voting_head in self.voting_heads:
                if re.fullmatch(head, voting_head):
                    voting = True
            # 检查这个头是否需要使用 Hough Voting 机制。如果需要，则初始化 Hough 类实例，
            # 这可能是用于某种基于Hough投票的空间变换操作（如关键点检测或对象中心点预测）
            if voting:
                out_classes = int(num_output / self.region_num)
                hough_voting = Hough(region_num=self.region_num,
                                     vote_field_size=self.vote_field_size,
                                     num_classes=out_classes,
                                     model_v1=model_v1)
                # e.g., self.voting_hm = hough_voting
                self.__setattr__('voting_' + head, hough_voting)
                voting = False
          # 没有voting map也就是没有使用houghNet的情况
          else:
            fc = nn.Conv2d(
              in_channels=64,
              out_channels=num_output,
              kernel_size=1,
              stride=1,
              padding=0
          )
          # self.hm = fc H*W*num_output=H*W*R*C
          self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _make_deconv_layer2(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = nn.Conv2d(self.inplanes, planes,
                    kernel_size=3, stride=1,
                    padding=1, dilation=1, bias=False)
            fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)


            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:

            if head in self.voting_heads:
                # access self.hm if head = 'hm'
                # self.__getattr__(head)(x) 用于动态调用模型中不同的输出头（head）对应的层，
                # 并将输入 x 传递给这些层，生成对应的输出
                # h*w*r*c
                # (B, region_num num_classes, 128, 128)
                #  if region_num = 9 and num_classes = 80, the shape would be:
                # (B, 720, 128, 128)
                voting_map_hm = self.__getattr__(head)(x)
                # self.voting_hm(voting_map_hm) h*w*c
                # (B, num_classes, 128, 128)
                # The Hough voting process essentially aggregates the votes from different 
                # regions for each class, reducing the channel dimension from 
                # region_num * num_classes to just num_classes, 
                # while maintaining the spatial dimensions (128x128 in this case)
                ret[head] = self.__getattr__('voting_' + head)(voting_map_hm)
            else:
                ret[head] = self.__getattr__(head)(x)

        return [ret]

    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            for _, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            # print('=> init final conv weights from normal distribution')
            for head in self.heads:
              final_layer = self.__getattr__(head)
              for i, m in enumerate(final_layer.modules()):
                  if isinstance(m, nn.Conv2d):
                      # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                      # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                      # print('=> init {}.bias as 0'.format(name))
                      if m.weight.shape[0] == self.heads[head]:
                          if 'hm' in head:
                              nn.init.constant_(m.weight, 0)
                              nn.init.constant_(m.bias, 0)
                          else:
                              nn.init.normal_(m.weight, std=0.001)
                              nn.init.constant_(m.bias, 0)
            #pretrained_state_dict = torch.load(pretrained)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

# opt.arch, opt.heads, opt.head_conv, opt.region_num, opt.vote_field_siz
# e.g.,
# arch: res_101
# num_layers=101, arch=res, get_model=get_houghnet_net
# heads={hm: # of class * # of region = 80 * 17,
#        wh: 2 * 80,
#        voting_head: {hm},
#        reg: 2 (center)},
# head_conv=64 for resnet,
# region_num=17,vote_field_size=65
def get_houghnet_net(num_layers, heads, head_conv, region_num, model_v1, vote_field_size):
  block_class, layers = resnet_spec[num_layers]

  model = HoughNetResNet(block_class, layers, heads, region_num, vote_field_size,  model_v1, head_conv=head_conv)
  model.init_weights(num_layers, pretrained=True)
  return model
