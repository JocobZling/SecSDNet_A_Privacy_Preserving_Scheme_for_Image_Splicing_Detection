import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import cv2
import numpy as np


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def rgb2yCbCr(input_im):
    im_flat = input_im.contiguous().view(32, [-1, 3]).float().cuda(1)
    mat = torch.tensor([[0.257, -0.148, 0.439],
                        [0.564, -0.291, -0.368],
                        [0.098, 0.439, -0.071]]).cuda(1)
    bias = torch.tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda(1)
    temp = im_flat.mm(mat) + bias
    out = temp.view(input_im.shape[0], input_im.shape[1], input_im.shape[2], input_im.shape[3]).cuda(1)
    return out


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU  # alias Swish  (torch>=1.7)

        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer())


class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c: int,  # block input channel
                 expand_c: int,  # block expand channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.ReLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class InvertedResidualConfig:
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    def __init__(self,
                 kernel: int,  # 3 or 5
                 input_c: int,
                 out_c: int,
                 expanded_ratio: int,  # 1 or 6
                 stride: int,  # 1 or 2
                 use_se: bool,  # True
                 drop_rate: float,
                 index: str,  # 1a, 2a, 2b, ...
                 width_coefficient: float):
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers = OrderedDict()
        activation_layer = nn.SiLU  # alias Swish

        # expand
        if cnf.expanded_c != cnf.input_c:
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                           cnf.expanded_c,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,
                                                           activation_layer=activation_layer)})

        # depthwise
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,
                                                  norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})

        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.input_c,
                                                   cnf.expanded_c)})

        # project
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)})

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        if cnf.drop_rate > 0:
            self.dropout = nn.Dropout2d(p=cnf.drop_rate, inplace=True)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x

        return result


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer


def se_model(chann_in, reduction):
    layer = nn.Sequential(
        nn.Linear(chann_in, chann_in // reduction, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(chann_in // reduction, chann_in, bias=False),
        nn.Sigmoid()
    )
    return layer


class Model(nn.Module):
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 1000,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        super(Model, self).__init__()
        # 自适应残差模块
        self.layer1 = conv_layer(3, 3, 3, 1)
        self.layer2 = conv_layer(3, 3, 3, 1)
        self.layer3 = conv_layer(3, 3, 3, 1)
        self.layer4 = conv_layer(6, 6, 3, 1)
        self.layer5 = conv_layer(6, 6, 3, 1)
        # SE模块
        self.layer6 = nn.AdaptiveAvgPool2d(1)
        self.layer7 = se_model(15, 2)

        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # build inverted_residual_setting
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        b = 0
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:
                    # strides equal 1 except first cnf
                    cnf[-3] = 1  # strides
                    cnf[1] = cnf[2]  # input_channel equal output_channel

                cnf[-1] *= b / num_blocks  # update dropout ratio
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        # create layers
        layers = OrderedDict()

        # first conv
        layers.update({"stem_conv": ConvBNActivation(in_planes=15,
                                                     out_planes=adjust_channels(32),
                                                     kernel_size=3,
                                                     stride=2,
                                                     norm_layer=norm_layer,
                                                     activation_layer=nn.Identity)})

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # build top
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,
                                               out_planes=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        classifier.append(nn.SiLU())
        self.classifier = nn.Sequential(*classifier)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                # nn.init.zeros(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # 自适应残差模块
        out = self.layer1(x)
        out = out - x
        fres = out  # 3
        out = self.layer2(out)
        out = self.layer3(out)

        out = torch.cat((out, fres), 1)  # 6

        f1 = out  # 6

        out = nn.Sigmoid()(out) * out + out

        out = self.layer4(out)  # 6
        out = self.layer5(out)  # 6

        out = torch.cat((out, f1), 1)  # 12
        out = torch.cat((out, fres), 1)  # 12+3

        out = torch.sigmoid(out) * out + out

        # out2 = rgb2yCbCr(x)
        # out = torch.cat((out, out2), 1)
        # SE模块
        b, c, _, _ = out.size()
        y = self.layer6(out).view(b, c)
        y = self.layer7(y)
        y = y.view(b, c, 1, 1)
        out = out + out * y.expand_as(out)  # 做一个残差 如果SE模块表现不好 也可以获取之前的信息

        x = self.features(out)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return Model(width_coefficient=1.0,
                 depth_coefficient=1.0,
                 dropout_rate=0.2,
                 num_classes=num_classes)


def efficientnet_b1(num_classes=1000):
    # input image size 240x240
    return Model(width_coefficient=1.0,
                 depth_coefficient=1.1,
                 dropout_rate=0.2,
                 num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    # input image size 260x260
    return Model(width_coefficient=1.1,
                 depth_coefficient=1.2,
                 dropout_rate=0.3,
                 num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    # input image size 300x300
    return Model(width_coefficient=1.2,
                 depth_coefficient=1.4,
                 dropout_rate=0.3,
                 num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    # input image size 380x380
    return Model(width_coefficient=1.4,
                 depth_coefficient=1.8,
                 dropout_rate=0.4,
                 num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    # input image size 456x456
    return Model(width_coefficient=1.6,
                 depth_coefficient=2.2,
                 dropout_rate=0.4,
                 num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    # input image size 528x528
    return Model(width_coefficient=1.8,
                 depth_coefficient=2.6,
                 dropout_rate=0.5,
                 num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    # input image size 600x600
    return Model(width_coefficient=2.0,
                 depth_coefficient=3.1,
                 dropout_rate=0.5,
                 num_classes=num_classes)
