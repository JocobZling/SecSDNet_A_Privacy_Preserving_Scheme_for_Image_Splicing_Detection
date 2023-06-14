import torch.nn as nn
import torch.quantization
import torchvision.transforms as transforms
import datetime
from torch.multiprocessing import Manager, Pipe
import numpy as np
import torchvision.datasets as dsets
from multiprocessing import Process
from zlProject.model import efficientnet_b0 as create_model
from torch.nn import functional as F
from PIL import Image
import os

import torch
import torch.quantization
import random
import time



def formatSigmoid1(out, dict_manager, event1, event2):
    pre = out
    out = torch.where(out < -10, out + 5, out)
    out = torch.where(out > 10, out - 5, out)
    out = torch.where(out < -20, out + 10, out)
    out = torch.where(out > 20, out - 10, out)
    out = torch.where(out < -30, out + 15, out)
    out = torch.where(out > 30, out - 15, out)
    out = torch.where(out < -40, out + 20, out)
    out = torch.where(out > 40, out - 20, out)
    out = torch.where(out < -50, out + 25, out)
    out = torch.where(out > 50, out - 25, out)
    mask = out - pre
    dict_manager.update({'subtract1': mask})
    event1.set()
    event2.wait()

    subtract2 = dict_manager['subtract2']
    event1.clear()
    out = out - torch.ones_like(out) * subtract2
    return out


def formatSigmoid2(out, dict_manager, event1, event2):
    pre = out
    out = torch.where(out < -10, out + 5, out)
    out = torch.where(out > 10, out - 5, out)
    out = torch.where(out < -20, out + 10, out)
    out = torch.where(out > 20, out - 10, out)
    out = torch.where(out < -30, out + 15, out)
    out = torch.where(out > 30, out - 15, out)
    out = torch.where(out < -40, out + 20, out)
    out = torch.where(out > 40, out - 20, out)
    out = torch.where(out < -50, out + 25, out)
    out = torch.where(out > 50, out - 25, out)
    mask = out - pre
    dict_manager.update({'subtract2': mask})
    event2.set()
    event1.wait()

    subtract1 = dict_manager['subtract1']
    event2.clear()
    out = out - torch.ones_like(out) * subtract1
    return out




def generate_share3(Conv, a, b, c):
    # 生成与input形状相同、元素全为1的张量
    A = torch.ones_like(Conv) * a
    B = torch.ones_like(Conv) * b
    C = torch.ones_like(Conv) * c

    # V = np.random.random(Conv.shape)
    V = torch.ones_like(Conv) * random.uniform(0, 1)
    Alpha1 = Conv - A
    Beta1 = V - B

    return A.data, B.data, C.data, V.data, Alpha1.data, Beta1.data


def reluOnCiph(F_enc, F):
    one = torch.ones_like(F_enc)
    zero = torch.zeros_like(F_enc)
    F_label = torch.where(F_enc > 0, one, zero)

    return F.mul(F_label)


def reluForServer1(input, dict_manager, event1, event2, event3, event4, a1, b1, c1):
    (A1, B1, C1, _, Alpha1, Beta1) = generate_share3(input, a1, b1, c1)

    dict_manager.update({'Alpha1': Alpha1, 'Beta1': Beta1})
    # write(Alpha1)
    # write(Beta1)
    event2.set()
    event1.wait()

    Alpha2 = dict_manager['Alpha2']
    Beta2 = dict_manager['Beta2']

    F1 = C1 + B1.mul(Alpha1 + Alpha2) + A1.mul(Beta1 + Beta2)

    dict_manager.update({'F1': F1})
    # write(F1)
    event4.set()
    event3.wait()
    event2.clear()

    F_enc = F1 + dict_manager['F2']
    event4.clear()
    return reluOnCiph(F_enc, input)


def reluForServer2(input, dict_manager, event1, event2, event3, event4, a2, b2, c2):
    (A2, B2, C2, _, Alpha2, Beta2) = generate_share3(input, a2, b2, c2)
    dict_manager.update({'Alpha2': Alpha2, 'Beta2': Beta2})
    # write(Alpha2)
    # write(Beta2)
    event1.set()
    event2.wait()

    Alpha1 = dict_manager['Alpha1']
    Beta1 = dict_manager['Beta1']

    F2 = C2 + B2.mul(Alpha1 + Alpha2) + A2.mul(Beta1 + Beta2) + (Alpha1 + Alpha2).mul(Beta1 + Beta2)

    dict_manager.update({'F2': F2})
    # print(getsizeof(F2.storage()))
    # write(F2)
    event3.set()
    event4.wait()
    event1.clear()

    F1 = dict_manager['F1']

    F_enc = F1 + F2
    event3.clear()

    return reluOnCiph(F_enc, input)


def serverSigmoidAll1(u1, c1_mul, c1_res, t1, a1, b1, c1, dict_manager, event1, event2, event3, event4, event5, event6):
    u1 = formatSigmoid1(u1, dict_manager, event5, event6)
    x = torch.exp(-u1)
    A1 = x / c1_mul

    # exp
    dict_manager.update({'A1': A1})
    # write(A1)

    event2.set()
    event1.wait()

    A2 = dict_manager['A2']

    A = A1.mul(A2)
    e_u1 = A.mul(c1_res)

    x1 = torch.ones_like(u1) * 1 / 2
    y1 = e_u1 + torch.ones_like(u1) * 1 / 2

    # div
    # mul
    a1, b1, c1, _, Alpha1, Beta1 = generate_share2(y1, a1, b1, c1)

    e1 = y1 - a1
    f1 = t1 - b1
    e3 = x1 - a1
    f3 = t1 - b1

    dict_manager.update({'e1': e1, 'f1': f1, 'e3': e3, 'f3': f3})
    # write(e1)
    # write(f1)
    # write(f3)
    # write(f3)
    event4.set()
    event3.wait()
    event2.clear()

    e2 = dict_manager['e2']
    f2 = dict_manager['f2']
    e4 = dict_manager['e4']
    f4 = dict_manager['f4']

    eA = e1 + e2
    fA = f1 + f2
    e = e3 + e4
    f = f3 + f4

    ty1 = c1 + b1 * eA + a1 * fA + eA * fA
    tx1 = c1 + b1 * e + a1 * f + e * f

    dict_manager.update({'ty1': ty1})

    # write(ty1)

    event6.set()
    event5.wait()
    event4.clear()

    ty2 = dict_manager['ty2']
    event6.clear()

    ty = ty1 + ty2
    res1 = tx1 / ty

    return res1


def serverSigmoidAll2(u2, c2_mul, c2_res, t2, a2, b2, c2, dict_manager, event1, event2, event3, event4, event5, event6):
    # exp
    u2 = formatSigmoid2(u2, dict_manager, event5, event6)
    x = torch.exp(-u2)
    A2 = x / c2_mul

    dict_manager.update({'A2': A2})
    # write(A2)
    event1.set()
    event2.wait()

    A1 = dict_manager['A1']

    A = A1.mul(A2)
    e_u2 = A.mul(c2_res)

    x2 = torch.ones_like(u2) * 1 / 2
    y2 = e_u2 + torch.ones_like(u2) * 1 / 2

    # div
    # mul1
    a2, b2, c2, _, Alpha1, Beta1 = generate_share2(y2, a2, b2, c2)
    e2 = y2 - a2
    f2 = t2 - b2
    e4 = x2 - a2
    f4 = t2 - b2

    dict_manager.update({'e2': e2, 'f2': f2, 'e4': e4, 'f4': f4})
    # write(e2)
    # write(f2)
    # write(e4)
    # write(f4)
    event3.set()
    event4.wait()
    event1.clear()

    e1 = dict_manager['e1']
    f1 = dict_manager['f1']
    e3 = dict_manager['e3']
    f3 = dict_manager['f3']

    eA = e1 + e2
    fA = f1 + f2
    e = e3 + e4
    f = f3 + f4

    tx2 = c2 + b2 * e + a2 * f
    ty2 = c2 + b2 * eA + a2 * fA

    dict_manager.update({'ty2': ty2})
    # write(ty2)
    event5.set()
    event6.wait()
    event3.clear()

    ty1 = dict_manager['ty1']
    event5.clear()

    ty = ty1 + ty2
    res1 = tx2 / ty

    return res1


def generate_share2(Conv, a, b, c):
    A = torch.ones_like(Conv) * a
    B = torch.ones_like(Conv) * b
    C = torch.ones_like(Conv) * c

    V = torch.ones_like(Conv) * random.uniform(0, 1)
    Alpha1 = Conv - A
    Beta1 = V - B

    return A.data, B.data, C.data, V.data, Alpha1.data, Beta1.data


def SecMulServer1(x1, y1, a1, b1, c1, dict_manager, event1, event2):
    a1, b1, c1, _, Alpha1, Beta1 = generate_share2(x1, a1, b1, c1)
    e1 = x1 - a1
    f1 = y1 - b1

    dict_manager.update({'e1': e1, 'f1': f1})
    # write(e1)
    # write(f1)
    event2.set()
    event1.wait()

    e2 = dict_manager['e2']
    f2 = dict_manager['f2']

    e = e1 + e2
    f = f1 + f2

    res1 = c1 + b1 * e + a1 * f + e * f
    event2.clear()
    return res1


def SecMulServer2(x2, y2, a2, b2, c2, dict_manager, event1, event2):
    a2, b3, c3, _, Alpha1, Beta1 = generate_share2(x2, a2, b2, c2)
    e2 = x2 - a2
    f2 = y2 - b2

    dict_manager.update({'e2': e2, 'f2': f2})
    # write(e2)
    # write(f2)
    event1.set()
    event2.wait()

    e1 = dict_manager['e1']
    f1 = dict_manager['f1']

    e = e1 + e2
    f = f1 + f2

    res2 = c2 + b2 * e + a2 * f
    event1.clear()
    return res2


def GetBN(beforeBn, channel, eps):
    newBN = nn.BatchNorm2d(channel, eps=eps, momentum=0.1, affine=True, track_running_stats=True)
    newBN.eval()

    newBNGrammer = beforeBn.weight
    newBNBeita = beforeBn.bias / 2.0
    newBNMean = beforeBn.running_mean / 2.0
    newBNVar = beforeBn.running_var

    newBN.weight = nn.Parameter(newBNGrammer)
    newBN.bias = nn.Parameter(newBNBeita)
    newBN.running_mean = newBNMean
    newBN.running_var = newBNVar

    return newBN


def GetSEConv(beforeConv, in_channels, out_channels):
    newConv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))
    newConv.weight = nn.Parameter(beforeConv.weight)
    newConv.bias = nn.Parameter(beforeConv.bias / 2.0)
    return newConv


def GetLinear(beforeLinear, in_features, out_features):
    newLinear = nn.Linear(in_features, out_features, bias=True)
    newLinear.weight = nn.Parameter(beforeLinear.weight)
    newLinear.bias = nn.Parameter(beforeLinear.bias / 2.0)
    return newLinear




def ConvBNReluActivationServer1(block, input, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_mul,
                                c1_res, t1, flag, BN_channel, dict_manager1, event5, event6, event7, event8):
    block = nn.Sequential(*list(block.children()))[:]
    conv = block[0]
    BN = GetBN(block[1], BN_channel, 0.001)
    conv1 = conv(input)
    BN1 = BN(conv1)
    relu = reluForServer1(BN1.data, dict_manager, event1, event2, event3, event4, a1, b1, c1)
    return relu


def ConvBNReluActivationServer2(block, input, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_mul,
                                c1_res, t1, flag, BN_channel, dict_manager1, event5, event6, event7, event8):
    block = nn.Sequential(*list(block.children()))[:]
    conv = block[0]
    BN = GetBN(block[1], BN_channel, 0.001)
    conv1 = conv(input)
    BN1 = BN(conv1)
    relu = reluForServer2(BN1.data, dict_manager, event1, event2, event3, event4, a1, b1, c1)
    return relu


def ConvBNSiLuActivationServer1(block, input, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_mul,
                                c1_res, t1, flag, BN_channel, dict_manager1, event5, event6, event7, event8):
    block = nn.Sequential(*list(block.children()))[:]
    conv = block[0]
    BN = GetBN(block[1], BN_channel, 0.001)
    conv1 = conv(input)
    BN1 = BN(conv1)
    if flag:
        Silu1 = sigmoidMulForServer1(BN1, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_res, c1_mul, t1,
                                     dict_manager1, event5, event6)
        return Silu1
    else:
        return BN1


def ConvBNSiLuActivationServer2(block, input, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_mul,
                                c1_res, t1, flag, BN_channel, dict_manager1, event5, event6):
    block = nn.Sequential(*list(block.children()))[:]
    conv = block[0]
    BN = GetBN(block[1], BN_channel, 0.001)
    conv1 = conv(input)
    BN1 = BN(conv1)
    if flag:
        Silu1 = sigmoidMulForServer2(BN1.data, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_res, c1_mul,
                                     t1, dict_manager1, event5, event6)
        return Silu1
    else:
        return BN1


def AvgLinearSiLuServer1(block, input, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_mul, c1_res, t1,
                         in_features, out_features, dict_manager1, event5, event6):
    block = nn.Sequential(*list(block.children()))[:]
    avg = nn.AdaptiveAvgPool2d(1)(input)
    avg = torch.flatten(avg, 1)
    linear1 = GetLinear(block[1], in_features, out_features)(avg)
    Silu = sigmoidMulForServer1(linear1, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_res, c1_mul, t1,
                                dict_manager1, event5, event6)
    return Silu


def AvgLinearSiLuServer2(block, input, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_mul, c1_res, t1,
                         in_features, out_features, dict_manager1, event5, event6):
    block = nn.Sequential(*list(block.children()))[:]
    avg = nn.AdaptiveAvgPool2d(1)(input)
    avg = torch.flatten(avg, 1)
    linear1 = GetLinear(block[1], in_features, out_features)(avg)
    Silu = sigmoidMulForServer2(linear1, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_res, c1_mul, t1,
                                dict_manager1, event5, event6)
    return Silu


def InvertedResidualServer1(block, input, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_mul, c1_res, t1,
                            in_channels, out_channels, BN_channel1, BN_channel2, dict_manager1, event5, event6, event7,
                            event8):
    block = nn.Sequential(*list(block.children()))[:][0]
    blockNew = nn.Sequential(*list(block.children()))[:]
    convBNActivation1 = ConvBNSiLuActivationServer1(blockNew[0], input, dict_manager, event1, event2, event3, event4,
                                                    a1, b1, c1, c1_mul, c1_res, t1, True, BN_channel1, dict_manager1,
                                                    event5, event6, event7, event8)
    squeezeExcitation = SEBlock1(block[1], convBNActivation1, dict_manager, event1, event2, event3, event4, a1, b1,
                                 c1, c1_mul, c1_res, t1, in_channels, out_channels, dict_manager1, event5, event6)
    convBNActivation2 = ConvBNSiLuActivationServer1(block[2], squeezeExcitation, dict_manager, event1, event2, event3,
                                                    event4, a1, b1, c1, c1_mul, c1_res, t1, False, BN_channel2,
                                                    dict_manager1, event5, event6, event7, event8)
    return convBNActivation2


def InvertedResidualServer2(block, input, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_mul, c1_res, t1,
                            in_channels, out_channels, BN_channel1, BN_channel2, dict_manager1, event5, event6, event7,
                            event8):
    block = nn.Sequential(*list(block.children()))[:][0]
    block = nn.Sequential(*list(block.children()))[:]
    convBNActivation1 = ConvBNSiLuActivationServer2(block[0], input, dict_manager, event1, event2, event3, event4, a1,
                                                    b1, c1, c1_mul, c1_res, t1, True, BN_channel1, dict_manager1,
                                                    event5, event6)
    squeezeExcitation = SEBlock2(block[1], convBNActivation1, dict_manager, event1, event2, event3, event4, a1, b1,
                                 c1, c1_mul, c1_res, t1, in_channels, out_channels, dict_manager1, event5, event6)
    convBNActivation2 = ConvBNSiLuActivationServer2(block[2], squeezeExcitation, dict_manager, event1, event2, event3,
                                                    event4, a1, b1, c1, c1_mul, c1_res, t1, False, BN_channel2,
                                                    dict_manager1, event5, event6)

    return convBNActivation2


def InvertedResidualWithExpandServer1(block, input, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_mul,
                                      c1_res, t1,
                                      in_channels, out_channels, BN_channel1, BN_channel2, dict_manager1, event5,
                                      event6, event7, event8):
    block1 = nn.Sequential(*list(block.children()))[:]
    blockNew = nn.Sequential(*list(block1[0].children()))[:]
    convBNActivation1 = ConvBNSiLuActivationServer1(blockNew[0], input, dict_manager, event1, event2, event3, event4,
                                                    a1, b1, c1, c1_mul, c1_res, t1, True, BN_channel1, dict_manager1,
                                                    event5, event6, event7, event8)
    convBNActivation2 = ConvBNSiLuActivationServer1(blockNew[1], convBNActivation1, dict_manager, event1, event2,
                                                    event3, event4, a1, b1, c1, c1_mul, c1_res, t1, True, BN_channel1,
                                                    dict_manager1, event5, event6, event7, event8)
    squeezeExcitation = SEBlock1(blockNew[2], convBNActivation2, dict_manager, event1, event2, event3, event4, a1, b1,
                                 c1, c1_mul, c1_res, t1, in_channels, out_channels, dict_manager1, event5, event6)
    convBNActivation3 = ConvBNSiLuActivationServer1(blockNew[3], squeezeExcitation, dict_manager, event1, event2,
                                                    event3, event4, a1, b1, c1, c1_mul, c1_res, t1, False, BN_channel2,
                                                    dict_manager1, event5, event6, event7, event8)

    return convBNActivation3


def InvertedResidualWithExpandServer2(block, input, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_mul,
                                      c1_res, t1,
                                      in_channels, out_channels, BN_channel1, BN_channel2, dict_manager1, event5,
                                      event6, event7, event8):
    block1 = nn.Sequential(*list(block.children()))[:]
    blockNew = nn.Sequential(*list(block1[0].children()))[:]
    convBNActivation1 = ConvBNSiLuActivationServer2(blockNew[0], input, dict_manager, event1, event2, event3, event4,
                                                    a1, b1, c1, c1_mul, c1_res, t1, True, BN_channel1, dict_manager1,
                                                    event5, event6)

    convBNActivation2 = ConvBNSiLuActivationServer2(blockNew[1], convBNActivation1, dict_manager, event1, event2,
                                                    event3, event4, a1, b1, c1, c1_mul, c1_res, t1, True, BN_channel1,
                                                    dict_manager1, event5, event6)

    squeezeExcitation = SEBlock2(blockNew[2], convBNActivation2, dict_manager, event1, event2, event3, event4, a1, b1,
                                 c1, c1_mul, c1_res, t1, in_channels, out_channels, dict_manager1, event5, event6)

    convBNActivation3 = ConvBNSiLuActivationServer2(blockNew[3], squeezeExcitation, dict_manager, event1, event2,
                                                    event3,
                                                    event4, a1, b1, c1, c1_mul, c1_res, t1, False, BN_channel2,
                                                    dict_manager1, event5, event6)
    return convBNActivation3


def SEBlock1(block, input, dict_manager, event1, event2, event3, event4, a1, b1,
             c1, c1_mul, c1_res, t1, in_channels, out_channels, dict_manager1, event5, event6):
    block = nn.Sequential(*list(block.children()))[:]
    out = F.adaptive_avg_pool2d(input, output_size=(1, 1))
    conv2d1 = GetSEConv(block[0], in_channels, out_channels)(out)
    relu = reluForServer1(conv2d1, dict_manager, event1, event2, event3, event4, a1, b1, c1)
    conv2d2 = GetSEConv(block[2], out_channels, in_channels)(relu)
    sigmoid = sigmoidServer1(conv2d2, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_res, c1_mul,
                             t1, event5, event6)

    sigmoidMul = MulForServer1(sigmoid, input, dict_manager, event1, event2, event3, event4, a1, b1, c1)
    return sigmoidMul


def SEBlock2(block, input, dict_manager, event1, event2, event3, event4, a1, b1,
             c1, c1_mul, c1_res, t1, in_channels, out_channels, dict_manager1, event5, event6):
    block = nn.Sequential(*list(block.children()))[:]
    out = F.adaptive_avg_pool2d(input, output_size=(1, 1))
    conv2d1 = GetSEConv(block[0], in_channels, out_channels)(out)
    relu = reluForServer2(conv2d1, dict_manager, event1, event2, event3, event4, a1, b1, c1)
    conv2d2 = GetSEConv(block[2], out_channels, in_channels)(relu)
    sigmoid = sigmoidServer2(conv2d2, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_res, c1_mul,
                             t1, event5, event6)
    sigmoidMul = MulForServer2(sigmoid, input, dict_manager, event1, event2, event3, event4, a1, b1, c1)
    return sigmoidMul


def generate_sigmoid_share(input, a1, b1, c1, c1_res, c1_mul, t1):
    c1_mul_ = torch.ones_like(input) * c1_mul
    c1_res_ = torch.ones_like(input) * c1_res
    t1_ = torch.ones_like(input) * t1
    a1_ = torch.ones_like(input) * a1
    b1_ = torch.ones_like(input) * b1
    c1_ = torch.ones_like(input) * c1

    return c1_mul_, c1_res_, t1_, a1_, b1_, c1_


def generate_share(input, a1, b1, c1):
    a1_ = torch.ones_like(input) * a1
    b1_ = torch.ones_like(input) * b1
    c1_ = torch.ones_like(input) * c1

    return a1_, b1_, c1_


def sigmoidServer1(input, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_res, c1_mul,
                   t1, event5, event6):
    (c1_mul_, c1_res_, t1_, a1_, b1_, c1_) = generate_sigmoid_share(input, a1, b1, c1, c1_res, c1_mul, t1)
    sig = serverSigmoidAll1(input.data, c1_mul_, c1_res_, t1_, a1_, b1_, c1_, dict_manager, event1, event2, event3,
                            event4, event5, event6)
    return sig


def sigmoidServer2(input, dict_manager, event1, event2, event3, event4, a2, b2, c2, c2_res, c2_mul,
                   t2, event5, event6):
    (c2_mul_, c2_res_, t2_, a2_, b2_, c2_) = generate_sigmoid_share(input, a2, b2, c2, c2_res, c2_mul, t2)
    sig = serverSigmoidAll2(input.data, c2_mul_, c2_res_, t2_, a2_, b2_, c2_, dict_manager, event1, event2, event3,
                            event4, event5, event6)
    return sig


def sigmoidMulAddForServer1(input, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_res, c1_mul,
                            t1, dict_manager1, event5, event6):
    # sigmoid*x+x
    (c1_mul_, c1_res_, t1_, a1_, b1_, c1_) = generate_sigmoid_share(input, a1, b1, c1, c1_res, c1_mul, t1)
    sig = serverSigmoidAll1(input.data, c1_mul_, c1_res_, t1_, a1_, b1_, c1_, dict_manager, event1, event2, event3,
                            event4, event5, event6)
    sigMulAdd = SecMulServer1(sig.data, input.data, a1_, b1_, c1_, dict_manager, event1, event2) + input
    return sigMulAdd


def sigmoidMulAddForServer2(input, dict_manager, event1, event2, event3, event4, a2, b2, c2, c2_res, c2_mul,
                            t2, dict_manager1, event5, event6):
    (c2_mul_, c2_res_, t2_, a2_, b2_, c2_) = generate_sigmoid_share(input, a2, b2, c2, c2_res, c2_mul, t2)
    sig = serverSigmoidAll2(input.data, c2_mul_, c2_res_, t2_, a2_, b2_, c2_, dict_manager, event1, event2, event3,
                            event4, event5, event6)
    sigMulAdd = SecMulServer2(sig.data, input.data, a2_, b2_, c2_, dict_manager, event1, event2) + input
    return sigMulAdd


def sigmoidMulForServer1(input, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_res, c1_mul,
                         t2, dict_manager1, event5, event6):
    (c1_mul_, c1_res_, t1_, a1_, b1_, c1_) = generate_sigmoid_share(input, a1, b1, c1, c1_res, c1_mul, t2)
    sig = serverSigmoidAll1(input.data, c1_mul_, c1_res_, t1_, a1_, b1_, c1_, dict_manager1, event1, event2, event3,
                            event4, event5, event6)
    sigMul = SecMulServer1(sig.data, input.data, a1_, b1_, c1_, dict_manager, event1, event2)
    return sigMul


def sigmoidMulForServer2(input, dict_manager, event1, event2, event3, event4, a2, b2, c2, c2_res, c2_mul,
                         t2, dict_manager1, event5, event6):
    (c2_mul_, c2_res_, t2_, a2_, b2_, c2_) = generate_sigmoid_share(input, a2, b2, c2, c2_res, c2_mul, t2)
    sig = serverSigmoidAll2(input.data, c2_mul_, c2_res_, t2_, a2_, b2_, c2_, dict_manager1, event1, event2, event3,
                            event4, event5, event6)
    sigMul = SecMulServer2(sig.data, input.data, a2_, b2_, c2_, dict_manager, event1, event2)
    return sigMul


def MulForServer1(input1, input2, dict_manager, event1, event2, event3, event4, a1, b1, c1):
    (a1_, b1_, c1_) = generate_share(input1, a1, b1, c1)
    mul = SecMulServer1(input1.data, input2.data, a1_, b1_, c1_, dict_manager, event1, event2)
    return mul


def MulForServer2(input1, input2, dict_manager, event1, event2, event3, event4, a2, b2, c2):
    (a2_, b2_, c2_) = generate_share(input1, a2, b2, c2)
    mul = SecMulServer2(input1.data, input2.data, a2_, b2_, c2_, dict_manager, event1, event2)
    return mul


def random_c():
    c = random.uniform(0, 0.5)
    while (c - 0 < 1e-32):
        c = random.uniform(0, 0.5)
    return c


def random_c_mul():
    c = random.uniform(1, 2)
    while (c - 0 < 1e-32):
        c = random.uniform(1, 2)
    return c


def get_model():
    model = create_model(num_classes=2).to('cpu')
    model.load_state_dict(torch.load('', map_location='cpu'))
    return model


tran = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def seModule1(input, liner1, liner2, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_res, c1_mul, t1,
              dict_manager1, event5, event6):
    b, c, _, _ = input.size()
    out = torch.mean(input, (2, 3), keepdim=True).view(b, c)
    se1 = liner1(out)
    se_relu = reluForServer1(se1, dict_manager, event1, event2, event3, event4, a1, b1, c1)
    se2 = liner2(se_relu)
    se_sigmoid = sigmoidServer1(se2, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_res, c1_mul, t1,
                                event5, event6)
    se = se_sigmoid.view(b, c, 1, 1)
    expand = se.expand(input.size())
    out = input + SecMulServer1(input.data, expand.data, a1, b1, c1, dict_manager, event1, event2)
    return out


def seModule2(input, liner1, liner2, dict_manager, event1, event2, event3, event4, a2, b2, c2, c2_res, c2_mul, t2,
              dict_manager1, event5, event6):
    b, c, _, _ = input.size()
    out = torch.mean(input, (2, 3), keepdim=True).view(b, c)
    se1 = liner1(out)
    se_relu = reluForServer2(se1, dict_manager, event1, event2, event3, event4, a2, b2, c2)
    se2 = liner2(se_relu)
    se_sigmoid = sigmoidServer2(se2, dict_manager, event1, event2, event3, event4, a2, b2, c2, c2_res, c2_mul, t2,
                                event5, event6)
    se = se_sigmoid.view(b, c, 1, 1)
    expand = se.expand(input.size())
    out = input + SecMulServer2(input.data, expand.data, a2, b2, c2, dict_manager, event1, event2)
    return out


def server1_Model(event1, event2, event3, event4, image_1, a1, b1, c1, c1_mul, c1_res, t1, dict_manager, conv1_0,
                  conv1_1BN, conv2_0, conv2_1BN, conv3_0, conv3_1BN, conv4_0, conv4_1BN, conv5_0, conv5_1BN,
                  liner6_1, liner6_2, new_model, p1, p2, dict_manager1, event5, event6, event7, event8):
    start = datetime.datetime.now()

    # layer1
    image_1 = torch.tensor(image_1, dtype=torch.float32)
    conv1 = conv1_0(image_1)
    conv1BN = conv1_1BN(conv1)
    conv1_relu = reluForServer1(conv1BN, dict_manager, event1, event2, event3, event4, a1, b1, c1)

    # 减法
    conv2 = conv1_relu - image_1
    conv2 = torch.tensor(conv2, dtype=torch.float32)
    fres = conv2

    # layer2
    conv2_1 = conv2_0(conv2)
    conv2_1_BN = conv2_1BN(conv2_1)
    conv2_1_relu = reluForServer1(conv2_1_BN, dict_manager, event1, event2, event3, event4, a1, b1, c1)

    # layer3
    conv3_0 = conv3_0(conv2_1_relu)
    conv3_1_BN = conv3_1BN(conv3_0)
    conv3_1_relu = reluForServer1(conv3_1_BN, dict_manager, event1, event2, event3, event4, a1, b1, c1)

    # concat
    conv4_0_0 = torch.cat((conv3_1_relu, fres), 1)
    conv4_0_0 = formatSigmoid1(conv4_0_0.data, dict_manager, event1, event2)
    conv4_0_0 = formatSigmoid1(conv4_0_0.data, dict_manager, event1, event2)

    f1 = conv4_0_0

    # sigmoid*x+x
    sigMulAdd = sigmoidMulAddForServer1(conv4_0_0.data, dict_manager, event1, event2, event3, event4, a1, b1, c1,
                                        c1_res,
                                        c1_mul, t1, dict_manager1, event5, event6)

    # print(f1)

    # layer4
    conv4_0 = conv4_0(sigMulAdd)
    conv4_1_BN = conv4_1BN(conv4_0)
    conv4_1_relu = reluForServer1(conv4_1_BN, dict_manager, event1, event2, event3, event4, a1, b1, c1)

    # layer5
    conv5_0 = conv5_0(conv4_1_relu)
    conv5_1_BN = conv5_1BN(conv5_0)
    conv5_1_relu = reluForServer1(conv5_1_BN, dict_manager, event1, event2, event3, event4, a1, b1, c1)

    # concat
    conv6_0_1 = torch.cat((conv5_1_relu, f1), 1)
    conv6_0_2 = torch.cat((conv6_0_1, fres), 1)

    conv6_0_2 = formatSigmoid1(conv6_0_2.data, dict_manager, event1, event2)
    conv6_0_2 = formatSigmoid1(conv6_0_2.data, dict_manager, event1, event2)

    # sigmoid*x+x
    sigMulAdd = sigmoidMulAddForServer1(conv6_0_2.data, dict_manager, event1, event2, event3, event4, a1, b1, c1,
                                        c1_res,
                                        c1_mul, t1, dict_manager1, event5, event6)

    # se
    se = seModule1(sigMulAdd, liner6_1, liner6_2, dict_manager, event1, event2, event3, event4, a1, b1, c1, c1_res,
                   c1_mul, t1, dict_manager1, event5, event6)

    #
    #
    # efficientNet
    efficientNet = nn.Sequential(*list(new_model[7].children()))[:]
    block0 = ConvBNSiLuActivationServer1(efficientNet[0], se, dict_manager, event1, event2, event3, event4, a1, b1, c1,
                                         c1_mul, c1_res, t1, False, 32, dict_manager1, event5, event6, event7, event8)
    print("blockA--0完成")

    block1 = InvertedResidualServer1(efficientNet[1], block0, dict_manager, event1, event2, event3, event4, a1, b1, c1,
                                     c1_mul, c1_res, t1, 32, 8, 32, 16, dict_manager1, event5, event6, event7, event8)
    print("blockA--1完成")
    block2 = InvertedResidualWithExpandServer1(efficientNet[2], block1, dict_manager, event1, event2, event3, event4,
                                               a1, b1, c1, c1_mul, c1_res, t1, 96, 4, 96, 24, dict_manager1, event5,
                                               event6, event7, event8)
    print("blockA--2完成")
    block3 = block2 + InvertedResidualWithExpandServer1(efficientNet[3], block2, dict_manager, event1, event2, event3,
                                                        event4, a1, b1, c1, c1_mul, c1_res, t1, 144, 6, 144, 24,
                                                        dict_manager1, event5, event6, event7, event8)
    print("blockA--3完成")
    block4 = InvertedResidualWithExpandServer1(efficientNet[4], block3, dict_manager, event1, event2, event3, event4,
                                               a1, b1, c1, c1_mul, c1_res, t1, 144, 6, 144, 40, dict_manager1, event5,
                                               event6, event7, event8)
    print("blockA--4完成")
    block5 = block4 + InvertedResidualWithExpandServer1(efficientNet[5], block4, dict_manager, event1, event2, event3,
                                                        event4, a1, b1, c1, c1_mul, c1_res, t1, 240, 10, 240, 40,
                                                        dict_manager1, event5, event6, event7, event8)
    print("blockA--5完成")
    block6 = InvertedResidualWithExpandServer1(efficientNet[6], block5, dict_manager, event1, event2, event3, event4,
                                               a1, b1, c1, c1_mul, c1_res, t1, 240, 10, 240, 80, dict_manager1, event5,
                                               event6, event7, event8)
    print("blockA--6完成")
    block7 = block6 + InvertedResidualWithExpandServer1(efficientNet[7], block6, dict_manager, event1, event2, event3,
                                                        event4, a1, b1, c1, c1_mul, c1_res, t1, 480, 20, 480, 80,
                                                        dict_manager1, event5, event6, event7, event8)
    print("blockA--7完成")
    block8 = block7 + InvertedResidualWithExpandServer1(efficientNet[8], block7, dict_manager, event1, event2, event3,
                                                        event4, a1, b1, c1, c1_mul, c1_res, t1, 480, 20, 480, 80,
                                                        dict_manager1, event5, event6, event7, event8)
    print("blockA--8完成")
    block9 = InvertedResidualWithExpandServer1(efficientNet[9], block8, dict_manager, event1, event2, event3, event4,
                                               a1, b1, c1, c1_mul, c1_res, t1, 480, 20, 480, 112, dict_manager1, event5,
                                               event6, event7, event8)
    print("blockA--9完成")
    block10 = block9 + InvertedResidualWithExpandServer1(efficientNet[10], block9, dict_manager, event1, event2, event3,
                                                         event4, a1, b1, c1, c1_mul, c1_res, t1, 672, 28, 672, 112,
                                                         dict_manager1, event5, event6, event7, event8)
    print("blockA--10完成")
    block11 = block10 + InvertedResidualWithExpandServer1(efficientNet[11], block10, dict_manager, event1, event2,
                                                          event3, event4, a1, b1, c1, c1_mul, c1_res, t1, 672, 28, 672,
                                                          112, dict_manager1, event5, event6, event7, event8)
    print("blockA--11完成")
    block12 = InvertedResidualWithExpandServer1(efficientNet[12], block11, dict_manager, event1, event2, event3, event4,
                                                a1, b1, c1, c1_mul, c1_res, t1, 672, 28, 672, 192, dict_manager1,
                                                event5, event6, event7, event8)
    print("blockA--12完成")
    block13 = block12 + InvertedResidualWithExpandServer1(efficientNet[13], block12, dict_manager, event1, event2,
                                                          event3, event4,
                                                          a1, b1, c1, c1_mul, c1_res, t1, 1152, 48, 1152, 192,
                                                          dict_manager1, event5, event6, event7, event8)
    print("blockA--13完成")
    block14 = block13 + InvertedResidualWithExpandServer1(efficientNet[14], block13, dict_manager, event1, event2,
                                                          event3, event4,
                                                          a1, b1, c1, c1_mul, c1_res, t1, 1152, 48, 1152, 192,
                                                          dict_manager1, event5, event6, event7, event8)
    print("blockA--14完成")
    block15 = block14 + InvertedResidualWithExpandServer1(efficientNet[15], block14, dict_manager, event1, event2,
                                                          event3, event4,
                                                          a1, b1, c1, c1_mul, c1_res, t1, 1152, 48, 1152, 192,
                                                          dict_manager1, event5, event6, event7, event8)
    print("blockA--15完成")
    block16 = InvertedResidualWithExpandServer1(efficientNet[16], block15, dict_manager, event1, event2, event3, event4,
                                                a1, b1, c1, c1_mul, c1_res, t1, 1152, 48, 1152, 320, dict_manager1,
                                                event5, event6, event7, event8)
    print("blockA--16完成")
    block17 = ConvBNReluActivationServer1(efficientNet[17], block16, dict_manager, event1, event2, event3, event4,
                                          a1, b1, c1, c1_mul, c1_res, t1, True, 1280, dict_manager1, event5, event6,
                                          event7, event8)
    print("blockA--17完成")
    block18 = AvgLinearSiLuServer1(new_model[9], block17, dict_manager, event1, event2, event3, event4,
                                   a1, b1, c1, c1_mul, c1_res, t1, 1280, 2, dict_manager1, event5, event6)

    # print(se)
    p1.send(block18.detach().numpy())
    print("----------111----------------\n")


#   q.put(conv1_1_1BN.detach().numpy())

def server2_Model(event1, event2, event3, event4, image_2, a2, b2, c2, c2_mul,
                  c2_res, t2,
                  dict_manager,
                  conv1_0, conv1_1BN, conv2_0, conv2_1BN, conv3_0, conv3_1BN, conv4_0, conv4_1BN, conv5_0, conv5_1BN,
                  liner6_1, liner6_2, new_model, p1, p2, dict_manager1, event5, event6, event7, event8):
    start = datetime.datetime.now()

    # layer1
    image_1 = torch.tensor(image_2, dtype=torch.float32)
    conv1 = conv1_0(image_1)
    conv1BN = conv1_1BN(conv1)

    conv1_relu = reluForServer2(conv1BN, dict_manager, event1, event2, event3, event4, a2, b2, c2)

    # 减法
    conv2 = conv1_relu - image_2
    conv2 = torch.tensor(conv2, dtype=torch.float32)
    fres = conv2

    # layer2
    conv2_1 = conv2_0(conv2)
    conv2_1_BN = conv2_1BN(conv2_1)
    conv2_1_relu = reluForServer2(conv2_1_BN, dict_manager, event1, event2, event3, event4, a2, b2, c2)

    # layer3
    conv3_0 = conv3_0(conv2_1_relu)
    conv3_1_BN = conv3_1BN(conv3_0)
    conv3_1_relu = reluForServer2(conv3_1_BN, dict_manager, event1, event2, event3, event4, a2, b2, c2)

    # concat
    conv4_0_0 = torch.cat((conv3_1_relu, fres), 1)
    conv4_0_0 = formatSigmoid2(conv4_0_0.data, dict_manager, event1, event2)
    conv4_0_0 = formatSigmoid2(conv4_0_0.data, dict_manager, event1, event2)

    f1 = conv4_0_0

    sigMulAdd = sigmoidMulAddForServer2(conv4_0_0.data, dict_manager, event1, event2, event3, event4, a2, b2, c2,
                                        c2_res,
                                        c2_mul, t2, dict_manager1, event5, event6)
    # print(f1)

    # layer4
    conv4_0 = conv4_0(sigMulAdd)
    conv4_1_BN = conv4_1BN(conv4_0)
    conv4_1_relu = reluForServer2(conv4_1_BN, dict_manager, event1, event2, event3, event4, a2, b2, c2)

    # layer5
    conv5_0 = conv5_0(conv4_1_relu)
    conv5_1_BN = conv5_1BN(conv5_0)
    conv5_1_relu = reluForServer2(conv5_1_BN, dict_manager, event1, event2, event3, event4, a2, b2, c2)

    # concat
    conv6_0_1 = torch.cat((conv5_1_relu, f1), 1)
    conv6_0_2 = torch.cat((conv6_0_1, fres), 1)

    conv6_0_2 = formatSigmoid2(conv6_0_2.data, dict_manager, event1, event2)
    conv6_0_2 = formatSigmoid2(conv6_0_2.data, dict_manager, event1, event2)

    # sigmoid*x+x
    sigMulAdd = sigmoidMulAddForServer2(conv6_0_2.data, dict_manager, event1, event2, event3, event4, a2, b2, c2,
                                        c2_res,
                                        c2_mul, t2, dict_manager1, event5, event6)
    # se
    se = seModule2(sigMulAdd, liner6_1, liner6_2, dict_manager, event1, event2, event3, event4, a2, b2, c2, c2_res,
                   c2_mul, t2, dict_manager1, event5, event6)
    #
    # efficientNet
    efficientNet = nn.Sequential(*list(new_model[7].children()))[:]
    block0 = ConvBNSiLuActivationServer2(efficientNet[0], se, dict_manager, event1, event2, event3, event4, a2, b2, c2,
                                         c2_mul, c2_res, t2, False, 32, dict_manager1, event5, event6)
    print("blockB--0完成")

    block1 = InvertedResidualServer2(efficientNet[1], block0, dict_manager, event1, event2, event3, event4, a2, b2, c2,
                                     c2_mul, c2_res, t2, 32, 8, 32, 16, dict_manager1, event5, event6, event7, event8)
    print("blockB--1完成")
    block2 = InvertedResidualWithExpandServer2(efficientNet[2], block1, dict_manager, event1, event2, event3, event4,
                                               a2, b2, c2, c2_mul, c2_res, t2, 96, 4, 96, 24, dict_manager1, event5,
                                               event6, event7, event8)
    print("blockB--2完成")
    block3 = block2 + InvertedResidualWithExpandServer2(efficientNet[3], block2, dict_manager, event1, event2, event3,
                                                        event4, a2, b2, c2, c2_mul, c2_res, t2, 144, 6, 144, 24,
                                                        dict_manager1, event5, event6, event7, event8)
    print("blockB--3完成")
    block4 = InvertedResidualWithExpandServer2(efficientNet[4], block3, dict_manager, event1, event2, event3, event4,
                                               a2, b2, c2, c2_mul, c2_res, t2, 144, 6, 144, 40, dict_manager1, event5,
                                               event6, event7, event8)
    print("blockB--4完成")
    block5 = block4 + InvertedResidualWithExpandServer2(efficientNet[5], block4, dict_manager, event1, event2, event3,
                                                        event4, a2, b2, c2, c2_mul, c2_res, t2, 240, 10, 240, 40,
                                                        dict_manager1, event5, event6, event7, event8)
    print("blockB--5完成")
    block6 = InvertedResidualWithExpandServer2(efficientNet[6], block5, dict_manager, event1, event2, event3, event4,
                                               a2, b2, c2, c2_mul, c2_res, t2, 240, 10, 240, 80, dict_manager1, event5,
                                               event6, event7, event8)
    print("blockB--6完成")
    block7 = block6 + InvertedResidualWithExpandServer2(efficientNet[7], block6, dict_manager, event1, event2, event3,
                                                        event4, a2, b2, c2, c2_mul, c2_res, t2, 480, 20, 480, 80,
                                                        dict_manager1, event5, event6, event7, event8)
    print("blockB--7完成")
    block8 = block7 + InvertedResidualWithExpandServer2(efficientNet[8], block7, dict_manager, event1, event2, event3,
                                                        event4, a2, b2, c2, c2_mul, c2_res, t2, 480, 20, 480, 80,
                                                        dict_manager1, event5, event6, event7, event8)
    print("blockB--8完成")
    block9 = InvertedResidualWithExpandServer2(efficientNet[9], block8, dict_manager, event1, event2, event3,
                                               event4, a2, b2, c2, c2_mul, c2_res, t2, 480, 20, 480, 112, dict_manager1,
                                               event5, event6, event7, event8)
    print("blockB--9完成")
    block10 = block9 + InvertedResidualWithExpandServer2(efficientNet[10], block9, dict_manager, event1, event2, event3,
                                                         event4,
                                                         a2, b2, c2, c2_mul, c2_res, t2, 672, 28, 672, 112,
                                                         dict_manager1, event5, event6, event7, event8)
    print("blockB--10完成")
    block11 = block10 + InvertedResidualWithExpandServer2(efficientNet[11], block10, dict_manager, event1, event2,
                                                          event3, event4,
                                                          a2, b2, c2, c2_mul, c2_res, t2, 672, 28, 672, 112,
                                                          dict_manager1, event5, event6, event7, event8)
    print("blockB--11完成")
    block12 = InvertedResidualWithExpandServer2(efficientNet[12], block11, dict_manager, event1, event2, event3, event4,
                                                a2, b2, c2, c2_mul, c2_res, t2, 672, 28, 672, 192, dict_manager1,
                                                event5, event6, event7, event8)
    print("blockB--12完成")
    block13 = block12 + InvertedResidualWithExpandServer2(efficientNet[13], block12, dict_manager, event1, event2,
                                                          event3, event4,
                                                          a2, b2, c2, c2_mul, c2_res, t2, 1152, 48, 1152, 192,
                                                          dict_manager1, event5, event6, event7, event8)
    print("blockB--13完成")
    block14 = block13 + InvertedResidualWithExpandServer2(efficientNet[14], block13, dict_manager, event1, event2,
                                                          event3, event4,
                                                          a2, b2, c2, c2_mul, c2_res, t2, 1152, 48, 1152, 192,
                                                          dict_manager1, event5, event6, event7, event8)
    print("blockB--14完成")
    block15 = block14 + InvertedResidualWithExpandServer2(efficientNet[15], block14, dict_manager, event1, event2,
                                                          event3, event4,
                                                          a2, b2, c2, c2_mul, c2_res, t2, 1152, 48, 1152, 192,
                                                          dict_manager1, event5, event6, event7, event8)
    print("blockB--15完成")
    block16 = InvertedResidualWithExpandServer2(efficientNet[16], block15, dict_manager, event1, event2, event3, event4,
                                                a2, b2, c2, c2_mul, c2_res, t2, 1152, 48, 1152, 320, dict_manager1,
                                                event5, event6, event7, event8)
    print("blockB--16完成")
    block17 = ConvBNReluActivationServer2(efficientNet[17], block16, dict_manager, event1, event2, event3, event4,
                                          a2, b2, c2, c2_mul, c2_res, t2, True, 1280, dict_manager1, event5, event6,
                                          event7, event8)
    print("blockB--17完成")
    block18 = AvgLinearSiLuServer2(new_model[9], block17, dict_manager, event1, event2, event3, event4,
                                   a2, b2, c2, c2_mul, c2_res, t2, 1280, 2, dict_manager1, event5, event6)
    # print(block0)
    p2.send(block18.detach().numpy())
    print("----------222----------------\n")


# q.put(conv1_1_1BN.detach().numpy())


def Model(image_1, image_2, conv1_0, conv1_1BN, conv2_0, conv2_1BN, conv3_0, conv3_1BN, conv4_0, conv4_1BN, conv5_0,
          conv5_1BN, liner6_1, liner6_2, new_model):
    event1 = torch.multiprocessing.Event()
    event2 = torch.multiprocessing.Event()
    event3 = torch.multiprocessing.Event()
    event4 = torch.multiprocessing.Event()

    event5 = torch.multiprocessing.Event()
    event6 = torch.multiprocessing.Event()
    event7 = torch.multiprocessing.Event()
    event8 = torch.multiprocessing.Event()

    # global dict_manager
    dict_manager = Manager().dict()
    dict_manager1 = Manager().dict()
    p1, p2 = Pipe()
    # 乘法的
    a = random_c()
    b = random_c()
    a1 = random_c()
    a2 = a - a1
    b1 = random_c()
    b2 = b - b1
    c = a * b
    c1 = random_c()
    c2 = c - c1

    # 指数的
    c1_mul = random_c_mul()
    c2_mul = random_c_mul()
    c = c1_mul * c2_mul
    c1_res = random_c()
    c2_res = c - c1_res

    # 除法的
    t = random_c()
    t1 = random_c()
    t2 = t - t1

    server1_process = Process(target=server1_Model,
                              args=(
                                  event1, event2, event3, event4, image_1, a1, b1, c1, c1_mul, c1_res, t1, dict_manager,
                                  conv1_0, conv1_1BN, conv2_0, conv2_1BN, conv3_0, conv3_1BN, conv4_0, conv4_1BN,
                                  conv5_0, conv5_1BN, liner6_1, liner6_2, new_model, p1, p2, dict_manager1, event5,
                                  event6, event7, event8))

    server2_process = Process(target=server2_Model,
                              args=(
                                  event1, event2, event3, event4, image_2, a2, b2, c2, c2_mul,
                                  c2_res, t2, dict_manager, conv1_0, conv1_1BN, conv2_0, conv2_1BN, conv3_0, conv3_1BN,
                                  conv4_0, conv4_1BN, conv5_0, conv5_1BN, liner6_1, liner6_2,
                                  new_model, p1, p2, dict_manager1, event5, event6, event7, event8))

    server1_process.start()
    server2_process.start()

    server1_process.join()
    server2_process.join()

    f1 = p1.recv()
    f2 = p2.recv()
    return f1 + f2


if __name__ == '__main__':
    model = get_model()
    new_model = nn.Sequential(*list(model.children()))[:]
    model = model.state_dict()

    # layer1
    conv1_0 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
    conv1_1BN = nn.BatchNorm2d(3, track_running_stats=True)
    conv1_1BN.eval()

    # layer2
    conv2_0 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
    conv2_1BN = nn.BatchNorm2d(3, track_running_stats=True)
    conv2_1BN.eval()

    # layer3
    conv3_0 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
    conv3_1BN = nn.BatchNorm2d(3, track_running_stats=True)
    conv3_1BN.eval()

    # layer4
    conv4_0 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
    conv4_1BN = nn.BatchNorm2d(3, track_running_stats=True)
    conv4_1BN.eval()

    # layer5
    conv5_0 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
    conv5_1BN = nn.BatchNorm2d(3, track_running_stats=True)
    conv5_1BN.eval()

    # layer se
    liner6_1 = nn.Linear(in_features=15, out_features=7, bias=False)
    liner6_2 = nn.Linear(in_features=7, out_features=15, bias=False)

    # 1 param
    con_w1_0 = model["layer1.0.weight"]
    con_b1_0 = model["layer1.0.bias"] / 2.0
    con1_1_grammer = model["layer1.1.weight"]
    con1_1_beita = model["layer1.1.bias"] / 2.0
    con1_1_mean = model["layer1.1.running_mean"] / 2.0
    con1_1_var = model["layer1.1.running_var"]

    conv1_0.weight = nn.Parameter(con_w1_0)
    conv1_0.bias = nn.Parameter(con_b1_0)
    conv1_1BN.weight = nn.Parameter(con1_1_grammer)
    conv1_1BN.bias = nn.Parameter(con1_1_beita)
    conv1_1BN.running_mean = con1_1_mean
    conv1_1BN.running_var = con1_1_var

    # 2 param
    con_w2_0 = model["layer2.0.weight"]
    con_b2_0 = model["layer2.0.bias"] / 2.0
    con2_1_grammer = model["layer2.1.weight"]
    con2_1_beita = model["layer2.1.bias"] / 2.0
    con2_1_mean = model["layer2.1.running_mean"] / 2.0
    con2_1_var = model["layer2.1.running_var"]

    conv2_0.weight = nn.Parameter(con_w2_0)
    conv2_0.bias = nn.Parameter(con_b2_0)
    conv2_1BN.weight = nn.Parameter(con2_1_grammer)
    conv2_1BN.bias = nn.Parameter(con2_1_beita)
    conv2_1BN.running_mean = con2_1_mean
    conv2_1BN.running_var = con2_1_var

    # 3 param
    con_w3_0 = model["layer3.0.weight"]
    con_b3_0 = model["layer3.0.bias"] / 2.0
    con3_1_grammer = model["layer3.1.weight"]
    con3_1_beita = model["layer3.1.bias"] / 2.0
    con3_1_mean = model["layer3.1.running_mean"] / 2.0
    con3_1_var = model["layer3.1.running_var"]

    conv3_0.weight = nn.Parameter(con_w3_0)
    conv3_0.bias = nn.Parameter(con_b3_0)
    conv3_1BN.weight = nn.Parameter(con3_1_grammer)
    conv3_1BN.bias = nn.Parameter(con3_1_beita)
    conv3_1BN.running_mean = con3_1_mean
    conv3_1BN.running_var = con3_1_var

    # 4 param
    con_w4_0 = model["layer4.0.weight"]
    con_b4_0 = model["layer4.0.bias"] / 2.0
    con4_1_grammer = model["layer4.1.weight"]
    con4_1_beita = model["layer4.1.bias"] / 2.0
    con4_1_mean = model["layer4.1.running_mean"] / 2.0
    con4_1_var = model["layer4.1.running_var"]

    conv4_0.weight = nn.Parameter(con_w4_0)
    conv4_0.bias = nn.Parameter(con_b4_0)
    conv4_1BN.weight = nn.Parameter(con4_1_grammer)
    conv4_1BN.bias = nn.Parameter(con4_1_beita)
    conv4_1BN.running_mean = con4_1_mean
    conv4_1BN.running_var = con4_1_var

    # 5 param
    con_w5_0 = model["layer5.0.weight"]
    con_b5_0 = model["layer5.0.bias"] / 2.0
    con5_1_grammer = model["layer5.1.weight"]
    con5_1_beita = model["layer5.1.bias"] / 2.0
    con5_1_mean = model["layer5.1.running_mean"] / 2.0
    con5_1_var = model["layer5.1.running_var"]

    conv5_0.weight = nn.Parameter(con_w5_0)
    conv5_0.bias = nn.Parameter(con_b5_0)
    conv5_1BN.weight = nn.Parameter(con5_1_grammer)
    conv5_1BN.bias = nn.Parameter(con5_1_beita)
    conv5_1BN.running_mean = con5_1_mean
    conv5_1BN.running_var = con5_1_var

    # 6 param
    liner6_1_w = model["layer7.0.weight"]
    liner6_2_w = model["layer7.2.weight"]

    liner6_1.weight = nn.Parameter(liner6_1_w)
    liner6_2.weight = nn.Parameter(liner6_2_w)

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor()
         ])


    testData = dsets.ImageFolder('', data_transform)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=1, shuffle=False)
    correct = 0
    total = 0
    list = []
    for image, labels in testLoader:
        # image = image.cuda()
        image_1 = torch.tensor(np.random.uniform(0, 1, image.shape))
        image_2 = image - image_1

        result = Model(image_1, image_2, conv1_0, conv1_1BN, conv2_0, conv2_1BN, conv3_0, conv3_1BN, conv4_0, conv4_1BN,
                       conv5_0, conv5_1BN, liner6_1, liner6_2, new_model)
        print(result)
        result = torch.tensor(result)
        _, predicted = torch.max(torch.sigmoid(result.data), 1)
        total += labels.size(0)
        print(labels.size(0))
        correct += (predicted.cpu() == labels).sum()
        print(predicted, labels, correct, total)
        print("avg acc: %f" % (100 * correct / total))
        list.append(100 * correct / total)
