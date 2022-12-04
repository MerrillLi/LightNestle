#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 27/05/2022 12:40
# @Author : YuHui Li(MerylLynch)
# @File : ParamsNet.py
# @Comment : Created By Liyuhui,12:40
# @Completed : No
# @Tested : No


import torch as t
import time
from torch.nn import *
from torch.nn import functional as F

# From ResMLP
class ResMLPMixer(Module):

    def __init__(self, dim, depth):
        super(ResMLPMixer, self).__init__()

        self.hori_mix = Linear(dim, dim)
        self.vert_mix = Linear(depth, depth)
        self.linear = Linear(dim, dim)

    def forward(self, x:t.Tensor):
        res = x
        x = self.hori_mix(x) # type:t.Tensor
        x = F.relu(x)
        x += res
        x = self.vert_mix(x.permute(0, 2, 1))
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        x += res
        return x



class ConvMixer(Module):

    def __init__(self, rank ,depth):
        super(ConvMixer, self).__init__()
        self.conv1 = Sequential(LazyConv2d(rank // 4, kernel_size=(depth, 1)), ReLU())
        self.conv2 = Sequential(LazyConv2d(rank // 4, kernel_size=(1, rank)), ReLU())
        self.linear = LazyLinear(rank)

    def forward(self, x):
        x = t.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = t.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


# From Attention Free Transformer
class TimeMixer(Module):

    def __init__(self, dim):
        super(TimeMixer, self).__init__()
        self.fc_q = Linear(dim, dim)
        self.fc_k = Linear(dim, dim)
        self.fc_v = Linear(dim, dim)
        self.dim = dim
        self.sigmoid = Sigmoid()
        self.linear = Linear(dim, dim)

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, Linear):
    #             init.normal_(m.weight, std=0.001)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)

    def forward(self, prev, curr):
        # Current : Query
        # Previous: Key=Value

        prev = prev.unsqueeze(1)
        curr = curr.unsqueeze(1)

        bs, n, dim = curr.shape
        q = self.fc_q(curr)
        k = self.fc_k(prev).view(1, bs, n, dim)
        v = self.fc_v(prev).view(1, bs, n, dim)

        numerator = t.sum(t.exp(k) * v, dim=2)
        denominator = t.sum(t.exp(k), dim=2)

        out = (numerator / denominator)
        out = self.sigmoid(q) * (out.permute(1, 0, 2))
        out = out.squeeze()
        out = self.linear(out)
        return out

