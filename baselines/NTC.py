#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 03/06/2022 15:41
# @Author : YuHui Li(MerylLynch)
# @File : NTC.py
# @Comment : Created By Liyuhui,15:41
# @Completed : No
# @Tested : No
import math

import torch as t

from torch.nn import *


class NeuralTensorCompletion(Module):

    def __init__(self, args):
        super(NeuralTensorCompletion, self).__init__()

        self.user_embeds = Embedding(args.num_users, args.rank)
        self.item_embeds = Embedding(args.num_items, args.rank)
        self.time_embeds = Embedding(args.num_times, args.rank)

        self.cnn = ModuleList()

        for i in range(int(math.log2(args.rank))):
            in_channels = 1 if i == 0 else args.channels
            conv_layer = Conv3d(in_channels=in_channels, out_channels=args.channels,
                                kernel_size=2, stride=2)
            self.cnn.append(conv_layer)
            self.cnn.append(LeakyReLU())

        self.score = Sequential(
            LazyLinear(1),
            Sigmoid()
        )


    def forward(self, user, item, time):

        user_vector = self.user_embeds(user)
        item_vector = self.item_embeds(item)
        time_vector = self.time_embeds(time)
        outer_prod = t.einsum('ni, nj, nk-> nijk', user_vector, item_vector, time_vector)
        outer_prod = t.unsqueeze(outer_prod, dim=1)
        rep = outer_prod
        for layer in self.cnn:
            rep = layer(rep)
        y = self.score(rep.squeeze())
        return y.flatten()






