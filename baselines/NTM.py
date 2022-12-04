#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 09/06/2022 11:23
# @Author : YuHui Li(MerylLynch)
# @File : NTM.py
# @Comment : Created By Liyuhui,11:23
# @Completed : No
# @Tested : No


import torch as t
from torch.nn import *


class TMLPBlock(Module):

    def __init__(self, in_feats, out_feats):
        super(TMLPBlock, self).__init__()

        self.A = Linear(in_feats, out_feats, bias=False)
        self.B = Linear(in_feats, out_feats, bias=False)
        self.C = Linear(in_feats, out_feats, bias=False)
        self.bias = Parameter(t.zeros((out_feats, out_feats, out_feats)))

    def forward(self, input):

        # input = [n, r, r, r]
        x = input
        x = t.einsum('nijk, ri->nrjk', x, self.A.weight)
        x = t.einsum('nijk, rj->nirk', x, self.B.weight)
        x = t.einsum('nijk, rk->nijr', x, self.C.weight)
        x += self.bias
        return x



class NeuralTensorModel(Module):

    def __init__(self, args):
        super(NeuralTensorModel, self).__init__()

        self.user_embeds = Embedding(args.num_users, args.rank)
        self.item_embeds = Embedding(args.num_items, args.rank)
        self.time_embeds = Embedding(args.num_times, args.rank)

        self.outer = Sequential(
            TMLPBlock(args.rank, args.rank // 2),
            TMLPBlock(args.rank // 2, args.rank // 4)
        )

        self.output =  Sequential(
            LazyLinear(1),
            Sigmoid()
        )

    def forward(self, user, item, time):

        user_vector = self.user_embeds(user)
        item_vector = self.item_embeds(item)
        time_vector = self.time_embeds(time)
        gcp = user_vector * item_vector * time_vector
        out_prod = t.einsum('ni, nj, nk-> nijk', user_vector, item_vector, time_vector)
        out_prod = self.outer(out_prod)
        rep = t.cat([gcp, out_prod.flatten(start_dim=1, end_dim=-1)], dim=-1)
        y = self.output(rep)
        return y.flatten()



if __name__ == '__main__':

    x = t.rand((16, 64, 64, 64))

    model = TMLPBlock(64, 16)

    y = model(x)
    print(y.shape)

