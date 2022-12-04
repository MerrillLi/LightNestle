#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 03/06/2022 15:41
# @Author : YuHui Li(MerylLynch)
# @File : NTF.py
# @Comment : Created By Liyuhui,15:41
# @Completed : No
# @Tested : No

import torch as t
from torch.nn import *

class NeuralTensorFactorization(Module):

    def __init__(self, args):
        super(NeuralTensorFactorization, self).__init__()

        self.windows = args.windows
        self.user_embeds = Embedding(args.num_users, args.rank)
        self.item_embeds = Embedding(args.num_items, args.rank)
        self.time_embeds = Embedding(args.num_times + 1, args.rank, padding_idx=0)

        self.lstm = LSTM(args.rank, args.rank)

        in_size = 3 * args.rank
        self.score = Sequential(
            Linear(in_size, in_size // 2),
            ReLU(),
            Linear(in_size // 2, in_size // 4),
            ReLU(),
            Linear(in_size // 4, 1),
            Sigmoid()
        )


    def forward(self, user, item, time):

        rnn_times = []
        for each_time in time:
            time_list = [max(0, int(each_time) + 1 - i) for i in range(self.windows)]
            time_list.reverse()
            rnn_times.append(time_list)

        rnn_times = t.tensor(rnn_times).permute(1, 0)

        user_vector = self.user_embeds(user)
        item_vector = self.item_embeds(item)
        time_vector = self.time_embeds(rnn_times)
        _, (time_vector, _) = self.lstm.forward(time_vector)
        rep = t.cat([user_vector, item_vector, time_vector.squeeze()], dim=-1)
        y = self.score(rep)
        return y.flatten()

