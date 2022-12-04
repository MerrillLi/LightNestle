import numpy as np
import torch
import torch as t
from torch.utils.data import DataLoader, Dataset
from torch.nn import *

class CoSTCo(Module):

    def __init__(self, num_times, num_users, num_items, num_shapes, rank):
        super(CoSTCo, self).__init__()
        self.num_shapes = num_shapes
        self.num_channels = rank
        self.time_embeds = Embedding(num_times, rank)
        self.user_embeds = Embedding(num_users, rank)
        self.item_embeds = Embedding(num_items, rank)
        self.conv1 = Sequential(LazyConv2d(self.num_channels, kernel_size=(self.num_shapes, 1)), ReLU())
        self.conv2 = Sequential(LazyConv2d(self.num_channels, kernel_size=(1, rank)), ReLU())
        self.flatten = Flatten()
        self.linear = Sequential(LazyLinear(rank), ReLU())
        self.output = Sequential(LazyLinear(1), Sigmoid())

    def forward(self, rIdx, cIdx, tIdx):

        # read embeds [batch, dim]
        time_embeds = self.time_embeds(tIdx)
        user_embeds = self.user_embeds(rIdx)
        item_embeds = self.item_embeds(cIdx)

        # stack as [batch, N, dim]
        x = t.stack([time_embeds, user_embeds, item_embeds], dim=1)

        # reshape to [batch, 1, N, dim]
        x = t.unsqueeze(x, dim=1)

        # conv
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)
        return x.flatten()



