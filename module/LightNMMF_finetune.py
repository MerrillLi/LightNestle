import torch as t
from torch.nn import *
from module.ParamsNet import ResMLPMixer, TimeMixer
from torch.nn import functional as F

class TensorEmbeddings(Module):

    def __init__(self, num_user, num_item, num_time, dim):
        super(TensorEmbeddings, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.num_time = num_time
        self.dim = dim
        self.user_embed = Embedding(num_user, dim)
        self.item_embed = Embedding(num_item, dim)
        self.time_embed = Embedding(num_time, dim)

    def next(self):
        pass
        # init.normal_(self.user_embed.weight)
        # init.normal_(self.item_embed.weight)
        # init.normal_(self.time_embed.weight)


class LightTC(Module):

    def __init__(self, args):
        super(LightTC, self).__init__()
        # Initialize Model Configuration
        self.num_user = args.users
        self.num_item = args.items
        self.num_time = args.times
        self.dim = args.dim

        # Initialize
        self.tensor_base = TensorEmbeddings(self.num_user, self.num_item, self.num_time, self.dim)

        self.user_linear = Linear(args.dim, args.dim)
        self.item_linear = Linear(args.dim, args.dim)
        self.time_linear = Linear(args.dim, args.dim)


    def forward(self, user, item, time):

        user_embeds = self.tensor_base.user_embed(user)
        item_embeds = self.tensor_base.item_embed(item)
        time_embeds = self.tensor_base.time_embed(time)

        time_embeds = self.time_linear(time_embeds)
        user_embeds = self.user_linear(user_embeds)
        item_embeds = self.item_linear(item_embeds)

        pred = t.sum(user_embeds * item_embeds * time_embeds, dim=-1).sigmoid()
        return pred
