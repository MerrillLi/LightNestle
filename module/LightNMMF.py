import torch as t
from torch.nn import *
from module.ParamsNet import ResMLPMixer, TimeMixer, ConvMixer
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
        self.score = Linear(dim, 1)

        self.user_reinit = self.user_embed.weight.detach()
        self.item_reinit = self.item_embed.weight.detach()
        self.time_reinit = self.time_embed.weight.detach()


        self.prev_user_embed = t.randn_like(self.user_embed.weight)
        self.prev_item_embed = t.randn_like(self.item_embed.weight)
        self.prev_time_embed = t.randn_like(self.time_embed.weight)




class ParamsTransfer(Module):

    def __init__(self, dim):
        super(ParamsTransfer, self).__init__()
        self.mixer = ResMLPMixer(dim, 4)

    def forward(self, prev_weight, curr_weight):
        dot = curr_weight * prev_weight
        sub = curr_weight - prev_weight
        input = t.stack([prev_weight, curr_weight, dot, sub], dim=1)
        out = self.mixer(input)
        out = out.mean(dim=1)
        return out



class EmbeddingAttentionTransfer(Module):

    def __init__(self, dim):
        super(EmbeddingAttentionTransfer, self).__init__()

        self.user_transfer = ParamsTransfer(dim)
        self.item_transfer = ParamsTransfer(dim)
        self.time_transfer = TimeMixer(dim)

    def forward(self, prev_weight, curr_weight, select):
        if select == 'user':
            return self.user_transfer.forward(prev_weight, curr_weight)
        elif select == 'item':
            return self.item_transfer.forward(prev_weight, curr_weight)
        elif select == 'time':
            return self.time_transfer.forward(prev_weight, curr_weight)


class EmbeddingMLPTransfer(Module):

    def __init__(self, dim):
        super(EmbeddingMLPTransfer, self).__init__()

        self.user_transfer = ParamsTransfer(dim)
        self.item_transfer = ParamsTransfer(dim)
        self.time_transfer = ParamsTransfer(dim)

    def forward(self, prev_weight, curr_weight, select):
        if select == 'user':
            return self.user_transfer.forward(prev_weight, curr_weight)
        elif select == 'item':
            return self.item_transfer.forward(prev_weight, curr_weight)
        elif select == 'time':
            return self.time_transfer.forward(prev_weight, curr_weight)


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
        if args.attention:
            self.transfer = EmbeddingAttentionTransfer(self.dim)
        else:
            self.transfer = EmbeddingMLPTransfer(self.dim)

    def update(self):
        curr_user = self.tensor_base.user_embed.weight
        prev_user = self.tensor_base.prev_user_embed
        curr_item = self.tensor_base.item_embed.weight
        prev_item = self.tensor_base.prev_item_embed
        curr_time = self.tensor_base.time_embed.weight
        prev_time = self.tensor_base.prev_time_embed

        user_embed = self.transfer.forward(prev_user, curr_user, select='user')
        item_embed = self.transfer.forward(prev_item, curr_item, select='item')
        time_embed = self.transfer.forward(prev_time, curr_time, select='time')

        self.tensor_base.user_embed.from_pretrained(user_embed)
        self.tensor_base.item_embed.from_pretrained(item_embed)
        self.tensor_base.time_embed.from_pretrained(time_embed)

        self.tensor_base.prev_user_embed = user_embed.detach()
        self.tensor_base.prev_item_embed = item_embed.detach()
        self.tensor_base.prev_time_embed = time_embed.detach()


    def forward(self, user, item, time):

        curr_user = self.tensor_base.user_embed(user)
        prev_user = self.tensor_base.prev_user_embed[user]
        curr_item = self.tensor_base.item_embed(item)
        prev_item = self.tensor_base.prev_item_embed[item]
        curr_time = self.tensor_base.time_embed(time)
        prev_time = self.tensor_base.prev_time_embed[time]

        user_embed = self.transfer.forward(prev_user, curr_user, select='user')
        item_embed = self.transfer.forward(prev_item, curr_item, select='item')
        time_embed = self.transfer.forward(prev_time, curr_time, select='time')

        dot = user_embed * item_embed * time_embed
        # pred = t.sum(dot, dim=-1).sigmoid()
        pred = self.tensor_base.score(dot).sigmoid().squeeze()
        return pred
