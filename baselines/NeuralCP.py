import numpy as np
import torch
import torch as t
from torch.utils.data import DataLoader, Dataset
from torch.nn import *
from tqdm import *

class TensorDataset(Dataset):

    def __init__(self, sparseTensor, test=False, test_start=3000):
        self.sparseTensor = sparseTensor
        self.offset = test_start
        self.test = test
        self.tIdx, self.rIdx, self.cIdx = self.sparseTensor.nonzero()

    def __len__(self):
        return len(self.tIdx)

    def __getitem__(self, id):
        tIdx = self.tIdx[id]
        rIdx = self.rIdx[id]
        cIdx = self.cIdx[id]
        mVal = self.sparseTensor[tIdx, rIdx, cIdx]
        if self.test:
            tIdx += self.offset
        return tIdx, rIdx, cIdx, mVal


def Metrics(pred, true):
    nonzeroIdx = true.nonzero()
    true = true[nonzeroIdx]
    pred = pred[nonzeroIdx]
    ER = np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum(true ** 2))
    NMAE = np.sum(np.abs(true - pred)) / np.sum(true)
    return ER, NMAE


class NeuralCP(Module):

    def __init__(self, num_times, num_users, num_items, num_shapes, rank):
        super(NeuralCP, self).__init__()
        self.num_shapes = num_shapes
        self.num_channels = rank
        self.time_embeds = Embedding(num_times, rank)
        self.user_embeds = Embedding(num_users, rank)
        self.item_embeds = Embedding(num_items, rank)

        self.user_linear = Linear(rank, rank)
        self.item_linear = Linear(rank, rank)
        self.time_linear = Linear(rank, rank)



    def forward(self, tIdx, rIdx, cIdx):

        # read embeds [batch, dim]
        time_embeds = self.time_embeds(tIdx)
        user_embeds = self.user_embeds(rIdx)
        item_embeds = self.item_embeds(cIdx)

        time_embeds = self.time_linear(time_embeds)
        user_embeds = self.user_linear(user_embeds)
        item_embeds = self.item_linear(item_embeds)


        x = t.sum(time_embeds * user_embeds * item_embeds, dim=-1)
        return x.flatten()



def get_dataloader(density=0.10, batch_size=50):
    tensor = np.load('../datasets/geant.npy')[:8000]
    thsh = np.percentile(tensor, q=99)
    tensor[tensor > thsh] = thsh
    tensor /= thsh

    tIdx, srcIdx, dstIdx = tensor.nonzero()
    p = np.random.permutation(len(tIdx))
    tIdx, srcIdx, dstIdx = tIdx[p], srcIdx[p], dstIdx[p]
    sample = int(np.prod(tensor.shape) * density)
    stIdx = tIdx[:sample]
    ssrcIdx = srcIdx[:sample]
    sdstIdx = dstIdx[:sample]
    trainTensor = np.zeros_like(tensor)
    trainTensor[stIdx, ssrcIdx, sdstIdx] = tensor[stIdx, ssrcIdx, sdstIdx]


    testTensor = np.zeros_like(tensor)
    stIdx = tIdx[sample:]
    ssrcIdx = srcIdx[sample:]
    sdstIdx = dstIdx[sample:]
    testTensor[stIdx, ssrcIdx, sdstIdx] = tensor[stIdx, ssrcIdx, sdstIdx]


    trainset = TensorDataset(trainTensor)
    trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = TensorDataset(testTensor[3000:], test=True, test_start=3000)
    testLoader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainLoader, testLoader, thsh


trainLoader, testLoader, thsh = get_dataloader(density=0.3, batch_size=256)
model = NeuralCP(8000, 23, 23, num_shapes=3, rank=30)
optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
LossFunc = MSELoss()
for epoch in range(50):

    model.train()
    losses = []
    for trainBatch in tqdm(trainLoader):
        optimizer.zero_grad()
        tIdx, rIdx, cIdx, label = trainBatch
        pred = model.forward(tIdx, rIdx, cIdx)
        loss = LossFunc(pred, label.float())
        loss.backward()
        optimizer.step()
        losses += [loss.item()]

    preds = []
    reals = []
    model.eval()
    with torch.no_grad():
        for testBatch in tqdm(testLoader):
            tIdx, rIdx, cIdx, label = testBatch
            pred = model.forward(tIdx, rIdx, cIdx)
            preds += pred.numpy().tolist()
            reals += label.numpy().tolist()

    ER, NMAE = Metrics(np.array(preds) * thsh, np.array(reals) * thsh)

    print('Epoch %02d, Loss=%.4f, ER=%.4f, NMAE=%.4f' %
          (epoch, np.mean(losses), ER, NMAE))

    if np.mean(losses) < 1e-5:
        break




