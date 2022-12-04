#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 10/06/2022 14:34
# @Author : YuHui Li(MerylLynch)
# @File : dataset.py
# @Comment : Created By Liyuhui,14:34
# @Completed : No
# @Tested : No

import numpy as np
from torch.utils.data import DataLoader, Dataset


def ErrMetrics(pred, true):
    nonzeroIdx = true.nonzero()
    true = true[nonzeroIdx]
    pred = pred[nonzeroIdx]
    ER = np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum(true ** 2))
    NMAE = np.sum(np.abs(true - pred)) / np.sum(true)
    return ER, NMAE



class TensorDataset(Dataset):

    def __init__(self, sparseTensor):
        self.sparseTensor = sparseTensor
        self.tIdx, self.rIdx, self.cIdx = self.sparseTensor.nonzero()

    def __len__(self):
        return len(self.tIdx)

    def __getitem__(self, id):
        tIdx = self.tIdx[id]
        rIdx = self.rIdx[id]
        cIdx = self.cIdx[id]
        mVal = self.sparseTensor[tIdx, rIdx, cIdx]
        return tIdx, rIdx, cIdx, mVal


def get_tensor(args):
    tensor = None
    if args.dataset == 'abilene':
        tensor = np.load('../../datasets/abilene.npy').astype('float32')[:48000]

    if args.dataset == 'geant':
        tensor = np.load('../../datasets/geant.npy').astype('float32')[:10000]
    return tensor


def get_loaders(args):

    tensor = get_tensor(args)
    quantile = np.percentile(tensor, q=99)
    tensor[tensor > quantile] = quantile
    tensor /= quantile
    density = args.density
    mask = np.random.rand(*tensor.shape).astype('float32')
    mask[mask > density] = 1
    mask[mask < density] = 0

    trainTensor = tensor * (1 - mask)
    testTensor = tensor * mask

    trainset = TensorDataset(trainTensor)
    testset = TensorDataset(testTensor)
    trainLoader = DataLoader(trainset, batch_size=256, shuffle=True)
    testLoader = DataLoader(testset, batch_size=1024, shuffle=True)
    return trainLoader, testLoader

