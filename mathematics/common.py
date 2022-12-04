#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 01/06/2022 21:58
# @Author : YuHui Li(MerylLynch)
# @File : common.py
# @Comment : Created By Liyuhui,21:58
# @Completed : No
# @Tested : No


import numpy as np

def ErrMetrics(true, pred):
    nonzeroIdx = true.nonzero()
    true = true[nonzeroIdx]
    pred = pred[nonzeroIdx]
    ER = np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum(true ** 2))
    NMAE = np.sum(np.abs(true - pred)) / np.sum(true)
    return ER, NMAE


def getTensor(args):
    dataTensor = None
    if args.dataset == 'abilene':
        dataTensor = np.load('../datasets/abilene.npy')
    elif args.dataset == 'geant':
        dataTensor = np.load('../datasets/geant.npy')
    else:
        raise NotImplementedError('Unknwon Dataset')
    return dataTensor


def SparsifyTensor(denseTensor, density):
    rowIdx, colIdx, timeIdx = denseTensor.nonzero()
    p = np.random.permutation(len(rowIdx))
    rowIdx, colIdx, timeIdx = rowIdx[p], colIdx[p], timeIdx[p]
    sampleSize = int(density * np.prod(denseTensor.shape))

    sparseTensor = np.zeros_like(denseTensor)
    testTensor = np.zeros_like(denseTensor)
    maskTensor = np.zeros_like(denseTensor)

    rowSampleIdx = rowIdx[:sampleSize]
    colSampleIdx = colIdx[:sampleSize]
    timeSampleIdx = timeIdx[:sampleSize]

    rowUnknownIdx = rowIdx[sampleSize:]
    colUnknownIdx = colIdx[sampleSize:]
    timeUnknownIdx = timeIdx[sampleSize:]

    sparseTensor[rowSampleIdx, colSampleIdx, timeSampleIdx] = denseTensor[rowSampleIdx, colSampleIdx, timeSampleIdx]
    testTensor[rowUnknownIdx, colUnknownIdx, timeUnknownIdx] = denseTensor[rowUnknownIdx, colUnknownIdx, timeUnknownIdx]
    maskTensor[rowSampleIdx, colSampleIdx, timeSampleIdx] = 1
    return sparseTensor, testTensor, maskTensor
