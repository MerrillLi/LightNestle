#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 01/06/2022 21:00
# @Author : YuHui Li(MerylLynch)
# @File : FaLRTC.py
# @Comment : Created By Liyuhui,21:00
# @Completed : No
# @Tested : No

import pyten
import numpy as np
import argparse
import time
from common import getTensor, ErrMetrics, SparsifyTensor
import logging

global logger

def Execute(args):

    ERs = []
    NMAEs = []
    Elapseds = []
    for roundId in range(args.rounds):
        ER, NMAE, Elapsed = ExecuteOnce(args)
        ERs += [ER]
        NMAEs += [NMAE]
        Elapseds += [Elapsed]
        logger.info(f'Round ID={roundId:02d}, ER={ER:.3f}, NMAE={NMAE:.3f}, Time={Elapsed:.3f}s')

    meanER = np.mean(ERs)
    meanNMAE = np.mean(NMAEs)
    meanTime = np.mean(Elapseds)

    logger.info(f'Average ER={meanER:.3f}, NMAE={meanNMAE:.3f}, Time={meanTime:.3f}s')



def ExecuteOnce(args):

    dataTensor = getTensor(args)

    thsh = np.percentile(dataTensor, q=99)
    dataTensor[dataTensor > thsh] = thsh
    dataTensor /= thsh

    sparseTensor, testTensor, maskTensor = SparsifyTensor(dataTensor, args.density)
    subs = np.stack(sparseTensor.nonzero()).T
    vals = sparseTensor[sparseTensor.nonzero()].reshape(-1, 1)

    # ttTensor = pyten.tenclass.Sptensor(subs, vals, sparseTensor.shape)
    ttTensor = pyten.tenclass.Tensor(sparseTensor)
    start = time.perf_counter()
    reconstruct = pyten.method.falrtc(x=ttTensor, omega=maskTensor,
                                         epsilon=args.tol, max_iter=args.maxiter, printitn=1)
    end = time.perf_counter()

    testIdx = (1- maskTensor).nonzero()
    reconstruct = reconstruct.tondarray()
    ER, NMAE = ErrMetrics(testTensor[testIdx], reconstruct[testIdx])

    Elapsed = (end - start) * 1

    return ER, NMAE, Elapsed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='abilene')
    parser.add_argument('--density', type=float, default=0.02)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--tol', type=int, default=1e-4)
    parser.add_argument('--maxiter', type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=f'./results/falrtc/{args.dataset}_{args.density}.log', filemode='w')
    logger = logging.getLogger('FaLRTC')
    logger.info(f'----------------------------')
    logger.info(f'Params Info={args}')
    Execute(args)












