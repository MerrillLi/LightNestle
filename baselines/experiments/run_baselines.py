#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 09/06/2022 16:16
# @Author : YuHui Li(MerylLynch)
# @File : run_baselines.py
# @Comment : Created By Liyuhui,16:16
# @Completed : No
# @Tested : No

import numpy as np
import logging
import argparse
import time
import torch
import torch as t

from baselines.CoSTCo import CoSTCo
from baselines.NTC import NeuralTensorCompletion
from baselines.NTF import NeuralTensorFactorization
from baselines.NTM import NeuralTensorModel
from baselines.utils import get_loaders, ErrMetrics
from tqdm import *
global logger



def get_model(args):

    model = None
    if args.model == 'CoSTCo':
        model = CoSTCo(args.num_times, args.num_users, args.num_items, 3, args.rank)
    if args.model == 'NTC':
        model = NeuralTensorCompletion(args)
    if args.model == 'NTF':
        model = NeuralTensorFactorization(args)
    if args.model == 'NTM':
        model = NeuralTensorModel(args)
    return model

def run(runid, args):

    model = get_model(args)
    optimizer = t.optim.Adam(model.parameters())
    criterion = t.nn.MSELoss()

    trainLoader, testLoader = get_loaders(args)

    for epoch in trange(args.epochs):
        losses = []
        for trainBatch in trainLoader:
            tIdx, rIdx, cIdx, mVal = trainBatch
            pred = model.forward(rIdx, cIdx, tIdx)
            loss = criterion(pred, mVal)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += [loss.item()]

        print(f'{args.model}, Epoch={epoch}, Loss={np.mean(losses)}')

    torch.set_grad_enabled(False)
    reals, preds = [], []
    for testBatch in tqdm(testLoader):
        tIdx, rIdx, cIdx, mVal = testBatch
        pred = model.forward(rIdx, cIdx, tIdx)
        reals += mVal.numpy().tolist()
        preds += pred.numpy().tolist()
    reals = np.array(reals)
    preds = np.array(preds)
    ER, NMAE = ErrMetrics(reals, preds)
    torch.set_grad_enabled(True)
    logger.info(f'Round ID={runid}, ER={ER:.3f}, NMAE={NMAE:.3f}')
    return ER, NMAE

def main(args):
    RunERs, RunNMAEs = [], []
    for runid in range(args.rounds):
        ER, NMAE = run(runid, args)
        RunERs += [ER]
        RunNMAEs += [NMAE]

    logger.info(f'Run ER={np.mean(RunERs):.3f}, Run NAME={np.mean(RunNMAEs):.3f}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='NTF')
    parser.add_argument('--dataset', type=str, default='abilene')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--num_users', type=int, default=12)
    parser.add_argument('--num_items', type=int, default=12)
    parser.add_argument('--num_times', type=int, default=10000)
    parser.add_argument('--windows', type=int, default=5)
    parser.add_argument('--density', type=float, default=0.1)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=5)

    args = parser.parse_args()
    ts = time.asctime()
    logging.basicConfig(level=logging.INFO, filename=f'./results/{args.model}/{args.dataset}_{args.density}_{ts}.log', filemode='w')
    logger = logging.getLogger('LightNestle')
    logger.info(f'----------------------------')
    logger.info(f'Params Info={args}')
    main(args)
