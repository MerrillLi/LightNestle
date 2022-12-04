import argparse
import numpy as np
import torch
import torch as t
from tqdm import trange
from module.LightNMMF_finetune import LightTC
from module.SeqDataset import SequentialDataset
import time

import logging

global logger

def ErrMetrics(true, pred):
    nonzeroIdx = true.nonzero()
    true = true[nonzeroIdx]
    pred = pred[nonzeroIdx]
    ER = np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum(true ** 2))
    NMAE = np.sum(np.abs(true - pred)) / np.sum(true)
    return ER, NMAE


def get_tensor(args):
    tensor = None
    if args.dataset == 'abilene':
        tensor = np.load('datasets/abilene.npy').astype('float32')

    if args.dataset == 'geant':
        tensor = np.load('datasets/geant.npy').astype('float32')
    return tensor


def run(runid, args):
    tensor = get_tensor(args)
    seqset = SequentialDataset(tensor, windows=args.window, density=args.density)
    model = LightTC(args)

    criterion = t.nn.MSELoss()

    globalPreds, globalLabels = [], []

    stagewise_nmaes, stagewise_ers = [], []


    for i in range(args.num_pass):
        seqset.reset()
        stage = 0
        meanTime = []
        while seqset.move_next():
            startTime = time.time()
            trainLoader, testLoader = seqset.get_loaders()
            optim_tc = t.optim.Adam(model.tensor_base.parameters(), lr=0.01, weight_decay=1e-5)

            # Tensor Completion
            model.train()
            iter_round = args.tc_iter if stage > 5 else args.tc_iter + 20
            for epoch in range(iter_round):
                losses = []
                for trainBatch in trainLoader:
                    tIdx, rIdx, cIdx, mVal = trainBatch
                    pred = model.forward(rIdx, cIdx, tIdx)
                    loss = criterion(pred, mVal)
                    optim_tc.zero_grad()
                    loss.backward()
                    optim_tc.step()
                    losses += [loss.item()]
                # print(f'Pass={i+1}, Stage={stage}, Completion Loss = {np.mean(losses):.7f}')

            preds = []
            reals = []
            with torch.no_grad():
                model.eval()
                for testBatch in testLoader:
                    tIdx, rIdx, cIdx, mVal = testBatch
                    pred = model.forward(rIdx, cIdx, tIdx)
                    reals += mVal.numpy().tolist()
                    preds += pred.numpy().tolist()

                    globalPreds += mVal.numpy().tolist()
                    globalLabels += pred.numpy().tolist()

            reals = np.array(reals)
            preds = np.array(preds)
            stageER, stageNMAE = ErrMetrics(reals, preds)
            # print(f'Pass={i+1}, Stage={stage}, ER={stageER:.4f}, NMAE={stageNMAE:.4f}')
            # logger.info(f'Run={runid}, Pass={i+1}, Stage={stage}, ER={stageER:.4f}, NMAE={stageNMAE:.4f}')
            stagewise_ers += [stageER]
            stagewise_nmaes += [stageNMAE]

            # maintain some values
            model.tensor_base.next()
            stage += 1

            endTime = time.time()
            # print(f'Time={(endTime - startTime)}s')
            meanTime.append(endTime - startTime)
            if len(meanTime) > 30:
                print(np.mean(meanTime))
        # np.savetxt(f"./results/LightNestle/ER_{args.dataset}_{args.density}.txt", stagewise_ers)
        # np.savetxt(f"./results/LightNestle/NMAE_{args.dataset}_{args.density}.txt", stagewise_nmaes)

    globalLabels = np.array(globalLabels)
    globalPreds = np.array(globalPreds)
    GlobalER, GlobalNMAE = ErrMetrics(globalLabels, globalPreds)
    return GlobalER, GlobalNMAE



def main(args):
    RunERs, RunNMAEs = [], []
    for runid in range(args.rounds):
        ER, NMAE = run(runid, args)
        RunERs += [ER]
        RunNMAEs += [NMAE]
        logger.info(f'Round GER={np.mean(ER):.3f}, RoundGNAME={np.mean(NMAE):.3f}')
    logger.info(f'Run ER={np.mean(RunERs):.3f}, Run NAME={np.mean(RunNMAEs):.3f}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='abilene')
    parser.add_argument('--dim', type=int, default=30)
    parser.add_argument('--users', type=int, default=12)
    parser.add_argument('--items', type=int, default=12)
    parser.add_argument('--times', type=int, default=400)
    parser.add_argument('--window', type=int, default=400)
    parser.add_argument('--density', type=float, default=0.10)
    parser.add_argument('--tc_iter', type=int, default=20)
    parser.add_argument('--tf_iter', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--num_pass', type=int, default=1)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, filename=f'./results/Finetune/{args.dataset}_{args.density}.log', filemode='w')
    logger = logging.getLogger('Finetune')
    logger.info(f'----------------------------')
    logger.info(f'Params Info={args}')
    main(args)
