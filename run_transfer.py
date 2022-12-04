import argparse
import numpy as np
import torch
import torch as t
from tqdm import trange
from torch.nn import functional as F
from module.LightNMMF import LightTC
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


def BalanceMSE(pred, target, noise_var=0.01):
    pred = pred.reshape(-1, 1)
    target = target.reshape(-1, 1)
    logits = -(pred - target.T).pow(2) / (2 * noise_var)
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))
    return loss


def run(runid, args):
    tensor = get_tensor(args)
    seqset = SequentialDataset(tensor, windows=args.window, density=args.density)
    model = LightTC(args)

    criterion = t.nn.MSELoss()
    # criterion = BalanceMSE

    globalPreds, globalLabels = [], []

    stagewise_nmaes, stagewise_ers = [], []

    for i in range(args.num_pass):
        seqset.reset()
        stage = 0
        meanTime = []
        while seqset.move_next():
            startTime = time.time()
            trainLoader, testLoader = seqset.get_loaders()
            optim_tc = t.optim.RMSprop(model.tensor_base.parameters(), lr=0.01, weight_decay=1e-5)
            optim_tf = t.optim.RMSprop(model.transfer.parameters(), lr=0.001, weight_decay=1e-5)

            # Tensor Completion
            model.transfer.eval()
            model.tensor_base.train()

            iter_round = args.tc_iter if stage > 5 else args.tc_iter + 10
            last_loss = None
            last_params = None
            for epoch in range(iter_round):
                losses = []
                for trainBatch in trainLoader:
                    tIdx, rIdx, cIdx, mVal = trainBatch
                    pred = model.forward(rIdx, cIdx, tIdx)
                    pred = pred.reshape(mVal.shape)
                    loss = criterion(pred, mVal)
                    optim_tc.zero_grad()
                    loss.backward()
                    optim_tc.step()
                    losses += [loss.item()]
                avg_loss = np.mean(losses)
                # print(f'Pass={i+1}, Stage={stage}, Completion Loss = {avg_loss:.7f}')
                if last_loss is None:
                    last_loss = avg_loss
                    last_params = model.state_dict()
                else:
                    if last_loss < avg_loss:
                        model.load_state_dict(last_params)
                        break
                    last_loss = avg_loss
                    last_params = model.state_dict()

            # Transfer Learning
            model.transfer.train()
            model.tensor_base.eval()
            last_loss = None
            last_params = None

            iter_round = args.tf_iter if stage < 60 else args.tf_iter
            for epoch in range(iter_round):
                losses = []
                for trainBatch in trainLoader:
                    tIdx, rIdx, cIdx, mVal = trainBatch
                    pred = model.forward(rIdx, cIdx, tIdx)
                    pred = pred.reshape(mVal.shape)
                    loss = criterion(pred, mVal)
                    optim_tf.zero_grad()
                    loss.backward()
                    optim_tf.step()
                    losses += [loss.item()]
                avg_loss = np.mean(losses)
                # print(f'Pass={i+1}, Stage={stage}, Transfer Loss = {np.mean(losses):.7f}')
                if last_loss is None:
                    last_loss = avg_loss
                    last_params = model.state_dict()
                else:
                    if last_loss < avg_loss:
                        model.load_state_dict(last_params)
                        break
                    last_loss = avg_loss
                    last_params = model.state_dict()

            preds = []
            reals = []
            with torch.no_grad():
                model.tensor_base.eval()
                model.transfer.eval()
                for testBatch in testLoader:
                    tIdx, rIdx, cIdx, mVal = testBatch
                    pred = model.forward(rIdx, cIdx, tIdx)
                    pred = pred.reshape(mVal.shape)
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
            model.update()
            stage += 1

            endTime = time.time()
            meanTime.append(endTime - startTime)
            if len(meanTime) > 20:
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

    parser.add_argument('--dataset', type=str, default='geant')
    parser.add_argument('--dim', type=int, default=20)
    parser.add_argument('--users', type=int, default=23)
    parser.add_argument('--items', type=int, default=23)
    parser.add_argument('--times', type=int, default=200)
    parser.add_argument('--window', type=int, default=200)
    parser.add_argument('--density', type=float, default=0.06)
    parser.add_argument('--tc_iter', type=int, default=10)
    parser.add_argument('--tf_iter', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--num_pass', type=int, default=1)
    parser.add_argument('--attention', type=bool, default=True)


    args = parser.parse_args()
    ts = time.asctime()
    logging.basicConfig(level=logging.INFO, filename=f'./results/LightNestle/{args.dataset}_{args.density}_{ts}.log', filemode='w')
    logger = logging.getLogger('LightNestle')
    logger.info(f'----------------------------')
    logger.info(f'Params Info={args}')
    main(args)
