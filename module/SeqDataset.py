
import numpy as np
from torch.utils.data import Dataset, DataLoader



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


class SequentialDataset:

    def __init__(self, dense, windows, density):

        self.dense = dense
        quantile = np.percentile(self.dense, q=99)
        self.dense[self.dense > quantile] = quantile
        self.dense /= quantile
        self.start = -windows
        self.windows = windows
        self.density = density
        self.mask = np.random.rand(*dense.shape).astype('float32')
        self.mask[self.mask > self.density] = 1
        self.mask[self.mask < self.density] = 0

    def move_next(self):
        self.start += self.windows
        return self.start < self.dense.shape[0]

    def reset(self):
        self.start = -self.windows

    def get_loaders(self):
        curr_tensor = self.dense[self.start:self.start + self.windows]
        curr_mask = self.mask[self.start:self.start + self.windows]
        trainTensor = curr_tensor * (1 - curr_mask)
        testTensor = curr_tensor * curr_mask
        trainset = TensorDataset(trainTensor)
        testset = TensorDataset(testTensor)
        trainLoader = DataLoader(trainset, batch_size=128, shuffle=True)
        testLoader = DataLoader(testset, batch_size=1024)
        return trainLoader, testLoader


class SequentialFutureDataset:

    def __init__(self, dense, windows, density):

        self.dense = dense
        quantile = np.percentile(self.dense, q=99)
        self.dense[self.dense > quantile] = quantile
        self.dense /= quantile
        self.start = -windows
        self.windows = windows
        self.density = density
        self.mask = np.random.rand(*dense.shape).astype('float32')
        self.mask[self.mask > self.density] = 1
        self.mask[self.mask < self.density] = 0

    def move_next(self):
        self.start += self.windows
        return self.start < self.dense.shape[0]

    def reset(self):
        self.start = -self.windows

    def get_loaders(self):
        curr_tensor = self.dense[self.start:self.start + self.windows]
        curr_mask = self.mask[self.start:self.start + self.windows]
        currTrain = curr_tensor * (1 - curr_mask)
        currTest = curr_tensor * curr_mask
        trainset = TensorDataset(currTrain)
        testset = TensorDataset(currTest)
        curr_trainLoader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)
        curr_testLoader = DataLoader(testset, batch_size=1024)

        if self.start + 2 * self.windows < self.dense.shape[0]:
            future_tensor = self.dense[self.start + self.windows: self.start + 2 * self.windows]
            future_mask = self.mask[self.start + self.windows: self.start + 2 * self.windows]
            futureTrain = future_tensor * (1 - future_mask)
            futureset = TensorDataset(futureTrain)
            future_trainLoader = DataLoader(futureset, batch_size=128, shuffle=True, drop_last=True)
            return curr_trainLoader, curr_testLoader, future_trainLoader

        return curr_trainLoader, curr_testLoader, None




