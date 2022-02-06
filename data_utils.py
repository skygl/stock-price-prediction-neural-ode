import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, data, m, n):
        self.data = data
        self.m = m
        self.n = n

    def __getitem__(self, idx):
        data = self.data.iloc[idx: idx + self.m + self.n]

        first = torch.FloatTensor(data.iloc[0].to_numpy()[1:-1].astype(np.float64))

        x = torch.FloatTensor(data.iloc[:self.m].to_numpy()[:, 1:-1].astype(np.float64))
        y = torch.FloatTensor(data.iloc[1:].to_numpy()[:, 1:-1].astype(np.float64))

        # normalize data
        x = x / first
        y = y / first

        return x, y, first

    def __len__(self):
        return len(self.data) - self.m - self.n
