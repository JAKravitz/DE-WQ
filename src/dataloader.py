"""
Custom dataloader for water quality

Laurel Hopkins Manella
June 26, 2023
"""


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WQDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y #['chl']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X.iloc[idx, :].to_numpy()
        if len(self.y.shape) == 1:  # single target variable
            y = np.array(self.y.iloc[idx])
        else:
            y = self.y.iloc[idx, :].to_numpy()
        # convert arrays to torch tensors
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        return X, y


def wq_dataloader(X, y, batch_size=2, shuffle=True, num_workers=2):
    dataset = WQDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader
