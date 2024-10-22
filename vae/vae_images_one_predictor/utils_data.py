import numpy as np
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class SplitDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SplitDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.vae_indices = self.indices[:len(self.indices)//2]
        self.predictor_indices = self.indices[len(self.indices)//2:]

    def __iter__(self):
        vae_loader = DataLoader(
            SplitDataset([self.dataset[i] for i in self.vae_indices]),
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )
        predictor_loader = DataLoader(
            SplitDataset([self.dataset[i] for i in self.predictor_indices]),
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )
        return zip(vae_loader, predictor_loader)

    def __len__(self):
        return min(len(self.vae_indices), len(self.predictor_indices)) // self.batch_size