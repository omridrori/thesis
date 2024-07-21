import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.joint_positions = self.data[[f'x{i+1}_{j+1}' for i in range(4) for j in range(2)]].values

    def __len__(self):
        return len(self.joint_positions)

    def __getitem__(self, idx):
        sample = torch.tensor(self.joint_positions[idx], dtype=torch.float32)
        return sample

def get_dataloader(csv_file, batch_size=32, shuffle=True):
    dataset = ToyDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)