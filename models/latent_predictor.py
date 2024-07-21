import torch
import torch.nn as nn

class LatentPredictor_x2(nn.Module):
    def __init__(self):
        super(LatentPredictor_x2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred


class LatentPredictor_x3(nn.Module):
    def __init__(self):
        super(LatentPredictor_x3, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred


class LatentPredictor_x4(nn.Module):
    def __init__(self):
        super(LatentPredictor_x4, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred