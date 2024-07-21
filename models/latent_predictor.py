import torch
import torch.nn as nn

class LatentPredictor(nn.Module):
    def __init__(self):
        super(LatentPredictor, self).__init__()
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