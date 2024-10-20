from torch import nn


class LatentPredictor_x0(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1)

        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred