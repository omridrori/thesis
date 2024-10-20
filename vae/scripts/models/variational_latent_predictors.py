import torch
import torch.nn as nn

class LatentPredictor_x2(nn.Module):
    def __init__(self):
        super(LatentPredictor_x2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 16),
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
            nn.Linear(16, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred


class LatentPredictor_x5(nn.Module):
    def __init__(self):
        super(LatentPredictor_x5, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred


class LatentPredictor_x6(nn.Module):
    def __init__(self):
        super(LatentPredictor_x6, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred


class LatentPredictor_x7(nn.Module):
    def __init__(self):
        super(LatentPredictor_x7, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred


class LatentPredictor_x8(nn.Module):
    def __init__(self):
        super(LatentPredictor_x8, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred


class LatentPredictor_x9(nn.Module):
    def __init__(self):
        super(LatentPredictor_x9, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred


class LatentPredictor_x10(nn.Module):
    def __init__(self):
        super(LatentPredictor_x10, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred


class LatentPredictor_x11(nn.Module):
    def __init__(self):
        super(LatentPredictor_x11, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred


class LatentPredictor_x12(nn.Module):
    def __init__(self):
        super(LatentPredictor_x12, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred


class LatentPredictor_x13(nn.Module):
    def __init__(self):
        super(LatentPredictor_x13, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred


class LatentPredictor_x14(nn.Module):
    def __init__(self):
        super(LatentPredictor_x14, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred


class LatentPredictor_x15(nn.Module):
    def __init__(self):
        super(LatentPredictor_x15, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(14, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred


class LatentPredictor_x16(nn.Module):
    def __init__(self):
        super(LatentPredictor_x16, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred




class LatentPredictor_x0(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(15, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)

        )

    def forward(self, z):
        z_pred = self.fc(z)
        return z_pred