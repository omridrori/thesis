from torch import nn


# class Predictor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Sequential(
#     nn.Linear(10, 256),
#     nn.ReLU(),
#     nn.Linear(256, 1024),
#     nn.BatchNorm1d(1024),  # BatchNorm after two linear layers
#     nn.ReLU(),
#
#     nn.Linear(1024, 2048),
#     nn.ReLU(),
#     nn.Linear(2048, 4096),
#     nn.BatchNorm1d(4096),  # BatchNorm after two more linear layers
#     nn.ReLU(),
#
#     nn.Linear(4096, 8192),
#     nn.ReLU(),
#     nn.Linear(8192, 4096),
#     nn.BatchNorm1d(4096),  # BatchNorm after two more linear layers
#     nn.ReLU(),
#
#     nn.Linear(4096, 2048),
#     nn.ReLU(),
#     nn.Linear(2048, 1024),
#     nn.BatchNorm1d(1024),  # BatchNorm after two more linear layers
#     nn.ReLU(),
#
#     nn.Linear(1024, 256),
#     nn.ReLU(),
#     nn.Linear(256, 10)  # Last linear layer, no BatchNorm needed
# )
#
#     def forward(self, z):
#         z_pred = self.fc(z)
#         return z_pred




class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
    nn.Linear(10, 256),
    nn.ReLU(),
    nn.Linear(256, 1024),
    nn.ReLU(),

    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 4096),
    nn.ReLU(),

    nn.Linear(4096, 8192),
    nn.ReLU(),
    nn.Linear(8192, 4096),
    nn.ReLU(),

    nn.Linear(4096, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1024),
    nn.ReLU(),

    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Linear(256, 10)  # Last linear layer, no BatchNorm needed
)

    def forward(self, z):
        # Save the original input for the residual connection
        z_residual = z
        # Pass through the fully connected layers
        z_pred = self.fc(z)
        # Add the residual connection from input to output
        z_pred += z_residual
        return z_pred
#

#
# class Predictor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Sequential(
#     nn.Linear(10, 1024),
#     nn.ReLU(),
#     nn.Linear(1024, 8096),
#     nn.ReLU(),
#     nn.Linear(8096, 1024),
#     nn.ReLU(),
#     nn.Linear(1024, 10),
# )
#
#     def forward(self, z):
#         # Save the original input for the residual connection
#         z_residual = z
#         # Pass through the fully connected layers
#         z_pred = self.fc(z)
#         # Add the residual connection from input to output
#         z_pred += z_residual
#         return z_pred