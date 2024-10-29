import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)




# # Define the VAE model
# class VAE(nn.Module):
#     def __init__(self, z_dim=10):
#         super(VAE, self).__init__()
#
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 32, 4, 2, 1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 4, 2, 1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, 4, 2, 1),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, 4, 2, 1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(256 * 4 * 4, 512),
#             nn.ReLU()
#         )
#         self.fc_mu = nn.Linear(512, z_dim)
#         self.fc_logvar = nn.Linear(512, z_dim)
#
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(z_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256 * 4 * 4),
#             nn.ReLU(),
#             nn.Unflatten(1, (256, 4, 4)),
#             nn.ConvTranspose2d(256, 128, 4, 2, 1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, 4, 2, 1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 1, 4, 2, 1),
#             nn.Sigmoid()
#         )
#
#     def encode(self, x):
#         h = self.encoder(x)
#         return self.fc_mu(h), self.fc_logvar(h)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def decode(self, z):
#         return self.decoder(z)
#
#     def forward(self, x,using_mu=False):
#         if using_mu:
#             mu, logvar = self.encode(x)
#             z = mu
#             return z, self.decode(z), mu, logvar
#
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return z,self.decode(z), mu, logvar
#

class VAE(nn.Module):
    def __init__(self, z_dim=10):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_logvar = nn.Linear(512, z_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x,using_mu=False):
        if using_mu:
            mu, logvar = self.encode(x)
            z = mu
            return z, self.decode(z), mu, logvar

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z,self.decode(z), mu, logvar


def loss_function_individual(recon_x, x, mu, logvar,  beta=0.1):
    n= x.size(0)

    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum').div(n)
    KLD = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    individual_loss = BCE + KLD*beta
    return individual_loss, BCE.detach().mean(), KLD.detach().mean()