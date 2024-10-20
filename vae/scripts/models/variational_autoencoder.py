import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(8, 1024),
            nn.Tanh(),
            nn.Linear(1024, 256),
            # nn.BatchNorm1d(128),
            # nn.Tanh(),
            # nn.Linear(128, 64)
        )

        # Latent space
        self.fc_mu = nn.Linear(256, 16)

        self.fc_logvar = nn.Linear(256, 16)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8),

            # nn.BatchNorm1d(64),
            # nn.Tanh(),
            # nn.Linear(64, 8)
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # z=F.normalize(z,dim=1)
        return z,self.decode(z), mu, logvar


