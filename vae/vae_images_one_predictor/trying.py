import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set random seed for reproducibility
torch.manual_seed(42)
import matplotlib.pyplot as plt
import numpy as np


def plot_reconstructions(data, recon_batch, n=5):
    """
    Plots the first n original images and their reconstructions.

    Parameters:
        data (torch.Tensor): Batch of original images.
        recon_batch (torch.Tensor): Batch of reconstructed images.
        n (int): Number of images to display (default is 5).
    """
    # Detach the tensors and move them to CPU
    data = data.cpu().detach().numpy()
    recon_batch = recon_batch.cpu().detach().numpy()

    # Reshape the data to (batch_size, 28, 28) for MNIST images
    data = data.reshape(-1, 28, 28)
    recon_batch = recon_batch.reshape(-1, 28, 28)

    # Plot the images
    fig, axes = plt.subplots(n, 2, figsize=(10, n * 2))
    for i in range(n):
        # Plot original image
        axes[i, 0].imshow(data[i], cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        # Plot reconstructed image
        axes[i, 1].imshow(recon_batch[i], cmap='gray')
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Define the VAE architecture
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')


# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download and prepare the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Initialize the VAE model
    latent_dim = 20
    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)

    # Save the trained model
    torch.save(model.state_dict(), "vae_mnist.pth")
    print("Training completed. Model saved as vae_mnist.pth")


if __name__ == "__main__":
    main()