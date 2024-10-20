from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from vae.vae_images.tensorboard_utils import  log_reconstruction_to_tensorboard
from vae.vae_images.vae_model.vae_model import VAE
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dSprites dataset
data = np.load(
    "C:\\Users\\User\\Desktop\\thesis\\vae\\vae_images\\dsprites-dataset\\dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
    allow_pickle=True)
imgs = data['imgs']

# Create a subset of 1000 images
subset_indices = np.random.choice(len(imgs), 200, replace=False)
imgs_subset = imgs[subset_indices]

# Convert the subset to torch tensor, add channel dimension, and convert to float
imgs_subset = torch.from_numpy(imgs_subset).float().unsqueeze(1)

# Create DataLoader
batch_size = 128
dataset = TensorDataset(imgs_subset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD




vae = VAE(z_dim=10).to(device)
# Initialize the optimizer
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

time_stamp =  datetime.now().strftime("%Y%m%d-%H%M%S")
# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=r"logs_regular_vae/logs_regular_vae " + time_stamp)








# Training loop
num_epochs = 2000000
for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    for batch_idx, (data,) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        z,recon_batch, mu, logvar = vae(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_loss = train_loss / len(dataloader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Average loss: {avg_loss:.4f}')



    if (epoch + 1) % 5 == 0:
        log_reconstruction_to_tensorboard(vae, dataloader, device, writer, epoch, avg_loss)



    # torch.save(vae.state_dict(), f'vae_epoch_{epoch + 1}.pth')

# Close the TensorBoard writer
writer.close()

print("Training completed. You can view the results in TensorBoard.")