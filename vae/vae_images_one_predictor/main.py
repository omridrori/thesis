import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from vae.vae_images_one_predictor.main_training import train_models
from vae.vae_images_one_predictor.tensorboard_utils import  log_reconstruction_to_tensorboard
from vae.vae_images_one_predictor.training_utils import setup_models, setup_optimizers
from vae.vae_images_one_predictor.vae_model.vae_model import VAE
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dSprites dataset
data = np.load(
    r"C:\Users\User\Desktop\thesis\vae\vae_images_one_predictor\dsprites-dataset\dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
    allow_pickle=True)
imgs = data['imgs']

# Create a subset of 1000 images
subset_indices = np.random.choice(len(imgs), 20000, replace=False)
imgs_subset = imgs[subset_indices]

# Convert the subset to torch tensor, add channel dimension, and convert to float
imgs_subset = torch.from_numpy(imgs_subset).float().unsqueeze(1)

imgs=imgs_subset
# Create DataLoader
batch_size = 128
dataset = TensorDataset(imgs_subset)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Ensure the data is in the range [0, 1] for Bernoulli distribution
if imgs.max() > 1:
    imgs = imgs / 255.0

# imgs = (imgs - imgs.mean()) / imgs.std()

vae_model, predictor = setup_models(device)
optimizer_vae, optimizers_predictors = setup_optimizers(vae_model, predictor)

# Debug: Print model information
print(f"VAE model:\n{vae_model}")


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(r'C:\Users\User\Desktop\thesis\vae\vae_images_one_predictor\runs_vae', f"{timestamp}")


train_models(
    train_loader, vae_model, predictor,
    optimizer_vae, optimizers_predictors,
    num_epochs=1000, log_dir=log_dir, delta=1000, beta=1,lambda_norm=0,
    log_tensorboard=True , device=device,rho=15)


