import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from vae.vae_images_one_predictor.main_training import train_models
from vae.vae_images_one_predictor.one_epoch_training import train_models_one_epoch
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
subset_indices = np.random.choice(len(imgs), 40000, replace=False)
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



# Define possible values for each hyperparameter
delta_values = [0,10,100] # Example values for delta
beta_values = [4]  # Example values for beta
lambda_values = [0]  # Example values for lambda_norm
rho_values = [10,100,200]  # Example values for rho


lambda_norm=0
for rho in rho_values:
    for beta in beta_values:
        for delta in delta_values:
            if delta ==0 and rho!=10:
                continue


            experiment_name = f"train_vae_predictor_8000_images_epochs_1000_delta_{delta}_beta_{beta}_lambda_{lambda_norm}_rho_{rho}"

            # Add timestamp to the experiment name
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_name = f"{experiment_name}_{timestamp}"

            # Define log directory
            log_dir = os.path.join(r'C:\Users\User\Desktop\thesis\vae\vae_images_one_predictor\runs_vae',
                                   experiment_name)

            # Debug: Print model information and current configuration
            print(f"VAE model:\n{vae_model}")
            print(f"Starting training with delta={delta}, beta={beta}, lambda_norm={lambda_norm}, rho={rho}")

            # Train the model with the current set of hyperparameters
            train_models_one_epoch(
                train_loader, vae_model, predictor,
                optimizer_vae, optimizers_predictors,
                num_epochs=1500, log_dir=log_dir, delta=delta, beta=beta, lambda_norm=lambda_norm,
                log_tensorboard=True, output_videos=experiment_name, device=device, rho=rho
            )
