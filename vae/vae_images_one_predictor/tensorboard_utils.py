import random

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import torch
import torchvision
from vae.vae_images_one_predictor.predictors.estimated_z import generate_estimated_z

def log_to_tensorboard(writer, epoch, vae_model, predictor, train_loader, loss_vae, loss_total,
                       loss_predictors, kld, delta, current_beta, device):
    # Log scalar metrics
    writer.add_scalar('Loss/train_ae', loss_vae, epoch)
    writer.add_scalar('Loss/train_total', loss_total, epoch)
    writer.add_scalar('Loss/train_predictors', loss_predictors, epoch)
    writer.add_scalar('Loss/train_kld', kld, epoch)
    writer.add_scalar('Hyperparameters/delta', delta, epoch)
    writer.add_scalar('Hyperparameters/beta', current_beta, epoch)

    # Choose one random image from the dataset
    vae_model.eval()
    with torch.no_grad():
        # Get a random batch and select the first image
        data = next(iter(train_loader))[0].to(device)
        original_image = data[0:1]  # Shape: [1, 1, 64, 64]

        # Original image
        writer.add_image('Images/Original', original_image.squeeze(0), epoch, dataformats='CHW')

        # Encoded and decoded image
        encoded, decoded, _, _ = vae_model(original_image)
        writer.add_image('Images/Decoded', decoded.squeeze(0), epoch, dataformats='CHW')

        # Decoded image with estimated z
        z_estimated = generate_estimated_z(predictor, encoded, device)
        decoded_estimated = vae_model.decoder(z_estimated)
        writer.add_image('Images/Decoded_Estimated', decoded_estimated.squeeze(0), epoch, dataformats='CHW')

    vae_model.train()





def log_reconstruction_to_tensorboard(vae, dataloader, device, writer, epoch, train_loss, predictor):
    vae.eval()
    predictor.eval()


    with torch.no_grad():
        # Get a random batch from the dataloader
        original = next(iter(dataloader))[0].to(device)

        # Select a single image from the batch
        single_image = original[0].unsqueeze(0)

        # Get the reconstruction
        z, reconstructed, _, _ = vae(single_image)

        # Get the reconstruction with estimated z
        z_estimated = generate_estimated_z(predictor, z, device)
        reconstructed_estimated = vae.decoder(z_estimated)

        # Create a side-by-side comparison of original, standard reconstruction, and estimated z reconstruction
        comparison = torch.cat([single_image, reconstructed, reconstructed_estimated], dim=2)

        # Create a grid (in this case, it's just one row with three images)
        img_grid = torchvision.utils.make_grid(comparison, nrow=1, normalize=True)

        # Log the image trio to TensorBoard
        writer.add_image('Original vs Reconstructed vs Estimated Z', img_grid, global_step=epoch + 1)

    # Log other metrics (if any)
    writer.add_scalar('Loss/train', train_loss, epoch)
