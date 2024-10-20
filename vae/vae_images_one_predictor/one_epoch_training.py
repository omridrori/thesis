import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from vae.vae_images_one_predictor.predictors.estimated_z import generate_estimated_z
from vae.vae_images_one_predictor.predictors.predictors_training import train_predictor
from vae.vae_images_one_predictor.tensorboard_utils import log_reconstruction_to_tensorboard
from vae.vae_images_one_predictor.vae_model.training_vae import train_vae
from vae.vae_images_one_predictor.vae_model.vae_model import loss_function_individual
from vae.vae_images_one_predictor.visualizations import create_disentanglement_videos, create_latent_traversal_grid

import torch.nn.functional as F


def train_on_batch_vae(data, vae, predictor, optimizer_vae, delta, beta, rho, lambda_norm, device):
    data = data[0]
    data = data.to(device)

    encoded, decoded, mu, logvar = vae(data)

    encoded_normalized = encoded
    z_estimated = generate_estimated_z(predictor, encoded, device)

    decoded_estimated = vae.decoder(z_estimated)

    predictors_loss = F.binary_cross_entropy(decoded_estimated, data, reduction='none').sum(
        axis=(1, 2, 3))  # todo check if this is correct

    loss_vae, bce, kld = loss_function_individual(decoded, data, mu, logvar, beta)

    total_loss = (loss_vae + delta * torch.exp(-predictors_loss / rho)).mean()
    # calculate norm of z
    z_norm = torch.norm(encoded_normalized, dim=1).mean()
    total_loss += lambda_norm * z_norm

    running_norm = z_norm.cpu().item()
    total_loss.backward()

    return bce.cpu().item(), total_loss.cpu().item(), kld.cpu().item(), predictors_loss.mean().cpu().item(), z_norm.cpu().item(), mu, logvar


def train_on_batch_predictor(data, vae, predictor, optimizer_predictor, delta, device):
    data = data[0]
    data = data.to(device)
    encoded, deccoded, mu, logvar = vae(data)

    generated_z = generate_estimated_z(predictor, encoded, device)
    decoded_estimated = vae.decoder(generated_z)
    loss_predictors = F.binary_cross_entropy(decoded_estimated, data, reduction='none').sum(axis=(1, 2, 3)).mean()
    loss_predictors.backward()

    return loss_predictors.cpu().item(), generated_z


def train_models_one_epoch(train_loader, vae_model, predictor, optimizer_vae, optimizer_predictor,
                           num_epochs, log_dir, latent_dim=10, delta=1, beta=1, lambda_norm=0.1,
                           log_tensorboard=False,output_videos=None ,device='cuda', rho=20):
    vae_model.to(device)
    predictor.to(device)

    writer = None
    if log_tensorboard:
        writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        all_bce_loss = all_loss_total = all_kld_loss = all_predictors_loss_in_vae = all_running_norm = all_loss_predictors = 0
        all_mu = torch.zeros(latent_dim).to(device)
        all_logvar = torch.zeros(latent_dim).to(device)
        all_mu_estimated = torch.zeros(latent_dim).to(device)

        # Lists to store values for histograms, one for each dimension
        all_mu_list = [[] for _ in range(latent_dim)]
        all_logvar_list = [[] for _ in range(latent_dim)]
        all_mu_estimated_list = [[] for _ in range(latent_dim)]

        total_batches = 0

        for batch_idx, data in enumerate(train_loader):
            optimizer_vae.zero_grad()
            optimizer_predictor.zero_grad()

            # VAE training step
            for param in vae_model.parameters():
                param.requires_grad = True
            for param in predictor.parameters():
                param.requires_grad = False

            bce, loss_total, kld, predictors_loss_in_vae, running_norm, mu, logvar = train_on_batch_vae(
                data, vae_model, predictor, optimizer_vae, delta, beta, rho, lambda_norm, device)

            # Predictor training steps
            for param in vae_model.parameters():
                param.requires_grad = False
            for param in predictor.parameters():
                param.requires_grad = True

            loss_predictors = 0
            num_epochs_predictors = 1
            for _ in range(num_epochs_predictors):
                loss_current, mu_estimated = train_on_batch_predictor(data, vae_model, predictor, optimizer_predictor,
                                                                      delta, device)
                loss_predictors += loss_current

            if num_epochs_predictors > 0:
                loss_predictors /= num_epochs_predictors

            optimizer_predictor.step()
            for param in vae_model.parameters():
                param.requires_grad = True
            for param in predictor.parameters():
                param.requires_grad = False

            optimizer_vae.step()

            all_bce_loss += bce
            all_loss_total += loss_total
            all_kld_loss += kld
            all_predictors_loss_in_vae += predictors_loss_in_vae
            all_running_norm += running_norm
            all_loss_predictors += loss_predictors
            all_mu += mu.mean(dim=0)
            all_logvar += logvar.mean(dim=0)
            all_mu_estimated += mu_estimated.mean(dim=0)

            # Collect values for histograms, separated by dimension
            for dim in range(latent_dim):
                all_mu_list[dim].extend(mu[:, dim].detach().cpu().numpy())
                all_logvar_list[dim].extend(logvar[:, dim].detach().cpu().numpy())
                all_mu_estimated_list[dim].extend(mu_estimated[:, dim].detach().cpu().numpy())

            total_batches += 1

        all_bce_loss /= len(train_loader)
        all_loss_total /= len(train_loader)
        all_kld_loss /= len(train_loader)
        all_predictors_loss_in_vae /= len(train_loader)
        all_running_norm /= len(train_loader)
        all_loss_predictors /= len(train_loader)
        mean_mu = all_mu / total_batches
        mean_logvar = all_logvar / total_batches
        mean_mu_estimated = all_mu_estimated / total_batches

        if log_tensorboard:
            writer.add_scalar('Loss/train_total', all_loss_total, epoch)
            writer.add_scalar('Loss/train_vae', all_bce_loss, epoch)
            writer.add_scalar('Loss/train_predictors', all_loss_predictors, epoch)
            writer.add_scalar('Loss/train_kld', all_kld_loss, epoch)
            writer.add_scalar('running_norm_z', all_running_norm, epoch)

            # Log mean mu, mean logvar, and mean mu_estimated
            for i in range(latent_dim):
                writer.add_scalar(f'mean_mu/dim_{i}', mean_mu[i], epoch)
                writer.add_scalar(f'mean_logvar/dim_{i}', mean_logvar[i], epoch)
                writer.add_scalar(f'mean_mu_estimated/dim_{i}', mean_mu_estimated[i], epoch)

            # Log histograms for each dimension
            for dim in range(latent_dim):
                writer.add_histogram(f'mu/dim_{dim}', np.array(all_mu_list[dim]), epoch)
                writer.add_histogram(f'logvar/dim_{dim}', np.array(all_logvar_list[dim]), epoch)
                writer.add_histogram(f'mu_estimated/dim_{dim}', np.array(all_mu_estimated_list[dim]), epoch)

            # if epoch % 1 == 0:
            #     log_reconstruction_to_tensorboard(vae_model, train_loader, device, writer, epoch, all_loss_total,
            #                                       predictor)

            if epoch % 15 == 0 and epoch > 0:
                #apply create_latent_traversal_grid(vae, dataloader, device, experiment_name, output_dir="results", num_dimensions=10, num_steps=5, range_value=3):
                create_latent_traversal_grid(vae_model, train_loader, device, output_videos, num_dimensions=10, num_steps=5, range_value=1)

        print("Epoch ", epoch, "Total loss: ", all_loss_total, "VAE loss: ", all_bce_loss, "Predictor loss: ",
              all_loss_predictors, "KLD: ", all_kld_loss, "Norm: ", all_running_norm)

    if log_tensorboard:
        writer.close()

    print("Training finished successfully!")