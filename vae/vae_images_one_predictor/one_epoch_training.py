import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from vae.vae_images_one_predictor.predictors.estimated_z import generate_estimated_z
from vae.vae_images_one_predictor.vae_model.vae_model import loss_function_individual
from vae.vae_images_one_predictor.visualizations import create_latent_traversal_grid
from torch.nn.utils import clip_grad_norm_

import torch.nn.functional as F


def train_on_batch_vae(data, vae, predictor, optimizer_vae, delta, beta, rho, lambda_norm, device):
    data = data[0]
    data = data.to(device)

    encoded, decoded, mu, logvar = vae(data)

    encoded_normalized = encoded
    z_estimated = generate_estimated_z(predictor, mu, device)

    decoded_estimated = vae.decoder(z_estimated)
    # predictors_loss = F.binary_cross_entropy(decoded_estimated, data, reduction='none').sum(
    #     axis=(1, 2, 3))  # todo check if this is correct
    predictors_loss=nn.functional.binary_cross_entropy(decoded_estimated, data, reduction='sum').div(data.size(0))


    loss_vae, bce, kld = loss_function_individual(decoded, data, mu, logvar, beta)
    predictors_loss_exp= torch.exp(-predictors_loss / rho)

    total_loss = loss_vae + delta * predictors_loss_exp
    # calculate norm of z
    z_norm = torch.norm(encoded_normalized, dim=1).mean()
    # total_loss += lambda_norm * z_norm

    running_norm = z_norm.cpu().item()
    total_loss.backward()
    # clip_grad_norm_(vae.parameters(), 1)

    return bce.cpu().item(), total_loss.cpu().item(), kld.cpu().item(), predictors_loss.cpu().item(), z_norm.cpu().item(), mu, logvar


def train_on_batch_predictor(data, vae, predictor, optimizer_predictor, delta, device):
    data = data[0]
    data = data.to(device)
    encoded, deccoded, mu, logvar = vae(data)

    # generated_z = generate_estimated_z(predictor, mu, device)
    # decoded_estimated = vae.decoder(generated_z)
    # # loss_predictors = F.binary_cross_entropy(decoded_estimated, data, reduction='none').sum(axis=(1, 2, 3)).mean()
    # loss_predictors=nn.functional.binary_cross_entropy(decoded_estimated, data, reduction='sum').div(data.size(0))

    generated_z = generate_estimated_z(predictor, mu, device)
    #calculate the mse loss between the generated z and the actual mu
    loss_predictors = nn.functional.mse_loss(generated_z, mu, reduction='sum').div(data.size(0))


    loss_predictors.backward()
    # clip_grad_norm_(predictor.parameters(), 1)

    return loss_predictors.cpu().item(), generated_z


def train_models_one_epoch(train_loader, vae_model, predictor, optimizer_vae, optimizer_predictor,
                           num_epochs, log_dir, latent_dim=10, delta=1, beta=1, lambda_norm=0.1,
                           output_videos=None, device='cuda', rho=20,
                           log_to_tensorboard=True):
    vae_model.to(device)
    predictor.to(device)
    writer = None
    if  log_to_tensorboard:
        writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        train_both = epoch % 2 == 0  # Alternate between training both and just predictor

        all_bce_loss = all_loss_total = all_kld_loss = all_predictors_loss_in_vae = all_running_norm = all_loss_predictors = 0
        all_mu = torch.zeros(latent_dim).to(device)
        all_logvar = torch.zeros(latent_dim).to(device)
        all_mu_estimated = torch.zeros(latent_dim).to(device)

        # Lists to store values for histograms, one for each dimension
        all_mu_list = [[] for _ in range(latent_dim)]
        all_logvar_list = [[] for _ in range(latent_dim)]
        all_mu_estimated_list = [[] for _ in range(latent_dim)]

        total_batches = 0

        for batch_idx, (batch_vae, batch_predictors) in enumerate(train_loader):
            optimizer_vae.zero_grad()
            optimizer_predictor.zero_grad()

            if train_both:
                # VAE training step
                for param in vae_model.parameters():
                    param.requires_grad = True
                for param in predictor.parameters():
                    param.requires_grad = False

                bce, loss_total, kld, predictors_loss_in_vae, running_norm, mu, logvar = train_on_batch_vae(
                    batch_vae, vae_model, predictor, optimizer_vae, delta, beta, rho, lambda_norm, device)

                optimizer_vae.step()

            # Predictor training steps
            for param in vae_model.parameters():
                param.requires_grad = False
            for param in predictor.parameters():
                param.requires_grad = True

            loss_current, mu_estimated = train_on_batch_predictor(
                batch_predictors, vae_model, predictor, optimizer_predictor, delta, device)

            optimizer_predictor.step()

            # Reset requires_grad
            for param in vae_model.parameters():
                param.requires_grad = True
            for param in predictor.parameters():
                param.requires_grad = True

            if train_both:
                all_bce_loss += bce
                all_loss_total += loss_total
                all_kld_loss += kld
                all_predictors_loss_in_vae += predictors_loss_in_vae
                all_running_norm += running_norm
                all_mu += mu.mean(dim=0)
                all_logvar += logvar.mean(dim=0)

            all_loss_predictors += loss_current
            all_mu_estimated += mu_estimated.mean(dim=0)

            # Collect values for histograms, separated by dimension
            if train_both:
                for dim in range(latent_dim):
                    all_mu_list[dim].extend(mu[:, dim].detach().cpu().numpy())
                    all_logvar_list[dim].extend(logvar[:, dim].detach().cpu().numpy())
                    all_mu_estimated_list[dim].extend(mu_estimated[:, dim].detach().cpu().numpy())

            total_batches += 1

        # Calculate averages
        all_loss_predictors /= len(train_loader)
        mean_mu_estimated = all_mu_estimated / total_batches

        if train_both:
            all_bce_loss /= len(train_loader)
            all_loss_total /= len(train_loader)
            all_kld_loss /= len(train_loader)
            all_predictors_loss_in_vae /= len(train_loader)
            all_running_norm /= len(train_loader)
            mean_mu = all_mu / total_batches
            mean_logvar = all_logvar / total_batches

        # Logging
        if log_to_tensorboard:
            writer.add_scalar('Loss/train_predictors', all_loss_predictors, epoch)

            if train_both:
                writer.add_scalar('Loss/train_total', all_loss_total, epoch)
                writer.add_scalar('Loss/train_vae', all_bce_loss, epoch)
                writer.add_scalar('Loss/train_kld', all_kld_loss, epoch)
                writer.add_scalar('running_norm_z', all_running_norm, epoch)

                # Log mean values and histograms only when training both
                for i in range(latent_dim):
                    writer.add_scalar(f'mean_mu/dim_{i}', mean_mu[i], epoch)
                    writer.add_scalar(f'mean_logvar/dim_{i}', mean_logvar[i], epoch)
                    writer.add_scalar(f'mean_mu_estimated/dim_{i}', mean_mu_estimated[i], epoch)

                for dim in range(latent_dim):
                    writer.add_histogram(f'mu/dim_{dim}', np.array(all_mu_list[dim]), epoch)
                    writer.add_histogram(f'logvar/dim_{dim}', np.array(all_logvar_list[dim]), epoch)
                    writer.add_histogram(f'mu_estimated/dim_{dim}', np.array(all_mu_estimated_list[dim]), epoch)

            if epoch % 15 == 0 and epoch > 0:
                create_latent_traversal_grid(vae_model, train_loader, device, output_videos,
                                             num_dimensions=10, num_steps=5, range_value=1)

        # Print appropriate message based on what was trained
        if train_both:
            print(f"Epoch {epoch} (Training Both) - Total loss: {all_loss_total:.4f}, VAE loss: {all_bce_loss:.4f}, "
                  f"Predictor loss: {all_predictors_loss_in_vae:.4f}, KLD: {all_kld_loss:.4f}, Norm: {all_running_norm:.4f}")
        else:
            print(f"Epoch {epoch} (Training Predictor Only) - Predictor loss: {all_loss_predictors:.4f}")

    if log_to_tensorboard:
        writer.close()

    print("Training finished successfully!")