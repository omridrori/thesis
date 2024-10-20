import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from vae.vae_images.predictors.predictors_training import train_predictors
from vae.vae_images.tensorboard_utils import log_reconstruction_to_tensorboard
from vae.vae_images.vae_model.training_vae import train_vae
from vae.vae_images.visualizations import create_disentanglement_videos



def train_models(train_loader, vae_model, predictors, optimizer_vae, optimizer_predictors,
                 num_epochs, log_dir, delta=1, initial_beta=1, lambda_norm=0.1, log_tensorboard=False, device='cuda', rho=20):
    vae_model.to(device)
    for predictor in predictors:
        predictor.to(device)
    writer = None
    if log_tensorboard:
        writer = SummaryWriter(log_dir=log_dir)

    # Calculate beta decay rate
    beta_decay_rate = np.log(initial_beta / 0.5) / 10
    beta_update_interval = num_epochs // 10

    for epoch in range(num_epochs):
        # Calculate current beta value
        current_step = epoch // beta_update_interval
        beta = max(initial_beta * np.exp(-beta_decay_rate * current_step), 0.2)

        # VAE training step
        for param in vae_model.parameters():
            param.requires_grad = True
        for predictor in predictors:
            for param in predictor.parameters():
                param.requires_grad = False
        loss_vae, loss_total, kld, predictors_loss_in_vae, running_norm = train_vae(
            train_loader, vae_model, predictors, optimizer_vae, delta, beta, rho, lambda_norm, device)

        # Predictor training steps
        for param in vae_model.parameters():
            param.requires_grad = False
        for predictor in predictors:
            for param in predictor.parameters():
                param.requires_grad = True

        loss_predictors = 0
        num_epochs_predictors =1


        # if epoch>3:
        #     num_epochs_predictors = 3
        # if epoch>20:
        #     num_epochs_predictors = 10
        #
        # if epoch>100:
        #     num_epochs_predictors = 10

        for _ in range(num_epochs_predictors):
            loss_predictors += train_predictors(train_loader, vae_model, predictors, optimizer_predictors, delta,
                                                device)

        loss_predictors /= num_epochs_predictors

        for param in vae_model.parameters():
            param.requires_grad = True


        if log_tensorboard:

            writer.add_scalar('Loss/train_total', loss_total, epoch)
            writer.add_scalar('Loss/train_vae', loss_vae, epoch)
            writer.add_scalar('Loss/train_predictors', loss_predictors, epoch)
            writer.add_scalar('Loss/train_kld', kld, epoch)
            writer.add_scalar('running_norm_z', running_norm, epoch)

            if epoch % 1 == 0:
                log_reconstruction_to_tensorboard(vae_model, train_loader, device, writer, epoch, loss_total,predictors)

            if epoch %20 == 0 and epoch>0:
                create_disentanglement_videos(vae_model, train_loader, device, "videos", range_value=5, num_steps=100, fps=30)




        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss Total: {loss_total:.4f}, Loss AE: {loss_vae:.4f}, Loss predictors: {loss_predictors:.4f}, KLD: {kld:.4f}, Beta: {beta:.5f}, Delta: {delta:.4f}')

    if log_tensorboard:
        writer.close()
    return


