import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from vae.vae_images_one_predictor.predictors.estimated_z import generate_estimated_z
from vae.vae_images_one_predictor.predictors.predictors_training import train_predictor
from vae.vae_images_one_predictor.tensorboard_utils import log_reconstruction_to_tensorboard
from vae.vae_images_one_predictor.vae_model.training_vae import train_vae
from vae.vae_images_one_predictor.vae_model.vae_model import loss_function_individual
from vae.vae_images_one_predictor.visualizations import create_disentanglement_videos

import torch.nn.functional as F




def train_models(train_loader, vae_model, predictor, optimizer_vae, optimizer_predictor,
                 num_epochs, log_dir, delta=1, beta=1, lambda_norm=0.1, log_tensorboard=False, device='cuda', rho=20):

    delta_temp=delta
    delta=0

    initial_beta = beta
    vae_model.to(device)
    predictor.to(device)

    writer = None
    if log_tensorboard:
        writer = SummaryWriter(log_dir=log_dir)

    # Calculate beta decay rate
    # beta_decay_rate = np.log(initial_beta / 0.5) / 10
    # beta_update_interval = num_epochs // 10

    for epoch in range(num_epochs):
        # Calculate current beta value
        # current_step = epoch // beta_update_interval
        # beta = max(initial_beta * np.exp(-beta_decay_rate * current_step), 0.2)

        # VAE training step
        for param in vae_model.parameters():
            param.requires_grad = True

        for param in predictor.parameters():
            param.requires_grad = False


        loss_vae, loss_total, kld, predictors_loss_in_vae, running_norm = train_vae(
            train_loader, vae_model, predictor, optimizer_vae, delta, beta, rho, lambda_norm, device)

        # Predictor training steps
        for param in vae_model.parameters():
            param.requires_grad = False
        for param in predictor.parameters():
            param.requires_grad = True

        loss_predictors = 0
        num_epochs_predictors =1


        if epoch>10:
            num_epochs_predictors = 5
            delta=delta_temp
        # if epoch>20:
        #     num_epochs_predictors = 10
        #
        # if epoch>100:
        #     num_epochs_predictors = 10

        for _ in range(num_epochs_predictors):
            loss_current=  train_predictor(train_loader, vae_model, predictor, optimizer_predictor, delta,
                                                device)
            loss_predictors +=loss_current

        if num_epochs_predictors>0:
            loss_predictors /= num_epochs_predictors

        for param in vae_model.parameters():
            param.requires_grad = True


        if log_tensorboard:

            writer.add_scalar('Loss/train_total', loss_total, epoch)
            writer.add_scalar('Loss/train_vae', loss_vae, epoch)
            writer.add_scalar('Loss/train_predictors', loss_predictors, epoch)
            writer.add_scalar('Loss/train_kld', kld, epoch)
            writer.add_scalar('running_norm_z', running_norm, epoch)

            # if epoch %100 == 0:
            #     log_reconstruction_to_tensorboard(vae_model, train_loader, device, writer, epoch, loss_total,predictor)

            if epoch %20 == 0 and epoch>0:
                create_disentanglement_videos(vae_model, train_loader, device, "videos", range_value=5, num_steps=100, fps=30)




        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss Total: {loss_total:.4f}, Loss AE: {loss_vae:.4f}, Loss predictors: {predictors_loss_in_vae:.4f}, KLD: {kld:.4f}, Beta: {beta:.5f}, Delta: {delta:.4f}')

    if log_tensorboard:
        writer.close()
    return


