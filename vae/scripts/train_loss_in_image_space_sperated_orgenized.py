import math
from random import random

import numpy as np
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.save_models import save_models
from vae.scripts.models.scedualers import WarmupExponentialDecayScheduler
from vae.scripts.models.variational_autoencoder import VAE
from vae.scripts.models.initializations import get_initializer

from vae.scripts.models.variational_latent_predictors import LatentPredictor_x2, LatentPredictor_x3, LatentPredictor_x4, \
    LatentPredictor_x5, \
    LatentPredictor_x6, LatentPredictor_x7, LatentPredictor_x8, LatentPredictor_x9, LatentPredictor_x10, \
    LatentPredictor_x11, LatentPredictor_x12, LatentPredictor_x13, LatentPredictor_x14, LatentPredictor_x15, \
    LatentPredictor_x16, LatentPredictor_x0
from utils.data_loader import get_dataloader
from utils.parse_args import get_args
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from dotenv import load_dotenv

from vae.scripts.plot_modifications import generate_data_to_plot, generate_plots_modifications

torch.autograd.set_detect_anomaly(True)
# Assume LatentPredictor_x2, LatentPredictor_x3, LatentPredictor_x4 are defined similarly to LatentPredictor_x4

import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from vae.scripts.models.annealer import Annealer


import torch

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

import torch.optim.lr_scheduler as lr_scheduler

def setup_schedulers(optimizer_vae, optimizers_predictors):
    scheduler_vae = lr_scheduler.StepLR(optimizer_vae, step_size=5, gamma=0.9)
    schedulers_predictors = [lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) for optimizer in optimizers_predictors]
    return scheduler_vae, schedulers_predictors
def setup_models(device):
    vae_model = VAE().to(device)
    predictors = [LatentPredictor_x0().to(device) for _ in range(16)]
    # predictors = [LatentPredictor_x0(),LatentPredictor_x2(), LatentPredictor_x3(), LatentPredictor_x4(), LatentPredictor_x5(),
    #                LatentPredictor_x6(), LatentPredictor_x7(), LatentPredictor_x8(), LatentPredictor_x9(),
    #                LatentPredictor_x10(), LatentPredictor_x11(), LatentPredictor_x12(), LatentPredictor_x13(),
    #                LatentPredictor_x14(), LatentPredictor_x15(), LatentPredictor_x16()]

    # predictors= [predictor.to(device) for predictor in predictors]
    return vae_model, predictors


def setup_optimizers(vae_model, predictors):
    optimizer_vae = optim.AdamW(vae_model.parameters(), lr=1e-3)
    optimizers_predictors = [optim.AdamW(predictor.parameters(), lr=1e-3) for predictor in predictors]
    return optimizer_vae, optimizers_predictors


def generate_estimated_z(predictors, z, device):
    z_copy = z.clone().to(device)
    z_estimated = []
    for i in range(z.shape[1]):
        z_input = torch.cat([z_copy[:, :i], z_copy[:, i + 1:]], dim=1)
        z_pred = predictors[i](z_input)
        z_estimated.append(z_pred)
    return torch.cat(z_estimated, dim=1)

# def generate_estimated_z(predictors, z,device):
#     z_copy = z.clone().to(device)
#     z_estimated = [predictors[0](z_copy[:, 1:])]  # Predict z0 using z1 to z16
#
#     for i in range(1, z.shape[1]):
#         z_input = z_copy[:, :i]  # Use original z values up to index i
#         z_pred = predictors[i](z_input)
#         z_estimated.append(z_pred)
#
#     return torch.cat(z_estimated, dim=1)


def predictor_task(predictor, z_input, z_val, device):
    z_pred = predictor(z_input.to(device))
    return (z_pred - z_val.to(device)) ** 2


def loss_predict_with_predictors(predictors, data, encoded_normalized, device):
    z = encoded_normalized.to(device)
    val_predictors_losses_individuals = torch.zeros(data.size(0), device=device)
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_predictor = {
            executor.submit(predictor_task, predictors[i - 1], z[:, :i], z[:, i].unsqueeze(1), device): i
            for i in range(1, len(predictors))}
        for future in as_completed(future_to_predictor):
            result = future.result()
            val_predictors_losses_individuals += result.squeeze()
    return val_predictors_losses_individuals


def train_predictors(train_loader, vae, predictors, optimizer_predictors, delta, device):
    for predictor in predictors:
        predictor.train()
    loss_predictors_total = 0.0
    for data in train_loader:
        for optimizer in optimizer_predictors:
            optimizer.zero_grad()
        data = data.to(device)
        encoded, _, _, _ = vae(data)
        # encoded_normalized = F.normalize(encoded, p=2, dim=1)
        encoded_normalized = encoded
        z_estimated = generate_estimated_z(predictors, encoded_normalized, device)
        # z_estimated_normalized = F.normalize(z_estimated, p=2, dim=1)
        z_estimated_normalized = z_estimated
        decoded_estimated = vae.decoder(z_estimated_normalized)
        loss_predictors = F.mse_loss(decoded_estimated, data)
        loss_predictors_total += loss_predictors.item()
        loss_predictors.backward()
        for optimizer in optimizer_predictors:
            optimizer.step()
    return loss_predictors_total / len(train_loader)


def train_vae(train_loader, vae, predictors, optimizer_vae, delta, annealing_agent, beta_max, device):
    vae.train()
    running_loss_vae = running_loss_total = running_kld = predictors_loss_mean = 0.0
    for data in train_loader:
        optimizer_vae.zero_grad()
        data = data.to(device)
        encoded, decoded, mu, logvar = vae(data)
        # encoded_normalized = F.normalize(encoded, p=2, dim=1)
        encoded_normalized = encoded
        z_estimated = generate_estimated_z(predictors, encoded_normalized, device)
        # z_estimated_normalized = F.normalize(z_estimated, p=2, dim=1)
        decoded_estimated = vae.decoder(z_estimated)
        # decoded_estimated = vae.decoder(z_estimated)
        predictors_loss = F.mse_loss(decoded_estimated, data, reduction='none').mean(axis=1)
        loss_vae, mse, kld = loss_function_individual(decoded, data, mu, logvar, annealing_agent, beta_max)
        total_loss = (loss_vae + delta * torch.exp(-predictors_loss/0.5)+torch.norm(encoded_normalized, p=1, dim=1)).mean()
        running_loss_vae += mse.item()
        running_kld += kld.item()
        running_loss_total += total_loss.item()
        predictors_loss_mean += predictors_loss.mean().item()
        total_loss.backward()
        optimizer_vae.step()
    n = len(train_loader)
    return running_loss_vae / n, running_loss_total / n, running_kld / n, predictors_loss_mean / n


def loss_function_individual(recon_x, x, mu, logvar, annealing_agent, beta_max):
    MSE = F.mse_loss(recon_x, x, reduction='none').mean(axis=1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    KLD_add = annealing_agent(KLD) * beta_max
    individual_loss = MSE + KLD_add
    return individual_loss, MSE.detach().mean(), KLD.detach().mean()


def log_training_info(writer, epoch, loss_vae, loss_total, loss_predictors, kld, delta, beta):
    writer.add_scalar('Loss/train_ae', loss_vae, epoch)
    writer.add_scalar('Loss/train_total', loss_total, epoch)
    writer.add_scalar('Loss/train_predictors', loss_predictors, epoch)
    writer.add_scalar('Loss/train_kld', kld, epoch)
    writer.add_scalar('Hyperparameters/delta', delta, epoch)
    writer.add_scalar('Hyperparameters/beta', beta, epoch)


def log_z_distributions(vae_model, predictors, train_loader, writer, epoch, device):
    vae_model.eval()
    for predictor in predictors:
        predictor.eval()
    z_values, z_estimated_values = [], []
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            encoded, _, _, _ = vae_model(batch)
            z_values.append(encoded)
            z_estimated = generate_estimated_z(predictors, encoded, device)
            z_estimated_values.append(z_estimated)
    z_values = torch.cat(z_values, dim=0)
    z_estimated_values = torch.cat(z_estimated_values, dim=0)
    for i in range(z_values.shape[1]):
        writer.add_histogram(f'z_distribution/dim_{i}', z_values[:, i].cpu(), epoch)
        writer.add_histogram(f'z_estimated_distribution/dim_{i}', z_estimated_values[:, i].cpu(), epoch)
    writer.add_scalar('z_distribution/mean', z_values.mean().item(), epoch)
    writer.add_scalar('z_distribution/std', z_values.std().item(), epoch)
    writer.add_scalar('z_estimated_distribution/mean', z_estimated_values.mean().item(), epoch)
    writer.add_scalar('z_estimated_distribution/std', z_estimated_values.std().item(), epoch)
    z_diff = z_values - z_estimated_values
    writer.add_scalar('z_difference/mean', z_diff.abs().mean().item(), epoch)
    writer.add_scalar('z_difference/std', z_diff.std().item(), epoch)
    vae_model.train()
    for predictor in predictors:
        predictor.train()


def train_models(train_loader, val_loader, vae_model, predictors, optimizer_vae, optimizer_predictors,
                 num_epochs, log_dir, delta=0.1, beta_max=0.1, log_tensorboard=False,
                 do_eval=False, data_to_plot=None, device="cpu", scheduler_vae=None, schedulers_predictors=None):
    vae_model.to(device)
    for predictor in predictors:
        predictor.to(device)

    if log_tensorboard:
        writer = SummaryWriter(log_dir=log_dir)

    annealing_agent_beta = Annealer(total_steps=50, shape='cosine', baseline=0.0, cyclical=True)
    annealing_agent_delta = Annealer(total_steps=50, shape='cosine', baseline=0.0, cyclical=True)
    factor = delta

    for epoch in range(num_epochs):
        #delta = annealing_agent_delta.slope() * factor
        factor *= 0.999
        current_beta = annealing_agent_beta.slope() * beta_max

        # VAE training step
        for param in vae_model.parameters():
            param.requires_grad = True
        for predictor in predictors:
            for param in predictor.parameters():
                param.requires_grad = False
        loss_vae, loss_total, kld, predictors_loss_in_vae = train_vae(
            train_loader, vae_model, predictors, optimizer_vae, delta, annealing_agent_beta, beta_max, device
        )

        # Predictor training steps (3 steps for each VAE step)
        for param in vae_model.parameters():
            param.requires_grad = False
        for predictor in predictors:
            for param in predictor.parameters():
                param.requires_grad = True

        loss_predictors=0
        num_epochs_predictors = 1

        if epoch > 20:
            num_epochs_predictors = 3

        if epoch > 100:
            num_epochs_predictors = 5
        # # while (loss_predictors > 15 * abs(loss_vae + 0 * current_beta * kld)):
        # #     loss_predictors = train_predictors(train_loader, vae_model, predictors, optimizer_predictors,
        # #                                                    delta, device)
        # #     print(f"mean_train_losses_predictors: {loss_predictors:.4f}")

        for _ in range(num_epochs_predictors):
            loss_predictors += train_predictors(train_loader, vae_model, predictors, optimizer_predictors, delta, device)

            loss_predictors /=num_epochs_predictors
        for param in vae_model.parameters():
            param.requires_grad = True

        annealing_agent_beta.step()
        annealing_agent_delta.step()
        scheduler_vae.step()
        for scheduler in schedulers_predictors:
            scheduler.step()



        print(f'Epoch {epoch + 1}/{num_epochs}')

        if log_tensorboard:
            log_training_info(writer, epoch, loss_vae, loss_total, loss_predictors, kld, delta, current_beta)
            log_z_distributions(vae_model, predictors, train_loader, writer, epoch, device)
            if data_to_plot is not None and epoch%10==0:
                with torch.no_grad():
                    vae_model.eval()
                    # data_to_plot= generate_data_to_plot()
                    #generate random in in 0 to 1999
                    idx = math.floor(random()*2000)
                    data_to_plot = next(iter(train_loader))[idx]
                    plots = generate_plots_modifications(vae_model,device=device)
                for i, fig in enumerate(plots):
                    writer.add_figure(f"Latent Variable {i + 1} Modification", fig, global_step=epoch)

        #save the model
        torch.save(vae_model.state_dict(), r"C:\Users\User\Desktop\thesis\vae\saved models\vae.pth")
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss Total: {loss_total:.4f}, Loss AE: {loss_vae:.4f}, Loss predictors: {loss_predictors:.4f}, KLD: {kld:.4f}, Beta: {current_beta:.5f}, Delta: {delta:.4f}')

    if log_tensorboard:
        writer.close()
    return


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_dotenv()
    args = get_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    train_loader = get_dataloader(r"C:\Users\User\Desktop\thesis\data\toy_dataset_train.csv", batch_size=3000)
    val_loader = get_dataloader(args.val_data_path, batch_size=16384)

    vae_model, predictors = setup_models(device)
    optimizer_vae, optimizers_predictors = setup_optimizers(vae_model, predictors)
    scheduler_vae, schedulers_predictors = setup_schedulers(optimizer_vae, optimizers_predictors)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(r'C:\Users\User\Desktop\thesis\vae\runs_vae',
                           f"{timestamp}_{args.experiment_name}" if args.experiment_name else timestamp)

    data_to_plot = np.load(r'C:\Users\User\Desktop\thesis\data\data_to_plot.npy')

    vae_model, predictors, history = train_models(
        train_loader, val_loader, vae_model, predictors,
        optimizer_vae, optimizers_predictors,
        num_epochs=10000, log_dir=log_dir, delta=2, beta_max=0,
        log_tensorboard=True, do_eval=args.do_eval, data_to_plot=data_to_plot, device=device,
        scheduler_vae=scheduler_vae, schedulers_predictors=schedulers_predictors)



if __name__ == "__main__":
    main()