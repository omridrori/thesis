import math

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


def generate_estimated_z(predictors, z):
    z_copy = z.clone().cpu()
    z_estimated = [predictors[0](z_copy[:, 1:])]  # Predict z0 using z1 to z16

    for i in range(1, z.shape[1]):
        z_input = z_copy[:, :i]  # Use original z values up to index i
        z_pred = predictors[i](z_input)
        z_estimated.append(z_pred)

    return torch.cat(z_estimated, dim=1)
def predictor_task(predictor, z_input, z_val):
    z_pred = predictor(z_input)
    return (z_pred - z_val) ** 2

def loss_predict_with_predictors(predictors, data, encoded_normalized):
    z = encoded_normalized.cpu()
    val_predictors_losses_individuals = torch.zeros(data.size(0))

    # Create a ThreadPoolExecutor with 5 workers
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit tasks for each predictor
        future_to_predictor = {}
        for i in range(1, len(predictors)):
            future = executor.submit(predictor_task, predictors[i-1], z[:, :i], z[:, i].unsqueeze(1))
            future_to_predictor[future] = i

        # Collect results as they complete
        for future in as_completed(future_to_predictor):
            result = future.result()
            val_predictors_losses_individuals += result.squeeze()

    return val_predictors_losses_individuals

def predictors_step(train_loader,vae, predictors, optimizer_predictors, delta,device):

    for predictor in predictors:
        predictor.train()

    loss_predictors_total = 0.0

    for batch_idx, data in enumerate(train_loader):

        for optimizer in optimizer_predictors:
            optimizer.zero_grad()

        data = data.to(device)
        # Forward pass through autoencoder
        encoded, decoded, mu, logvar = vae_model(data)

        # encoded_normalized = F.normalize(encoded, p=2, dim=1)

        z_estimated = generate_estimated_z(predictors, encoded_normalized).to(device)
        z_estimated_normalized = F.normalize(z_estimated, p=2, dim=1)
        decoded_estimated = vae_model.decoder(z_estimated_normalized)
        # normalized_decoded_estimated = F.normalize(decoded_estimated, p=2, dim=1)
        # normalized_data=F.normalize(data, p=2, dim=1)

        loss_predictors = F.mse_loss(decoded_estimated, data)

        loss_predictors_total += loss_predictors.item()

        loss_predictors.backward()

        for optimizer in optimizer_predictors:
            optimizer.step()



    return loss_predictors_total/ len(train_loader)

def vae_step(train_loader, vae, predictors, optimizer_vae, delta,annealing_agent,beta_max,device):
    vae.train()

    running_loss_vae = 0.0
    running_loss_total = 0.0
    runing_kld=0.0
    predictors_loss_individual_examples_mean = 0.0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        optimizer_vae.zero_grad()
        data = data.to(device)
        # Forward pass through autoencoder
        encoded, decoded, mu, logvar = vae_model(data)

        encoded_normalized_for_predictors = F.normalize(encoded, p=2, dim=1).clone().detach()
        encoded_normalized = F.normalize(encoded, p=2, dim=1)
        encoded = encoded_normalized.clone()
        decoded = vae_model.decoder(encoded)

        z_estimated = generate_estimated_z(predictors, encoded_normalized).to(device)
        z_estimated_normalized = F.normalize(z_estimated, p=2, dim=1)

        decoded_estimated = vae_model.decoder(z_estimated_normalized)

        # normalized_decoded_estimated = F.normalize(decoded_estimated, p=2, dim=1)
        # normalized_data=F.normalize(data, p=2, dim=1)

        predictors_loss_individual_examples =F.mse_loss(decoded_estimated, data, reduction='none').mean(axis=1)
        # predictors_loss_individual_examples =torch.exp(predictors_loss_individual_examples/0.1)
        # predictors_loss_individual_examples = predictors_loss_individual_examples / len(predictors)
        # Calculate total loss for each example
        loss_vae_individual, mse_mean, kld_mean = loss_function_individual(decoded, data, mu, logvar, annealing_agent,
                                                                           beta_max)
        total_loss_individual = loss_vae_individual - delta * predictors_loss_individual_examples.to(device)

        running_loss_vae += mse_mean
        runing_kld += kld_mean
        total_loss = total_loss_individual.mean()
        predictors_loss_individual_examples_mean += predictors_loss_individual_examples.mean()

        running_loss_total += total_loss.item()

        total_loss.backward()


        optimizer_vae.step()


    return running_loss_vae / len(train_loader), running_loss_total / len(train_loader),runing_kld / len(train_loader), predictors_loss_individual_examples_mean / len(train_loader)

def loss_function(recon_x, x, mu, logvar,beta=0.1):
    # Reconstruction loss (Mean Squared Error)
    MSE = nn.MSELoss(reduction='sum')(recon_x, x)

    # KL divergence
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

    return MSE,KLD

def loss_function_individual(recon_x, x, mu, logvar, annealing_agent,beta_max):
    # Reconstruction loss (Mean Squared Error) for each example
    # recon_x_normalized = F.normalize(recon_x, p=2, dim=1)
    # x_normalized = F.normalize(x, p=2, dim=1)
    MSE = F.mse_loss(recon_x, x, reduction='none').mean(axis=1)

    # KL divergence for each example
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    KLD_add=annealing_agent(KLD)*beta_max
    # Total loss for each example, using the current beta value
    individual_loss = MSE + KLD_add

    return individual_loss, MSE.detach().mean(), KLD.detach().mean()


def log_z_distribution(vae_model, train_loader, writer, epoch, device):
    vae_model.eval()
    z_values = []

    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            encoded, _, _, _ = vae_model(batch)
            z_values.append(encoded)

    z_values = torch.cat(z_values, dim=0)

    for i in range(z_values.shape[1]):
        writer.add_histogram(f'z_distribution/dim_{i}', z_values[:, i], epoch)

    # Optional: log overall statistics
    writer.add_scalar('z_distribution/mean', z_values.mean(), epoch)
    writer.add_scalar('z_distribution/std', z_values.std(), epoch)

    vae_model.train()


def train_models(train_loader, val_loader, vae_model, predictors, criterion,
                 optimizer_vae, optimizer_predictors, scheduler_ae, schedulers, num_epochs, log_dir, delta=0.1, beta_vae=0.1,
                 log_tensorboard=False, do_eval=False, data_to_plot=None, device="cpu", beta_max=None):

    vae_model.to(device)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(r'C:\Users\User\Desktop\thesis\vae\runs_vae', f"{timestamp}")


    if log_tensorboard:
        writer = SummaryWriter(log_dir=log_dir)
    mean_train_losses_predictors = 0.0
    mean_train_loss_total = 0.0
    history = []
    total_steps = 50
    annealing_agent_beta  = Annealer(total_steps=50, shape='cosine', baseline=0.0, cyclical=True)
    annealing_agent_delta = Annealer(total_steps=50, shape='cosine', baseline=0.0, cyclical=True)
    factor=0.1
    for epoch in range(num_epochs):
        vae_model.train()
        for predictor in predictors:
            predictor.train()

        # if (epoch + 1) % 80 == 0:
        #     turning= not annealing_agent_beta.cyclical
        #     annealing_agent_beta.cyclical_setter(not annealing_agent_beta.cyclical)
        #     print(f"Switched annealing to {'cyclical' if annealing_agent_beta.cyclical else 'non-cyclical'} at epoch {epoch + 1}")
        #     annealing_agent_beta.current_step=0
        #

        # current_beta = beta_max * annealing_agent.slope()
        delta= annealing_agent_delta.slope()*factor
        factor= factor*0.999
        current_beta= annealing_agent_beta.slope()*beta_max







        # Unfreeze latent predictor parameters
        for param in vae_model.parameters():
            param.requires_grad = True
        for predictor in predictors:
            for param in predictor.parameters():
                param.requires_grad = False


        for i in range(1):
            mean_train_loss_vae, mean_train_loss_total, mean_kld ,predictors_loss_in_vae = vae_step(train_loader, vae_model, predictors, optimizer_vae, delta, annealing_agent_beta, beta_max, device)
            #            print(f" mean_train_loss_vae: {mean_train_loss_vae}, mean_train_loss_total: {mean_train_loss_total}, mean_kld: {mean_kld}, predictors_loss_in_vae: {predictors_loss_in_vae}")
            # print(f"  mean_train_loss_total: {mean_train_loss_total :.4f},mean_train_loss_vae: {mean_train_loss_vae :.4f}, mean_kld: {mean_kld :.4f}, predictors_loss_in_vae: {predictors_loss_in_vae :.4f}")

        # Unfreeze latent predictor parameters
        for param in vae_model.parameters():
            param.requires_grad = False
        for predictor in predictors:
            for param in predictor.parameters():
                param.requires_grad = True
        # if mean_train_losses_predictors > 5 * abs(mean_train_loss_vae + current_beta*mean_kld):
        #     number_of_epochs_predictors = 1
        # else:
        #     number_of_epochs_predictors = 1
        #
        # # if epoch>10:
        # #     number_of_epochs_predictors = 10
        # #
        # # else :
        # #     number_of_epochs_predictors = 1
        for i in range(2):
            mean_train_losses_predictors = predictors_step(train_loader, vae_model, predictors, optimizer_predictors, delta,
                                                       device)
        # while (mean_train_losses_predictors > 5 * abs(mean_train_loss_vae + 0*current_beta*mean_kld)):
        #     mean_train_losses_predictors = predictors_step(train_loader, vae_model, predictors, optimizer_predictors, delta, device)
        #     print(f"mean_train_losses_predictors: {mean_train_losses_predictors:.4f}")





        annealing_agent_beta.step()
        annealing_agent_delta.step()

        current_lr = scheduler_vae.get_last_lr()[0]
        if current_lr > 0.000008:
            scheduler_vae.step()

            for optimizer in optimizer_predictors:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr*5




        current_lr = scheduler_vae.get_last_lr()[0]


        print(f'Epoch {epoch + 1}/{num_epochs}, Current Learning Rate: {current_lr:.4f}')


        if log_tensorboard:

            for name, param in vae_model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'VAE_gradients/{name}', param.grad, epoch)

                # Visualize predictor gradients
            for i, predictor in enumerate(predictors):
                for name, param in predictor.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'Predictor_{i}_gradients/{name}', param.grad, epoch)

                # Optionally, you can also log the norm of the gradients
            vae_total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in vae_model.parameters() if p.grad is not None]), 2)
            writer.add_scalar('Gradient_norm/VAE', vae_total_norm, epoch)

            for i, predictor in enumerate(predictors):
                predictor_total_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in predictor.parameters() if p.grad is not None]),
                    2)
                writer.add_scalar(f'Gradient_norm/Predictor_{i}', predictor_total_norm, epoch)

            # Log the mean training loss
            writer.add_scalar('Loss/train_ae', mean_train_loss_vae, epoch)
            writer.add_scalar('Loss/train_total', mean_train_loss_total, epoch)
            writer.add_scalar('Loss/train_predictors', mean_train_losses_predictors, epoch)
            writer.add_scalar('Loss/train_kld', mean_kld, epoch)
            writer.add_scalar('Hyperparameters/beta', delta, epoch)
            log_z_distribution(vae_model, train_loader, writer, epoch, device)
            if data_to_plot is not None and epoch%1==0:
                with torch.no_grad():
                    vae_model.eval()
                    data_to_plot= generate_data_to_plot()
                    plots = generate_plots_modifications(vae_model, data_to_plot,device=device)
                for i, fig in enumerate(plots):
                    writer.add_figure(f"Latent Variable {i + 1} Modification", fig, global_step=epoch)
        epoch_metrics = {
            'train_loss_ae': mean_train_loss_vae,
            'train_loss_total': mean_train_loss_total,
            'train_loss_predictors': mean_train_losses_predictors,
            delta:': current_beta'
        }



        history.append(epoch_metrics)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss Total: {mean_train_loss_total:.4f}, Loss AE: {mean_train_loss_vae:.4f}, Loss predictors: {mean_train_losses_predictors:.4f}, KLD: {mean_kld:.4f}, Beta: {current_beta:.5f}, Delta: {delta:.4f}')


    if log_tensorboard:
        writer.close()
    return vae_model, predictors, history


# Example of how to call train_models in your main script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_dotenv()
    args = get_args()
    # Print all the argument values
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    torch.autograd.set_detect_anomaly(True)
    train_loader = get_dataloader(r"C:\Users\User\Desktop\thesis\data\toy_dataset_train.csv",
                                  batch_size=5000)
    val_loader = get_dataloader(args.val_data_path,
                                batch_size=16384)

    vae_model = VAE()
    predictors = [LatentPredictor_x0(),LatentPredictor_x2(), LatentPredictor_x3(), LatentPredictor_x4(), LatentPredictor_x5(),
                   LatentPredictor_x6(), LatentPredictor_x7(), LatentPredictor_x8(), LatentPredictor_x9(),
                   LatentPredictor_x10(), LatentPredictor_x11(), LatentPredictor_x12(), LatentPredictor_x13(),
                   LatentPredictor_x14(), LatentPredictor_x15(), LatentPredictor_x16()]

    initializer = get_initializer("orthogonal")
    vae_model.apply(initializer)
    for predictor in predictors:
        predictor.apply(initializer)

    criterion = nn.MSELoss()

    optimizer_vae = optim.AdamW(vae_model.parameters(), lr=0.005,weight_decay=0.001)
    learning_rate_predictors =0.0001
    optimizers_predictors = [optim.Adam(predictor.parameters(), lr=learning_rate_predictors) for predictor in predictors]





    scheduler_vae = lr_scheduler.StepLR(optimizer_vae, step_size=10, gamma=0.9)

    # Cosine Annealing scheduler

    # schedulers = [lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) for optimizer in optimizers]
    schedulers=[]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(r'C:\Users\User\Desktop\thesis\vae\runs_vae', f"{timestamp}_{args.experiment_name}" if args.experiment_name else timestamp)

    data_to_plot= np.load(r'C:\Users\User\Desktop\thesis\data\data_to_plot.npy')

    train_models(train_loader, val_loader, vae_model,
                 predictors, criterion, optimizer_vae, optimizers_predictors,
                 scheduler_vae, schedulers, num_epochs=10000,
                 log_dir=log_dir, delta=0.001, beta_vae=0.00001,beta_max = 0.000001,
                 log_tensorboard=True, do_eval=args.do_eval,data_to_plot=data_to_plot,device=device)

    save_models(vae_model, predictors, path=r"/vae/saved models")

