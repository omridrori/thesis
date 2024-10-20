import numpy as np
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.save_models import save_models
from vae.scripts.models.variational_autoencoder import VAE
from vae.scripts.models.initializations import get_initializer

from vae.scripts.models.variational_latent_predictors import LatentPredictor_x2, LatentPredictor_x3, LatentPredictor_x4, LatentPredictor_x5, \
    LatentPredictor_x6, LatentPredictor_x7, LatentPredictor_x8, LatentPredictor_x9, LatentPredictor_x10, \
    LatentPredictor_x11, LatentPredictor_x12, LatentPredictor_x13, LatentPredictor_x14, LatentPredictor_x15, \
    LatentPredictor_x16
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

def predictors_step(train_loader,autoencoder, predictors, optimizers, delta,device):
    autoencoder.eval()
    for predictor in predictors:
        predictor.train()

    loss_predictors_total = 0.0
    for batch_idx, data in enumerate(train_loader):
        data=data.to(device)
        for optimizer in optimizers:
            optimizer.zero_grad()
        encoded = autoencoder.encoder(data)
        encoded_normalized = F.normalize(encoded, p=2, dim=1)
        val_predictors_losses_individuals = loss_predict_with_predictors(predictors, data, encoded_normalized)
        losses_individuals = val_predictors_losses_individuals / len(predictors)
        loss_predictors = losses_individuals.mean()
        loss_predictors.backward()
        for optimizer in optimizers:
            optimizer.step()

        loss_predictors_total += loss_predictors.item()

    return loss_predictors_total/ len(train_loader)

def autoencoder_step(train_loader, autoencoder, predictors, optimizer_ae, delta,beta,device):
    autoencoder.train()

    running_loss_ae = 0.0
    running_loss_total = 0.0
    runnning_loss_kld=0.0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer_ae.zero_grad()
        # Forward pass through autoencoder
        encoded,decoded, mu, logvar = autoencoder(data)
        # Normalize the encoded vector
        encoded_normalized = F.normalize(encoded, p=2, dim=1)
        decoded = autoencoder.decoder(encoded)

        val_predictors_losses_individuals = loss_predict_with_predictors(predictors, data, encoded_normalized)
        losses_individuals = val_predictors_losses_individuals / len(predictors)
        # Calculate total loss for each example
        total_loss_individual,mse_mean,kld_mean = loss_function_individual(decoded,data,mu,logvar,beta) - delta * losses_individuals.to(device)
        total_loss= total_loss_individual.mean()

        total_loss.backward()
        optimizer_ae.step()

        running_loss_total += total_loss.item()
        running_loss_ae+= mse_mean
        runnning_loss_kld+= kld_mean
        # with torch.no_grad():
        #    mse,kld = loss_function(decoded,data,mu,logvar,beta)
        #    mse_loss = mse.item()/len(data)
        #    kld_loss =kld.item()/len(data)
        #    running_loss_ae += mse_loss
        #    runnning_loss_kld += kld_loss

    return running_loss_total/ len(train_loader), running_loss_ae/ len(train_loader) ,runnning_loss_kld/ len(train_loader)

def loss_function(recon_x, x, mu, logvar,beta=0.1):
    # Reconstruction loss (Mean Squared Error)
    MSE = nn.MSELoss(reduction='sum')(recon_x, x)

    # KL divergence
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

    return MSE,KLD

def loss_function_individual(recon_x, x, mu, logvar, annealing_agent,beta_max):
    # Reconstruction loss (Mean Squared Error) for each example
    MSE = F.mse_loss(recon_x, x, reduction='none').mean(axis=1)

    # KL divergence for each example
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    KLD_add=annealing_agent(KLD)*beta_max
    # Total loss for each example, using the current beta value
    individual_loss = MSE + KLD_add

    return individual_loss, MSE.detach().mean(), KLD.detach().mean()


def train_models(train_loader, val_loader, vae_model, predictors, criterion,
                 optimizer_vae, optimizer_predictors, scheduler_ae, schedulers, num_epochs, log_dir, delta=0.1, beta_vae=0.1,
                 log_tensorboard=False, do_eval=False, data_to_plot=None, device="cpu", beta_max=None):

    vae_model.to(device)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(r'C:\Users\User\Desktop\thesis\vae\runs_vae', f"{timestamp}")


    if log_tensorboard:
        writer = SummaryWriter(log_dir=log_dir)

    history = []
    total_steps = 20
    annealing_agent  = Annealer(total_steps=total_steps, shape='cosine', baseline=0.0, cyclical=True)
    for epoch in range(num_epochs):
        vae_model.train()
        for predictor in predictors:
            predictor.train()

        if (epoch + 1) % 80 == 0:
            turning= not annealing_agent.cyclical
            annealing_agent.cyclical_setter(not annealing_agent.cyclical)
            print(f"Switched annealing to {'cyclical' if annealing_agent.cyclical else 'non-cyclical'} at epoch {epoch + 1}")
            annealing_agent.current_step=0


        current_beta = beta_max * annealing_agent.slope()




        runing_kld =0.0
        running_loss_ae = 0.0
        running_loss_total = 0.0
        loss_predictors_total = 0.0
        for batch_idx, data in enumerate(train_loader):


            for optimizer in optimizer_predictors:
                optimizer.zero_grad()

            optimizer_vae.zero_grad()
            data = data.to(device)
            # Forward pass through autoencoder
            encoded, decoded, mu, logvar = vae_model(data)
            # Normalize the encoded vector
            #todo normelized also for decoder
            encoded_normalized_for_predictors = F.normalize(encoded, p=2, dim=1).clone().detach()
            encoded_normalized = F.normalize(encoded, p=2, dim=1)
            encoded= encoded_normalized.clone()
            decoded = vae_model.decoder(encoded)

            val_predictors_losses_individuals = loss_predict_with_predictors(predictors, data, encoded_normalized)
            val_predictors_losses_individuals_for_prdictors = loss_predict_with_predictors(predictors, data, encoded_normalized_for_predictors)
            losses_individuals = val_predictors_losses_individuals / len(predictors)
            losses_individuals_for_predictors = val_predictors_losses_individuals_for_prdictors / len(predictors)
            # Calculate total loss for each example
            loss_vae_individual,mse_mean,kld_mean = loss_function_individual(decoded, data, mu, logvar,annealing_agent,beta_max)
            total_loss_individual = loss_vae_individual - delta * losses_individuals.to(device)

            running_loss_ae += mse_mean
            runing_kld += kld_mean
            total_loss = total_loss_individual.mean()

            loss_predictors = losses_individuals_for_predictors.mean()

            running_loss_total += total_loss.item()

            # with torch.no_grad():
            #     mse, kld = loss_function(decoded, data, mu, logvar, beta_vae)
            #     mse_loss = mse.item() / len(data)
            #     running_loss_ae += mse_loss
            #     runing_kld += kld.item() / len(data)

            loss_predictors_total += loss_predictors.item()

            # Unfreeze latent predictor parameters
            for param in vae_model.parameters():
                param.requires_grad = True
            for predictor in predictors:
                for param in predictor.parameters():
                    param.requires_grad = False

            total_loss.backward(retain_graph=True)

            # Unfreeze latent predictor parameters
            for param in vae_model.parameters():
                param.requires_grad = False
            for predictor in predictors:
                for param in predictor.parameters():
                    param.requires_grad = True

            loss_predictors.backward()

            for param in vae_model.parameters():
                param.requires_grad = True

            optimizer_vae.step()
            for optimizer in optimizer_predictors:
                optimizer.step()

        annealing_agent.step()

        mean_kld = runing_kld / len(train_loader)
        mean_train_loss_ae = running_loss_ae / len(train_loader)
        mean_train_loss_total = running_loss_total / len(train_loader)
        mean_train_losses_predictors = loss_predictors_total / len(train_loader)











        if log_tensorboard:
            # Log the mean training loss
            writer.add_scalar('Loss/train_ae', mean_train_loss_ae, epoch)
            writer.add_scalar('Loss/train_total', mean_train_loss_total, epoch)
            writer.add_scalar('Loss/train_predictors', mean_train_losses_predictors, epoch)
            writer.add_scalar('Loss/train_kld', mean_kld, epoch)
            writer.add_scalar('Hyperparameters/beta', current_beta, epoch)
            if data_to_plot is not None and epoch%30==0:
                plots = generate_plots_modifications(vae_model, data_to_plot,device=device)
                for i, fig in enumerate(plots):
                    writer.add_figure(f"Latent Variable {i + 1} Modification", fig, global_step=epoch)
        epoch_metrics = {
            'train_loss_ae': mean_train_loss_ae,
            'train_loss_total': mean_train_loss_total,
            'train_loss_predictors': mean_train_losses_predictors,
            current_beta:': current_beta'
        }

        if do_eval:
            # Evaluate on validation set
            vae_model.eval()
            for predictor in predictors:
                predictor.eval()
            val_loss_ae = 0.0
            val_loss_total = 0.0
            val_loss_predictors = 0.0
            with torch.no_grad():
                for data in val_loader:
                    mean_train_loss_ae, mean_train_loss_total,kld_loss = autoencoder_step(train_loader, vae_model, predictors,
                                                                                 optimizer_ae, delta,beta_vae, device)
                    val_loss_ae += mean_train_loss_ae
                    val_loss_total += mean_train_loss_total
                    val_loss_predictors += predictors_step(train_loader, vae_model, predictors, optimizers_predictors,
                                                           delta, device)
                    Val_loss_predictors = val_loss_predictors / len(val_loader)

                mean_val_loss_ae = val_loss_ae / len(val_loader)
                mean_val_loss_total = val_loss_total / len(val_loader)
                mean_val_loss_predictors = val_loss_predictors / len(val_loader)

                # Log the mean validation loss
                if log_tensorboard:
                    writer.add_scalar('Loss/val_ae', mean_val_loss_ae, epoch)
                    writer.add_scalar('Loss/val_total', mean_val_loss_total, epoch)
                    writer.add_scalar('Loss/val_predictors', mean_val_loss_predictors, epoch)



            epoch_metrics.update({
                'val_loss_ae': mean_val_loss_ae,
                'val_loss_total': mean_val_loss_total,
                'val_loss_predictors': mean_val_loss_predictors
            })
            # # Step the learning rate scheduler
            # scheduler_ae.step()
            # for scheduler in schedulers:
            #     scheduler.step()
        history.append(epoch_metrics)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss Total: {mean_train_loss_total:.4f}, Loss AE: {mean_train_loss_ae:.4f}, Loss predictors: {mean_train_losses_predictors:.4f}, KLD: {mean_kld:.4f}, Beta: {current_beta:.4f}')

        if do_eval:
            print(
                f' \n Val Loss Total: {mean_val_loss_total:.4f}, Val Loss AE: {mean_val_loss_ae:.4f}, Val loss_predictors: {mean_val_loss_predictors:.4f}')

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
    train_loader = get_dataloader(args.train_data_path,
                                  batch_size=16384)
    val_loader = get_dataloader(args.val_data_path,
                                batch_size=16384)

    vae_model = VAE()
    predictors = [LatentPredictor_x2(), LatentPredictor_x3(), LatentPredictor_x4(), LatentPredictor_x5(),
                   LatentPredictor_x6(), LatentPredictor_x7(), LatentPredictor_x8(), LatentPredictor_x9(),
                   LatentPredictor_x10(), LatentPredictor_x11(), LatentPredictor_x12(), LatentPredictor_x13(),
                   LatentPredictor_x14(), LatentPredictor_x15(), LatentPredictor_x16()]

    initializer = get_initializer("orthogonal")
    vae_model.apply(initializer)
    for predictor in predictors:
        predictor.apply(initializer)

    criterion = nn.MSELoss()

    optimizer_ae = optim.Adam(vae_model.parameters(), lr=0.0001)
    learning_rate_predictors =0.0001
    optimizers_predictors = [optim.Adam(predictor.parameters(), lr=learning_rate_predictors) for predictor in predictors]
    scheduler_ae = lr_scheduler.StepLR(optimizer_ae, step_size=args.step_size, gamma=args.gamma)
    # schedulers = [lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) for optimizer in optimizers]
    schedulers=[]


    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(r'C:\Users\User\Desktop\thesis\vae\runs_vae', f"{timestamp}_{args.experiment_name}" if args.experiment_name else timestamp)

    data_to_plot= np.load(r'C:\Users\User\Desktop\thesis\data\data_to_plot.npy')

    train_models(train_loader, val_loader, vae_model,
                 predictors, criterion, optimizer_ae, optimizers_predictors,
                 scheduler_ae, schedulers, num_epochs=400,
                 log_dir=log_dir, delta=10, beta_vae=0.00001,beta_max = 0.5,
                 log_tensorboard=True, do_eval=args.do_eval,data_to_plot=data_to_plot,device=device)

    save_models(vae_model, predictors, path=r"C:\Users\User\Desktop\thesis\vae\saved models")

