import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.autoencoder import Autoencoder
from models.initializations import constant_init, normal_init, kaiming_uniform_init, xavier_uniform_init, \
    orthogonal_init
from models.latent_predictor import LatentPredictor
from utils.data_loader import get_dataloader
from utils.parse_args import get_args
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F


def get_initializer(method):
    if method == 'xavier':
        return xavier_uniform_init
    elif method == 'kaiming':
        return kaiming_uniform_init
    elif method == 'normal':
        return normal_init
    elif method == 'orthogonal':
        return orthogonal_init
    elif method == 'constant':
        return lambda m: constant_init(m, value=0.1)


def train_models(train_loader, val_loader, autoencoder, latent_predictor, criterion, optimizer_ae, optimizer_lp,
                 scheduler_ae, scheduler_lp, num_epochs, log_dir, delta=0.1):
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        autoencoder.train()
        latent_predictor.train()
        running_loss_ae = 0.0
        running_loss_lp = 0.0
        running_loss_total = 0.0

        for batch_idx, data in enumerate(train_loader):
            optimizer_ae.zero_grad()
            optimizer_lp.zero_grad()

            encoded = autoencoder.encoder(data)
            decoded = autoencoder.decoder(encoded)

            encoded_normalized = F.normalize(encoded, p=2, dim=1)
            z1_z2_z3 = encoded_normalized[:, :3]
            z4 = encoded_normalized[:, 3]

            # Forward pass through latent predictor
            z4_pred = latent_predictor(z1_z2_z3)
            # norm_z1_z2_z3 = torch.sum(z1_z2_z3 ** 2, dim=1, keepdim=True)
            # Calculate the squared norm of the difference
            lp_loss = (z4_pred - z4.unsqueeze(1)) ** 2
            lp_loss_mean = lp_loss.mean()

            # Calculate the reconstruction loss
            ae_loss = torch.mean((decoded - data) ** 2)

            # Normalize by the squared norm of (z1, z2, z3)
            normalized_loss_lp = lp_loss

            # Calculate total loss for each example
            total_loss_individual = torch.mean((decoded - data) ** 2, dim=1) - delta * normalized_loss_lp.squeeze()
            total_loss = total_loss_individual.mean()

            # Backward pass
            # Freeze latent predictor parameters
            for param in latent_predictor.parameters():
                param.requires_grad = False

            total_loss.backward(retain_graph=True)

            # Unfreeze latent predictor parameters
            for param in latent_predictor.parameters():
                param.requires_grad = True

            for param in autoencoder.parameters():
                param.requires_grad = False

            lp_loss_mean.backward()

            for param in autoencoder.parameters():
                param.requires_grad = True

            optimizer_ae.step()
            optimizer_lp.step()

            running_loss_ae += ae_loss.item()
            running_loss_lp += lp_loss_mean.item()
            running_loss_total += total_loss.item()

        # Compute mean training loss
        mean_train_loss_ae = running_loss_ae / len(train_loader)
        mean_train_loss_lp = running_loss_lp / len(train_loader)
        mean_train_loss_total = running_loss_total / len(train_loader)

        # Log the mean training loss
        writer.add_scalar('Loss/train_ae', mean_train_loss_ae, epoch)
        writer.add_scalar('Loss/train_lp', mean_train_loss_lp, epoch)
        writer.add_scalar('Loss/train_total', mean_train_loss_total, epoch)

        # Evaluate on validation set
        autoencoder.eval()
        latent_predictor.eval()
        val_loss_ae = 0.0
        val_loss_lp = 0.0
        val_loss_total = 0.0
        with torch.no_grad():
            for data in val_loader:
                encoded = autoencoder.encoder(data)
                decoded = autoencoder.decoder(encoded)

                encoded_normalized = F.normalize(encoded, p=2, dim=1)



                ae_loss = torch.sum((decoded - data) ** 2, dim=1).mean()

                z1_z2_z3 = encoded_normalized[:, :3]
                z4 = encoded_normalized[:, 3]
                norm_z1_z2_z3 = torch.mean(z1_z2_z3 ** 2, dim=1, keepdim=True)
                z4_pred = latent_predictor(z1_z2_z3)
                lp_loss = (z4_pred - z4.unsqueeze(1)) ** 2
                lp_loss_mean = lp_loss.mean()


                normalized_loss_lp = lp_loss
                total_loss_individual = torch.sum((decoded - data) ** 2, dim=1) - delta * normalized_loss_lp.squeeze()
                total_loss = total_loss_individual.mean()

                val_loss_ae += ae_loss.item()
                val_loss_lp += lp_loss_mean.item()
                val_loss_total += total_loss.item()

        # Compute mean validation loss
        mean_val_loss_ae = val_loss_ae / len(val_loader)
        mean_val_loss_lp = val_loss_lp / len(val_loader)
        mean_val_loss_total = val_loss_total / len(val_loader)

        # Log the mean validation loss
        writer.add_scalar('Loss/val_ae', mean_val_loss_ae, epoch)
        writer.add_scalar('Loss/val_lp', mean_val_loss_lp, epoch)
        writer.add_scalar('Loss/val_total', mean_val_loss_total, epoch)

        # Step the learning rate scheduler
        scheduler_ae.step()
        scheduler_lp.step()

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss AE: {mean_train_loss_ae:.4f}, Val Loss AE: {mean_val_loss_ae:.4f}, Train Loss LP: {mean_train_loss_lp:.4f}, Val Loss LP: {mean_val_loss_lp:.4f}, Train Loss Total: {mean_train_loss_total:.4f}, Val Loss Total: {mean_val_loss_total:.4f}')

    writer.close()


if __name__ == "__main__":
    args = get_args()

    train_loader = get_dataloader(args.data_path.replace('toy_dataset.csv', 'toy_dataset_train.csv'),
                                  batch_size=args.batch_size)
    val_loader = get_dataloader(args.data_path.replace('toy_dataset.csv', 'toy_dataset_val.csv'),
                                batch_size=args.batch_size)

    autoencoder = Autoencoder()
    latent_predictor = LatentPredictor()

    initializer = get_initializer(args.init_method)
    autoencoder.apply(initializer)

    criterion = nn.MSELoss()

    optimizer_ae = optim.Adam(autoencoder.parameters(), lr=args.learning_rate)
    optimizer_lp = optim.Adam(latent_predictor.parameters(), lr=args.learning_rate)

    scheduler_ae = lr_scheduler.StepLR(optimizer_ae, step_size=args.step_size, gamma=args.gamma)
    scheduler_lp = lr_scheduler.StepLR(optimizer_lp, step_size=args.step_size, gamma=args.gamma)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('./runs', f"{timestamp}_{args.experiment_name}" if args.experiment_name else timestamp)
    train_models(train_loader, val_loader, autoencoder, latent_predictor, criterion, optimizer_ae, optimizer_lp, scheduler_ae, scheduler_lp, num_epochs=args.num_epochs,delta=args.delta, log_dir=log_dir)

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(autoencoder.state_dict(), args.model_path)
    torch.save(latent_predictor.state_dict(), args.model_path.replace('autoencoder', 'latent_predictor'))
    print(
        f"Models trained and saved to '{args.model_path}' and '{args.model_path.replace('autoencoder', 'latent_predictor')}'.")