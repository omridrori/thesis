import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.autoencoder import Autoencoder
from models.initializations import constant_init, normal_init, kaiming_uniform_init, xavier_uniform_init, \
    orthogonal_init
from models.latent_predictor import LatentPredictor_x2, LatentPredictor_x3, LatentPredictor_x4
from utils.data_loader import get_dataloader
from utils.parse_args import get_args
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F


# Assume LatentPredictor_x2, LatentPredictor_x3, LatentPredictor_x4 are defined similarly to LatentPredictor_x4

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


def train_models(train_loader, val_loader, autoencoder, predictor_x2, predictor_x3, predictor_x4, criterion,
                 optimizer_ae, optimizer_x2, optimizer_x3, optimizer_x4, scheduler_ae, scheduler_x2, scheduler_x3,
                 scheduler_x4, num_epochs, log_dir, delta=0.1, l2_lambda=1e-5, clip_value=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        autoencoder.train()
        predictor_x2.train()
        predictor_x3.train()
        predictor_x4.train()
        running_loss_ae = 0.0
        running_loss_x2 = 0.0
        running_loss_x3 = 0.0
        running_loss_x4 = 0.0
        running_loss_total = 0.0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer_ae.zero_grad()
            optimizer_x2.zero_grad()
            optimizer_x3.zero_grad()
            optimizer_x4.zero_grad()

            # Forward pass through autoencoder
            encoded = autoencoder.encoder(data)
            # Normalize the encoded vector
            encoded = F.normalize(encoded, p=2, dim=1)
            decoded = autoencoder.decoder(encoded)

            z1 = encoded[:, :1].cpu()
            z1_z2 = encoded[:, :2].cpu()
            z1_z2_z3 = encoded[:, :3].cpu()
            z2 = encoded[:, 1].cpu()
            z3 = encoded[:, 2].cpu()
            z4 = encoded[:, 3].cpu()

            # Forward pass through latent predictors
            z2_pred = predictor_x2(z1)
            z3_pred = predictor_x3(z1_z2)
            z4_pred = predictor_x4(z1_z2_z3)

            # Calculate the squared norm of the differences
            loss_x2 = (z2_pred - z2.unsqueeze(1)) ** 2
            loss_x2_mean = loss_x2.mean()
            loss_x3 = (z3_pred - z3.unsqueeze(1)) ** 2
            loss_x3_mean = loss_x3.mean()
            loss_x4 = (z4_pred - z4.unsqueeze(1)) ** 2
            loss_x4_mean = loss_x4.mean()

            # Calculate the reconstruction loss using criterion
            ae_loss = criterion(decoded, data)

            # Normalize by the squared norm of the respective predictors
            norm_z1 = torch.sum(z1 ** 2, dim=1, keepdim=True)
            norm_z1_z2 = torch.sum(z1_z2 ** 2, dim=1, keepdim=True)
            norm_z1_z2_z3 = torch.sum(z1_z2_z3 ** 2, dim=1, keepdim=True)
            normalized_loss_x2 = loss_x2
            normalized_loss_x3 = loss_x3
            normalized_loss_x4 = loss_x4

            # Calculate total loss for each example
            total_loss_individual = torch.sum((decoded - data) ** 2, dim=1) - delta * (
                        normalized_loss_x2.squeeze() + normalized_loss_x3.squeeze() + normalized_loss_x4.squeeze()).to(device)
            total_loss = total_loss_individual.mean()
            l2_norm = sum(p.pow(2.0).sum() for p in autoencoder.parameters())
            total_loss += l2_lambda * l2_norm

            # Backward pass
            # Freeze latent predictor parameters
            for param in predictor_x2.parameters():
                param.requires_grad = False
            for param in predictor_x3.parameters():
                param.requires_grad = False
            for param in predictor_x4.parameters():
                param.requires_grad = False

            total_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), clip_value)

            # Unfreeze latent predictor parameters
            for param in predictor_x2.parameters():
                param.requires_grad = True
            for param in predictor_x3.parameters():
                param.requires_grad = True
            for param in predictor_x4.parameters():
                param.requires_grad = True

            for param in autoencoder.parameters():
                param.requires_grad = False

            loss_x2_mean.backward(retain_graph=True)
            loss_x3_mean.backward(retain_graph=True)
            loss_x4_mean.backward()

            for param in autoencoder.parameters():
                param.requires_grad = True

            optimizer_ae.step()
            optimizer_x2.step()
            optimizer_x3.step()
            optimizer_x4.step()

            running_loss_ae += ae_loss.item()
            running_loss_x2 += loss_x2_mean.item()
            running_loss_x3 += loss_x3_mean.item()
            running_loss_x4 += loss_x4_mean.item()
            running_loss_total += total_loss.item()

        # Compute mean training loss
        mean_train_loss_ae = running_loss_ae / len(train_loader)
        mean_train_loss_x2 = running_loss_x2 / len(train_loader)
        mean_train_loss_x3 = running_loss_x3 / len(train_loader)
        mean_train_loss_x4 = running_loss_x4 / len(train_loader)
        mean_train_loss_total = running_loss_total / len(train_loader)

        # Log the mean training loss
        writer.add_scalar('Loss/train_ae', mean_train_loss_ae, epoch)
        writer.add_scalar('Loss/train_x2', mean_train_loss_x2, epoch)
        writer.add_scalar('Loss/train_x3', mean_train_loss_x3, epoch)
        writer.add_scalar('Loss/train_x4', mean_train_loss_x4, epoch)
        writer.add_scalar('Loss/train_total', mean_train_loss_total, epoch)

        # Evaluate on validation set
        autoencoder.eval()
        predictor_x2.eval()
        predictor_x3.eval()
        predictor_x4.eval()
        val_loss_ae = 0.0
        val_loss_x2 = 0.0
        val_loss_x3 = 0.0
        val_loss_x4 = 0.0
        val_loss_total = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                encoded = autoencoder.encoder(data)
                # Normalize the encoded vector
                encoded = F.normalize(encoded, p=2, dim=1)
                decoded = autoencoder.decoder(encoded)
                ae_loss = criterion(decoded, data)

                z1 = encoded[:, :1].cpu()
                z1_z2 = encoded[:, :2].cpu()
                z1_z2_z3 = encoded[:, :3].cpu()
                z2 = encoded[:, 1].cpu()
                z3 = encoded[:, 2].cpu()
                z4 = encoded[:, 3].cpu()

                z2_pred = predictor_x2(z1)
                z3_pred = predictor_x3(z1_z2)
                z4_pred = predictor_x4(z1_z2_z3)

                loss_x2 = (z2_pred - z2.unsqueeze(1)) ** 2
                loss_x2_mean = loss_x2.mean()
                loss_x3 = (z3_pred - z3.unsqueeze(1)) ** 2
                loss_x3_mean = loss_x3.mean()
                loss_x4 = (z4_pred - z4.unsqueeze(1)) ** 2
                loss_x4_mean = loss_x4.mean()

                norm_z1 = torch.sum(z1 ** 2, dim=1, keepdim=True)
                norm_z1_z2 = torch.sum(z1_z2 ** 2, dim=1, keepdim=True)
                norm_z1_z2_z3 = torch.sum(z1_z2_z3 ** 2, dim=1, keepdim=True)
                normalized_loss_x2 = loss_x2
                normalized_loss_x3 = loss_x3
                normalized_loss_x4 = loss_x4
                total_loss_individual = ae_loss - delta * (
                            normalized_loss_x2.squeeze() + normalized_loss_x3.squeeze() + normalized_loss_x4.squeeze()).to(device)
                total_loss = total_loss_individual.mean()
                l2_norm = sum(p.pow(2.0).sum() for p in autoencoder.parameters())
                total_loss += l2_lambda * l2_norm
                val_loss_ae += ae_loss.item()
                val_loss_x2 += loss_x2_mean.item()
                val_loss_x3 += loss_x3_mean.item()
                val_loss_x4 += loss_x4_mean.item()
                val_loss_total += total_loss.item()

        # Compute mean validation loss
        mean_val_loss_ae = val_loss_ae / len(val_loader)
        mean_val_loss_x2 = val_loss_x2 / len(val_loader)
        mean_val_loss_x3 = val_loss_x3 / len(val_loader)
        mean_val_loss_x4 = val_loss_x4 / len(val_loader)
        mean_val_loss_total = val_loss_total / len(val_loader)

        # Log the mean validation loss
        writer.add_scalar('Loss/val_ae', mean_val_loss_ae, epoch)
        writer.add_scalar('Loss/val_x2', mean_val_loss_x2, epoch)
        writer.add_scalar('Loss/val_x3', mean_val_loss_x3, epoch)
        writer.add_scalar('Loss/val_x4', mean_val_loss_x4, epoch)
        writer.add_scalar('Loss/val_total', mean_val_loss_total, epoch)

        # Step the learning rate scheduler
        scheduler_ae.step()
        scheduler_x2.step()
        scheduler_x3.step()
        scheduler_x4.step()

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss AE: {mean_train_loss_ae:.4f}, Val Loss AE: {mean_val_loss_ae:.4f}, Train Loss X2: {mean_train_loss_x2:.4f}, Val Loss X2: {mean_val_loss_x2:.4f}, Train Loss X3: {mean_train_loss_x3:.4f}, Val Loss X3: {mean_val_loss_x3:.4f}, Train Loss X4: {mean_train_loss_x4:.4f}, Val Loss X4: {mean_val_loss_x4:.4f}, Train Loss Total: {mean_train_loss_total:.4f}, Val Loss Total: {mean_val_loss_total:.4f}')

    writer.close()


def save_models(autoencoder, predictor_x2, predictor_x3, predictor_x4):
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(autoencoder.state_dict(), args.model_path)
    torch.save(predictor_x2.state_dict(), args.model_path.replace('autoencoder', 'predictor_x2'))
    torch.save(predictor_x3.state_dict(), args.model_path.replace('autoencoder', 'predictor_x3'))
    torch.save(predictor_x4.state_dict(), args.model_path.replace('autoencoder', 'predictor_x4'))
    print(
        f"Models trained and saved to '{args.model_path}', '{args.model_path.replace('autoencoder', 'predictor_x2')}', '{args.model_path.replace('autoencoder', 'predictor_x3')}', and '{args.model_path.replace('autoencoder', 'predictor_x4')}'.")


# Example of how to call train_models in your main script
if __name__ == "__main__":
    args = get_args()

    train_loader = get_dataloader(args.data_path.replace('toy_dataset.csv', 'toy_dataset_train.csv'),
                                  batch_size=args.batch_size)
    val_loader = get_dataloader(args.data_path.replace('toy_dataset.csv', 'toy_dataset_val.csv'),
                                batch_size=args.batch_size)

    autoencoder = Autoencoder()
    predictor_x2 = LatentPredictor_x2()
    predictor_x3 = LatentPredictor_x3()
    predictor_x4 = LatentPredictor_x4()

    initializer = get_initializer(args.init_method)
    autoencoder.apply(initializer)
    predictor_x2.apply(initializer)
    predictor_x3.apply(initializer)
    predictor_x4.apply(initializer)

    criterion = nn.MSELoss()

    optimizer_ae = optim.Adam(autoencoder.parameters(), lr=args.learning_rate)
    optimizer_x2 = optim.Adam(predictor_x2.parameters(), lr=args.learning_rate)
    optimizer_x3 = optim.Adam(predictor_x3.parameters(), lr=args.learning_rate)
    optimizer_x4 = optim.Adam(predictor_x4.parameters(), lr=args.learning_rate)

    scheduler_ae = lr_scheduler.StepLR(optimizer_ae, step_size=args.step_size, gamma=args.gamma)
    scheduler_x2 = lr_scheduler.StepLR(optimizer_x2, step_size=args.step_size, gamma=args.gamma)
    scheduler_x3 = lr_scheduler.StepLR(optimizer_x3, step_size=args.step_size, gamma=args.gamma)
    scheduler_x4 = lr_scheduler.StepLR(optimizer_x4, step_size=args.step_size, gamma=args.gamma)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('./runs', f"{timestamp}_{args.experiment_name}" if args.experiment_name else timestamp)

    train_models(train_loader, val_loader, autoencoder, predictor_x2, predictor_x3, predictor_x4, criterion,
                 optimizer_ae, optimizer_x2, optimizer_x3, optimizer_x4, scheduler_ae, scheduler_x2, scheduler_x3,
                 scheduler_x4, num_epochs=args.num_epochs, log_dir=log_dir, delta=args.delta)

    save_models(autoencoder, predictor_x2, predictor_x3, predictor_x4)