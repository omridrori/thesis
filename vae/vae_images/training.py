import numpy as np
import os
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from vae.vae_images.vae_model.vae_model import VAE
from vae.vae_images.predictors.predictors import LatentPredictor_x0

import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from vae.scripts.models.annealer import Annealer

torch.autograd.set_detect_anomaly(True)

from concurrent.futures import ThreadPoolExecutor, as_completed


def setup_schedulers(optimizer_vae, optimizers_predictors):
    scheduler_vae = lr_scheduler.StepLR(optimizer_vae, step_size=5, gamma=0.9)
    schedulers_predictors = [lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) for optimizer in
                             optimizers_predictors]
    return scheduler_vae, schedulers_predictors


def setup_models(device):
    vae_model = VAE().to(device)
    predictors = [LatentPredictor_x0().to(device) for _ in range(10)]
    return vae_model, predictors


def setup_optimizers(vae_model, predictors):
    optimizer_vae = optim.AdamW(vae_model.parameters(), lr=1e-4)
    optimizers_predictors = [optim.AdamW(predictor.parameters(), lr=1e-4) for predictor in predictors]
    return optimizer_vae, optimizers_predictors


def generate_estimated_z(predictors, z, device):
    z_copy = z.clone().to(device)
    z_estimated = []
    for i in range(z.shape[1]):
        z_input = torch.cat([z_copy[:, :i], z_copy[:, i + 1:]], dim=1)
        z_pred = predictors[i](z_input)
        z_estimated.append(z_pred)
    return torch.cat(z_estimated, dim=1)


def predictor_task(predictor, z_input, z_val, device):
    z_pred = predictor(z_input.to(device))
    return (z_pred - z_val.to(device)) ** 2


def loss_predict_with_predictors(predictors, data, encoded_normalized, device):
    z = encoded_normalized.to(device)
    val_predictors_losses_individuals = torch.zeros(data.size(0), device=device)
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_predictor = {
            executor.submit(predictor_task, predictors[i], z[:, :i], z[:, i].unsqueeze(1), device): i
            for i in range(len(predictors))}
        for future in as_completed(future_to_predictor):
            result = future.result()
            val_predictors_losses_individuals += result.squeeze()
    return val_predictors_losses_individuals


def train_predictors(train_loader, vae, predictors, optimizer_predictors, delta, device):
    for predictor in predictors:
        predictor.train()
    loss_predictors_total = 0.0
    for batch_idx, data in enumerate(train_loader):
        for optimizer in optimizer_predictors:
            optimizer.zero_grad()
        data = data[0]
        data = data.to(device)

        print(
            f"Predictor training - Batch {batch_idx}: Input shape: {data.shape}, Mean: {data.mean():.4f}, Std: {data.std():.4f}")

        encoded, _, _, _ = vae(data)
        encoded_normalized = encoded

        print(
            f"Predictor training - Batch {batch_idx}: Encoded shape: {encoded.shape}, Mean: {encoded.mean():.4f}, Std: {encoded.std():.4f}")

        z_estimated = generate_estimated_z(predictors, encoded_normalized, device)
        z_estimated_normalized = z_estimated

        print(
            f"Predictor training - Batch {batch_idx}: Estimated z shape: {z_estimated.shape}, Mean: {z_estimated.mean():.4f}, Std: {z_estimated.std():.4f}")

        decoded_estimated = vae.decoder(z_estimated_normalized)

        print(
            f"Predictor training - Batch {batch_idx}: Decoded estimated shape: {decoded_estimated.shape}, Mean: {decoded_estimated.mean():.4f}, Std: {decoded_estimated.std():.4f}")

        loss_predictors = F.binary_cross_entropy(decoded_estimated, data, reduction='sum')
        loss_predictors_total += loss_predictors.item()

        print(f"Predictor training - Batch {batch_idx}: Loss: {loss_predictors.item():.4f}")

        loss_predictors.backward()
        for optimizer in optimizer_predictors:
            optimizer.step()
    return loss_predictors_total / len(train_loader)


def train_vae(train_loader, vae, predictors, optimizer_vae, delta, annealing_agent, beta_max, device):
    vae.train()
    running_loss_vae = running_loss_total = running_kld = predictors_loss_mean = 0.0
    for batch_idx, data in enumerate(train_loader):
        optimizer_vae.zero_grad()
        data = data[0]
        data = data.to(device)

        print(
            f"VAE training - Batch {batch_idx}: Input shape: {data.shape}, Mean: {data.mean():.4f}, Std: {data.std():.4f}")

        encoded, decoded, mu, logvar = vae(data)

        print(
            f"VAE training - Batch {batch_idx}: Encoded shape: {encoded.shape}, Mean: {encoded.mean():.4f}, Std: {encoded.std():.4f}")
        print(f"VAE training - Batch {batch_idx}: Mu shape: {mu.shape}, Mean: {mu.mean():.4f}, Std: {mu.std():.4f}")
        print(
            f"VAE training - Batch {batch_idx}: Logvar shape: {logvar.shape}, Mean: {logvar.mean():.4f}, Std: {logvar.std():.4f}")

        encoded_normalized = encoded
        z_estimated = generate_estimated_z(predictors, encoded_normalized, device)

        print(
            f"VAE training - Batch {batch_idx}: Estimated z shape: {z_estimated.shape}, Mean: {z_estimated.mean():.4f}, Std: {z_estimated.std():.4f}")

        decoded_estimated = vae.decoder(z_estimated)

        print(
            f"VAE training - Batch {batch_idx}: Decoded shape: {decoded.shape}, Mean: {decoded.mean():.4f}, Std: {decoded.std():.4f}")
        print(
            f"VAE training - Batch {batch_idx}: Decoded estimated shape: {decoded_estimated.shape}, Mean: {decoded_estimated.mean():.4f}, Std: {decoded_estimated.std():.4f}")

        predictors_loss = F.binary_cross_entropy(decoded_estimated, data, reduction='none').sum(axis=(1, 2, 3))
        loss_vae, bce, kld = loss_function_individual(decoded, data, mu, logvar, annealing_agent, beta_max)
        total_loss = (loss_vae + delta * torch.exp(-predictors_loss / 0.5)).mean()

        print(
            f"VAE training - Batch {batch_idx}: BCE: {bce:.4f}, KLD: {kld:.4f}, Predictors Loss: {predictors_loss.mean():.4f}, Total Loss: {total_loss:.4f}")

        running_loss_vae += bce.item()
        running_kld += kld.item()
        running_loss_total += total_loss.item()
        predictors_loss_mean += predictors_loss.mean().item()
        total_loss.backward()
        optimizer_vae.step()
    n = len(train_loader)
    return running_loss_vae / n, running_loss_total / n, running_kld / n, predictors_loss_mean / n


def loss_function_individual(recon_x, x, mu, logvar, annealing_agent, beta_max):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='none').sum(axis=(1, 2, 3))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    KLD_add = annealing_agent(KLD) * beta_max
    individual_loss = BCE + KLD_add
    return individual_loss, BCE.detach().mean(), KLD.detach().mean()


def log_training_info(writer, epoch, loss_vae, loss_total, loss_predictors, kld, delta, beta):
    writer.add_scalar('Loss/train_ae', loss_vae, epoch)
    writer.add_scalar('Loss/train_total', loss_total, epoch)
    writer.add_scalar('Loss/train_predictors', loss_predictors, epoch)
    writer.add_scalar('Loss/train_kld', kld, epoch)
    writer.add_scalar('Hyperparameters/delta', delta, epoch)
    writer.add_scalar('Hyperparameters/beta', beta, epoch)




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

        # Predictor training steps
        for param in vae_model.parameters():
            param.requires_grad = False
        for predictor in predictors:
            for param in predictor.parameters():
                param.requires_grad = True

        loss_predictors = 0
        num_epochs_predictors = 1

        if epoch > 20:
            num_epochs_predictors = 3

        if epoch > 100:
            num_epochs_predictors = 5

        for _ in range(num_epochs_predictors):
            loss_predictors += train_predictors(train_loader, vae_model, predictors, optimizer_predictors, delta,
                                                device)

        loss_predictors /= num_epochs_predictors

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


        torch.save(vae_model.state_dict(), r"C:\Users\User\Desktop\thesis\vae\saved models\vae_dsprites.pth")
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss Total: {loss_total:.4f}, Loss AE: {loss_vae:.4f}, Loss predictors: {loss_predictors:.4f}, KLD: {kld:.4f}, Beta: {current_beta:.5f}, Delta: {delta:.4f}')

    if log_tensorboard:
        writer.close()
    return


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dSprites dataset
    data = np.load(
        r"C:\Users\User\Desktop\thesis\vae\vae_images\dsprites-dataset\dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        allow_pickle=True)
    imgs = data['imgs']
    np.random.shuffle(imgs)

    # Debug: Print dataset information
    print(f"Dataset shape: {imgs.shape}")
    print(f"Dataset min: {imgs.min()}, max: {imgs.max()}, mean: {imgs.mean():.4f}")

    # Use more images for better training, but not too many to start
    imgs = imgs[:100]
    imgs = torch.from_numpy(imgs).float().unsqueeze(1)

    # Ensure the data is in the range [0, 1] for Bernoulli distribution
    if imgs.max() > 1:
        imgs = imgs / 255.0

    imgs = (imgs - imgs.mean()) / imgs.std()

    # Debug: Print processed dataset information
    print(f"Processed dataset shape: {imgs.shape}")
    print(f"Processed dataset min: {imgs.min():.4f}, max: {imgs.max():.4f}, mean: {imgs.mean():.4f}")

    # Create DataLoader
    dataset = TensorDataset(imgs)
    train_loader = DataLoader(dataset, batch_size=25, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=25, shuffle=False)

    # Debug: Print DataLoader information
    print(f"Number of batches in train_loader: {len(train_loader)}")

    vae_model, predictors = setup_models(device)
    optimizer_vae, optimizers_predictors = setup_optimizers(vae_model, predictors)
    scheduler_vae, schedulers_predictors = setup_schedulers(optimizer_vae, optimizers_predictors)

    # Debug: Print model information
    print(f"VAE model:\n{vae_model}")
    print(f"Number of predictors: {len(predictors)}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(r'C:\Users\User\Desktop\thesis\vae\vae_images\runs_vae', f"{timestamp}")

    # Debug: Print training parameters
    print(f"Log directory: {log_dir}")
    print(f"Number of epochs: 1000")
    print(f"Initial delta: 0.1")
    print(f"Initial beta_max: 0.1")

    train_models(
        train_loader, val_loader, vae_model, predictors,
        optimizer_vae, optimizers_predictors,
        num_epochs=1000, log_dir=log_dir, delta=0.1, beta_max=0.1,
        log_tensorboard=True, do_eval=False, data_to_plot=None, device=device,
        scheduler_vae=scheduler_vae, schedulers_predictors=schedulers_predictors)

    # After training, test the model
    print("Training completed. Testing the model...")
    vae_model.eval()
    with torch.no_grad():
        test_batch = next(iter(val_loader))[0].to(device)
        reconstructed, _, _, _ = vae_model(test_batch)

        # Debug: Print test results
        print(f"Test batch shape: {test_batch.shape}")
        print(f"Reconstructed batch shape: {reconstructed.shape}")
        print(f"Test batch mean: {test_batch.mean():.4f}, std: {test_batch.std():.4f}")
        print(f"Reconstructed batch mean: {reconstructed.mean():.4f}, std: {reconstructed.std():.4f}")

        # Save some test images and their reconstructions
        import torchvision.utils as vutils
        vutils.save_image(test_batch, 'test_original.png')
        vutils.save_image(reconstructed, 'test_reconstructed.png')
        print("Test images saved as 'test_original.png' and 'test_reconstructed.png'")


if __name__ == "__main__":
    main()