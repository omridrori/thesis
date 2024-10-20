import torch
from torch import nn

from vae.vae_images_one_predictor.predictors.estimated_z import generate_estimated_z
import torch.nn.functional as F


def loss_function_individual(recon_x, x, mu, logvar,  beta=0.1):

    BCE = F.binary_cross_entropy(recon_x, x, reduction='none').sum(axis=(1, 2, 3))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    individual_loss = BCE + KLD*beta
    return individual_loss, BCE.detach().mean(), KLD.detach().mean()

def train_vae(train_loader, vae, predictor, optimizer_vae, delta, beta,rho,lambda_norm,  device):
    vae.train()
    running_loss_vae = running_loss_total = running_kld = predictors_loss_mean =  running_norm = 0
    for batch_idx, data in enumerate(train_loader):
        optimizer_vae.zero_grad()
        data = data[0]
        data = data.to(device)

        encoded, decoded, mu, logvar = vae(data)


        encoded_normalized = encoded
        z_estimated = generate_estimated_z(predictor, mu, device)


        decoded_estimated = vae.decoder(z_estimated)

        predictors_loss = F.binary_cross_entropy(decoded_estimated, data, reduction='none').sum(axis=(1, 2, 3)) #todo check if this is correct

        loss_vae, bce, kld = loss_function_individual(decoded, data, mu, logvar,beta)

        total_loss = (loss_vae + delta * torch.exp(-predictors_loss /rho)).mean()
        #calculate norm of z
        z_norm = torch.norm(encoded_normalized, dim=1).mean()
        total_loss += lambda_norm * z_norm



        running_norm += z_norm.cpu().item()

        running_loss_vae += bce.cpu().item()
        running_kld += kld.cpu().item()
        running_loss_total += total_loss.cpu().item()
        predictors_loss_mean += predictors_loss.mean().cpu().item()
        total_loss.backward()
        optimizer_vae.step()
    n = len(train_loader)
    return running_loss_vae / n, running_loss_total / n, running_kld / n, predictors_loss_mean / n, running_norm / n





