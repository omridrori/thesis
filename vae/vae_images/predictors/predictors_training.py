import torch
from vae.vae_images.predictors.estimated_z import generate_estimated_z
import torch.nn.functional as F


def train_predictors(train_loader, vae, predictors, optimizer_predictors, delta, device):
    for param in vae.parameters():
        param.requires_grad = False
    for predictor in predictors:
        predictor.train()
    loss_predictors_total = 0.0
    for batch_idx, data in enumerate(train_loader):
        for optimizer in optimizer_predictors:
            optimizer.zero_grad()
        data = data[0]
        data = data.to(device)
        encoded, deccoded, mu, logvar = vae(data)

        z_estimated = generate_estimated_z(predictors, mu, device)
        z_estimated_normalized = z_estimated
        decoded_estimated = vae.decoder(z_estimated_normalized)
        loss_predictors = F.binary_cross_entropy(decoded_estimated, data, reduction='none').mean(axis=(1,2,3)).mean() #todo check if this is correct
        loss_predictors_total += loss_predictors.item()
        loss_predictors.backward()
        for optimizer in optimizer_predictors:
            optimizer.step()
    return loss_predictors_total / len(train_loader)