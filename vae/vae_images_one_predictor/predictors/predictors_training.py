from random import random

import torch
from vae.vae_images_one_predictor.predictors.estimated_z import generate_estimated_z
import torch.nn.functional as F


def train_predictor(train_loader, vae, predictor, optimizer_predictor,delta, device):
    for param in vae.parameters():
        param.requires_grad = False


    predictor.train()
    for param in predictor.parameters():
        param.requires_grad = True

    loss_predictors_total = 0.0
    for batch_idx, data in enumerate(train_loader):
        optimizer_predictor.zero_grad()
        data = data[0]
        data = data.to(device)
        encoded, deccoded, mu, logvar = vae(data)


        generated_z = generate_estimated_z(predictor, mu, device)
        decoded_estimated = vae.decoder(generated_z)
        loss_predictors = F.binary_cross_entropy(decoded_estimated, data, reduction='none').sum(axis=(1, 2, 3)).mean()
        loss_predictors_total += loss_predictors.cpu().item()
        loss_predictors.backward()
        optimizer_predictor.step()

    return loss_predictors_total / len(train_loader)