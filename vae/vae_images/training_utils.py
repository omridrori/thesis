from torch import optim

from vae.vae_images.predictors.predictors import LatentPredictor_x0
from vae.vae_images.vae_model.vae_model import VAE


def setup_models(device):
    vae_model = VAE().to(device)
    predictors = [LatentPredictor_x0().to(device) for _ in range(10)]
    return vae_model, predictors


def setup_optimizers(vae_model, predictors):
    optimizer_vae = optim.AdamW(vae_model.parameters(), lr=1e-3)
    optimizers_predictors = [optim.AdamW(predictor.parameters(), lr=1e-3) for predictor in predictors]
    return optimizer_vae, optimizers_predictors
