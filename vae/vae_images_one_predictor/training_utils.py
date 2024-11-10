from torch import optim

from vae.vae_images_one_predictor.predictors.predictors import Predictor
from vae.vae_images_one_predictor.vae_model.vae_model import VAE


def setup_models(device):
    vae_model = VAE().to(device)
    predictors = Predictor().to(device)
    return vae_model, predictors


def setup_optimizers(vae_model, predictor):
    optimizer_vae = optim.AdamW(vae_model.parameters(), lr=5e-4,betas=(0.9,0.999))
    optimizers_predictor =   optim.AdamW(predictor.parameters(), lr=5e-3,betas=(0.5,0.9))
    return optimizer_vae, optimizers_predictor
