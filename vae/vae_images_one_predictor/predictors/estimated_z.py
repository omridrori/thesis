import random
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_estimated_z(predictor, z, device,zero_rate=0.1):
    z= z.to(device)
    predictor = predictor.to(device)

    # Choose random indices to zero out
    batch_size, latent_dim = z.shape
    num_zeros = int(latent_dim * zero_rate)  # delta is the fraction of indices to zero out

    mask = torch.ones_like(z)
    for i in range(batch_size):
        zero_indices = random.sample(range(latent_dim), num_zeros)
        mask[i, zero_indices] = 0

    # Zero out the chosen indices
    mu_masked = z * mask

    # Use predictor to guess the original vector
    predicted_mu = predictor(mu_masked)
    return predicted_mu
