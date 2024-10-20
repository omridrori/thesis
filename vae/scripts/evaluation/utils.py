import torch
import numpy as np
import sklearn
from sklearn.metrics import mutual_info_score

import torch


def generate_batch_factor_code(model, num_points, device, batch_size=64):
    representations = []
    factors = []

    for i in range(0, num_points, batch_size):
        current_batch_size = min(batch_size, num_points - i)

        # Generate factors (angles)
        phi = np.random.uniform(0, 2 * np.pi, size=(current_batch_size, 4))

        # Generate data points
        d = [1, 1, 1, 1, 1]  # Lengths d1, d2, d3, d4, d5
        x = np.zeros((current_batch_size, 4, 2))

        x[:, 0, 0] = d[0] * np.cos(phi[:, 0])
        x[:, 0, 1] = d[0] * np.sin(phi[:, 0])
        x[:, 1, 0] = x[:, 0, 0] + d[1] * np.cos(phi[:, 1])
        x[:, 1, 1] = x[:, 0, 1] + d[1] * np.sin(phi[:, 1])
        x[:, 2, 0] = x[:, 1, 0] + d[2] * np.cos(phi[:, 2])
        x[:, 2, 1] = x[:, 1, 1] + d[2] * np.sin(phi[:, 2])
        x[:, 3, 0] = x[:, 2, 0] + d[3] * np.cos(phi[:, 3])
        x[:, 3, 1] = x[:, 2, 1] + d[3] * np.sin(phi[:, 3])

        # Flatten x coordinates
        x_flat = x.reshape(current_batch_size, -1)

        # Flatten x coordinates

        # Convert to PyTorch tensor and move to device
        x_tensor = torch.FloatTensor(x_flat).to(device)

        # Get latent representations
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(x_tensor)
            current_representations = mu.cpu().numpy()

        representations.append(current_representations)
        factors.append(phi)

    # Concatenate all batches
    representations = np.concatenate(representations, axis=0)
    factors = np.concatenate(factors, axis=0)

    return np.transpose(representations), np.transpose(factors)


def make_discretizer(target, num_bins=20,):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
    return discretized


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = mutual_info_score(ys[j, :], mus[i, :])
    return m


def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = mutual_info_score(ys[j, :], ys[j, :])
    return h


def normalize_data(data, mean=None, stddev=None):
    if mean is None:
        mean = np.mean(data, axis=1)
    if stddev is None:
        stddev = np.std(data, axis=1)
    return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev