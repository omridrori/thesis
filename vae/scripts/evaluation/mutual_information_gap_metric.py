import json
import numpy as np
import torch
from vae.scripts.evaluation.utils import generate_batch_factor_code, make_discretizer, discrete_mutual_info, \
    discrete_entropy
from vae.scripts.models.variational_autoencoder import VAE


def log_mig(output_file_path, evaluation_time, mig_score):
    res = {
        'eval_time' : evaluation_time,
        'mig_score' : mig_score
    }
    with open(output_file_path, 'w') as fp:
        json.dump(res, fp)

def compute_mig(model, num_train=10000, batch_size=16, seed=17):
    """Computes the mutual information gap."""
    model.cpu()
    device = torch.device("cpu")
    np.random.seed(seed)
    mus_train, ys_train = generate_batch_factor_code(model, num_train, device, batch_size)
    assert mus_train.shape[1] == num_train
    return _compute_mig(mus_train, ys_train)

def _compute_mig(mus_train, ys_train):
    """Computes score based on both training and testing codes and factors."""
    discretized_mus = make_discretizer(mus_train)
    discretized_ys = make_discretizer(ys_train)
    m = discrete_mutual_info(discretized_mus, discretized_ys)
    assert m.shape[0] == mus_train.shape[0]
    assert m.shape[1] == ys_train.shape[0]
    # m is [num_latents, num_factors]
    entropy = discrete_entropy(discretized_ys)
    sorted_m = np.sort(m, axis=0)[::-1]
    return np.mean([res for res in np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]) if res < np.inf])


if __name__ == "__main__":
    # Load model and data
    model = VAE()
    model.load_state_dict(torch.load(r'C:\Users\User\Desktop\thesis\vae\saved models\autoencoder.pth'))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # Compute MIG metric
    mig_score = compute_mig(model, num_train=10000, batch_size=32, seed=17)
    print(f"MIG Score: {mig_score}")