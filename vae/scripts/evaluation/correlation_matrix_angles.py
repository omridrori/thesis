import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from vae.scripts.models.variational_autoencoder import VAE

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def generate_data(angles):
    d = [4,4,4,4]
    x = np.zeros((4, 2))
    x[0] = [d[0] * np.cos(angles[0]), d[0] * np.sin(angles[0])]
    x[1] = x[0] + [d[1] * np.cos(angles[1]), d[1] * np.sin(angles[1])]
    x[2] = x[1] + [d[2] * np.cos(angles[2]), d[2] * np.sin(angles[2])]
    x[3] = x[2] + [d[3] * np.cos(angles[3]), d[3] * np.sin(angles[3])]
    return  x.flatten()


def evaluate_disentanglement(vae_model, n_examples=1000, n_steps=20):
    device = next(vae_model.parameters()).device
    vae_model.eval()

    correlation_matrices = []

    for example_i in tqdm(range(n_examples), desc="Processing examples"):
        # Generate a random set of base angles for this example
        base_angles = np.random.uniform(0,  2*np.pi, size=4)

        example_correlations = np.zeros((4, 16))

        for angle_j in range(4):  # For each angle
            angle_values = np.linspace(0,  2*np.pi, n_steps)
            z_values = np.zeros((16, n_steps))

            for step, angle_val in enumerate(angle_values):
                # Create a copy of base angles and modify only the j-th angle
                modified_angles = base_angles.copy()
                modified_angles[angle_j] = angle_val

                # Generate data and convert to tensor
                data = generate_data(modified_angles)
                data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

                # Get latent representation
                with torch.no_grad():
                    mu, _ = vae_model.encode(data_tensor)

                # Store the latent representation
                z_values[:, step] = mu.cpu().numpy().flatten()

            # Compute correlation for this angle
            for k in range(16):
                example_correlations[angle_j, k] = np.abs(np.corrcoef(angle_values, z_values[k, :])[0, 1])

        correlation_matrices.append(example_correlations)

    # Compute average correlation matrix
    avg_correlation_matrix = np.mean(correlation_matrices, axis=0)

    return avg_correlation_matrix


def plot_correlation_matrix(correlation_matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Average Correlation between Angles and Latent Dimensions')
    plt.xlabel('Latent Dimensions')
    plt.ylabel('Angles')
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(correlation_matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title('Average Correlation between Angles and Latent Dimensions')
    plt.xlabel('Latent Dimensions')
    plt.ylabel('Angles')
    plt.tight_layout()
    plt.show()

vae_model= VAE()
vae_model.load_state_dict(torch.load(r'C:\Users\User\Desktop\thesis\vae\saved models\vae.pth'))
# Load the VAE model


# Evaluate disentanglement
avg_correlation_matrix = evaluate_disentanglement(vae_model)

# Plot the results
plot_correlation_matrix(avg_correlation_matrix)