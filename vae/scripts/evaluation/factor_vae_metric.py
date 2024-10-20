# import tqdm
# import torch
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from vae.scripts.models.variational_autoencoder import VAE  # Assuming your VAE class is in a file named VAE.py
#
#
# def load_data(file_path):
#     df = pd.read_csv(file_path)
#     # Extract only the angle columns (phi1, phi2, phi3, phi4)
#     angles = df[['phi1', 'phi2', 'phi3', 'phi4']].values
#     # Extract the x coordinates (assuming they're the last 8 columns)
#     x_coords = df.iloc[:, -8:].values
#     return angles, x_coords
#
#
# def representation_function(model, x, device):
#     model.eval()
#     model=model.to(device)
#     with torch.no_grad():
#         # Convert numpy array to PyTorch tensor, move to device, and ensure it's float
#         x = torch.tensor(x, dtype=torch.float32, requires_grad=False).to(device)
#         mu, _ = model.encode(x)
#         return mu.cpu().numpy()  # Move back to CPU and convert to numpy
#
#
#
# def compute_variances(representations):
#     return np.var(representations, axis=0, ddof=1)
#
#
# def generate_training_sample(representation_function, global_variances, active_dims,device):
#     d = [1, 1, 1, 1, 1]
#     factor_index = np.random.randint(4)
#     fixed_angle = np.random.uniform(0, 2 * np.pi)
#     batch_size = 64
#
#     data = []
#     for _ in range(batch_size):
#         phi = np.random.uniform(0, 2 * np.pi, size=4)
#         phi[factor_index] = fixed_angle
#
#         x = np.zeros((4, 2))
#         x[0] = [d[0] * np.cos(phi[0]), d[0] * np.sin(phi[0])]
#         x[1] = x[0] + [d[1] * np.cos(phi[1]), d[1] * np.sin(phi[1])]
#         x[2] = x[1] + [d[2] * np.cos(phi[2]), d[2] * np.sin(phi[2])]
#         x[3] = x[2] + [d[3] * np.cos(phi[3]), d[3] * np.sin(phi[3])]
#
#         data.append(x.flatten())
#
#     data = np.array(data)
#     representations = representation_function(data,device)
#     local_variances = np.var(representations, axis=0, ddof=1)
#     argmin = np.argmin(local_variances[active_dims] / global_variances[active_dims])
#
#     return factor_index, argmin
#
#
# def generate_training_batch(representation_function, num_points, global_variances, active_dims,device):
#     votes = np.zeros((4, len(active_dims)), dtype=np.int64)
#     for _ in tqdm(range(num_points)):
#         factor_index, argmin = generate_training_sample(representation_function, global_variances, active_dims,device)
#         votes[factor_index, argmin] += 1
#     return votes
#
#
# def metric_factor_vae(representation_function, num_train=10000, num_eval=5000,device=None):
#     # Generate a large batch of data to compute global variances
#     d = [1, 1, 1, 1, 1]
#     batch_size = 500
#     data = []
#     for _ in range(batch_size):
#         phi = np.random.uniform(0, 2 * np.pi, size=4)
#         x = np.zeros((4, 2))
#         x[0] = [d[0] * np.cos(phi[0]), d[0] * np.sin(phi[0])]
#         x[1] = x[0] + [d[1] * np.cos(phi[1]), d[1] * np.sin(phi[1])]
#         x[2] = x[1] + [d[2] * np.cos(phi[2]), d[2] * np.sin(phi[2])]
#         x[3] = x[2] + [d[3] * np.cos(phi[3]), d[3] * np.sin(phi[3])]
#         data.append(x.flatten())
#     data = np.array(data)
#
#     representations = representation_function(data,device)
#     global_variances = compute_variances(representations)
#     active_dims = global_variances >= 0.0
#
#     if not active_dims.any():
#         return {"factor_vae.train_accuracy": 0.0, "factor_vae.eval_accuracy": 0.0, "factor_vae.num_active_dims": 0}
#
#     training_votes = generate_training_batch(representation_function, num_train, global_variances, active_dims,device)
#     classifier = np.argmax(training_votes, axis=0)
#     other_index = np.arange(training_votes.shape[1])
#
#     train_accuracy = np.sum(training_votes[classifier, other_index]) * 1.0 / np.sum(training_votes)
#
#     eval_votes = generate_training_batch(representation_function, num_eval, global_variances, active_dims,device)
#     eval_accuracy = np.sum(eval_votes[classifier, other_index]) * 1.0 / np.sum(eval_votes)
#
#     return {
#         "factor_vae.train_accuracy": train_accuracy,
#         "factor_vae.eval_accuracy": eval_accuracy,
#         "factor_vae.num_active_dims": len(active_dims),
#     }
#
# if __name__ ==  "__main__":
#     # Load the model
#     model = VAE()
#     model.load_state_dict(torch.load(r'C:\Users\User\Desktop\thesis\vae\saved models\vae.pth'))
#     model.eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     def rep_function(x,device):
#         return representation_function(model, x,device)
#
#     # Compute the Factor VAE metric
#     results = metric_factor_vae(rep_function)
#
#     print(results)
#
# # Define the representation function
#
#
#
#
import torch
import matplotlib.pyplot as plt
import numpy as np


def generate_trajectory_pair():
    d = [4, 4, 4, 4, 4]  # Lengths d1, d2, d3, d4, d5
    phi = np.random.uniform(0, np.pi, size=4)  # Angles phi1, phi2, phi3, phi4

    # Generate first trajectory
    x1 = np.zeros((4, 2))
    x1[0] = [d[0] * np.cos(phi[0]), d[0] * np.sin(phi[0])]
    x1[1] = x1[0] + [d[1] * np.cos(phi[1]), d[1] * np.sin(phi[1])]
    x1[2] = x1[1] + [d[2] * np.cos(phi[2]), d[2] * np.sin(phi[2])]
    x1[3] = x1[2] + [d[3] * np.cos(phi[3]), d[3] * np.sin(phi[3])]

    # Choose a random angle to modify
    angle_index = np.random.randint(0, 4)
    phi[angle_index] += np.pi / 4  # Modify the chosen angle

    # Generate second trajectory with the modified angle
    x2 = np.zeros((4, 2))
    x2[0] = [d[0] * np.cos(phi[0]), d[0] * np.sin(phi[0])]
    x2[1] = x2[0] + [d[1] * np.cos(phi[1]), d[1] * np.sin(phi[1])]
    x2[2] = x2[1] + [d[2] * np.cos(phi[2]), d[2] * np.sin(phi[2])]
    x2[3] = x2[2] + [d[3] * np.cos(phi[3]), d[3] * np.sin(phi[3])]

    return x1.flatten(), x2.flatten(), angle_index


def plot_interpolation(original1, original2, interpolated_points, title):
    fig, ax = plt.subplots(figsize=(20, 12))

    # Plot original points
    ax.plot(original1[::2], original1[1::2], 'bs-', label='Original', markersize=12, linewidth=2)
    ax.plot(original2[::2], original2[1::2], 'rs-', label='Modified', markersize=12, linewidth=2)

    # Plot interpolated points
    colors = plt.cm.viridis(np.linspace(0, 1, len(interpolated_points)))
    for i, points in enumerate(interpolated_points):
        ax.plot(points[::2], points[1::2], 'o-', color=colors[i],
                label=f'Interpolation {i / (len(interpolated_points) - 1):.2f}', markersize=5)

    # Starting point (0,0)
    ax.plot(0, 0, 'ko', markersize=10)

    ax.set_xlim([-12, 12])
    ax.set_ylim([-12, 12])
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)

    return fig


def generate_plots_modifications(model, device):
    model.eval()
    with torch.no_grad():
        model = model.to(device)

        # Generate a pair of trajectories
        trajectory1, trajectory2, modified_angle = generate_trajectory_pair()

        # Encode both trajectories
        input_tensor1 = torch.tensor(trajectory1, dtype=torch.float32).unsqueeze(0).to(device)
        input_tensor2 = torch.tensor(trajectory2, dtype=torch.float32).unsqueeze(0).to(device)

        encoded1, _, _, _ = model(input_tensor1)
        encoded2, _, _, _ = model(input_tensor2)

        plots = []
        for dim in range(16):  # Assuming 16-dimensional latent space
            # Interpolate in the current dimension of latent space
            num_steps = 10
            alphas = np.linspace(0, 1, num_steps)
            interpolated_points = []

            for alpha in alphas:
                interpolated_z = encoded1.clone()
                interpolated_z[0, dim] = (1 - alpha) * encoded1[0, dim] + alpha * encoded2[0, dim]
                decoded = model.decode(interpolated_z).detach().cpu().numpy().flatten()
                values_to_prepend = [0, 0]
                decoded = np.insert(decoded, 0, values_to_prepend)
                interpolated_points.append(decoded)

            # Prepare trajectories for plotting
            values_to_prepend = [0, 0]
            trajectory1_plot = np.insert(trajectory1, 0, values_to_prepend)
            trajectory2_plot = np.insert(trajectory2, 0, values_to_prepend)

            # Generate plot
            plots.append(plot_interpolation(trajectory1_plot, trajectory2_plot, interpolated_points,
                                            f"Interpolation along dimension {dim + 1} of latent space\n"
                                            f"Original pair differs in angle {modified_angle + 1}"))

    return plots