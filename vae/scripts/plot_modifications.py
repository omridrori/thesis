import matplotlib.pyplot as plt
import torch
import numpy as np

import torch.nn.functional as F
def generate_data_to_plot():
    data = []

    d = [4,4,4,4,4]  # Lengths d1, d2, d3, d4, d5

    phi = np.random.uniform(0, np.pi, size=4)  # Angles phi1, phi2, phi3, phi4
    x = np.zeros((4, 2))  # Assume 2D for simplicity
    x[0] = [d[0] * np.cos(phi[0]), d[0] * np.sin(phi[0])]
    x[1] = x[0] + [d[1] * np.cos(phi[1]), d[1] * np.sin(phi[1])]
    x[2] = x[1] + [d[2] * np.cos(phi[2]), d[2] * np.sin(phi[2])]
    x[3] = x[2] + [d[3] * np.cos(phi[3]), d[3] * np.sin(phi[3])]

    return x



# def plot_points(original, modified_points, values_to_add, title):
#     fig, ax = plt.subplots(figsize=(20, 12))
#
#     # Plotting the original points with bold blue line
#     ax.plot(original[::2], original[1::2], 'bs-', label='Original', markersize=12, linewidth=2)
#
#     # Starting point (0,0)
#     ax.plot(0, 0, 'ko', markersize=10)  # Plots the point (0,0)
#
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(modified_points)))
#     for i, points in enumerate(modified_points):
#         # Adjust style based on whether the modification is zero
#         if values_to_add[i] == 0.0:
#             ax.plot(points[::2], points[1::2], '+--', color='black', label='Modified z by 0', markersize=8, linewidth=2)
#         else:
#             ax.plot(points[::2], points[1::2], 'o-', color=colors[i], label=f'Modified z by {values_to_add[i]}',
#                     markersize=5)
#
#     # Setting axis limits to ensure they always stay between -3 and 3
#     ax.set_xlim([-12, 12])
#     ax.set_ylim([-12, 12])
#
#     ax.legend()
#     ax.set_title(title)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.grid(True)
#
#     return fig
#
#
# def generate_plots_modifications(model,data,device):
#     model.eval()
#     with torch.no_grad():
#         model= model.to(device)
#         input_point = data.reshape(-1)
#
#         input_tensor = torch.tensor(input_point, dtype=torch.float32).unsqueeze(0).to(device)
#
#         encoded, _ ,_,_= model(input_tensor)
#         encoded = encoded.detach().cpu().numpy()
#         values_to_add = [-0.5, -0.1, 0.0, 0.1, 0.5]
#         plots=  []
#         for i in range(encoded.shape[1]):
#             modified_points = []
#             for val in values_to_add:
#                 modified_encoded = encoded.copy()
#                 modified_encoded[0, i] += val
#                 modified_tensor = torch.tensor(modified_encoded, dtype=torch.float32)
#                 # modified_tensor = F.normalize(modified_tensor, dim=1)
#                 modified_tensor = modified_tensor.to(device)
#                 decoded = model.decode(modified_tensor).detach().cpu().numpy().flatten()
#                 values_to_prepend = [0, 0]
#                 decoded = np.insert(decoded, 0, values_to_prepend)
#                 modified_points.append(decoded)
#
#             values_to_prepend = [0, 0]
#             input_point=np.insert(input_point, 0, values_to_prepend)
#             plots.append(plot_points(input_point, modified_points, values_to_add, f"Effect of modifying z{i + 1}"))
#     return plots
#


def generate_trajectory_pair(angle_index, angle_diff=np.pi / 4):
    d = [4, 4, 4, 4, 4]  # Lengths d1, d2, d3, d4, d5
    phi = np.random.uniform(0, np.pi, size=4)  # Angles phi1, phi2, phi3, phi4

    # Generate first trajectory
    x1 = np.zeros((4, 2))
    x1[0] = [d[0] * np.cos(phi[0]), d[0] * np.sin(phi[0])]
    x1[1] = x1[0] + [d[1] * np.cos(phi[1]), d[1] * np.sin(phi[1])]
    x1[2] = x1[1] + [d[2] * np.cos(phi[2]), d[2] * np.sin(phi[2])]
    x1[3] = x1[2] + [d[3] * np.cos(phi[3]), d[3] * np.sin(phi[3])]

    # Modify the specified angle
    phi[angle_index] += angle_diff

    # Generate second trajectory with the modified angle
    x2 = np.zeros((4, 2))
    x2[0] = [d[0] * np.cos(phi[0]), d[0] * np.sin(phi[0])]
    x2[1] = x2[0] + [d[1] * np.cos(phi[1]), d[1] * np.sin(phi[1])]
    x2[2] = x2[1] + [d[2] * np.cos(phi[2]), d[2] * np.sin(phi[2])]
    x2[3] = x2[2] + [d[3] * np.cos(phi[3]), d[3] * np.sin(phi[3])]

    return x1.flatten(), x2.flatten()


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

        # Choose a fixed angle to modify
        fixed_angle_index = np.random.randint(0, 4)

        # Generate a pair of trajectories with the fixed angle difference
        trajectory1, trajectory2 = generate_trajectory_pair(fixed_angle_index)

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
                                            f"Original pair differs in angle {fixed_angle_index + 1}"))

    return plots