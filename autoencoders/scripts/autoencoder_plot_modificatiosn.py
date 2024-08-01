import matplotlib.pyplot as plt
import torch
import numpy as np


def generate_data_to_plot():
    data = []

    d = [1, 1, 1, 1, 1]  # Lengths d1, d2, d3, d4, d5

    phi = np.random.uniform(0, 2 * np.pi, size=4)  # Angles phi1, phi2, phi3, phi4
    x = np.zeros((4, 2))  # Assume 2D for simplicity
    x[0] = [d[0] * np.cos(phi[0]), d[0] * np.sin(phi[0])]
    x[1] = x[0] + [d[1] * np.cos(phi[1]), d[1] * np.sin(phi[1])]
    x[2] = x[1] + [d[2] * np.cos(phi[2]), d[2] * np.sin(phi[2])]
    x[3] = x[2] + [d[3] * np.cos(phi[3]), d[3] * np.sin(phi[3])]

    return x



def plot_points(original, modified_points, values_to_add, title):


    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting the original points with bold blue line
    ax.plot(original[::2], original[1::2], 'bs-', label='Original', markersize=12, linewidth=2)

    # Starting point (0,0)
    ax.plot(0, 0, 'ko', markersize=10)  # Plots the point (0,0)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(modified_points)))
    for i, points in enumerate(modified_points):
        # Connect (0,0) to the first point of each set of modified points
        # ax.plot([0, points[0]], [0, points[1]], '--', color=colors[i])

        # Adjust style based on whether the modification is zero
        if values_to_add[i] == 0.0:
            ax.plot(points[::2], points[1::2], '+--', color='black', label='Modified z by 0', markersize=8, linewidth=2)
        else:
            ax.plot(points[::2], points[1::2], 'o-', color=colors[i], label=f'Modified z by {values_to_add[i]}',
                     markersize=5)

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)

    return fig


def generate_plots_modifications(model,data,device):
    model.eval()
    with torch.no_grad():
        model= model.to(device)
        input_point = data.reshape(-1)

        input_tensor = torch.tensor(input_point, dtype=torch.float32).unsqueeze(0).to(device)

        encoded = model.encode(input_tensor).detach().cpu().numpy()
        values_to_add = [-0.5, -0.1, 0.0, 0.1, 0.5]
        plots=  []
        for i in range(encoded.shape[1]):
            modified_points = []
            for val in values_to_add:
                modified_encoded = encoded.copy()
                modified_encoded[0, i] += val  # Change the latent variable
                modified_tensor = torch.tensor(modified_encoded, dtype=torch.float32).to(device)
                decoded = model.decoder(modified_tensor).detach().cpu().numpy().flatten()
                values_to_prepend = [0, 0]
                decoded = np.insert(decoded, 0, values_to_prepend)
                modified_points.append(decoded)

            values_to_prepend = [0, 0]
            input_point=np.insert(input_point, 0, values_to_prepend)
            plots.append(plot_points(input_point, modified_points, values_to_add, f"Effect of modifying z{i + 1}"))
    return plots

