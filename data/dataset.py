import numpy as np
import pandas as pd


def generate_data(num_samples=1000):
    data = []

    d = np.random.uniform(1, 10, size=5)  # Lengths d1, d2, d3, d4, d5
    for _ in range(num_samples):
        # Sample lengths of poles


        # Sample angles between joints
        phi = np.random.uniform(0, np.pi, size=4)  # Angles phi1, phi2, phi3, phi4

        # Calculate joint positions
        x = np.zeros((4, 2))  # We assume 2D for simplicity

        x[0] = [d[0] * np.cos(phi[0]), d[0] * np.sin(phi[0])]
        x[1] = x[0] + [d[1] * np.cos(phi[1]), d[1] * np.sin(phi[1])]
        x[2] = x[1] + [d[2] * np.cos(phi[2]), d[2] * np.sin(phi[2])]
        x[3] = x[2] + [d[3] * np.cos(phi[3]), d[3] * np.sin(phi[3])]

        data.append(np.hstack([d, phi, x.flatten()]))

    columns = [f'd{i + 1}' for i in range(5)] + [f'phi{i + 1}' for i in range(4)] + [f'x{i + 1}_{j + 1}' for i in
                                                                                     range(4) for j in range(2)]

    df = pd.DataFrame(data, columns=columns)

    return df


