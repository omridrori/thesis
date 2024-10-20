import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_estimated_z(predictors, z, device):
    z_copy = z.clone().to(device)
    z_estimated = []
    for i in range(z.shape[1]):
        z_input = torch.cat([z_copy[:, :i], z_copy[:, i + 1:]], dim=1)
        z_pred = predictors[i](z_input)
        z_estimated.append(z_pred)
    return torch.cat(z_estimated, dim=1)

