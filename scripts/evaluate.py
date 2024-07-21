import torch
import matplotlib.pyplot as plt
from models.autoencoder import Autoencoder
from utils.data_loader import get_dataloader

def visualize_reconstruction(data_loader, model, num_examples=5):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num_examples:
                break
            outputs = model(data)
            plt.figure(figsize=(8, 4))
            for j in range(2):
                plt.subplot(1, 2, j+1)
                data_points = data[j].numpy().reshape(-1, 2)
                output_points = outputs[j].numpy().reshape(-1, 2)
                plt.plot(data_points[:, 0], data_points[:, 1], 'bo-', label='Original')
                plt.plot(output_points[:, 0], output_points[:, 1], 'ro-', label='Reconstructed')
                plt.legend()
            plt.show()

if __name__ == "__main__":
    data_loader = get_dataloader('data/toy_dataset_train.csv', batch_size=5)
    model = Autoencoder()
    model.load_state_dict(torch.load('models/autoencoder_delta0.1.pth'))
    visualize_reconstruction(data_loader, model, num_examples=5)