import os

import torch
def save_models(autoencoder,predictors,path):
    os.makedirs(path, exist_ok=True)
    torch.save(autoencoder.state_dict(), os.path.join(path, 'autoencoder.pth'))
    for i, predictor in enumerate(predictors):
        # Replace 'autoencoder' with 'predictor_x' followed by the index + 2
        new_filename = f'predictor_x{i + 2}.pth'
        # Join the base path with the new filename
        new_path = os.path.join(path,new_filename)
        # Save the model state dict to the new path
        torch.save(predictor.state_dict(), new_path)
    print(
        f"Models trained and saved to '{path}'")
