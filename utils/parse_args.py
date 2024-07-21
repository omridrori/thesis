import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train Autoencoder and Latent Predictor')
    parser.add_argument('--data_path', type=str, default='data/toy_dataset.csv', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of epochs for training')
    parser.add_argument('--model_path', type=str, default='./models/autoencoder.pth', help='Path to save the trained autoencoder model')
    parser.add_argument('--predictor_model_path', type=str, default='./models/latent_predictor.pth', help='Path to save the trained latent predictor model')
    parser.add_argument('--experiment_name', type=str, default='', help='Optional experiment name to add to the log directory')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for the learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for the learning rate scheduler')
    parser.add_argument('--init_method', type=str, default='xavier', choices=['xavier', 'kaiming', 'normal', 'orthogonal', 'constant'], help='Initialization method')
    parser.add_argument('--delta', type=float, default=0.1, help='weight for latent predictor loss in overall')

    return parser.parse_args()