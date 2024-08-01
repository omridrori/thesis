import argparse

def get_args():
    import os
    from dotenv import load_dotenv
    import argparse


    # Set up the parser with hardcoded defaults
    parser = argparse.ArgumentParser(description='Train Autoencoder and Latent Predictor')
    parser.add_argument('--train_data_path', type=str, default=r'C:\Users\User\Desktop\thesis\data\toy_dataset_train.csv', help='Path to the train dataset')
    parser.add_argument('--val_data_path', type=str, default=r'C:\Users\User\Desktop\thesis\data\toy_dataset_val.csv', help='Path to the val dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--do_eval', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Whether to do evaluation or not')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--model_path', type=str, default='./autoencoder_saved_models/autoencoder.pth',
                        help='Path to save the trained autoencoder model')
    parser.add_argument('--predictor_model_path', type=str,
                        default='./autoencoder_saved_models/latent_predictor.pth',
                        help='Path to save the trained latent predictor model')
    parser.add_argument('--log_tensorboard', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='log the training process to tensorboard')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Optional experiment name to add to the log directory')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for the learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for the learning rate scheduler')
    parser.add_argument('--init_method', type=str, default='xavier',
                        choices=['xavier', 'kaiming', 'normal', 'orthogonal', 'constant'],
                        help='Initialization method')
    parser.add_argument('--delta', type=float, default=0.1, help='weight for latent predictor loss in overall')
    parser.add_argument('--beta_vae', type=float, default=0.1,
                        help='beta parameter for the beta variational autoencoder')

    # Parse the arguments
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Override defaults with environment variables if they exist


    if 'train_data_path' in os.environ:
        args.train_data_path = os.path.normpath(os.environ['train_data_path'])

    if 'val_data_path' in os.environ:
        args.val_data_path = os.path.normpath(os.environ['val_data_path'])

    if 'batch_size' in os.environ:
        args.batch_size = int(os.environ['batch_size'])
    if 'do_eval' in os.environ:
        args.do_eval = os.environ['do_eval'].lower() == 'true'
    if 'learning_rate' in os.environ:
        args.learning_rate = float(os.environ['learning_rate'])
    if 'num_epochs' in os.environ:
        args.num_epochs = int(os.environ['num_epochs'])
    if 'model_path' in os.environ:
        args.model_path = os.environ['model_path']
    if 'predictor_model_path' in os.environ:
        args.predictor_model_path = os.environ['predictor_model_path']
    if 'log_tensorboard' in os.environ:
        args.log_tensorboard = os.environ['log_tensorboard'].lower() == 'true'
    if 'experiment_name' in os.environ:
        args.experiment_name = os.environ['experiment_name']
    if 'step_size' in os.environ:
        args.step_size = int(os.environ['step_size'])
    if 'gamma' in os.environ:
        args.gamma = float(os.environ['gamma'])
    if 'init_method' in os.environ:
        args.init_method = os.environ['init_method']
    if 'delta' in os.environ:
        args.delta = float(os.environ['delta'])
    if 'beta_vae' in os.environ:
        args.beta_vae = float(os.environ['beta_vae'])

    # Check if log_tensorboard is True and experiment_name is not provided
    if args.log_tensorboard and not args.experiment_name:
        parser.error("--experiment_name is required when --log_tensorboard is set to True")

    return args


