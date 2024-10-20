import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

from utils.data_loader import get_dataloader
from utils.parse_args import get_args
from vae.scripts.evaluation.factor_vae_metric import representation_function, metric_factor_vae
from vae.scripts.models.initializations import get_initializer
from vae.scripts.models.variational_autoencoder import VAE
from vae.scripts.train_vae_loss_predicotrs import train_models
from vae.scripts.models.variational_latent_predictors import LatentPredictor_x2, LatentPredictor_x3, LatentPredictor_x4, LatentPredictor_x5, \
    LatentPredictor_x6, LatentPredictor_x7, LatentPredictor_x8, LatentPredictor_x9, LatentPredictor_x10, \
    LatentPredictor_x11, LatentPredictor_x12, LatentPredictor_x13, LatentPredictor_x14, LatentPredictor_x15, \
    LatentPredictor_x16

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_dotenv()
    args = get_args()

    # Print all the argument values
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    torch.autograd.set_detect_anomaly(True)
    train_loader = get_dataloader(args.train_data_path, batch_size=args.batch_size)
    val_loader = get_dataloader(args.val_data_path, batch_size=args.batch_size)

    # Fix delta
    delta = 0.3

    # Define epoch numbers to evaluate
    epoch_numbers = [30, 100, 400, 1000]

    # Create a SummaryWriter for TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(r'C:\Users\User\Desktop\thesis\vae\experiments\experiments_run', f"{timestamp}_fixed_delta_{delta}_varying_epochs")
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Training with fixed delta = {delta}")

    data_to_plot = np.load(r'C:\Users\User\Desktop\thesis\data\data_to_plot.npy')

    for num_epochs in epoch_numbers:
        print(f"Training for {num_epochs} epochs")

        vae_model = VAE()
        predictors = [LatentPredictor_x2(), LatentPredictor_x3(), LatentPredictor_x4(), LatentPredictor_x5(),
                      LatentPredictor_x6(), LatentPredictor_x7(), LatentPredictor_x8(), LatentPredictor_x9(),
                      LatentPredictor_x10(), LatentPredictor_x11(), LatentPredictor_x12(), LatentPredictor_x13(),
                      LatentPredictor_x14(), LatentPredictor_x15(), LatentPredictor_x16()]

        initializer = get_initializer(args.init_method)
        vae_model.apply(initializer)
        for predictor in predictors:
            predictor.apply(initializer)

        criterion = nn.MSELoss()

        optimizer_ae = optim.Adam(vae_model.parameters(), lr=args.learning_rate)
        optimizers_predictors = [optim.Adam(predictor.parameters(), lr=args.learning_rate) for predictor in predictors]
        scheduler_ae = lr_scheduler.StepLR(optimizer_ae, step_size=args.step_size, gamma=args.gamma)
        schedulers = []

        # Train the model
        vae_model, predictors, _ = train_models(train_loader, val_loader, vae_model,
                                                predictors, criterion, optimizer_ae, optimizers_predictors,
                                                scheduler_ae, schedulers, num_epochs=num_epochs,
                                                log_dir=log_dir, delta=delta, beta_vae=args.beta_vae,
                                                log_tensorboard=False, do_eval=args.do_eval, data_to_plot=data_to_plot, device=device)

        # Evaluate the model using Factor VAE metric
        vae_model.eval()

        def rep_function(x):
            return representation_function(vae_model, x, device=device)

        results = metric_factor_vae(rep_function, num_train=1000, num_eval=500)

        # Log Factor VAE metric to TensorBoard
        writer.add_scalar('FactorVAE/eval_accuracy', results['factor_vae.eval_accuracy'], num_epochs)
        writer.add_scalar('FactorVAE/train_accuracy', results['factor_vae.train_accuracy'], num_epochs)
        writer.add_scalar('FactorVAE/num_active_dims', results['factor_vae.num_active_dims'], num_epochs)
        writer.flush()

        print(f"Evaluation score for {num_epochs} epochs: {results['factor_vae.eval_accuracy']}")

    writer.close()

    print(f"TensorBoard logs saved to {log_dir}")
    print("To view the results, run:")
    print(f"tensorboard --logdir={log_dir}")