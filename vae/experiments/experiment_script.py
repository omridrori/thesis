import os
from datetime import datetime

import torch

from autoencoders.scripts.models.initializations import get_initializer
from autoencoders.scripts.models.latent_predictor import LatentPredictor_x2, LatentPredictor_x3, LatentPredictor_x4, LatentPredictor_x5, \
    LatentPredictor_x6, LatentPredictor_x7, LatentPredictor_x8, LatentPredictor_x9, LatentPredictor_x10, \
    LatentPredictor_x11, LatentPredictor_x12, LatentPredictor_x13, LatentPredictor_x14, LatentPredictor_x15, \
    LatentPredictor_x16
from utils.data_loader import get_dataloader
from utils.parse_args import get_args
from vae.scripts.models.variational_autoencoder import VAE
from vae.scripts.train_vae_loss_predicotrs import train_models

if __name__ == "__main__":
    args = get_args()
    torch.autograd.set_detect_anomaly(True)

    train_loader = get_dataloader(r"C:\Users\User\Desktop\thesis\data\toy_dataset_train.csv",
                                  batch_size=args.batch_size)
    val_loader = get_dataloader(r"C:\Users\User\Desktop\thesis\data\toy_dataset_val.csv", batch_size=args.batch_size)

    delta_values = [0.1, 0.5, 1.0]
    beta_vae_values = [0.1, 1.0, 5.0]

    for delta in delta_values:
        for beta_vae in beta_vae_values:
            print(f"Starting experiment with delta={delta}, beta_vae={beta_vae}")
            args.delta = delta
            args.beta_vae = beta_vae
            args.experiment_name = f"delta_{delta}_betaVAE_{beta_vae}"

            variational_autoencoder = VAE()
            predictors = [LatentPredictor_x2(), LatentPredictor_x3(), LatentPredictor_x4(), LatentPredictor_x5(),
                          LatentPredictor_x6(), LatentPredictor_x7(), LatentPredictor_x8(), LatentPredictor_x9(),
                          LatentPredictor_x10(), LatentPredictor_x11(), LatentPredictor_x12(), LatentPredictor_x13(),
                          LatentPredictor_x14(), LatentPredictor_x15(), LatentPredictor_x16()]

            initializer = get_initializer(args.init_method)
            variational_autoencoder.apply(initializer)
            for predictor in predictors:
                predictor.apply(initializer)

            criterion = torch.nn.MSELoss()
            optimizer_ae = torch.optim.Adam(variational_autoencoder.parameters(), lr=args.learning_rate)
            optimizers_predictors = [torch.optim.Adam(predictor.parameters(), lr=args.learning_rate) for predictor in
                                     predictors]
            scheduler_ae = torch.optim.lr_scheduler.StepLR(optimizer_ae, step_size=args.step_size, gamma=args.gamma)

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(r'C:\Users\User\Desktop\thesis\vae\runs_vae', f"{timestamp}_{args.experiment_name}")

            train_models(train_loader, val_loader, variational_autoencoder, predictors, criterion,
                         optimizer_ae, optimizers_predictors, scheduler_ae, [],
                         num_epochs=10, log_dir=log_dir, delta=args.delta, beta_vae=args.beta_vae,
                         log_tensorboard=True, do_eval=False)

