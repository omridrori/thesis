import pandas as pd
import itertools
import json

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
from vae.scripts.evaluation.mutual_information_gap_metric import compute_mig
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

    # Define delta values in log scale
    beta_vals = [0.000011] # 10 values from 10^-3 to 10^1
    delta_values = [1,0.5,0.1,0.0001,0.00001]
    lr_diff = [500]
    regular_lr=[0.01]
    # beta_vals = [0.1] # 10 values from 10^-3 to 10^1
    # delta_values = [0.1]
    # lr_diff = [500]
    # regular_lr=[0.01]


    configurations = list(itertools.product(delta_values, beta_vals, lr_diff))
    config_map = {i: config for i, config in enumerate(configurations)}


    # Create a SummaryWriter for TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(r'C:\Users\User\Desktop\thesis\vae\experiments\experiments_run', f"{timestamp}_delta_experiments")
    writer = SummaryWriter(log_dir=log_dir)
    with open(os.path.join(log_dir, 'config_map.json'), 'w') as f:
        json.dump({str(k): v for k, v in config_map.items()}, f, indent=2)
    step=0
    results_list = []

    for lr in regular_lr:
       for delta in delta_values:
           for beta in beta_vals:
               for learning_rate_diff in lr_diff:
                   step += 1

                   print(f"step {step} : Training with delta = {delta}, beta = {beta},lr = {lr} lr_diff = {learning_rate_diff}")

                   vae_model = VAE()
                   predictors = [LatentPredictor_x2(), LatentPredictor_x3(), LatentPredictor_x4(), LatentPredictor_x5(),
                                 LatentPredictor_x6(), LatentPredictor_x7(), LatentPredictor_x8(), LatentPredictor_x9(),
                                 LatentPredictor_x10(), LatentPredictor_x11(), LatentPredictor_x12(),
                                 LatentPredictor_x13(),
                                 LatentPredictor_x14(), LatentPredictor_x15(), LatentPredictor_x16()]

                   initializer = get_initializer(args.init_method)
                   vae_model.apply(initializer)
                   for predictor in predictors:
                       predictor.apply(initializer)

                   criterion = nn.MSELoss()

                   optimizer_ae = optim.Adam(vae_model.parameters(), lr=lr)
                   lr_predictors = args.learning_rate / learning_rate_diff
                   optimizers_predictors = [optim.Adam(predictor.parameters(), lr=lr_predictors) for predictor in
                                            predictors]
                   scheduler_ae = lr_scheduler.StepLR(optimizer_ae, step_size=args.step_size, gamma=args.gamma)
                   schedulers = []

                   data_to_plot = np.load(r'C:\Users\User\Desktop\thesis\data\data_to_plot.npy')

                   # Modify train_models to return the training history
                   vae_model, predictors, history = train_models(train_loader, val_loader, vae_model,
                                                                 predictors, criterion, optimizer_ae,
                                                                 optimizers_predictors,
                                                                 scheduler_ae, schedulers, num_epochs=350,
                                                                 log_dir=log_dir, delta=delta, beta_vae=beta,
                                                                 log_tensorboard=True, do_eval=args.do_eval,
                                                                 data_to_plot=data_to_plot, device=device)

                   # Evaluate the model using Factor VAE metric
                   vae_model.eval()


                   def rep_function(x, device):
                       return representation_function(vae_model, x, device=device)


                   results = metric_factor_vae(rep_function, num_train=10000, num_eval=5000, device=device)
                   vae_model.eval()
                   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                   # Compute MIG metric
                   mig_score = compute_mig(vae_model, num_train=10000, batch_size=32, seed=17)

                   # Log Factor VAE metric to TensorBoard
                   # writer.add_scalar('FactorVAE/eval_accuracy', results['factor_vae.eval_accuracy'], delta)
                   # writer.add_scalar('FactorVAE/train_accuracy', results['factor_vae.train_accuracy'], delta)
                   # writer.add_scalar('FactorVAE/num_active_dims', results['factor_vae.num_active_dims'], delta)
                   results_list.append({
                       'lr': lr,
                       'delta': delta,
                       'beta': beta,
                       'lr_diff': learning_rate_diff,
                       'factor_vae_eval_accuracy': results['factor_vae.eval_accuracy'],
                       'mig_score': mig_score
                   })
                   print(f"Factor VAE eval accuracy: {results['factor_vae.eval_accuracy']} MIG Score: {mig_score}")

    writer.close()
    df_results = pd.DataFrame(results_list)
    csv_path="./results3.csv"
    df_results.to_csv("./results3.csv", index=False)
    print(f"Results saved to {csv_path}")
    print(f"TensorBoard logs saved to {log_dir}")
    print("To view the results, run:")
    print(f"tensorboard --logdir={log_dir}")