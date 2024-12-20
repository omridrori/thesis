import numpy as np
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter

from autoencoders.scripts.autoencoder_plot_modificatiosn import generate_plots_modifications
from models.autoencoder import Autoencoder
from models.initializations import constant_init, normal_init, kaiming_uniform_init, xavier_uniform_init, \
    orthogonal_init, get_initializer
from models.latent_predictor import LatentPredictor_x2, LatentPredictor_x3, LatentPredictor_x4, LatentPredictor_x5, \
    LatentPredictor_x6, LatentPredictor_x7, LatentPredictor_x8, LatentPredictor_x9, LatentPredictor_x10, \
    LatentPredictor_x11, LatentPredictor_x12, LatentPredictor_x13, LatentPredictor_x14, LatentPredictor_x15, \
    LatentPredictor_x16
from utils.data_loader import get_dataloader
from utils.parse_args import get_args
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from utils.save_models import save_models

torch.autograd.set_detect_anomaly(True)




def loss_predict_with_predictors(predictors, data, encoded_normalized):
    z = encoded_normalized.cpu()
    z_preds = []
    val_predictors_losses_individuals = torch.zeros(data.size(0))
    for i in range(1, len(predictors)):
        z_input = z[:, :i]
        z_pred = predictors[i - 1](z_input)
        z_val = z[:, i].unsqueeze(1)
        curr_loss = (z_pred - z_val) ** 2
        val_predictors_losses_individuals = val_predictors_losses_individuals + curr_loss.squeeze()

    return val_predictors_losses_individuals



def predictors_step(train_loader,autoencoder, predictors, optimizers, delta,device):
    autoencoder.eval()
    for predictor in predictors:
        predictor.train()

    loss_predictors_total = 0.0
    for batch_idx, data in enumerate(train_loader):
        data=data.to(device)
        for optimizer in optimizers:
            optimizer.zero_grad()
        encoded = autoencoder.encoder(data)
        encoded_normalized = F.normalize(encoded, p=2, dim=1)
        val_predictors_losses_individuals = loss_predict_with_predictors(predictors, data, encoded_normalized)
        losses_individuals = val_predictors_losses_individuals / len(predictors)
        loss_predictors = losses_individuals.mean()
        loss_predictors.backward()
        for optimizer in optimizers:
            optimizer.step()

        loss_predictors_total += loss_predictors.item()

    return loss_predictors_total/ len(train_loader)



def autoencoder_step(train_loader, autoencoder, predictors, optimizer_ae, delta,device):
    autoencoder.train()

    running_loss_ae = 0.0
    running_loss_total = 0.0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer_ae.zero_grad()
        # Forward pass through autoencoder
        encoded = autoencoder.encoder(data)
        # Normalize the encoded vector
        encoded_normalized = F.normalize(encoded, p=2, dim=1)
        decoded = autoencoder.decoder(encoded)

        val_predictors_losses_individuals = loss_predict_with_predictors(predictors, data, encoded_normalized)
        losses_individuals = val_predictors_losses_individuals / len(predictors)
        # Calculate total loss for each example
        total_loss_individual = torch.sum((decoded - data) ** 2, dim=1) - delta * losses_individuals.to(device)
        total_loss= total_loss_individual.mean()

        total_loss.backward()
        optimizer_ae.step()

        running_loss_total += total_loss.item()
        with torch.no_grad():
           ae_loss = criterion(decoded, data)
        running_loss_ae += ae_loss.item()

    return running_loss_total/ len(train_loader), running_loss_ae/ len(train_loader)



def train_models(train_loader, val_loader, autoencoder, predictors, criterion,
                 optimizer_ae, optimizers_predictors, scheduler_ae, schedulers, num_epochs, log_dir, delta=0.1,
                 l2_lambda=1e-5, clip_value=1,do_eval=True,log_tensorboard=True,data_to_plot=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):



        for predictor in predictors:
            for param in predictor.parameters():
                param.requires_grad = False


        mean_train_loss_total,mean_train_loss_ae=autoencoder_step(train_loader, autoencoder, predictors, optimizer_ae, delta,device)

        # Unfreeze latent predictor parameters
        for param in autoencoder.parameters():
            param.requires_grad = False
        for predictor in predictors:
            for param in predictor.parameters():
                param.requires_grad = True

        mean_train_losses_predictors=predictors_step(train_loader, autoencoder, predictors, optimizers_predictors, delta,device)


        for param in autoencoder.parameters():
            param.requires_grad = True




        if log_tensorboard:
            # Log the mean training loss
            writer.add_scalar('Loss/train_ae', mean_train_loss_ae, epoch)
            writer.add_scalar('Loss/train_total', mean_train_loss_total, epoch)
            writer.add_scalar('Loss/train_predictors', mean_train_losses_predictors, epoch)
            if data_to_plot is not None:
                plots = generate_plots_modifications(autoencoder, data_to_plot,device=device)
                for i, fig in enumerate(plots):
                    writer.add_figure(f"Latent Variable {i + 1} Modification", fig, global_step=epoch)
        if do_eval:
            # Evaluate on validation set
            autoencoder.eval()
            for predictor in predictors:
                predictor.eval()
            val_loss_ae = 0.0
            val_loss_total = 0.0
            val_loss_predictors = 0.0
            with torch.no_grad():
                for data in val_loader:
                    mean_train_loss_ae, mean_train_loss_total = autoencoder_step(train_loader, autoencoder, predictors,
                                                                                 optimizer_ae, delta, device)
                    val_loss_ae += mean_train_loss_ae
                    val_loss_total += mean_train_loss_total
                    val_loss_predictors += predictors_step(train_loader, autoencoder, predictors, optimizers_predictors, delta, device)
                    Val_loss_predictors = val_loss_predictors / len(val_loader)




                mean_val_loss_ae = val_loss_ae / len(val_loader)
                mean_val_loss_total = val_loss_total / len(val_loader)
                mean_val_loss_predictors = val_loss_predictors / len(val_loader)


                # Log the mean validation loss
                if log_tensorboard:
                    writer.add_scalar('Loss/val_ae', mean_val_loss_ae, epoch)
                    writer.add_scalar('Loss/val_total', mean_val_loss_total, epoch)
                    writer.add_scalar('Loss/val_predictors', mean_val_loss_predictors, epoch)



            # # Step the learning rate scheduler
            # scheduler_ae.step()
            # for scheduler in schedulers:
            #     scheduler.step()

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss Total: {mean_train_loss_total:.4f}, Train Loss AE: {mean_train_loss_ae:.4f}, Train loss_predictors: {mean_train_losses_predictors:.4f}')

        if do_eval:
            print(
                f' \n Val Loss Total: {mean_val_loss_total:.4f}, Val Loss AE: {mean_val_loss_ae:.4f}, Val loss_predictors: {mean_val_loss_predictors:.4f}')

    writer.close()





# Example of how to call train_models in your main script
if __name__ == "__main__":
    load_dotenv()
    args = get_args()

    # Print all the argument values
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    train_loader = get_dataloader(args.train_data_path,
                                  batch_size=args.batch_size)
    val_loader = get_dataloader(args.val_data_path,
                                batch_size=args.batch_size)


    autoencoder = Autoencoder()
    predictors = [LatentPredictor_x2(), LatentPredictor_x3(), LatentPredictor_x4(), LatentPredictor_x5(),
                   LatentPredictor_x6(), LatentPredictor_x7(), LatentPredictor_x8(), LatentPredictor_x9(),
                   LatentPredictor_x10(), LatentPredictor_x11(), LatentPredictor_x12(), LatentPredictor_x13(),
                   LatentPredictor_x14(), LatentPredictor_x15(), LatentPredictor_x16()]

    initializer = get_initializer(args.init_method)
    autoencoder.apply(initializer)
    for predictor in predictors:
        predictor.apply(initializer)

    criterion = nn.MSELoss()

    optimizer_ae = optim.Adam(autoencoder.parameters(), lr=args.learning_rate)
    optimizers_predictors = [optim.Adam(predictor.parameters(), lr=args.learning_rate) for predictor in predictors]
    scheduler_ae = lr_scheduler.StepLR(optimizer_ae, step_size=args.step_size, gamma=args.gamma)
    # schedulers = [lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) for optimizer in optimizers]
    schedulers=[]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('./runs', f"{timestamp}_{args.experiment_name}" if args.experiment_name else timestamp)
    data_to_plot = np.load(r'C:\Users\User\Desktop\thesis\data\data_to_plot.npy')


    #todo  change do_eval to args.do_eval and log_tensorboard to args.log_tensorboard
    train_models(train_loader, val_loader, autoencoder, predictors, criterion,
                 optimizer_ae, optimizers_predictors, scheduler_ae, schedulers,
                 num_epochs=args.num_epochs, log_dir=log_dir, delta=args.delta,
                 log_tensorboard=args.log_tensorboard, do_eval=args.do_eval,data_to_plot=data_to_plot)

    save_models(autoencoder, predictors,path=args.model_path)