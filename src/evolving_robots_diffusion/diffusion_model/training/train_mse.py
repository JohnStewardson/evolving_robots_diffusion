import torch
import numpy as np
import torch.nn as nn
import torch
import os
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import csv

from evolving_robots_diffusion.diffusion_model.sample import sample_timesteps
from evolving_robots_diffusion.diffusion_model.adding_noise import forward_diffusion_sample


def train_mse(dataloader, model, alpha_hat, args, output_dir=None, save_model=False):
    device = args.device

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    if output_dir is None:
        output_dir = args.exp_name

    if output_dir is not None and not os.path.exists(output_dir) and save_model:
        os.makedirs(output_dir, exist_ok=True)
    

    logger = SummaryWriter(os.path.join(output_dir, "tensorboard"))
    csv_path = os.path.join(output_dir, "mse_log.csv")

    l = len(dataloader)
    

    # create csv file to log validity information
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(['Epoch', 'MSE_Loss'])
        writer.writerow(['Batch', 'Epoch', 'MSE_Loss'])
        batch_number = 0
        for epoch in range(args.epochs):
            losses = []
            pbar = tqdm(dataloader)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}")
            for i, robot_structures in enumerate(pbar):

                x = robot_structures.to(device).float()
                #print(f"x: {x}")
                # Algo 1, line 3
                t = sample_timesteps(x.shape[0], args).to(device) # samples random timestep for each image
                x_t, noise = forward_diffusion_sample(x, t, alpha_hat)
                #print(f"x_t: {x_t}")
                # Algo 1, line 4
                predicted_noise = model(x_t, t)
                loss = mse(noise, predicted_noise)
                losses.append(loss.item())
                # edit loss function to incorporate rewards
                # Algo line 5
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.writerow([batch_number, epoch, loss.item()])
                batch_number += 1
                pbar.set_postfix(MSE=loss.item())
                logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        
        if save_model:
            save_model_path = os.path.join(output_dir, "model")
            os.makedirs(save_model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_model_path, "weights.pt"))
        
        logger.close()
        print("Finished")



