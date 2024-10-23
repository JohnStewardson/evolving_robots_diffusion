import torch
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sample import sample_timesteps, prepare_noise_schedule
from adding_noise.adding_noise import forward_diffusion_sample
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import csv

def train_mse(dataloader, model, alpha_hat, args):
    device, T, exp_name, exp_path = (
        args.device,
        args.T,
        args.exp_name,
        args.exp_path
    )
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    home_path = os.path.join(exp_path)
    os.makedirs(home_path, exist_ok=True)
    csv_path = os.path.join(home_path, 'mse_log.csv')
    model_path = os.path.join(home_path, 'models')
    os.makedirs(model_path, exist_ok=True)

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

        #torch.save(model.state_dict(), os.path.join(model_path, "final.pt"))
        print("Finished")



