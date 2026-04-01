import sys
import os
import torch
from torch.utils.data import DataLoader

import argparse
from evolving_robots_diffusion.diffusion_model.model_architecture import UNet
from evolving_robots_diffusion.diffusion_model.training.train_mse import train_mse
from evolving_robots_diffusion.diffusion_model.datasets import SingleRobotDataset
from evolving_robots_diffusion.diffusion_model.diffusion_args import add_diffusion_args
from evolving_robots_diffusion.diffusion_model.parameter_tuning.plot_mse_loss import plot_average_mse_loss_per_epoch_multi
import numpy as np



def prepare_noise_schedule(noise_steps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, noise_steps)


device = "cuda" if torch.cuda.is_available() else "cpu"




min_epoch = 10
max_epoch = 20
lrs = [1e-2, 1e-3, 3e-4, 1e-4, 1e-5]
exp_names = ['lr_1e_2', 'lr_1e_3', 'lr_3e_4', 'lr_1_4', 'lr_1_5']
labels_array = []
csv_array = []
##in loop:
for idx in range(len(lrs)):
    b = 0.02

    T = 500
    l = lrs[idx]
    labels_array.append(f"lr {l}")
    exp_name = exp_names[idx]
    exp_dir = os.path.join("outputs", "hyperparameter_tuning", "learning_rate", exp_name)
    csv_temp = os.path.join(exp_dir, 'mse_log.csv')
    csv_array.append(csv_temp)
    parser = argparse.ArgumentParser(description='Arguments for the single robot script')
    add_diffusion_args(parser)
    parser.add_argument('--epochs', type=int, default=max_epoch, help='Epochs (500)')
    parser.add_argument('--lr', type=float, default=l, help='Learning rate for training (default: 3e-4)')
    
    
    parser.add_argument('--exp-name', type=str, default=exp_name,
                            help='Name of the experiment (default: 2024-08-30_First_exp_Walker_v0_00)')
    parser.add_argument('--exp-path', type=str, default=exp_name)
    
    args = parser.parse_args()
    
    beta = prepare_noise_schedule(noise_steps=args.T).to(device) # beta: variance of noise added at each time step in the forward diffusion
    alpha = 1. - beta # alpha_t = 1 - beta_t, proportion of signal retained after each time step
    alpha_hat = torch.cumprod(alpha, axis=0) # alpha_hat_t = PI(s=1 to t): alpha_s
    
    diff_model = UNet()
    print("Num params: ", sum(p.numel() for p in diff_model.parameters()))
    
    size_data = 64
    robot_matrix = np.array([
        [2, 2, 2, 2, 2],
        [4, 4, 0, 4, 4],
        [4, 4, 0, 4, 4],
        [4, 4, 0, 4, 4],
        [3, 3, 0, 3, 3]
    ])
    
    
    dataset = SingleRobotDataset(size_data, robot_matrix)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    out_dir = os.path.join("outputs", "hyperparameter_tuning", "learning_rate", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    train_mse(dataloader, diff_model, alpha_hat, args, output_dir=out_dir)
    del diff_model

output_dir = os.path.join("outputs", "hyperparameter_tuning", "learning_rate")
os.makedirs(output_dir, exist_ok=True)
plot_average_mse_loss_per_epoch_multi(csv_array, labels_array, min_epoch=min_epoch ,max_epoch=max_epoch, output_path=os.path.join(output_dir,f'mse_loss_lrs_first_{min_epoch}_{max_epoch}.pdf'))
