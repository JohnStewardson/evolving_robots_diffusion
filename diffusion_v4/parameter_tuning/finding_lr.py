import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from model_architecture import UNet
from training.train_mse import train_mse
from data_sets import SingleRobotDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from diffusion_args import add_diffusion_args
from plotting.plot_mse_loss import plot_average_mse_loss_per_epoch_multi


def prepare_noise_schedule(noise_steps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, noise_steps)


device = "cuda" if torch.cuda.is_available() else "cpu"





lrs = [1e-2, 1e-3, 3e-4, 1e-4, 1e-5]
exp_names = ['lr/1e_2', 'lr/1e_3', 'lr/3e_4', 'lr/1_4', 'lr/1_5']
labels_array = []
csv_array = []
##in loop:
for idx in range(len(lrs)):
    b = 0.02

    T = 500
    l = lrs[idx]
    labels_array.append(f"lr {l}")
    exp_name = exp_names[idx]
    csv_temp = os.path.join(exp_name, 'mse_log.csv')
    csv_array.append(csv_temp)
    # parser = argparse.ArgumentParser(description='Arguments for the single robot script')
    # add_diffusion_args(parser)
    # parser.add_argument('--epochs', type=int, default=500, help='Epochs')
    # parser.add_argument('--lr', type=float, default=l, help='Learning rate for training (default: 3e-4)')
    #
    #
    # parser.add_argument('--exp-name', type=str, default=exp_name,
    #                         help='Name of the experiment (default: 2024-08-30_First_exp_Walker_v0_00)')
    # parser.add_argument('--exp-path', type=str, default=exp_name)
    #
    # parser.add_argument('--T', type=int, default=T)
    # args = parser.parse_args()
    #
    # beta = prepare_noise_schedule(noise_steps=args.T).to(device) # beta: variance of noise added at each time step in the forward diffusion
    # alpha = 1. - beta # alpha_t = 1 - beta_t, proportion of signal retained after each time step
    # alpha_hat = torch.cumprod(alpha, axis=0) # alpha_hat_t = PI(s=1 to t): alpha_s
    #
    # diff_model = UNet()
    # print("Num params: ", sum(p.numel() for p in diff_model.parameters()))
    #
    # size_data = 64
    # robot_matrix = np.array([
    #     [2, 2, 2, 2, 2],
    #     [4, 4, 0, 4, 4],
    #     [4, 4, 0, 4, 4],
    #     [4, 4, 0, 4, 4],
    #     [3, 3, 0, 3, 3]
    # ])
    #
    #
    # dataset = SingleRobotDataset(size_data, robot_matrix)
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    #
    # train_mse(dataloader, diff_model, alpha_hat, args)
    # del diff_model

plot_average_mse_loss_per_epoch_multi(csv_array, labels_array,min_epoch=100 ,max_epoch=200, output_path='mse_loss_lrs_first_100_200.pdf')
