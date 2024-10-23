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
# def prepare_noise_schedule(noise_steps, beta_start=0.0001, beta_end=0.02):
#     return torch.linspace(beta_start, beta_end, noise_steps)
#
# parser = argparse.ArgumentParser(description='Arguments for the single robot script')
# add_diffusion_args(parser)
# parser.add_argument('--epochs', type=int, default=500, help='Epochs')
# exp_name = '2024-10_02_test'
# parser.add_argument('--exp-name', type=str, default=exp_name,
#                         help='Name of the experiment (default: 2024-08-30_First_exp_Walker_v0_00)')
# parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for training (default: 3e-4)')
# parser.add_argument('--exp-path', type=str, default=exp_name)
# parser.add_argument('--T', type=int, default=500)
# args = parser.parse_args()
#
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
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
#
# labels_array = ['beta_end: 0.02, T: 500']
# csv_array = ['2024-10_02_test/mse_log.csv']
# plot_average_mse_loss_per_epoch_multi(csv_array, labels_array, max_epoch=500)

def prepare_noise_schedule(noise_steps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, noise_steps)


device = "cuda" if torch.cuda.is_available() else "cpu"




bs = [0.01, 0.015, 0.02, 0.03, 0.05]
Ts = [600, 600, 500, 400, 200]
exp_names = ['b01_T600', 'b015_T600', 'b02T500', 'b03_T400', 'b05_T200']
labels_array = []
csv_array = []
##in loop:
for idx in range(len(bs)):
    b = bs[idx]
    labels_array.append(f"beta {b}")
    T = Ts[idx]
    exp_name = exp_names[idx]
    csv_temp = os.path.join(exp_name, 'mse_log.csv')
    csv_array.append(csv_temp)
    # parser = argparse.ArgumentParser(description='Arguments for the single robot script')
    # add_diffusion_args(parser)
    # parser.add_argument('--epochs', type=int, default=250, help='Epochs')
    # parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for training (default: 3e-4)')
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

plot_average_mse_loss_per_epoch_multi(csv_array, labels_array, max_epoch=100, min_epoch=5, output_path='mse_loss_betas_5_100.pdf')
