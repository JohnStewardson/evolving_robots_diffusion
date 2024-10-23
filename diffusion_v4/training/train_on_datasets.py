import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from model_architecture import UNet
from training.train_mse import train_mse
from transform_data import tensor_to_robots
from sample import sample, sample_with_print, sample_valid_robots
from data_sets import SingleRobotDataset, GenerationDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from diffusion_args import add_diffusion_args
from plotting.plot_mse_loss import plot_average_mse_loss_per_epoch_multi
import torch.nn as nn
from plotting.plot_robot import plot_matrix_and_save


def sample_compute_mse_full(dataloader, diff_model, alpha, alpha_hat, beta, args):
    device = args.device
    diff_model = diff_model.to(device)
    mse = nn.MSELoss()  # Mean Squared Error loss for training


    # Get the first batch from the dataloader (assuming dataloader returns [batch_size, 1, 5, 5])
    first_batch = next(iter(dataloader))  # Retrieve the first batch
    first_batch = first_batch.to(device)
    n = len(first_batch)

    repeated_first_element = first_batch[:n]

    x = sample(diff_model, n, alpha, alpha_hat, beta, args)

    loss = mse(x, repeated_first_element).mean()
    print(f"Avg MSE: {loss}")
    return loss


def prepare_noise_schedule(noise_steps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, noise_steps)

parser = argparse.ArgumentParser(description='Arguments for the single robot script')
add_diffusion_args(parser)
parser.add_argument('--epochs', type=int, default=500, help='Epochs')
exp_name = '2024-10_05_test'
parser.add_argument('--exp-name', type=str, default=exp_name,
                        help='Name of the experiment (default: 2024-10-05_First_exp_Walker_v0_00)')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training (default: 3e-4)')
parser.add_argument('--exp-path', type=str, default=exp_name)
parser.add_argument('--T', type=int, default=500)
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

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
robots_data = [robot_matrix, robot_matrix, robot_matrix, robot_matrix, robot_matrix]
os.makedirs('sampling_loss', exist_ok=True)
dataset = GenerationDataset(robots_data, size_data)
#dataset = SingleRobotDataset(size_data, robot_matrix)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


#x, x_begin = sample_with_print(diff_model, 1, alpha, alpha_hat, beta, args)
#print(f"x: {x}")
#print(f"x_begin: {x_begin}")
#robots = tensor_to_robots(x)
#robots_begin = tensor_to_robots(x_begin)

#plot_matrix_and_save(robots[0])

#plot_matrix_and_save(robots_begin[0])
#loss_before = sample_compute_mse_full(dataloader, diff_model, alpha, alpha_hat, beta, args)


train_mse(dataloader, diff_model, alpha_hat, args)
#loss_after = sample_compute_mse_full(dataloader, diff_model, alpha, alpha_hat, beta, args)


#x, x_begin = sample_with_print(diff_model, 100, alpha, alpha_hat, beta, args)
#print(f"x: {x}")
#print(f"x_begin: {x_begin}")
#robots = tensor_to_robots(x)
#robots_begin = tensor_to_robots(x_begin)

robot_matrix_hashes = {}
valid_robots = sample_valid_robots(diff_model, 8, alpha, alpha_hat, beta, args, robot_matrix_hashes)
for robot in valid_robots:
    plot_matrix_and_save(robot)

# overfitted = 0
# for i in range(100):
#
#     #plot_matrix_and_save(robots[i])
#     if np.array_equal(robots[i], robot_matrix):
#         overfitted+=1
#
#     plot_matrix_and_save(robots_begin[i])
#
# print(f"overfitted: {overfitted} out of 100")
