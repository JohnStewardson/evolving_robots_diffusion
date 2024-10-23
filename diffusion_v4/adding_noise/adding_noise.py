import torch
from torch.utils.data import DataLoader
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_sets import SingleRobotDataset
import numpy as np




def prepare_noise_schedule(noise_steps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, noise_steps)

def forward_diffusion_sample(x, t, alpha_hat):
    """
    Add noise to sample x, at time t, return x with added noise, and noise
    """
    noise = torch.randn_like(x)
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]

    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

# robot_matrix = np.array([
#     [2, 2, 2, 2, 2],
#     [4, 4, 0, 4, 4],
#     [4, 4, 0, 4, 4],
#     [4, 4, 0, 4, 4],
#     [3, 3, 0, 3, 3]
# ])
# size_data = 64
# dataset = SingleRobotDataset(size_data, robot_matrix)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# T = 300
# beta = prepare_noise_schedule(noise_steps=T).to(device) # beta: variance of noise added at each time step in the forward diffusion
# alpha = 1. - beta # alpha_t = 1 - beta_t, proportion of signal retained after each time step
# alpha_hat = torch.cumprod(alpha, axis=0) # alpha_hat_t = PI(s=1 to t): alpha_s

