import torch
from torch.utils.data import DataLoader
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_sets import SingleRobotDataset
from adding_noise import forward_diffusion_sample
from transform_data import tensor_to_robots
from plotting.plot_robot import plot_matrix_and_save
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np


robot_matrix = np.array([
    [2, 2, 2, 2, 2],
    [4, 4, 0, 4, 4],
    [4, 4, 0, 4, 4],
    [4, 4, 0, 4, 4],
    [3, 3, 0, 3, 3]
])
size_data = 64
dataset = SingleRobotDataset(size_data, robot_matrix)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

def prepare_noise_schedule(noise_steps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, noise_steps)

device = "cuda" if torch.cuda.is_available() else "cpu"
T = 500
beta = prepare_noise_schedule(noise_steps=T).to(device) # beta: variance of noise added at each time step in the forward diffusion
alpha = 1. - beta # alpha_t = 1 - beta_t, proportion of signal retained after each time step
alpha_hat = torch.cumprod(alpha, axis=0) # alpha_hat_t = PI(s=1 to t): alpha_s

robot_tensor = next(iter(dataloader))[0]


num_images = 10
stepsize = int(T/num_images)
os.makedirs("vis_forward", exist_ok=True)
for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64) # tensor containing current time_step
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    noisy_robot, noise = forward_diffusion_sample(robot_tensor.to(device), t.to(device), alpha_hat)
    #print(f"Noisy robot: {noisy_robot}")
    robots = tensor_to_robots(noisy_robot)
    #print(f"robots: {robots}")
    plot_matrix_and_save(robots[0], out_path=f"vis_forward/noisy_robot_step_{idx}_500.pdf")

