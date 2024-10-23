import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
from torch.utils.data import DataLoader
from model_architecture import UNet
from sample import prepare_noise_schedule, sample_valid_robots
from training.train_mse import train_mse
from data_sets import GenerationDataset

def train_and_sample_job(survivor_robots, robot_matrix_hashes, args):
    print("Starting combined train and sample job...")
    device, T, pop_size, num_survivors = args.device, args.T, args.pop_size, args.num_survivors
    diff_model = UNet().to(device)
    beta = prepare_noise_schedule(noise_steps=T).to(device)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, axis=0)

    # Train the diffusion model
    print("Starting MSE training...")
    num_samples = 64
    data_set = GenerationDataset(survivor_robots, num_samples=num_samples)
    dataloader = DataLoader(data_set, batch_size=num_survivors, shuffle=True)
    train_mse(dataloader, diff_model, alpha_hat, args)
    print("Finished MSE training.")

    print("Starting robot sampling...")
    with torch.no_grad():
        robots_matrices = sample_valid_robots(diff_model, pop_size, alpha, alpha_hat, beta, args, robot_matrix_hashes)

    robots_matrices = [robot.astype(float) for robot in robots_matrices]
    print("Finished sampling robots.")

    return robots_matrices

