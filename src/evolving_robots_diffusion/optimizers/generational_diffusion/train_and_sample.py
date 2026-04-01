import torch
from torch.utils.data import DataLoader
import os

from evolving_robots_diffusion.diffusion_model.model_architecture import UNet
from evolving_robots_diffusion.diffusion_model.training.train_mse import train_mse
from evolving_robots_diffusion.diffusion_model.datasets import GenerationDataset
from evolving_robots_diffusion.diffusion_model.adding_noise import prepare_noise_schedule
from evolving_robots_diffusion.diffusion_model.sample import sample_valid_robots

def train_and_sample_job(
        survivor_robots,
        robot_matrix_hashes, 
        args,
        generation_path=None,
        saved_weights_path=None):
    
    print("Starting combined train and sample job...")
    device, T, pop_size, num_survivors = args.device, args.T, args.pop_size, args.num_survivors
    diff_model = UNet().to(device)

    # if saved_weights_path is not None and os.path.exists(saved_weights_path):
    #     print(f"Warm-starting diffusion model from: {saved_weights_path}")
    #     diff_model.load_state_dict(torch.load(saved_weights_path, map_location=device))


    beta = prepare_noise_schedule(noise_steps=T).to(device)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, axis=0)

    # Train the diffusion model
    print("Starting MSE training...")
    num_samples = 64
    data_set = GenerationDataset(survivor_robots, num_samples=num_samples)
    dataloader = DataLoader(data_set, batch_size=num_survivors, shuffle=True)
    
    train_mse(dataloader, diff_model, alpha_hat, args, output_dir=generation_path)
    print("Finished MSE training.")

    print("Starting robot sampling...")
    with torch.no_grad():
        robots_matrices = sample_valid_robots(diff_model, pop_size, alpha, alpha_hat, beta, args, robot_matrix_hashes)

    robots_matrices = [robot.astype(float) for robot in robots_matrices]
    print("Finished sampling robots.")

    return robots_matrices

