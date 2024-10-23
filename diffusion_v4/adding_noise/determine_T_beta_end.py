import torch
from torch.utils.data import DataLoader
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_sets import TwoMatrixDataset
import numpy as np

def evaluate_randomness_at_noise_level(T, device, num_samples=100000, timestep=None):
    """
        Evaluate the randomness of the outputs after applying noise at a given timestep.
        i.e. take as input two matrix, apply noise at t for many, calculate variance and mean
        This function is used to determine what T and beta_max should be
        Parameters:
        - dataloader: DataLoader object to load the dataset.
        - T: Total number of timesteps.
        - device: Device to perform computations on.
        - num_samples: Number of samples to evaluate.
        - timestep: Timestep at which to apply noise. If None, defaults to max noise level (T-1).

    """
    dataset = TwoMatrixDataset(num_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Use the provided timestep or default to max noise level
    t_value = T - 1 if timestep is None else timestep

    all_noised_samples = []

    # Go through the dataset
    for i, data in enumerate(dataloader):
        if i >= num_samples:
            break

        # Move the data to the appropriate device
        x = data.to(device)

        # Apply forward diffusion at timestep t
        x_noised, noise = forward_diffusion_sample(x, t_value)

        # Collect the noised samples
        all_noised_samples.append(x_noised.cpu().numpy())  # Move to CPU and convert to numpy

    # Convert all samples to a single numpy array for analysis
    all_noised_samples = np.concatenate(all_noised_samples, axis=0)

    # Compute the mean and variance of the noised samples
    mean_noise = np.mean(all_noised_samples)
    variance_noise = np.var(all_noised_samples)

    return mean_noise, variance_noise
def prepare_noise_schedule(noise_steps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, noise_steps)


def forward_diffusion_sample(x, t):
    """
    Add noise to sample x, at time t, return x with added noise, and noise
    """
    # Generate Gaussian noise with the same shape as x
    noise = torch.randn_like(x)

    # Compute the scaling factors for x and noise at time step t
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t]).view(-1, 1, 1, 1)  # Scaling for the signal x
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t]).view(-1, 1, 1, 1)  # Scaling for the noise

    # Apply noise to the input x at time step t
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise


device = "cuda" if torch.cuda.is_available() else "cpu"


Ts = [100, 200, 300, 400, 500, 600]
bs = [0.01, 0.015, 0.03, 0.05]
for b in bs:
    print(f"beta_end: {b}")
    for T in Ts:
        beta = prepare_noise_schedule(noise_steps=T, beta_end=b).to(
            device)  # beta: variance of noise added at each time step in the forward diffusion
        alpha = 1. - beta  # alpha_t = 1 - beta_t, proportion of signal retained after each time step
        alpha_hat = torch.cumprod(alpha, axis=0)  # alpha_hat_t = PI(s=1 to t): alpha_s
        mean, variance = evaluate_randomness_at_noise_level(T, device)
        print(f"T: {T}, mean: {mean}, variance: {variance}")
