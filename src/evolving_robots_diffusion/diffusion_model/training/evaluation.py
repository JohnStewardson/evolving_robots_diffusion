import torch
import torch.nn as nn
import numpy as np

from evolving_robots_diffusion.diffusion_model.sample import sample
from evolving_robots_diffusion.diffusion_model.datasets import SingleRobotDataset


def compute_single_robot_sampling_mse(
    robot_matrix,
    diff_model,
    alpha,
    alpha_hat,
    beta,
    args,
    batch_size=8,
    size_data=64,
):  
    """
    Function to compute loss of samples to one robot
    """

    device = args.device
    diff_model = diff_model.to(device)
    mse = nn.MSELoss()  # Mean Squared Error loss for training


    dataset = SingleRobotDataset(size_data, robot_matrix)
    reference_batch = next(
        iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False))
    ).to(device)

    sampled_batch = sample(diff_model, batch_size, alpha, alpha_hat, beta, args)

    loss = mse(sampled_batch, reference_batch)
    print(f"Avg MSE: {loss.item()}")
    return loss.item()


def count_exact_matches(sampled_robots, target_robot):
    overfitted = 0
    num_sampled_robots = len(sampled_robots)

    for sampled_robot in sampled_robots:
        if np.array_equal(sampled_robot, target_robot):
            overfitted+=1
    
    print(f"Reconstructed: {overfitted} out of {num_sampled_robots}")

