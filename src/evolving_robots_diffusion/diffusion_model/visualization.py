import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from evolving_robots_diffusion.diffusion_model.datasets import SingleRobotDataset
from evolving_robots_diffusion.diffusion_model.adding_noise import forward_diffusion_sample, prepare_noise_schedule
from evolving_robots_diffusion.diffusion_model.transform_data import tensor_to_robots
from evolving_robots_diffusion.plotting.plot_robot import plot_matrix_and_save


def visualize_forward_process(
    robot_matrix,
    out_dir="outputs/vis_forward",
    size_data=64,
    T=500,
    num_images=10,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SingleRobotDataset(size_data, np.array(robot_matrix))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    beta = prepare_noise_schedule(noise_steps=T).to(device) # beta: variance of noise added at each time step in the forward diffusion
    alpha = 1.0 - beta # alpha_t = 1 - beta_t, proportion of signal retained after each time step
    alpha_hat = torch.cumprod(alpha, dim=0) # alpha_hat_t = PI(s=1 to t): alpha_s

    robot_tensor = next(iter(dataloader))[0]

    os.makedirs(out_dir, exist_ok=True)

    step_size = max(1, T // num_images)

    for idx in range(0, T, step_size):
        t = torch.tensor([idx], dtype=torch.int64, device=device) # tensor containing current time_step
        noisy_robot, _ = forward_diffusion_sample(
            robot_tensor.to(device), t, alpha_hat
        )
        robots = tensor_to_robots(noisy_robot)
        plot_matrix_and_save(
            robots[0],
            out_path=os.path.join(out_dir, f"noisy_robot_step_{idx}_{T}.pdf"),
        )