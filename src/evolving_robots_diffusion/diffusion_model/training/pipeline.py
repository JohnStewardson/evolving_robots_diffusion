import os
import torch
from torch.utils.data import DataLoader

from evolving_robots_diffusion.diffusion_model.model_architecture import UNet
from evolving_robots_diffusion.diffusion_model.datasets import GenerationDataset, SingleRobotDataset
from evolving_robots_diffusion.diffusion_model.adding_noise import prepare_noise_schedule
from evolving_robots_diffusion.diffusion_model.sample import sample_valid_robots, sample
from evolving_robots_diffusion.diffusion_model.training.train_mse import train_mse
from evolving_robots_diffusion.plotting.plot_robot import plot_matrix_and_save
from evolving_robots_diffusion.diffusion_model.transform_data import tensor_to_robots
from evolving_robots_diffusion.diffusion_model.training.evaluation import compute_single_robot_sampling_mse, count_exact_matches


def run_training_pipeline(args, robots_data):
    """
    This function receives robots (survivors) as input, trains on them and samples again
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    beta = prepare_noise_schedule(noise_steps=args.T).to(device) # beta: variance of noise added at each time step in the forward diffusion
    alpha = 1. - beta # alpha_t = 1 - beta_t, proportion of signal retained after each time step
    alpha_hat = torch.cumprod(alpha, axis=0) # alpha_hat_t = PI(s=1 to t): alpha_s

    diff_model = UNet()
    print("Num params: ", sum(p.numel() for p in diff_model.parameters()))

    dataset = GenerationDataset(robots_data, args.size_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    ## Train
    train_mse(dataloader, diff_model, alpha_hat, args, output_dir=args.output_dir)

    ## Sample
    robot_matrix_hashes = {}
    valid_robots = sample_valid_robots(
        diff_model, args.num_samples, alpha, alpha_hat, beta, args, robot_matrix_hashes
    )

    ## Save the sampled robots
    sample_plot_dir = os.path.join(args.output_dir, "sampled_robots")
    os.makedirs(sample_plot_dir, exist_ok=True)
    for i, robot in enumerate(valid_robots):
        plot_matrix_and_save(robot, out_path=f"{sample_plot_dir}/robot_{i}.pdf")


def run_overfitting_pipeline(robot_matrix, args):

    size_data = args.size_data
    batch_size = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    beta = prepare_noise_schedule(noise_steps=args.T).to(device) # beta: variance of noise added at each time step in the forward diffusion
    alpha = 1. - beta # alpha_t = 1 - beta_t, proportion of signal retained after each time step
    alpha_hat = torch.cumprod(alpha, axis=0) # alpha_hat_t = PI(s=1 to t): alpha_s

    diff_model = UNet()
    print("Num params: ", sum(p.numel() for p in diff_model.parameters()))

    dataset = SingleRobotDataset(size_data, robot_matrix)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    ## before training:
    loss_before = compute_single_robot_sampling_mse(
        robot_matrix,
        diff_model,
        alpha,
        alpha_hat,
        beta,
        args,
        batch_size=batch_size,
        size_data=size_data,
    )  

    ## Training
    train_mse(dataloader, diff_model, alpha_hat, args, output_dir=args.output_dir)

    ## after training:
    loss_after = compute_single_robot_sampling_mse(
        robot_matrix,
        diff_model,
        alpha,
        alpha_hat,
        beta,
        args,
        batch_size=batch_size,
        size_data=size_data,
    )
    print(f"Loss before training: {loss_before}")
    print(f"Loss after training: {loss_after}")

    ## Checking for overfitting:
    x = sample(diff_model, 100, alpha, alpha_hat, beta, args)
    sampled_robots = tensor_to_robots(x)

    count_exact_matches(sampled_robots, robot_matrix)


