import torch
from tqdm import tqdm
from transform_data import tensor_to_robots
from evogym import is_connected, has_actuator, hashable

def sample(model, n, alpha, alpha_hat, beta, args):
    """
    Returns sample as tensor
    """
    print("Sampling")
    device, T = (
        args.device,
        args.T
    )
    model.eval() # Sets model to evaluation mode
    with torch.no_grad(): # No gradient calculation for memory efficiency
        x = torch.randn((n, 1, 5, 5)).to(device) # Initialize x with random noise
        for i in tqdm(reversed(range(1, T)), position=0):
            t = (torch.ones(n) * i).long().to(device)
            predicted_noise = model(x, t)
            alpha_t = alpha[t][:, None, None, None]
            alpha_hat_t = alpha_hat[t][:, None, None, None]
            beta_t = beta[t][:, None, None, None]
            # Algorithm 2, line 3:
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            # Denoise (Algorithm line 4)
            x = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / (torch.sqrt(1 - alpha_hat_t))) * predicted_noise) + torch.sqrt(beta_t) * noise

    model.train()

    return x


def sample_timesteps(n, args):
    T = args.T
    return torch.randint(low=1, high=T, size=(n,))

def prepare_noise_schedule(noise_steps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, noise_steps)


def sample_with_print(model, n, alpha, alpha_hat, beta, args):
    """
    Returns sample as tensor
    """
    print("Sampling")
    device, T = (
        args.device,
        args.T
    )
    model.eval() # Sets model to evaluation mode
    with torch.no_grad(): # No gradient calculation for memory efficiency
        x_begin = torch.randn((n, 1, 5, 5)).to(device) # Initialize x with random noise
        x = x_begin.clone()
        for i in tqdm(reversed(range(1, T)), position=0):
            t = (torch.ones(n) * i).long().to(device)
            predicted_noise = model(x, t)
            alpha_t = alpha[t][:, None, None, None]
            alpha_hat_t = alpha_hat[t][:, None, None, None]
            beta_t = beta[t][:, None, None, None]
            # Algorithm 2, line 3:
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            # Denoise (Algorithm line 4)
            x = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / (torch.sqrt(1 - alpha_hat_t))) * predicted_noise) + torch.sqrt(beta_t) * noise

    model.train()

    return x, x_begin

def sample_valid_robots(model, batch_size, alpha, alpha_hat, beta, args, robot_matrix_hashes, initial_sample_size=50):
    valid_robots = []
    sampled_iters = 0
    while len(valid_robots) < batch_size:
        sampled_tensor = sample(model, initial_sample_size, alpha, alpha_hat, beta, args)
        sampled_robots = tensor_to_robots(sampled_tensor)

        for robot in sampled_robots:
            if is_connected(robot) and has_actuator(robot) and not (hashable(robot) in robot_matrix_hashes):
                robot_matrix_hashes[hashable(robot)] = True
                valid_robots.append(robot)

            if len(valid_robots) == batch_size:
                return valid_robots

        sampled_iters += 1
        if sampled_iters == 20:
            print("Sampling valid robots took too long")
            break