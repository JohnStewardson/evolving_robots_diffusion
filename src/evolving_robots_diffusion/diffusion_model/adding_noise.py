import torch


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

