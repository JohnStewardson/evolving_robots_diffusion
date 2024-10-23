import argparse
import torch

def add_diffusion_args(parser: argparse.ArgumentParser) -> None:
    """
    Add Diffusion model arguments to the parser
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    diff_parser: argparse.ArgumentParser = parser.add_argument_group('diffusion arguments')
    diff_parser.add_argument('--T', type=int, default=500, help='largest timestep (Default 300)')
    diff_parser.add_argument('--run-name', type=str, default='DDPM', help='Name for the run (default: DDPM)')

    diff_parser.add_argument('--device', type=str, default=device,
                        help='Device to use for training (default: cuda if available)')

