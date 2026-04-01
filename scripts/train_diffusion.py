import argparse
import numpy as np

from evolving_robots_diffusion.diffusion_model.diffusion_args import add_diffusion_args
from evolving_robots_diffusion.diffusion_model.training.pipeline import run_training_pipeline

def main():
    parser = argparse.ArgumentParser()
    add_diffusion_args(parser)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="outputs/train_diffusion")
    args = parser.parse_args()

    robot_matrix = np.array([
        [2, 2, 2, 2, 2],
        [4, 4, 0, 4, 4],
        [4, 4, 0, 4, 4],
        [4, 4, 0, 4, 4],
        [3, 3, 0, 3, 3]
    ])
    robots_data = [robot_matrix] * 5

    run_training_pipeline(args, robots_data)

if __name__ == "__main__":
    main()