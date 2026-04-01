import argparse
import multiprocessing as mp
import os

from evolving_robots_diffusion.optimizers.generation_dm_simple import run_dm_gen_simple
from evolving_robots_diffusion.diffusion_model.diffusion_args import add_diffusion_args
from evolving_robots_diffusion.plotting.plot_success import plot_success
from evolving_robots_diffusion.plotting.plot_robot import plot_all_generations_structures

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description='Arguments for dm_gm script simple')
    add_diffusion_args(parser)
    parser.add_argument('--epochs', type=int, default=20, help='Epochs (default 200)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training (default: 3e-4)')
    parser.add_argument('--exp-name', type=str, default='2024-09-18-dm_ga_simple', help='Name of the experiment (default: test_gd)')
    parser.add_argument('--pop-size', type=int, default=3, help='Population size (default: 25)')
    parser.add_argument('--structure_shape', type=tuple, default=(5, 5), help='Shape of the structure (default: (5,5))')
    parser.add_argument('--max-evaluations', type=int, default=6, help='Maximum number of robots that will be evaluated (default: 1000)')
    parser.add_argument('--num-cores', type=int, default=3, help='Number of robots to evaluate simultaneously (default: 3)')
    parser.add_argument('--num-survivors', type=int, default=1, help='Number of survivors (default 5)')
    args = parser.parse_args()

    run_dm_gen_simple(args)
    file_path = os.path.join('outputs', 'simple_env', args.exp_name)
    out_path = os.path.join('outputs', 'simple_env', args.exp_name, f'{args.exp_name}.pdf')
    plot_success(file_path, "Generational Diffusion", output_path=out_path)

    home_path = os.path.join("outputs", "simple_env", args.exp_name)
    plot_all_generations_structures(home_path)