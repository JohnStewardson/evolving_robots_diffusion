import argparse
from evolving_robots_diffusion.optimizers.cppn_simple import run_cppn_simple
from evolving_robots_diffusion.plotting.plot_success import plot_success
from evolving_robots_diffusion.plotting.plot_robot import plot_all_generations_structures
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for ga script simple')
    parser.add_argument('--exp-name', type=str, default='2024-10-05_cppn', help='Name of the experiment (default: test_cppn)')
    parser.add_argument('--env-name', type=str, default='Error_env_required', help='Name of the experiment')
    parser.add_argument('--pop-size', type=int, default=25, help='Population size (default: 3)')
    parser.add_argument('--structure_shape', type=tuple, default=(5, 5), help='Shape of the structure (default: (5,5))')
    parser.add_argument('--max-evaluations', type=int, default=1000, help='Maximum number of robots that will be evaluated (default: 6)')
    parser.add_argument('--num-cores', type=int, default=3, help='Number of robots to evaluate simultaneously (default: 3)')
    parser.add_argument('--num-survivors', type=int, default=5, help='Number of survivors')
    args = parser.parse_args()

    best_robot, best_fitness = run_cppn_simple(args)
    file_path = os.path.join('outputs', 'simple_env', args.exp_name)
    out_path = os.path.join('outputs', 'simple_env', args.exp_name, f'{args.exp_name}.pdf')
    plot_success(file_path, "CPPN-NEAT", output_path=out_path)

    home_path = os.path.join("outputs", "simple_env", args.exp_name)
    