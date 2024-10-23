import os
import sys
import shutil
import numpy as np
import torch
import neat
import argparse

# Set current directory and root paths
curr_dir = os.path.dirname(os.path.abspath(__file__))  # current directory (examples/diffusion_v3/reward_based_optim/)
root_dir = os.path.join(curr_dir, '..', '..')  # root directory (examples/)

# Adjust sys.path for custom and external imports
sys.path.append(os.path.abspath(root_dir))  # Add root directory to sys.path
sys.path.append(os.path.join(root_dir, 'externals', 'PyTorch-NEAT'))  # Add PyTorch-NEAT to sys.path

# Import from PyTorch-NEAT and evogym
from pytorch_neat.cppn import create_cppn
from evogym import is_connected, has_actuator, get_full_connectivity, hashable
from get_simple_reward import calc_two_peaks
# Import classes and functions from cppn_helper (which contains Population and ParallelEvaluator)
from cppn_helper import Population, ParallelEvaluator





def get_cppn_input(structure_shape):
    x, y = torch.meshgrid(torch.arange(structure_shape[0]), torch.arange(structure_shape[1]))
    x, y = x.flatten(), y.flatten()
    center = (np.array(structure_shape) - 1) / 2
    d = ((x - center[0]) ** 2 + (y - center[1]) ** 2).sqrt()
    return x, y, d

def get_robot_from_genome(genome, config):
    nodes = create_cppn(genome, config, leaf_names=['x', 'y', 'd'], node_names=['empty', 'rigid', 'soft', 'hori', 'vert'])
    structure_shape = config.extra_info['structure_shape']
    x, y, d = get_cppn_input(structure_shape)
    material = []
    for node in nodes:
        material.append(node(x=x, y=y, d=d).numpy())
    material = np.vstack(material).argmax(axis=0)
    robot = material.reshape(structure_shape)
    return robot


def eval_genome_fitness(genome, config, genome_id, generation):
    """Simplified fitness evaluation function."""
    robot = get_robot_from_genome(genome, config)

    # Simplified reward calculation
    reward = calc_two_peaks(robot)  # Example reward: sum of all material cells (simple heuristic)
    return reward

def eval_genome_constraint(genome, config, genome_id, generation):
    robot = get_robot_from_genome(genome, config)
    validity = is_connected(robot) and has_actuator(robot)
    if validity:
        robot_hash = hashable(robot)
        if robot_hash in config.extra_info['structure_hashes']:
            validity = False
        else:
            config.extra_info['structure_hashes'][robot_hash] = True
    return validity

class SaveResultReporter(neat.BaseReporter):
    """Simplified result reporter to save only rewards."""

    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.generation = None
        self.rewards_total = []  # Store rewards for all generations

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        # Collect rewards
        rewards = [genome.fitness for genome in population.values()]
        self.rewards_total.append(rewards)
        # Print rewards for the current generation
        print(f"Generation {self.generation}: Rewards - {rewards}")
        # Save rewards to file
        np.save(os.path.join(self.save_path, 'rewards_total.npy'), self.rewards_total)


def run_cppn_simple(args):
    env_name, num_cores, pop_size, max_evaluations, structure_shape, exp_name, num_survivors = (
        args.env_name,
        args.num_cores,
        args.pop_size,
        args.max_evaluations,
        args.structure_shape,
        args.exp_name,
        args.num_survivors,
    )

    save_path = exp_name
    try:
        os.makedirs(save_path)
    except:
        print(f'THIS EXPERIMENT ({exp_name}) ALREADY EXISTS')
        print('Override? (y/n): ', end='')
        ans = input()
        if ans.lower() == 'y':
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        else:
            return None, None
        print()

    structure_hashes = {}

    config_path = os.path.join(curr_dir, 'neat.cfg')
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
        extra_info={
            'structure_shape': structure_shape,
            'save_path': save_path,
            'structure_hashes': structure_hashes,
            'args': args,  # args for run_ppo
            'env_name': env_name,
        },
        custom_config=[
            ('NEAT', 'pop_size', pop_size),
        ],
    )

    pop = Population(config)
    reporters = [
        neat.StatisticsReporter(),
        neat.StdOutReporter(True),
        SaveResultReporter(save_path),
    ]
    for reporter in reporters:
        pop.add_reporter(reporter)

    evaluator = ParallelEvaluator(num_cores, eval_genome_fitness, eval_genome_constraint)

    pop.run(
        evaluator.evaluate_fitness,
        evaluator.evaluate_constraint,
        n=np.ceil(max_evaluations / pop_size))

    best_robot = get_robot_from_genome(pop.best_genome, config)
    best_fitness = pop.best_genome.fitness
    return best_robot, best_fitness






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for ga script simple')
    parser.add_argument('--exp-name', type=str, default='2024-10-05_cppn_two_peaks', help='Name of the experiment (default: test_ga)')
    parser.add_argument('--env-name', type=str, default='Error_env_required', help='Name of the experiment (default: test_ga)')
    parser.add_argument('--pop-size', type=int, default=25, help='Population size (default: 3)')
    parser.add_argument('--structure_shape', type=tuple, default=(5, 5), help='Shape of the structure (default: (5,5))')
    parser.add_argument('--max-evaluations', type=int, default=1000, help='Maximum number of robots that will be evaluated (default: 6)')
    parser.add_argument('--num-cores', type=int, default=3, help='Number of robots to evaluate simultaneously (default: 3)')
    parser.add_argument('--num-survivors', type=int, default=5, help='Number of survivors')
    args = parser.parse_args()

    best_robot, best_fitness = run_cppn_simple(args)