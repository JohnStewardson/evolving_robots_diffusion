import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from diffusion_args import add_diffusion_args
import shutil
from reward_based_survivors.helper_functions import sample_random_robots
from reward_based_survivors.train_generation import train_and_sample_job_same_inputs, train_and_sample_job
from reward_based_survivors.group_diff import Group_DM
from evogym import hashable, get_full_connectivity
from get_simple_reward import get_simple_rewards
import numpy as np
import torch

class Structure():

    def __init__(self, body, connections, label):
        self.body = body
        self.connections = connections

        self.reward = 0
        self.fitness = self.compute_fitness()

        self.is_survivor = False
        self.prev_gen_label = 0

        self.label = label

    def compute_fitness(self):
        self.fitness = self.reward
        return self.fitness

    def set_reward(self, reward):
        self.reward = reward
        self.compute_fitness()

    def __str__(self):
        return f'\n\nStructure:\n{self.body}\nF: {self.fitness}\tR: {self.reward}\tID: {self.label}'

    def __repr__(self):
        return self.__str__()



def run_dm_gen_simple(args, saved_dm_path=None):
    exp_name, pop_size, max_evaluations, num_survivors = (
        args.exp_name,
        args.pop_size,
        args.max_evaluations,
        args.num_survivors,
    )

    home_path = exp_name
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({exp_name}) ALREADY EXISTS')
        print("Override? (y/n): ", end="")
        ans = input()
        if ans.lower() == "y":
            os.makedirs(home_path, exist_ok=True)
            print()

        else:
            return

    # Sample generation 0 with random valid robots
    generation = 0
    robots = sample_random_robots(pop_size)  # array of robots (matrices)

    # Generate hash table
    robot_matrix_hashes = {}

    num_evaluations = 0
    total_rewards = []
    structures = []
    input_robots_total = []
    survivors = []
    sampling_inputs = torch.randn((25, 1, 5, 5))
    while num_evaluations <= max_evaluations:

        gen_rewards = []
        for robot in robots:
            robot_matrix_hashes[hashable(robot)] = True

        generation_path = os.path.join(home_path, f"generation_{generation}")
        os.makedirs(generation_path, exist_ok=True)
        gen_rewards = get_simple_rewards(robots)
        print(f"Gen: {generation}, rewards: {gen_rewards}")
        total_rewards.append(gen_rewards)
        structures = []
        for i in range(len(robots)):
            robot = robots[i]
            structure = Structure(robot, get_full_connectivity(robot), i)
            structure.set_reward(gen_rewards[i])
            r = structure.compute_fitness()
            structures.append(structure)

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)
        new_survivors = structures[:num_survivors]
        all_survivors = survivors + new_survivors #join together
        survivors  = sorted(all_survivors, key=lambda structure: structure.fitness, reverse=True)
        survivors = survivors[:num_survivors]
        survivor_robots = []
        print(f"Gen_{generation}:")
        input_robots = []
        for survivor in survivors:
            survivor_robots.append(survivor.body)
            input_robots.append(survivor.fitness)
            print(f"r: {survivor.fitness}")
        input_robots_total.append(input_robots)
        num_evaluations += len(robots)

        group_dm = Group_DM()
        robots_from_sampling = []
        # group_dm.add_job(
        #     train_and_sample_job_same_inputs,
        #     (generation_path, survivor_robots, robot_matrix_hashes, sampling_inputs, args, saved_dm_path),
        #     callback=lambda robots: robots_from_sampling.extend(robots)
        # )
        group_dm.add_job(
            train_and_sample_job,
            (generation_path, survivor_robots, robot_matrix_hashes, args, saved_dm_path),
            callback=lambda robots: robots_from_sampling.extend(robots)
        )

        # Run jobs and clean up
        group_dm.run_jobs(args.num_cores)
        robots = robots_from_sampling
        saved_dm_path = os.path.join(generation_path, "d_model.pt")

        generation += 1
        if generation % 50 == 0:
            temp_path = os.path.join(exp_name, f'rewards_input_robots_{generation}.npy')
            np.save(temp_path, input_robots_total)
            temp_path = os.path.join(exp_name, f'rewards_total_{generation}.npy')
            np.save(temp_path, total_rewards)

    temp_path = os.path.join(exp_name, 'rewards_total.npy')
    np.save(temp_path, total_rewards)
    temp_path = os.path.join(exp_name, 'rewards_input_robots.npy')
    np.save(temp_path, input_robots_total)
    print("Done")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for dm_gm script simple')
    add_diffusion_args(parser)
    parser.add_argument('--epochs', type=int, default=200, help='Epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training (default: 3e-4)')
    parser.add_argument('--exp-name', type=str, default='2024-09-18-dm_ga_same_noise', help='Name of the experiment (default: test_ga)')
    parser.add_argument('--pop-size', type=int, default=25, help='Population size (default: 3)')
    parser.add_argument('--structure_shape', type=tuple, default=(5, 5), help='Shape of the structure (default: (5,5))')
    parser.add_argument('--max-evaluations', type=int, default=1000, help='Maximum number of robots that will be evaluated (default: 6)')
    parser.add_argument('--num-cores', type=int, default=3, help='Number of robots to evaluate simultaneously (default: 3)')
    parser.add_argument('--num-survivors', type=int, default=5, help='Number of survivors')
    args = parser.parse_args()

    run_dm_gen_simple(args)