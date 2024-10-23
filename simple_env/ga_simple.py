import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.algo_utils import get_percent_survival_evals, mutate, Structure
from typing import List
from evogym import sample_robot, hashable
import argparse
import shutil
from get_simple_reward import get_simple_rewards, get_simple_reward_two_peaks
import numpy as np
import random


def run_ga_simple(args):
    num_cores, pop_size, max_evaluations, structure_shape, exp_name, num_survivors = (
        args.num_cores,
        args.pop_size,
        args.max_evaluations,
        args.structure_shape,
        args.exp_name,
        args.num_survivors,
    )
    ## Manage directories:
    home_path = exp_name
    os.makedirs(exp_name, exist_ok=True)
    


    ## Generate initial population:
    structures: List[Structure] = []
    population_structure_hashes = {}
    num_evaluations = 0
    generation = 0

    for i in range(pop_size):

        temp_structure = sample_robot(structure_shape)
        while (hashable(temp_structure[0]) in population_structure_hashes):
            temp_structure = sample_robot(structure_shape)

        structures.append(Structure(*temp_structure, i))
        population_structure_hashes[hashable(temp_structure[0])] = True
        num_evaluations += 1

    rewards_total = []
    generation_rewards = []
    robots = []

    while True:
        robots = []
        generation_rewards = []
        for structure in structures:
            robots.append(structure.body)
        generation_rewards = get_simple_reward_two_peaks(robots)
        print(f"rewards: {generation_rewards}")
        print(f"length: {len(generation_rewards)}")
        for i in range(len(generation_rewards)):
            #print(f"length of structures: {len(structures)}, length of rewards: {len(generation_rewards)}")
            structures[i].set_reward(generation_rewards[i])
            structures[i].compute_fitness()
        print(f"Generation {generation}, rewards: {generation_rewards}")
        rewards_total.append(generation_rewards)

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)

        if num_evaluations >= max_evaluations:
            print("Reached max_evaluations")
            temp_path = os.path.join(exp_name, 'rewards_total.npy')

            np.save(temp_path, rewards_total)
            return

        survivors = structures[:num_survivors]

        num_children = 0
        while num_children < (pop_size - num_survivors):

            parent_index = random.sample(range(num_survivors), 1)
            child = mutate(survivors[parent_index[0]].body.copy(), mutation_rate=0.1, num_attempts=50)

            if child != None and hashable(child[0]) not in population_structure_hashes:
                # overwrite structures array w new child
                structures[num_survivors + num_children] = Structure(*child, num_survivors + num_children)
                population_structure_hashes[hashable(child[0])] = True
                num_children += 1
                num_evaluations += 1

        structures = structures[:num_children + num_survivors]
        print(f"Length of structures: {len(structures)}")
        generation += 1









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for ga script simple')
    parser.add_argument('--exp-name', type=str, default='2024-10-05_ga_2peaks', help='Name of the experiment (default: test_ga)')
    parser.add_argument('--pop-size', type=int, default=25, help='Population size (default: 3)')
    parser.add_argument('--structure_shape', type=tuple, default=(5, 5), help='Shape of the structure (default: (5,5))')
    parser.add_argument('--max-evaluations', type=int, default=1000, help='Maximum number of robots that will be evaluated (default: 6)')
    parser.add_argument('--num-cores', type=int, default=3, help='Number of robots to evaluate simultaneously (default: 3)')
    parser.add_argument('--num-survivors', type=int, default=5, help='Number of survivors')
    args = parser.parse_args()

    run_ga_simple(args)