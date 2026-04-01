import os
from typing import List
from evogym import sample_robot, hashable
import argparse

import numpy as np
import random
from evolving_robots_diffusion.optimizers.algo_utils import mutate, Structure
from evolving_robots_diffusion.simple_env.rewards import get_simple_rewards


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
    exp_path = os.path.join("outputs", "simple_env",exp_name)
    os.makedirs(exp_path, exist_ok=True)
    


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
        generation_rewards = get_simple_rewards(robots)
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
            temp_path = os.path.join(exp_path, 'rewards_total.npy')

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


