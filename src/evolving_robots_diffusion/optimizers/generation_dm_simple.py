
import os

from evogym import hashable, get_full_connectivity

import numpy as np
import torch

from evolving_robots_diffusion.diffusion_model.diffusion_args import add_diffusion_args
from evolving_robots_diffusion.optimizers.algo_utils import Structure, sample_random_robots
from evolving_robots_diffusion.simple_env.rewards import get_simple_rewards
from evolving_robots_diffusion.optimizers.generational_diffusion.train_and_sample import train_and_sample_job
from evolving_robots_diffusion.optimizers.generational_diffusion.group_diffusion import Group_DM

def run_dm_gen_simple(args, saved_dm_path=None):
    exp_name, pop_size, max_evaluations, num_survivors = (
        args.exp_name,
        args.pop_size,
        args.max_evaluations,
        args.num_survivors,
    )

    home_path = os.path.join("outputs", "simple_env", exp_name)
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
    
    while num_evaluations <= max_evaluations:

        gen_rewards = []
        for robot in robots:
            robot_matrix_hashes[hashable(robot)] = True

        generation_path = os.path.join(home_path, f"generation_{generation}")
        os.makedirs(generation_path, exist_ok=True)

        # get rewards
        gen_rewards = get_simple_rewards(robots)
        print(f"Gen: {generation}, rewards: {gen_rewards}")

        # get new survivors
        total_rewards.append(gen_rewards)
        structures = []
        for i in range(len(robots)):
            robot = robots[i]
            structure = Structure(robot, get_full_connectivity(robot), i)
            structure.set_reward(gen_rewards[i])
            r = structure.compute_fitness()
            structures.append(structure)

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)
        
        # Save structures
        save_path_structure = os.path.join(generation_path, "structure")
        os.makedirs(save_path_structure, exist_ok=True)
        for i in range(len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)
        # save reward info:
        temp_path = os.path.join(generation_path, "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
        f.write(out)
        f.close()
        
        
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

        robots = train_and_sample_job(survivor_robots, robot_matrix_hashes, args, generation_path=generation_path)
        # group_dm = Group_DM()
        # robots_from_sampling = []
        # group_dm.add_job(
        #     train_and_sample_job_same_inputs,
        #     (generation_path, survivor_robots, robot_matrix_hashes, sampling_inputs, args, saved_dm_path),
        #     callback=lambda robots: robots_from_sampling.extend(robots)
        # )
        # group_dm.add_job(
        #     train_and_sample_job,
        #     (survivor_robots, robot_matrix_hashes, args, generation_path, saved_dm_path),
        #     callback=lambda robots: robots_from_sampling.extend(robots)
        # )

        # # Run jobs and clean up
        # group_dm.run_jobs(args.num_cores)
        # robots = robots_from_sampling
        # saved_dm_path = os.path.join(generation_path, "model", "weights.pt")
        
        generation += 1
        if generation % 5 == 0:
            temp_path = os.path.join(home_path, f'rewards_input_robots_{generation}.npy')
            np.save(temp_path, input_robots_total)
            temp_path = os.path.join(home_path, f'rewards_total_{generation}.npy')
            np.save(temp_path, total_rewards)

    temp_path = os.path.join(home_path, 'rewards_total.npy')
    np.save(temp_path, total_rewards)
    temp_path = os.path.join(home_path, 'rewards_input_robots.npy')
    np.save(temp_path, input_robots_total)
    print("Done")
    





