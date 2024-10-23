import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys


def plot_experiment(exp_dir, max_gen=0, output=None, title=None):
    generation_directories = glob.glob(os.path.join(exp_dir, 'generation_*'))

    all_rewards = []
    max_rewards = []
    min_rewards = []
    avg_rewards = []
    # Loop through each generation directory
    gen = 0
    for generation_directory in sorted(generation_directories):
        if max_gen:
            if gen > max_gen:
                break
            else:
                gen+=1
        gen_rewards = []
        temp_out_path = os.path.join(generation_directory, 'output.txt')
        # read in output.txt, add rewards to gen_rewards
        # Check if output.txt exists before reading
        if os.path.exists(temp_out_path):
            with open(temp_out_path, 'r') as f:
                # Read each line in the file
                for line in f:
                    # Split the line by whitespace and take the second column (the reward)
                    data = line.split()
                    if len(data) == 2:  # Make sure we have two columns
                        reward = float(data[1])
                        gen_rewards.append(reward)


        if gen_rewards:
            max_rewards.append(np.max(gen_rewards))
            min_rewards.append(np.min(gen_rewards))
            avg_rewards.append(np.average(gen_rewards))
        # Add the rewards of this generation to the overall rewards
        all_rewards.append(gen_rewards)



    plt.figure(figsize=(5,5))
    gens = []
    for gen in range(len(all_rewards)):
        gen_rewards = all_rewards[gen]
        top_5 = sorted(gen_rewards, reverse=True)[:5]
        plt.scatter(gen*np.ones(shape=(len(gen_rewards), 1)), gen_rewards, color='grey', alpha=0.6)
        if gen==0:
            plt.scatter(gen * np.ones(shape=(len(top_5), 1)), top_5, color='blue', label='suvivor', alpha=0.8)
        else:
            plt.scatter(gen * np.ones(shape=(len(top_5), 1)), top_5, color='blue', alpha=0.8)
        gens.append(gen)
    plt.plot(gens, max_rewards, color='green', label='Max. reward')
    plt.plot(gens, min_rewards, color='red', label='Min. reward')
    plt.plot(gens, avg_rewards, color='orange', label='Avg. reward')
    plt.xticks(gens)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    if title:
        plt.title(title)
    if output:
        output_path = os.path.join(exp_dir, output)
        # Save the plot to the specified path
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    plt.show()
    return

# exp_dir = "../../imported_exp_data/2024_09_27_Walker_v0_ga"
# plot_experiment(exp_dir, max_gen=4, output='Plot.pdf', title='Genetic Algorithm')
# exp_dir = "../../imported_exp_data/2024-09-13-Diffusion_v0"
# plot_experiment(exp_dir, max_gen=4, output='Plot.pdf', title='Generational Diffusion')
exp_dir = "../../imported_exp_data/2024-08-26-Walker_v0_cppn"
plot_experiment(exp_dir, max_gen=7, output='CPPN_7_generations' ,title='CPPN')

