import os.path

import numpy as np
import matplotlib.pyplot as plt


def plot_success(file_path, algorithm_name, output=None):
    data_file_path = os.path.join(file_path, 'rewards_total.npy')
    data = np.load(data_file_path)

    # Step 2: Extract the best and worst values for each generation
    generations = data.shape[0]  # Number of generations
    best_values = np.max(data, axis=1)  # Best (max) value for each generation
    worst_values = np.min(data, axis=1)  # Worst (min) value for each generation
    average_values = np.average(data, axis=1)
    # Step 3: Plot the best and worst values
    plt.figure(figsize=(10, 6))
    plt.plot(range(generations), best_values, label='Best Fitness', color='green', marker='o')
    plt.plot(range(generations), average_values, label='Average Fitness', color='orange', marker='o')
    plt.plot(range(generations), worst_values, label='Worst Fitness', color='red', marker='o')

    # Step 4: Customize the plot
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.title(algorithm_name)
    plt.legend(loc='lower right')
    plt.grid(True)
    if output:
        output_path = os.path.join(file_path, output)
        # Save the plot to the specified path
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    # Show the plot
    plt.show()

def plot_success_generationally(file_path, algorithm_name):

    max_values = []
    min_values = []
    avg_values = []
    generations = []
    generation = 0
    while True:
        gen_file_path = os.path.join(file_path, f'rewards_{generation}.npy')
        try:
            gen_data = np.load(gen_file_path)
            max_values.append(np.max(gen_data))
            min_values.append(np.min(gen_data))
            avg_values.append(np.average(gen_data))
            generations.append(generation)
            generation += 1
        except:
            break


    plt.figure(figsize=(10, 6))
    plt.plot(generations, max_values, label='Best Fitness', color='blue', marker='o')
    plt.plot(generations, min_values, label='Worst Fitness', color='red', marker='x')
    plt.plot(generations, avg_values, label='Average Fitness')

    # Step 4: Customize the plot
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.title(algorithm_name +' Progress: Best vs. Worst Fitness per Generation')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
def plot_top_rewards(file_path, exp_name):
    data = np.load(file_path)

    # Step 2: Extract the top 5 rewards for each generation
    generations = data.shape[0]  # Number of generations
    top_5_rewards = np.sort(data, axis=1)[:, -5:]  # Sort each generation and select the top 5 values

    # Step 3: Plot the top 5 rewards for each generation
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(range(generations), top_5_rewards[:, i], label=f'Top {5 - i} Reward', marker='o')

    # Step 4: Customize the plot
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.title(exp_name + ' Progress: Top 5 Rewards per Generation')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_input_rewards(file_path, exp_name=None):
    data = np.load(file_path)

    # Step 2: Extract the top 5 rewards for each generation
    generations = data.shape[0]  # Number of generations
    best_values = np.max(data, axis=1)  # Best (max) value for each generation
    worst_values = np.min(data, axis=1)  # Worst (min) value for each generation
    average_values = np.average(data, axis=1)
    # top_5_rewards = np.sort(data, axis=1)[:, -5:]  # Sort each generation and select the top 5 values
    #
    # # Step 3: Plot the top 5 rewards for each generation
    # plt.figure(figsize=(10, 6))
    # for i in range(5):
    #     plt.plot(range(generations), top_5_rewards[:, i])
    plt.figure(figsize=(10, 6))
    plt.plot(range(generations), best_values, label='Max. Fitness')
    plt.plot(range(generations), worst_values, label='Min. Fitness')
    plt.plot(range(generations), average_values, label='Avg. Fitness')

    # Step 4: Customize the plot
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.title('Rewards of input robots per Generation')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()



# # Step 1: Read the data from the file
# file_path = 'test_ga_simple/rewards_total.npy'  # Update this path if different
# plot_success(file_path, 'Genetic Algorithm')
file_path = '2024-10-05_cppn_two_peaks'
plot_success(file_path, "CPPN-NEAT", output='CPPN_2_Peaks.pdf')
# file_path = 'dm_ga_simple/rewards_total.npy'
# plot_success(file_path, "Diffusion model with survivors")

