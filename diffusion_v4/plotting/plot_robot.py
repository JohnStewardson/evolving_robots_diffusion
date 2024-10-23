import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

def plot_matrix_and_save(matrix, out_path=None):


    # Define the colors based on the provided image
    colors = {
        0: '#F4F5F7',            # Transparent
        1: '#262626',         # Dark grey/black
        2: '#BFBFBF',         # Light grey
        3: '#FD8E3E',         # Light orange
        4: '#6DAFD6'          # Light blue
    }

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot each entry in the matrix with the corresponding color
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            rect = plt.Rectangle((j, matrix.shape[0] - i - 1), 1, 1, facecolor=colors[matrix[i, j]], edgecolor='black')
            ax.add_patch(rect)

    # Set the limits and aspect ratio
    ax.set_xlim(0, matrix.shape[1])
    ax.set_ylim(0, matrix.shape[0])
    ax.set_aspect('equal')

    # Remove the ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    if out_path:
        # Save the plot to the specified path
        plt.savefig(out_path, format='pdf', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def read_np_and_plot_generation(structure_path):
    # Get the parent directory of structure_path
    parent_directory = os.path.dirname(structure_path)
    output_path = os.path.join(parent_directory, 'robot_images')

    os.makedirs(output_path, exist_ok=True)

    for file in os.listdir(structure_path):  # List all files in the directory
        if file.endswith(".npz"):
            file_path = os.path.join(structure_path, file)
            robot = read_in_npz_structure(file_path)

            file_without_ext = os.path.splitext(file)[0]  # Get the file name without the extension
            output_file_name = f'{file_without_ext}.pdf'  # Add .pdf extension
            output_path_full = os.path.join(output_path, output_file_name)  # Full path to save the .pdf file
            plot_matrix_and_save(robot, output_path_full)
    print("Done")


def read_in_npz_structure(file_path):
    data = np.load(file_path)
    # Assuming the structure is stored with the key 'structure' in the .npz file
    return data['arr_0']

def plot_all_generations_structures(exp_dir):
    # Find all directories matching "generation_{number}" in the experiment directory
    generation_directories = glob.glob(os.path.join(exp_dir, 'generation_*'))

    # Loop through each generation directory
    for generation_directory in generation_directories:
        temp_structure_path = os.path.join(generation_directory, 'structure')
        if os.path.isdir(temp_structure_path):
            read_np_and_plot_generation(temp_structure_path)
            print(f"finished {generation_directory}")


# directory = '../../imported_exp_data/2024_09_27_Walker_v0_ga'
#
# plot_all_generations_structures(directory)




