import time

import numpy as np
from evogym import sample_robot
import sys
import os
import torch
from helper_functions import calc_two_peaks
import time
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plotting.plot_robot import plot_matrix_and_save
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import plotly.graph_objects as go



# #def calc_two_peaks(robot):
#     # convert matrix into vector
#     robot_vector = robot.reshape(25, 1)
#     peak_1 =
#     peak_2 =
#     slope_1 =
#     slope_2 =
#     # Calculate euclidian distance between to each peak
#     delta_peak_1 = np.linalg.norm(robot_vector - peak_1)
#     delta_peak_2 = np.linalg.norm(robot_vector - peak_2)
#
#     reward = max(100*peak_1 - delta_peak_1 * slope_1, 50*peak_1 - delta_peak_2 * slope_2)
#
#     return reward


# p1, _ = sample_robot(robot_shape=(5,5))
# p1 = p1.reshape(25,1)
# p2, _ = sample_robot(robot_shape=(5,5))
# p2 = p2.reshape(25,1)
# print(p1)
# print(p2)
# delta_peak_2 = np.linalg.norm(p1 - p2)
# print(delta_peak_2)

# while True:
#
#     p1, _ = sample_robot(robot_shape=(5, 5))
#     p1_v = p1.reshape(25, 1)
#     p2, _ = sample_robot(robot_shape=(5, 5))
#     p2_v = p2.reshape(25, 1)
#     if np.linalg.norm(p1_v - p2_v) > 10:
#         print(f"p1: {p1_v}")
#         print(f"p2: {p2_v}")
#
#
#         plot_matrix_and_save(p1)
#         time.sleep(10)
#         plot_matrix_and_save(p2)
#         # Average is 10
#         break

p1 = np.array([[0.], [4.], [0.], [4.], [4.], [1.], [1.], [4.], [3.], [3.], [2.], [0.], [0.], [0.], [2.], [2.], [0.], [0.], [0.], [3.], [4.], [4.], [4.], [4.], [3.]])
p2 = np.array([[0.], [3.], [2.], [0.], [0.], [1.], [3.], [4.], [0.], [0.], [3.], [4.], [3.], [0.], [1.], [3.], [2.], [2.], [0.], [1.], [0.], [0.], [4.], [1.], [1.]])


# def vector_to_number(vector):
#     # Convert the 25x1 vector into a 1D array if needed
#     vector = vector.flatten()
#
#     # Initialize the number to 0
#     number = 0
#
#     # Loop through each element of the vector
#     for i, val in enumerate(vector):
#         number += val * (5 ** (len(vector) - 1 - i))
#
#     return number

def vector_to_number(vector):
    # Flatten the 25x1 vector
    vector = vector.flatten()

    # Integer part: sum of the vector elements
    integer_part = np.sum(vector)

    # Fractional part: a small value based on the unique structure of the vector
    # We'll use a weighted sum of the vector components, scaled down to ensure it's a small fraction
    fractional_part = 0
    for i, val in enumerate(vector):
        fractional_part += val * (10 ** -(i + 1))  # Fraction based on position and value

    # Combine integer and fractional parts
    number = integer_part + fractional_part

    return number

origin = np.zeros((25, 1))  # Origin vector (all zeros)
center = np.full((25, 1), 2)  # Center vector (all 2s)

def vector_to_2d_encoding(vector):
    # Calculate the Euclidean distance to the origin
    distance_to_origin = np.linalg.norm(vector - origin)

    # Calculate the Euclidean distance to the center (2, 2, 2,...)
    distance_to_center = np.linalg.norm(vector - center)

    return distance_to_origin, distance_to_center


print(f"p1: {vector_to_number(p1)}, reward: {calc_two_peaks(p1)}")
print(f"p1: {vector_to_2d_encoding(p2)}, reward: {calc_two_peaks(p2)}")
print(f" 0s: {vector_to_number(0*np.ones(shape=(1,25)))}, reward: {calc_two_peaks(0*np.ones(shape=(1,25)))}")
print(f" 1s: {vector_to_number(np.ones(shape=(1,25)))}, reward: {calc_two_peaks(1*np.ones(shape=(1,25)))}")
print(f" 2s: {vector_to_number(2*np.ones(shape=(1,25)))}, reward: {calc_two_peaks(2*np.ones(shape=(1,25)))}")
print(f" 3s: {vector_to_number(3*np.ones(shape=(1,25)))}, reward: {calc_two_peaks(3*np.ones(shape=(1,25)))}")
print(f" 4s: {vector_to_2d_encoding(4*np.ones(shape=(1,25)))}, reward: {calc_two_peaks(4*np.ones(shape=(1,25)))}")

# Function to generate a robot vector for given distances
# Function to generate a robot vector for given distances
def generate_robot_for_distances(dist_origin, dist_center):
    """
    This function will attempt to generate a robot vector whose distance to the origin is close to `dist_origin`
    and distance to the center is close to `dist_center`.
    """
    # Start with the center vector (as floats for floating point arithmetic)
    robot_vector = np.full((25, 1), 2, dtype=np.float64)

    # Adjust the vector to move toward the origin (reduce values)
    norm_origin = np.linalg.norm(robot_vector - origin)
    if norm_origin > 0:
        scale_origin = dist_origin / norm_origin
        robot_vector -= (robot_vector - origin) * scale_origin

    # Adjust the vector to move toward the center (adjust values towards [2, 2, 2,...])
    norm_center = np.linalg.norm(robot_vector - center)
    if norm_center > 0:
        scale_center = dist_center / norm_center
        robot_vector -= (robot_vector - center) * scale_center

    # Ensure the values are clipped between [0, 4] and rounded to integers
    robot_vector = np.clip(np.round(robot_vector), 0, 4).astype(int)

    return robot_vector

# Generate robots and calculate their encodings and rewards
# Generate robots and calculate their encodings and rewards
def plot_3d_encoding():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the grid of distances
    num_points = 1000
    distances_to_origin = np.linspace(0, 10, num_points)
    distances_to_center = np.linspace(0, 10, num_points)

    # Store the encodings and rewards
    X, Y, Z = [], [], []
    seen_robots = set()

    # Loop through the grid and generate robots
    for dist_origin in distances_to_origin:
        for dist_center in distances_to_center:
            # Generate a robot vector for the given distances
            robot_vector = generate_robot_for_distances(dist_origin, dist_center)

            # Create a hashable tuple for the robot's unique configuration
            robot_hash = tuple(robot_vector.flatten())

            # Skip if this robot configuration already exists
            if robot_hash in seen_robots:
                continue
            seen_robots.add(robot_hash)

            # Calculate the reward
            reward = calc_two_peaks(robot_vector)

            # Store the encoding (x: dist_origin, y: dist_center) and the reward (z)
            X.append(dist_origin)
            Y.append(dist_center)
            Z.append(reward)

    # Scatter plot in 3D
    scatter = ax.scatter(X, Y, Z, c=Z, cmap='viridis', marker='o')

    # Add color bar for the reward values
    fig.colorbar(scatter, ax=ax, label='Reward')

    # Set labels
    ax.set_xlabel("Distance to Origin")
    ax.set_ylabel("Distance to Center (2,2,2...)")
    ax.set_zlabel("Reward")

    # Enable interactive mode for rotating, zooming, and panning
    plt.show()

# Example usage
plot_3d_encoding()