from evogym import sample_robot, get_full_connectivity
import sys
import os
import torch
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np


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


def sample_random_robots(num_robots):
    # Sample robots
    robots = []
    for i in range(num_robots):
        robot_shape = (5,5)
        robot, _ = sample_robot(robot_shape)
        robots.append(robot)

    return robots

def get_simple_rewards_sum(robots):
    rewards = []
    for robot in robots:
        rewards.append(np.sum(robot))

    return rewards


def get_simple_reward_two_peaks(robots):
    rewards = []
    for robot in robots:
        rewards.append(calc_two_peaks(robot))

    return rewards

def calc_two_peaks(robot):
    robot_vector = robot.reshape(25, 1)
    p1 = np.array(
        [[0.], [4.], [0.], [4.], [4.], [1.], [1.], [4.], [3.], [3.], [2.], [0.], [0.], [0.], [2.], [2.], [0.], [0.],
         [0.], [3.], [4.], [4.], [4.], [4.], [3.]])
    p2 = np.array(
        [[0.], [3.], [2.], [0.], [0.], [1.], [3.], [4.], [0.], [0.], [3.], [4.], [3.], [0.], [1.], [3.], [2.], [2.],
         [0.], [1.], [0.], [0.], [4.], [1.], [1.]])
    slope_1 = 10
    slope_2 = 3

    delta_peak_1 = np.linalg.norm(robot_vector - p1)
    delta_peak_2 = np.linalg.norm(robot_vector - p2)

    reward = max(100 - delta_peak_1 * slope_1, 50 - delta_peak_2 * slope_2)
    return reward




