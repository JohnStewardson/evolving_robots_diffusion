import numpy as np


def get_simple_rewards(robots):
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
