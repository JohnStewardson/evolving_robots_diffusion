import numpy as np
from get_simple_reward import get_simple_rewards, get_simple_reward_two_peaks
import os

class GradientEntry():
    def __init__(self, array_values, array_robot_indices):
        self.array_values = array_values
        self.array_robot_indices = array_robot_indices
        self.rewards = []

    def enter_rewards(self, rewards):
        for index in self.array_robot_indices:
            self.rewards.append(rewards[index])

    def get_new_value(self):
        if rewards is None:
            print("No rewards added yet")
            return
        index = np.argmax(self.rewards)
        return self.array_values[index]

class GradientMatrix():

    def __init__(self):
        self.grad_matrix = np.full((5, 5), None)
    def add_grad_entry(self, k, j, grad_entry):
        self.grad_matrix[k, j] = grad_entry

    def enter_rewards_matrix(self, rewards):
        for k in range(5):
            for j in range(5):
                self.grad_matrix[k, j].enter_rewards(rewards)
    def get_new_robot(self):
        new_robot = np.zeros((5,5))
        for k in range(5):
            for j in range(5):
                grad_entry = self.grad_matrix[k, j]
                new_robot[k, j] = grad_entry.get_new_value()
        return new_robot

exp_name = "2024-10-05-direct_grad_2_peaks"
x_0 = np.full((5,5), 2)
x_i = x_0
generation = 0
max_evaluations = 1000
num_evaluations = 0
total_rewards = []
c = True
while c is True:
    robots = []
    robot_index = 0

    grad_matrix = GradientMatrix()
    robots.append(x_i)
    for k in range(5):
        for j in range(5):
            gradient_indices = []
            gradient_values = []

            # +1
            temp_robot = x_i.copy()
            if temp_robot[k,j] < 4:
                temp_robot[k,j] = temp_robot[k,j] + 1
                gradient_values.append(temp_robot[k,j])
                robot_index += 1
                gradient_indices.append(robot_index)
                robots.append(temp_robot)

            # +0
            temp_robot = x_i.copy()
            gradient_indices.append(0)
            gradient_values.append(temp_robot[k,j])

            # -1
            if temp_robot[k, j] > 0:
                temp_robot[k, j] = temp_robot[k, j] - 1
                robot_index += 1
                gradient_indices.append(robot_index)
                gradient_values.append(temp_robot[k, j])
                robots.append(temp_robot)

            grad_entry = GradientEntry(gradient_values, gradient_indices)
            grad_matrix.add_grad_entry(k, j, grad_entry)


    rewards = get_simple_reward_two_peaks(robots)
    print(f"length of rewards: {len(rewards)}")
    total_rewards.append(rewards)
    grad_matrix.enter_rewards_matrix(rewards)
    x_i_old = x_i.copy()
    x_i = grad_matrix.get_new_robot()
    print(f"Reward of generation {generation}: {rewards[0]}")
    print(x_i_old)
    num_evaluations += len(rewards)
    os.makedirs(exp_name, exist_ok=True)
    temp_path = os.path.join(exp_name, f'rewards_{generation}.npy')
    np.save(temp_path, rewards)
    if num_evaluations >= max_evaluations:
        c = False

    if np.array_equal(x_i, x_i_old):
        c = False

    generation += 1




