import torch
from torch.utils.data import Dataset, DataLoader
from transform_data import robots_to_tensor
from evogym import sample_robot
import os
import numpy as np
import matplotlib.pyplot as plt
import gc


class SingleRobotDataset(Dataset):
    """
    Dataset containing only a single robot
    """
    def __init__(self, num_samples, robot_matrix):
        self.num_samples = num_samples
        self.robot = robot_matrix

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        robot_tensor = robots_to_tensor([self.robot])
        #return robot_tensor
        return robot_tensor[0] # check this

class TwoMatrixDataset(Dataset):
    """
    Data set only containing a robot that is filled with 2s
    """
    def __init__(self, num_samples, structure_shape=(5, 5)):
        self.num_samples = num_samples
        self.structure_shape = structure_shape
        # Create a single robot filled with 2s
        robot = np.ones(shape=structure_shape) * 2

        # Convert the single robot to a tensor (and keep adding the batch in __getitem__)
        self.robot_tensor = robots_to_tensor([robot])  # robots_to_tensor expects a list

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.robot_tensor


class GenerationDataset(Dataset):

    def __init__(self, survivors, num_samples):
        self.robot_tensor = robots_to_tensor(survivors)
        num_repeats = max(1, num_samples // len(survivors) + 1)
        self.num_samples = num_samples
        self.expanded_robots = self.robot_tensor.repeat(num_repeats, 1, 1, 1)[:self.num_samples]

    def __len__(self):
        return len(self.expanded_robots)

    def __getitem__(self, idx):
        return self.expanded_robots[idx]