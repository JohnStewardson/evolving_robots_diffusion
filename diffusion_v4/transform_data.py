import numpy as np
import torch

def map_robot(robot):
    """
    Input: np.array of size 5x5, with evogym encoding
    Output: np.array of size 5x5 with new encoding
    """
    dictionary_of_mapping = {
        0: 0,
        1: 4,
        2: 1,
        3: 2,
        4: 3
    }
    for i in range(5):
        for j in range(5):
            robot[i, j] = dictionary_of_mapping[robot[i, j]]
    return robot

def remap_robot(robot):
    """
    Input: np.array of size 5x5, with new encoding
    Output: np.array of size 5x5 with evogym encoding
    """
    inverse_dictionary_of_mapping = {
        0: 0,
        4: 1,
        1: 2,
        2: 3,
        3: 4
    }

    robot = np.clip(robot, 0, 4)
    for i in range(5):
        for j in range(5):
            robot[i, j] = inverse_dictionary_of_mapping[int(robot[i, j])]
    return robot

def normalize_data(x):
    # Map from original to scaled range [-1, 1] with min=-0.5, max=4.5
    return 2 * ((x + 0.5) / 5) - 1

def rescale_data(x_scaled):
    # Map back from scaled range [-1, 1] to original range [-0.5, 4.5]
    return (x_scaled + 1) * 2.5 - 0.5

def robots_to_tensor(robots):
    """
    Input: array of robots, where each robot is np.array of size 5x5 in evogym encoding
    Output: tensor of size batch x 1 x 5 x 5
    The robot gets mapped and scaled and shifted to be within a range of [-1,1]
    """
    transformed_robots = []
    for robot in robots:
        robot = map_robot(robot) # change encoding
        robot = normalize_data(robot) # scale to [-1,1]
        transformed_robots.append(robot)

    transformed_robots = np.array(transformed_robots, dtype=np.float32)
    # Convert the numpy array to a tensor and add the channel dimension (unsqueeze)
    tensor_robots = torch.tensor(transformed_robots).unsqueeze(1)
    return tensor_robots

def tensor_to_robots(tensor):
    """
    Input: tensor of size batch x 1 x 5 x 5
    Output: array of lenght batch, with np.arrays of size 5x5, with evogym encoding
    """
    robots = []
    # Remove the extra channel dimension and convert the tensor to numpy arrays
    tensor = tensor.squeeze(1).numpy()
    for robot in tensor:
        robot = rescale_data(robot) #back to [-0.5, 4.5]
        robot = np.clip(robot, 0, 4) #clip at [0,4]
        robot = np.round(robot).astype(int) # round to [0,1,2,3,4]

        # Remap to evogym encoding
        robot = remap_robot(robot)

        # Append to the list of robots
        robots.append(robot)

    return robots



# robot_matrix = np.array([
#     [2, 2, 2, 2, 2],
#     [4, 4, 0, 4, 4],
#     [4, 4, 0, 4, 4],
#     [4, 4, 0, 4, 4],
#     [3, 3, 0, 3, 3]
# ])
# robots = [robot_matrix]
# tensor = robots_to_tensor(robots)
# robots = tensor_to_robots(tensor)
# print(robots)