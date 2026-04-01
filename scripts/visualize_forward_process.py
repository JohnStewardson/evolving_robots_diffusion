import numpy as np
from evolving_robots_diffusion.diffusion_model.visualization import visualize_forward_process

def main():
    robot_matrix = np.array([
        [2, 2, 2, 2, 2],
        [4, 4, 0, 4, 4],
        [4, 4, 0, 4, 4],
        [4, 4, 0, 4, 4],
        [3, 3, 0, 3, 3]
    ])

    visualize_forward_process(robot_matrix)

if __name__ == "__main__":
    main()