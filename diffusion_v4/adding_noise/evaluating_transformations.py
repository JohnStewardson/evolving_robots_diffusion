
import numpy as np
import matplotlib.pyplot as plt

# Define transformation functions

# Strategy 1: min = 0, max = 4
def scale_strategy_1(x):
    # Map from original to scaled range [-1, 1]
    return (x / 2) - 1

def inverse_scale_strategy_1(x_scaled):
    # Map back from scaled range [-1, 1] to original range [0, 4]
    return (x_scaled + 1) * 2

# Strategy 2: min = -0.5, max = 4.5
def scale_strategy_2(x):
    # Map from original to scaled range [-1, 1] with min=-0.5, max=4.5
    return 2 * ((x + 0.5) / 5) - 1

def inverse_scale_strategy_2(x_scaled):
    # Map back from scaled range [-1, 1] to original range [-0.5, 4.5]
    return (x_scaled + 1) * 2.5 - 0.5

# Generaterandom floats between [-1, 1]
num_samples = 10000000
random_scaled_values = np.random.uniform(-1, 1, num_samples)

# Apply inverse transformations to map back to original range
# Strategy 1 (min=0, max=4)
original_values_strategy_1 = np.round(inverse_scale_strategy_1(random_scaled_values))

# Strategy 2 (min=-0.5, max=4.5)
original_values_strategy_2 = np.round(inverse_scale_strategy_2(random_scaled_values))

# # Plot the results to compare
# plt.figure(figsize=(10, 5))
#
#
# plt.subplot(1, 2, 1)
# plt.hist(original_values_strategy_1, bins=np.arange(-0.5, 5.5, 1), edgecolor='black', density=True)
# plt.title('Strategy 1: min=0, max=4')
# plt.xlabel('Original Values')
# plt.ylabel('Frequency')
#
# plt.subplot(1, 2, 2)
# plt.hist(original_values_strategy_2, bins=np.arange(-0.5, 5.5, 1), edgecolor='black', density=True)
# plt.title('Strategy 2: min=-0.5, max=4.5')
# plt.xlabel('Original Values')
# plt.ylabel('Frequency')
#
# plt.tight_layout()
# plt.show()

# First plot for Strategy 1
plt.figure(figsize=(5, 5))
plt.hist(original_values_strategy_1, bins=np.arange(-0.5, 5.5, 1), edgecolor='black', density=True)
plt.title('Strategy 1: min=0, max=4')
plt.xlabel('Original Values')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('Option_1.pdf')  # Save the first plot as PDF
plt.close()

# Second plot for Strategy 2
plt.figure(figsize=(5, 5))
plt.hist(original_values_strategy_2, bins=np.arange(-0.5, 5.5, 1), edgecolor='black', density=True)
plt.title('Strategy 2: min=-0.5, max=4.5')
plt.xlabel('Original Values')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('Option_2.pdf')  # Save the second plot as PDF
plt.close()

