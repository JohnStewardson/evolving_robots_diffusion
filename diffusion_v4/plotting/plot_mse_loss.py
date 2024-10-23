import csv
import matplotlib.pyplot as plt
import numpy as np


def plot_average_mse_loss_per_batch_multi_new(csv_file_paths, labels_array, max_epoch=None, output_path=None):
    """
    Plot the average MSE loss per epoch from multiple CSV files on the same plot with specified labels.

    Args:
    - csv_file_paths (list of str): List of CSV file paths containing epoch and MSE loss data.
    - labels_array (list of str): List of labels corresponding to each CSV file for the legend.
    - output_path (str, optional): If provided, saves the plot to the specified path.
    """
    # Ensure the labels array matches the CSV file paths array length
    assert len(csv_file_paths) == len(labels_array), "The length of csv_file_paths and labels_array must match."

    # Loop through each CSV file path provided
    plt.figure(figsize=(10, 5))
    for csv_file_path, label in zip(csv_file_paths, labels_array):

        batch = []
        epoch = []
        mse_loss = []

        # Read the CSV file
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                batch.append(int(row[0]))
                epoch.append(float(row[1]))
                mse_loss.append(float(row[2]))
                # Only consider epochs up to max_epoch
                if max_epoch:
                    if epoch[-1] > max_epoch:
                        continue
                else:
                    continue

        if not max_epoch:
            max_epoch = len(mse_loss)
            #plt.plot(epoch, mse_loss, marker='o', linestyle='-', label=label)
        plt.plot(batch, mse_loss, marker='o', linestyle='-', label=label)
        #print(f"Sampling time total: {np.sum(sampling_time)}")
    # Plot settings
    plt.title('MSE_Loss of Robots')
    plt.xlabel('Batch')
    plt.ylabel('Average MSE Loss')
    plt.grid(True)
    plt.legend()  # Add a legend with provided labels

    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_average_mse_loss_per_epoch_multi(csv_file_paths, labels_array,min_epoch=0, max_epoch=1000, output_path=None):
    """
    Plot the average MSE loss per epoch from multiple CSV files on the same plot with specified labels.

    Args:
    - csv_file_paths (list of str): List of CSV file paths containing epoch and MSE loss data.
    - labels_array (list of str): List of labels corresponding to each CSV file for the legend.
    - output_path (str, optional): If provided, saves the plot to the specified path.
    """
    # Ensure the labels array matches the CSV file paths array length
    assert len(csv_file_paths) == len(labels_array), "The length of csv_file_paths and labels_array must match."

    # Loop through each CSV file path provided
    plt.figure(figsize=(10, 5))
    for csv_file_path, label in zip(csv_file_paths, labels_array):
        epoch_losses = {}

        # Read the CSV file
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                epoch = int(row[1])
                mse_loss = float(row[2])  # Assuming MSE loss is in the second column

                # Only consider epochs up to max_epoch
                if epoch > max_epoch:
                    continue
                if epoch >= min_epoch:
                    # Collect MSE losses for each epoch
                    if epoch in epoch_losses:
                        epoch_losses[epoch].append(mse_loss)
                    else:
                        epoch_losses[epoch] = [mse_loss]

        # Calculate the average MSE loss for each epoch
        epochs = []
        average_mse_losses = []

        for epoch, losses in epoch_losses.items():
            average_loss = sum(losses) / len(losses)
            epochs.append(epoch)
            average_mse_losses.append(average_loss)
        # epochs = epochs[-100:]
        # average_mse_losses = average_mse_losses[-100:]
        # Plotting each CSV file's data with the provided label
        plt.plot(epochs, average_mse_losses, linestyle='-', label=label)

    # Plot settings
    plt.title('MSE_Loss of Robots')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE Loss')
    plt.grid(True)
    plt.legend()  # Add a legend with provided labels

    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
    plt.show()


# labels_array = ['beta_end: 0.02, T: 500']
# csv_array = ['../parameter_tuning/finding_beta_end/mse_training/mse_log.csv']
# plot_average_mse_loss_per_batch_multi_new(csv_array, labels_array)
# plot_average_mse_loss_per_epoch_multi(csv_array, labels_array, max_epoch=500)
# print("Done")