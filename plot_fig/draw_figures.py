import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style


def plot_rmse_boxplots(folder_names):
    style.use('/home/zhangtianyi/Gibss-Gaussian-Filtering/style.mpl')
    rmse_values = {}

    for folder in folder_names:
        folder_path = os.path.join('../results', folder)
        json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]

        if len(json_files) != 1:
            raise ValueError(f"Expected one JSON file in folder {folder}, found {len(json_files)}")

        json_file_name = json_files[0]
        filter_method = json_file_name.split('-')[0]

        json_file = os.path.join(folder_path, json_file_name)
        npz_file = os.path.join(folder_path, 'data.npz')

        if not os.path.exists(npz_file):
            print(f"Missing NPY file for folder: {folder}")
            continue

        with open(json_file, 'r') as file:
            json_data = json.load(file)

        data = np.load(npz_file, allow_pickle=True)
        x_error = data['x_mc'] - data['x_hat_mc']
        x_error = np.linalg.norm(x_error, axis=2)  # This computes the norm along the last axis
        x_error = x_error.flatten()  # Flatten the array to 1D

        rmse_values.setdefault(filter_method, []).extend(x_error)  # Extend the list with flattened array

    fig, ax = plt.subplots()
    ax.boxplot(list(rmse_values.values()))
    ax.set_xticklabels(rmse_values.keys())
    ax.set_ylabel('Error')

    plt.show()

# # Example usage
# folder_names = ['2023-11-26-14-39', '2023-11-26-14-40', '2023-11-26-14-56']  # Example folder names
# plot_rmse_boxplots(folder_names)