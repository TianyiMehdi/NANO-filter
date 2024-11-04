from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import autograd.numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(file_path)
# plt.rcParams

def calculate_rmse(x_mc, x_hat_mc):
    """
    Calculate the Average Root Mean Square Error (RMSE) for each state.

    :param x_mc: Numpy array with shape (num_experiments, trajectory_length, num_states) representing the true states.
    :param x_hat_mc: Numpy array with shape (num_experiments, trajectory_length, num_states) representing the estimated states.
    :return: A numpy array containing the average RMSE for each state.
    """
    squared_errors = (x_mc - x_hat_mc) ** 2

    threshold=100
    # if np.sqrt(np.mean(squared_errors[-1], axis=0)).all()>threshold:
    #     raise ValueError
    rmse_sum=0
    index=0

    for squared_error in squared_errors:
        rmse=np.sqrt(np.mean(squared_error, axis=0))
        # if (rmse<threshold).all():#剔除实验异常值
        rmse_sum+=rmse
        index+=1
    rmse_perstate=rmse_sum/index
    return rmse_perstate #先求总的平均值再开根号 ARMSE 每一次实验都开个根号再求平均值


def save_per_exp(data_dict, **args_dict):
    data_dir = args_dict['result_dir']
    x_mc = data_dict['x_mc']
    x_hat_mc = data_dict['x_hat_mc']
    x_rmse = calculate_rmse(x_mc, x_hat_mc)
    print('RMSE:', x_rmse)
    print('time:', data_dict['mean_time'])
    data_dict['x_rmse'] = x_rmse
    args_dict['x_rmse'] = x_rmse.tolist()
    args_dict['time'] = data_dict['mean_time']

    # Get the current date and time in the format Year-Month-Day-Hour-Minute
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

    # Create a directory named after the current date and time
    if data_dir is None:
        data_dir = f"{directory_path}/results/{current_time}"
    os.makedirs(data_dir, exist_ok=True)

    # Create a file name for parameters based on 'filter_name', 'model_name', and 'noise_name' from the parameters dict
    parameter_file_name = f"{args_dict.get('filter_name', 'unknown')}-{args_dict.get('model_name', 'unknown')}-{args_dict.get('noise_name', 'unknown')}.json"
    data_file_name = "data.npz"

    # Save parameters to a JSON file in the created directory
    with open(os.path.join(data_dir, parameter_file_name), 'w') as file:
        json.dump(args_dict, file, indent=4)

    # Convert the data dictionary to arrays and save to an NPZ file in the created directory
    np.savez(os.path.join(data_dir, data_file_name), **data_dict)

    plot_state_error(data_dir, args_dict['filter_name'], **data_dict)

    return f"Files saved in directory: {data_dir}"


def plot_state_error(data_dir, filter_name, **data_dict):
    x_mc = data_dict['x_mc']
    x_hat_mc = data_dict['x_hat_mc']
    num_experiments, num_steps, num_states = x_mc.shape

    # 使用 Seaborn 的样式
    plt.style.use(f"{directory_path}/style.mpl")

    # 对每个状态绘制一个误差图
    for state_index in range(num_states):
        error = x_mc[:, :, state_index] - x_hat_mc[:, :, state_index]

        # 计算误差的均值和标准差
        error_mean = error.mean()
        error_std = error.std()

        # 准备用于 Seaborn 绘图的数据
        data = pd.DataFrame({
            'Step': np.tile(np.arange(num_steps), num_experiments),
            'Error': error.flatten()
        })

        # 绘制误差线图
        plt.figure(figsize=(8, 6))
        sns.lineplot(x='Step', y='Error', data=data, ci='sd')

        plt.xlabel("Step")
        plt.ylabel(r'$x_{}$ Error'.format(state_index + 1))  # 使用 TeX 公式
        plt.axhline(0, ls='-.', c='k', lw=1, alpha=0.5)
        plt.ylim(1.5 * (error_mean - error_std), 1.5 * (error_mean + error_std))  # 使用均值 ± sigma
        plt.xlim(0, num_steps - 1)

        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f'{filter_name}_state_{state_index + 1}_error.png'))
        plt.close()


def plot_box_methods(data_dicts):
    # box plot
    pass