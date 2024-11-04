import argparse
import importlib
import os
import sys
import time

import autograd.numpy as np
from tqdm import tqdm

sys.path.append("../")
from data_processing import load_data
from environ import UGV
from filter import EKF, NANO, UKF
from save_and_plot import calculate_rmse, save_per_exp

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="")

    # Add arguments
    parser.add_argument("--filter_name", default="NANO", type=str, help="Name of the filter")
    parser.add_argument("--model_name", default="UGV", type=str, help="Name of the model")
    parser.add_argument("--noise_name", default="Gaussian", type=str, help="Name of the model")
    parser.add_argument("--result_dir", default=None, type=str, help="Save dir")
    parser.add_argument("--outlier_type", default='direct', type=str,
                        help='Different types to add outliers, "indirect" and "direct"')
    parser.add_argument("--random_seed", default=42, type=int, help='Number of the random seed')

    # env arguments
    parser.add_argument("--state_outlier_flag", default=False, type=bool, help="")
    parser.add_argument("--measurement_outlier_flag", default=False, type=bool, help="")

    parser.add_argument("--n_iterations", default=1, type=int, help="Iterations for NANO")
    parser.add_argument("--lr", default=0.5, type=float, help="Learning Rate for NANO")
    parser.add_argument("--init_type", default='prior', type=str, help="Initialization type for Natural Gradient iteration")
    # init_type: 'prior', 'laplace', 'iekf'; usually 'prior' for linear and low nonlinearity system, 'iekf' for high nonlinearity system
    parser.add_argument("--derivate_type", default='stein', type=str, help="Derivate type for Natural Gradient iteration")
    # derivate_type: 'direct', 'stein'; 'direct' for linear and low nonlinearity system, 'stein' for high nonlinearity system
    parser.add_argument("--iekf_max_iter", default=1, type=float, help="Iterations for iekf init")

    parser.add_argument("--loss_type", default='log_likelihood_loss', type=str, help="Loss type for NANO")
    # loss_type: 'log_likelihood_loss', 'pseudo_huber_loss', 'weighted_log_likelihood_loss', 'beta_likelihood_loss'
    parser.add_argument("--delta", default=1, type=float, help="HyperParameter for Huber loss")
    parser.add_argument("--c", default=8, type=float, help="HyperParameter for Weight loss")
    parser.add_argument("--beta", default=2e-2, type=float, help="HyperParameter for beta divergence")
    
    # exp arguments
    parser.add_argument("--N_exp", default=46, type=int, help="Number of the MC experiments")
    parser.add_argument("--steps", default=101, type=int, help="Number of the steps in each trajectory")

    # Parse the arguments
    args = parser.parse_args()
    args_dict = vars(args)

    np.random.seed(args_dict['random_seed'])

    filepath_list=['/home/zhangtianyi/NANO-filter/data_processing/20230216-140452.npz',
                '/home/zhangtianyi/NANO-filter/data_processing/20230216-141321.npz',
                '/home/zhangtianyi/NANO-filter/data_processing/20230216-141616.npz',
                '/home/zhangtianyi/NANO-filter/data_processing/20230216-142042.npz']

    lr = args_dict['lr']
    measurement_outlier_flag = args_dict['measurement_outlier_flag']
    model = UGV(args_dict['state_outlier_flag'], args_dict['measurement_outlier_flag'],
                        args_dict['noise_name'])
    filter = NANO(model, loss_type=args_dict['loss_type'], init_type=args_dict['init_type'], 
                  derivate_type=args_dict['derivate_type'], iekf_max_iter=args_dict['iekf_max_iter'],
                  n_iterations=args_dict['n_iterations'], delta=args_dict['delta'], c=args_dict['c'], beta=args_dict['beta'])
    
    x_lists, y_lists, u_lists, x0_lists = model.get_sensor_data(filepath_list, min_len=args_dict['steps'])
    N_exp = len(x_lists)

    x_mc = []
    y_mc = []
    x_hat_mc = []
    all_time = []

    for n in tqdm(range(N_exp)):
        x_list, y_list, u_list, x0 = x_lists[n], y_lists[n], u_lists[n], x0_lists[n]
        x_hat_list = []
        run_time = []
        # initialize system
        x = x0
        y = y_list[0]
        filter.x = x0
        filter.P = np.diag(np.array([0.0001, 0.0001, 0.0001])) ** 2

        x_hat_list.append(x)

        for i in range(1, args_dict['steps']):
            u = u_list[i]
            y = y_list[i]
            if measurement_outlier_flag:
                prob = np.random.rand()
                if prob <= 0.01:
                    random_number = np.random.randint(0, 2)
                    y[random_number] = 20
            time1 = time.time()
            # perform filtering
            filter.predict(u)
            filter.update(y)
            time2 = time.time()
            x_hat_list.append(filter.x)
            run_time.append(time2 - time1)

        x_mc.append(np.array(x_list))
        y_mc.append(np.array(y_list))
        x_hat_mc.append(np.array(x_hat_list))
        all_time.append(np.mean(run_time))

    x_mc = np.array(x_mc)
    y_mc = np.array(y_mc)
    x_hat_mc = np.array(x_hat_mc)
    mean_time = np.mean(all_time)

    data_dict = {'x_mc': x_mc, 'y_mc': y_mc, 'x_hat_mc': x_hat_mc, 'mean_time': mean_time}
    # print(calculate_rmse(x_mc, x_hat_mc))

    save_per_exp(data_dict, **args_dict)