import os
import sys
import time
import argparse
import importlib

import autograd.numpy as np
from tqdm import tqdm
sys.path.append("../")
from filter import NANO, EKF, UKF, IEKF
from environ import RobotMove
from save_and_plot import calculate_rmse, save_per_exp




if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="")

    # Add arguments
    parser.add_argument("--filter_name", default="IEKF", type=str, help="Name of the filter")
    parser.add_argument("--model_name", default="RobotMove", type=str, help="Name of the model")
    parser.add_argument("--noise_name", default="Beta", type=str, help="Name of the model")
    parser.add_argument("--result_dir", default=None, type=str, help="Save dir")
    parser.add_argument("--outlier_type", default='direct', type=str,
                        help='Different types to add outliers, "indirect" and "direct"')
    parser.add_argument("--random_seed", default=42, type=int, help='Number of the random seed')

    # env arguments
    parser.add_argument("--state_outlier_flag", default=False, type=bool, help="")
    parser.add_argument("--measurement_outlier_flag", default=False, type=bool, help="")
    args = parser.parse_args()

    if args.filter_name == "PF":
        parser.add_argument("--N_particles", default=100, type=float, help="Parameter for PF")
    
    if args.filter_name == "IEKF":
        parser.add_argument("--max_iter", default=3, type=float, help="Parameter for iEKF")

    # exp arguments
    parser.add_argument("--N_exp", default=100, type=int, help="Number of the MC experiments")
    parser.add_argument("--steps", default=50, type=int, help="Number of the steps in each trajectory")

    # Parse the arguments
    args = parser.parse_args()
    args_dict = vars(args)

    np.random.seed(args_dict['random_seed'])

    model = RobotMove(args_dict['state_outlier_flag'], args_dict['measurement_outlier_flag'],
                        args_dict['noise_name'])
    filter = IEKF(model, args_dict['max_iter'])

    x_mc = []
    y_mc = []
    x_hat_mc = []
    all_time = []

    for _ in tqdm(range(args_dict['N_exp'])):
        x_list, y_list, x_hat_list = [], [], []
        run_time = []
        # initialize system
        x = model.x0
        y = model.h_withnoise(x)

        x_list.append(x)
        y_list.append(y)
        x_hat_list.append(x)

        for i in range(1, args_dict['steps']):
            # generate data
            x = model.f_withnoise(x)
            y = model.h_withnoise(x)
            x_list.append(x)
            y_list.append(y)

            time1 = time.time()
            # perform filtering
            filter.predict()
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