import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy import sin, cos, arctan, pi, arctan2, sqrt
from data_processing import load_data
from .model import Model

class UGV(Model):

    dt : float = 1.0 / 15

    def __init__(self, state_outlier_flag=False, 
                measurement_outlier_flag=False, noise_type='Gaussian'):
        super().__init__(self)
        self.dim_x = 3
        self.dim_y = 6
        self.x0 = np.array([0., 0., 0.])
        self.P0 = np.diag(np.array([0.0001, 0.0001, 0.0001])) ** 2

        self.state_outlier_flag = state_outlier_flag
        self.measurement_outlier_flag = measurement_outlier_flag
        self.noise_type = noise_type
        self.alpha = 2.0
        self.beta = 5.0

        self.process_std = np.array([0.0034, 0.0056, 0.0041])
        self.observation_std = np.array([0.0238, 0.0284, 0.0259, 0.0107, 0.0094, 0.0118])
        self.obs_var = np.ones(self.dim_y) * 0.01
        self.Q = np.diag(self.process_std**2)
        self.R = np.diag(self.observation_std**2)

    def f(self, x, u, dt=None):
        if dt is None:
            dt = self.dt
        x0, x1, x2 = x
        x0_ = x0 + dt * u[0] * np.cos(x2) - 0.0001675046729610055
        x1_ = x1 + dt * u[0] * np.sin(x2) - 0.0001963914687308423
        x2_ = x2 + dt * u[1] + 0.0005640178926637775
        return np.array([x0_, x1_, x2_])

    def h(self, x):
        px, py, theta = x
        obstacle_info = [[1.052, -2.695], [4.072, -1.752], [6.028, -3.324]]
        obstacle = np.array(obstacle_info)
        der_x_robot = 0.329578
        rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        dist = []
        obstacle_obs = []
        for M in obstacle:
            obstacle_obs.append(((M - x[:2])@rot.T) - np.array([der_x_robot, 0]))
        obstacle_obs = np.array(obstacle_obs)
        dist = np.linalg.norm(obstacle_obs, axis=1)
        angle = np.arctan2(obstacle_obs[:, 1], obstacle_obs[:, 0])
        return np.concatenate([dist, angle]) + np.array([-0.0312, -0.0581, -0.0557, 0.0053, 0.0059, 0.0125])
        

    def jac_f(self, x, u):
        px, py, theta = x
        v, w = u
        dt = self.dt
        return np.array([
            [1, 0, -v*sin(theta)*dt],
            [0, 1, v*cos(theta)*dt],
            [0, 0, 1]
        ])

    def jac_h(self, x_hat):
        return jacobian(lambda x: self.h(x))(x_hat)
    
    def get_sensor_data(self, filepath_list, min_len):
        state = []
        action = []
        obs = []
        for file_path in filepath_list:
            state_, action_, obs_ = load_data(file_path, min_len)
            state += state_
            action += action_
            obs += obs_
        NUM_RUNS = len(state)
        X_list = []
        Y_list = []
        U_list = []
        for i in range(NUM_RUNS):
            X_list.append(np.array(state[i]))
            Y_list.append(np.array(obs[i]))
            U_list.append(np.array(action[i]))
        X_arr = np.array(X_list)
        Y_arr = np.array(Y_list)
        U_arr = np.array(U_list)
        zero_states = X_arr[:, 0, :]
        return X_arr, Y_arr, U_arr, zero_states
    
    def f_withnoise(self, x, u=None):
        if self.state_outlier_flag:
            prob = np.random.rand()
            if prob <= 0.95:
                cov = self.Q  # 95%概率使用Q
            else:
                cov = 100 * self.Q  # 5%概率使用100Q
        else:
            cov = self.Q
        return self.f(x, u) + np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=cov)
    
    def h_withnoise(self, x):
        if self.noise_type == 'Gaussian':
            if self.measurement_outlier_flag:
                prob = np.random.rand()
                if prob <= 0.99:
                    cov = self.R  # 95%概率使用R
                else:
                    cov = 10 * self.R  # 5%概率使用100R
            else:
                cov = self.R
            return self.h(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_y), cov=cov)
        elif self.noise_type == 'Beta':
            noise = np.random.beta(self.alpha, self.beta, self.dim_y)
            noise = noise - np.mean(noise)
            return self.h(x) + noise
        else:
            return self.h(x) + np.random.laplace(loc=0, scale=self.obs_var, size=(self.dim_y, ))
