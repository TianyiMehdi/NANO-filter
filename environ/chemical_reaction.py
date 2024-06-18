from dataclasses import dataclass
import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy import sin, cos, arctan, pi, arctan2

class ChemicalReaction:

    dt : float = 0.10

    def __init__(self, state_outlier_flag=False, measurement_outlier_flag=False):

        self.dim_x = 2
        self.dim_y = 1
        self.x0 = np.array([0.1, 4.5])
        self.P0 = 1e-4 * np.eye(self.dim_x)

        self.state_outlier_flag = state_outlier_flag
        self.measurement_outlier_flag = measurement_outlier_flag
        self.Q = 1e-4 * np.eye(self.dim_x)
        self.R = 1e-2 * np.eye(self.dim_y)

    def f(self, x, dt=None):
        # reaction parmater
        k1 = 0.16
        k2 = 0.0064
        if dt is None:
            dt = self.dt
        x1, x2 = x
        x1_ = x1 + (-2 * k1 * x1 * x1 + 2 * k2 * x2) * dt
        x2_ = x2 + (k1 * x1 * x1 - k2 * x2) * dt
        # print('state:', np.array([x1_, x2_]))
        return np.array([x1_, x2_])

    def h(self, x):
        x1, x2 = x
        # print('obs:', np.array([x1 + x2]))
        return np.array([x1 + x2])
    
    def f_withnoise(self, x):
        if self.state_outlier_flag:
            prob = np.random.rand()
            if prob <= 0.95:
                cov = self.Q  # 95%概率使用Q
            else:
                cov = 100 * self.Q  # 5%概率使用100Q
        else:
            cov = self.Q
        return self.f(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=cov)
    
    def h_withnoise(self, x):
        if self.measurement_outlier_flag:
            prob = np.random.rand()
            if prob <= 0.8:
                cov = self.R  # 95%概率使用R
            else:
                cov = 1000 * self.R  # 5%概率使用100R
        else:
            cov = self.R
        return self.h(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_y), cov=cov)

    def jac_f(self, x):
        k1 = 0.16
        k2 = 0.0064
        x1, x2 = x
        jac_11 = 1 - 4 * k1 * x1 * self.dt
        jac_12 = -2 * k2 * self.dt
        jac_21 = 2 * k1 * x1 * self.dt
        jac_22 = 1 - k2 * self.dt

        # print('state:', np.array([x1_, x2_]))
        return np.array([[jac_11, jac_12], [jac_21, jac_22]])

    def jac_h(self, x):
        jac_h11 = 1
        jac_h12 = 1
        return np.array([[jac_h11, jac_h12]])