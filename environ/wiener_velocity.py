import autograd.numpy as np
from .model import Model


class WienerVelocity(Model):

    dt : float = 0.1
    def __init__(self, state_outlier_flag=False, 
                measurement_outlier_flag=False, noise_type='Gaussian'):
        super().__init__(self)
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],
                           ])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        self.dim_x = self.F.shape[0]
        self.dim_y = self.H.shape[0]
        self.x0 = np.array([0., 0., 1., 1.])
        self.P0 = np.eye(self.dim_x)

        self.state_outlier_flag = state_outlier_flag
        self.measurement_outlier_flag = measurement_outlier_flag
        self.noise_type = noise_type
        self.alpha = 2.0
        self.beta = 5.0

        dt = self.dt
        self.var = np.array([1, 1, 1, 1])
        self.obs_var = np.array([1, 1])
        self.Q = np.array([
            [dt**3/3, 0, dt**2/2, 0],
            [0, dt**3/3, 0, dt**2/2],
            [dt**2/2, 0, dt, 0],
            [0, dt**2/2, 0, dt]
        ])
        if noise_type == 'Beta':
            self.R = np.eye(self.dim_y) * (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        else:
            self.R = np.diag(self.obs_var)


    def f(self, x, dt=None, u=None):
        if u is None:
            return self.F @ x

    def h(self, x):
        return self.H @ x

    def jac_f(self, x, u=None):
        return self.F

    def jac_h(self, x):
        return self.H

    def f_withnoise(self, x):
        if self.state_outlier_flag:
            prob = np.random.rand()
            if prob <= 0.9:
                cov = self.Q  # 95%概率使用Q
            else:
                cov = 100 * self.Q  # 5%概率使用100Q
        else:
            cov = self.Q
        return self.f(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=cov)
    
    def h_withnoise(self, x):
        if self.noise_type == 'Gaussian':
            if self.measurement_outlier_flag:
                prob = np.random.rand()
                if prob <= 0.9:
                    cov = self.R  # 95%概率使用R
                else:
                    cov = 1000 * self.R  # 5%概率使用100R
            else:
                cov = self.R
            return self.h(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_y), cov=cov)
        elif self.noise_type == 'Beta':
            noise = np.random.beta(self.alpha, self.beta, self.dim_y)
            noise = noise - np.mean(noise)
            return self.h(x) + noise
        else:
            return self.h(x) + np.random.laplace(loc=0, scale=1, size=(self.dim_y, ))