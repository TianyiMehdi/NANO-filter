import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy import sin, cos, arctan, pi, arctan2
from .model import Model

class Air_Traffic(Model):

    q1 = 0.5
    q2 = 1e-6
    height = 50
    dt = 0.2

    def __init__(self, state_outlier_flag=False, 
                measurement_outlier_flag=False, noise_type='Gaussian'):
        super().__init__(self)
        q1 = self.q1
        q2 = self.q2
        height = self.height
        tau = self.dt
        
        self.dim_x = 5
        self.dim_y = 4
        self.P0 = np.diag(np.array([5, 5, 2e4, 10, 1e-7]))
        self.x0 =  np.array([130, 25, -20, 1, -4*pi/180])

        self.state_outlier_flag = state_outlier_flag
        self.measurement_outlier_flag = measurement_outlier_flag
        self.noise_type = noise_type
        self.alpha = 2.0
        self.beta = 5.0
        
        self.Q = np.array([
            [q1*tau**3/3, q1*tau**2/2, 0, 0, 0],
            [q1*tau**2/2, q1*tau, 0, 0, 0],
            [0, 0, q1*tau**3/3, q1*tau**2/2, 0],
            [0, 0, q1*tau**2/2, q1*tau, 0],
            [0, 0, 0, 0, q2*tau],
        ])
        self.obs_var = np.array([1000, (30*pi/180)**2, (30*pi/180)**2, 100])
        R1 = np.diag(np.array([1000, (30*pi/180)**2, (30*pi/180)**2, 100]))
        R2 = np.diag(np.array([1000, (1e-3*pi/180)**2, (30*pi/180)**2, 1e-4]))
        # self.R = R1
        if noise_type == 'Beta':
            self.R = np.eye(self.dim_y) * (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        else:
            self.R = R1

    def f(self, x, u=None):
        tau = self.dt
        Delta = x[4]
        F = np.array([
            [1, sin(Delta*tau)/Delta, 0, -(1 - cos(Delta*tau))/Delta, 0],
            [0, cos(Delta*tau), 0, -sin(Delta*tau), 0],
            [0, (1 - cos(Delta*tau))/Delta, 1, sin(Delta*tau)/Delta, 0],
            [0, sin(Delta*tau), 0, cos(Delta*tau), 0],
            [0, 0, 0, 0, 1]
        ])
        return F @ x
    
    def h(self, x):
        height = self.height
        px, dpx, py, dpy, Delta = x
        y1 = np.sqrt(px**2 + py ** 2 + height**2)
        y2 = arctan2(py, px)
        y3 = arctan(height / np.sqrt(px**2 + py ** 2))
        y4 = (px * dpx + py * dpy) / y1
        return np.array([y1, y2, y3, y4])
    
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
                if prob <= 0.9:
                    cov = self.R  # 95%概率使用R
                else:
                    cov = 100 * self.R  # 5%概率使用100R
            else:
                cov = self.R
            return self.h(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_y), cov=cov)
        elif self.noise_type == 'Beta':
            noise = np.random.beta(self.alpha, self.beta, self.dim_y)
            noise = noise - np.mean(noise)
            return self.h(x) + noise
        else:
            return self.h(x) + np.random.laplace(loc=0, scale=self.obs_var, size=(self.dim_y, ))

    def jac_f(self, x_hat, u=0):
        return jacobian(lambda x: self.f(x))(x_hat)
    
    def jac_h(self, x_hat, u=0):
        return jacobian(lambda x: self.h(x))(x_hat)
        
