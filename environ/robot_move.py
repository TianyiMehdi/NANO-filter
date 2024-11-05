import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy import sin, cos, arctan, pi, arctan2, sqrt
from .model import Model


class RobotMove(Model):

    dt : float = 1.0

    def __init__(self, state_outlier_flag=False, 
                measurement_outlier_flag=False, noise_type='Gaussian'):
        super().__init__(self)
        
        self.dim_x = 2
        self.dim_y = 4
        self.x0 = np.array([0., 0.])
        self.P0 = np.eye(self.dim_x)

        self.state_outlier_flag = state_outlier_flag
        self.measurement_outlier_flag = measurement_outlier_flag
        self.noise_type = noise_type
        self.alpha = 2.0
        self.beta = 5.0

        self.obs_var = np.ones(self.dim_y) * 0.01
        self.Q = np.eye(self.dim_x) * 0.0025
        if noise_type == 'Beta':
            self.R = np.eye(self.dim_y) * (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
        else:
            self.R = np.eye(self.dim_y) * 0.01

    def f(self, x, dt=None):
        if dt is None:
            dt = self.dt
        x1, x2 = x
        x1_ = x1 + dt
        x2_ = x2 + dt
        return np.array([x1_, x2_])

    def h(self, x):
        landmarks = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])
        hx = []
        px, py = x
        for i in range(len(landmarks)):
            dist = sqrt((px - landmarks[i][0])**2 + (py - landmarks[i][1])**2)
            hx.append(dist)
        return np.array(hx)
        

    def jac_f(self, x, u=0):
        return np.array([[1, 0], [0, 1]])

    def jac_h(self, x):
        landmarks = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])
        h_list=[]
        for landmark in landmarks:
            h1 = (x[0]-landmark[0])/(sqrt((x[0]-landmark[0])**2+(x[1]-landmark[1])**2))
            h2 = (x[1] - landmark[1]) / (sqrt((x[0] - landmark[0]) ** 2 + (x[1] - landmark[1]) ** 2))
            h_list.append([h1,h2])
        return np.array(h_list)
    
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
        if self.noise_type == 'Gaussian':
            if self.measurement_outlier_flag:
                prob = np.random.rand()
                if prob <= 0.95:
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
            return self.h(x) + np.random.laplace(loc=0, scale=self.obs_var, size=(self.dim_y, ))
