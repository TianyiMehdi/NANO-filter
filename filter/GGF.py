import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import KalmanFilter as KF
from filterpy.kalman import MerweScaledSigmaPoints, JulierSigmaPoints
from filterpy.kalman import unscented_transform
import autograd.numpy as np
from autograd import jacobian, hessian
import seaborn as sns
# logging.basicConfig(level=logging.DEBUG)
from scipy.optimize import minimize

from .utils import is_positive_semidefinite, cal_mean

class GGF:
    def __init__(self, model, n_iterations=10):    
        self.model = model
        self.dim_x = model.dim_x
        self.dim_y = model.dim_y    
        self.x = model.x0
        self.P = model.P0

        self.f = model.f
        self.h = model.h
        self.Q = model.Q
        self.R = model.R

        self.n_iterations = n_iterations
        self.points = MerweScaledSigmaPoints(self.dim_x, alpha=0.1, beta=2.0, kappa=1.0)
        self.UT  = unscented_transform
        self.x_prior = self.x
        self.P_prior = self.P
        self.x_post = self.x
        self.P_post = self.P
        
    def log_likelihood_loss(self, x, y):
        return 0.5 * np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))
    
    def log_likelihood_jacobian(self, x, y):
        # cal jacobian of loss function
        return jacobian(lambda x: self.log_likelihood_loss(x, y))(x)
        
    def log_likelihood_hessian(self, x, y):
        # cal hessian of loss function
        return hessian(lambda x: self.log_likelihood_loss(x, y))(x)
    
    def predict(self):
        # print('----------predict----------')

        sigmas = self.points.sigma_points(self.x, self.P)

        self.sigmas_f = np.zeros((len(sigmas), self.dim_x))
        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = self.f(s)        
        
        self.x, self.P = self.UT(self.sigmas_f, self.points.Wm, self.points.Wc, self.Q)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
    
    def map_loss(self, x_prior, P_prior, x_posterior, y):
        l1 = 0.5 * (x_posterior - x_prior).T @ np.linalg.inv(P_prior) @ (x_posterior - x_prior) 
        l2 = self.log_likelihood_loss(x_posterior, y)
        return l1 + l2
            
    def update_init(self, y, x_prior, P_prior):
        # Laplace近似
        loss = lambda x_posterior: self.map_loss(x_prior, P_prior, x_posterior, y)
        x_hat_posterior = minimize(loss, x0=x_prior, method='BFGS').x
        # print('----------update_init----------')
        # print('x_hat_posterior: ', x_hat_posterior)
        P_posterior_inv = hessian(lambda x: self.map_loss(x_prior, P_prior, x, y))(x_hat_posterior)
        # print('P_posterior_inv: ', P_posterior_inv)
        
        return x_hat_posterior, P_posterior_inv
    
    def update(self, y, n_iterations = None):
        # print('----------update----------')
        beta = 1
        epsilon = 1e-6
        
        if n_iterations is None:
            n_iterations = self.n_iterations
        
        # 求初始迭代步的Laplace近似后验估计
        x_hat_prior = self.x.copy()
        P_inv_prior = np.linalg.inv(self.P).copy()
        # x_hat, P_inv = self.update_init(y, x_hat_prior, self.P.copy())
        x_hat, P_inv = x_hat_prior, P_inv_prior
        L = np.linalg.cholesky(P_inv)

        for _ in range(n_iterations):
            P = np.linalg.inv(P_inv)
            E_hessian = P_inv_prior + cal_mean(lambda x: self.log_likelihood_hessian(x, y), x_hat, P, self.points)
            L = (1 - beta / 2) * L + beta / 2 * E_hessian @ np.linalg.inv(L).T
            P_inv_next = L @ L.T + epsilon * np.eye(self.dim_x)
            P_next = np.linalg.inv(P_inv_next)

            x_hat_next = x_hat - beta * (P_next @ cal_mean(lambda x: self.log_likelihood_jacobian(x, y), x_hat, P, self.points) - P_next @ P_inv_prior @ (x_hat - x_hat_prior))

            P_inv = P_inv_next.copy()
            x_hat = x_hat_next.copy()
        
            
        self.x = x_hat
        self.P = np.linalg.inv(P_inv)

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
