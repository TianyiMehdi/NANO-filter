import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import KalmanFilter as KF
from filterpy.kalman import MerweScaledSigmaPoints, JulierSigmaPoints
from filterpy.kalman import unscented_transform as UT
import autograd.numpy as np
from autograd import jacobian, hessian
import seaborn as sns
import time
# logging.basicConfig(level=logging.DEBUG)
from scipy.optimize import minimize

from .utils import is_positive_semidefinite, cal_mean, cal_mean_mc, kl_divergence

class GGF:

    beta : float = 1.0
    gamma : float = 1.0
    epsilon: float = 1e-12
    threshold: float = 0.1

    def __init__(self, model, loss_type = 'log_likelihood_loss', n_iterations=10):    
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
        self.x_prior = self.x
        self.P_prior = self.P
        self.x_post = self.x
        self.P_post = self.P

        if loss_type == 'pseudo_huber_loss':
            self.loss_func = self.pseudo_huber_loss
        elif loss_type == 'weighted_log_likelihood_loss':
            self.loss_func = self.weighted_log_likelihood_loss
        elif loss_type == 'beta_likelihood_loss':
            self.loss_func = self.beta_likelihood_loss
        else:
            self.loss_func = self.log_likelihood_loss
        
    def log_likelihood_loss(self, x, y):
        return 0.5 * np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))
    
    def pseudo_huber_loss(self, x, y, delta=150):
        mse_loss = np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))

        return delta**2 * (np.sqrt(1 + mse_loss / delta**2) - 1)
    
    def weighted_log_likelihood_loss(self, x, y, c=0.1):
        mse_loss = np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))
        weight = np.sqrt(1 + mse_loss / c**2)
        log_likelihood = -0.5 * (mse_loss + self.dim_y * np.log(2 * np.pi) + np.log(np.linalg.det(self.R)))
        
        return -weight * log_likelihood
    
    def beta_likelihood_loss(self, x, y, beta=5e-9):
        R_inv = np.linalg.inv(self.R)
        return 1 / ((beta + 1)**1.5*(2*np.pi)**(self.dim_y*beta/2))\
                - (1 / beta) * 1 / ((2 * np.pi) ** (beta*self.dim_y/2)) * np.exp(-0.5*beta*(y-self.h(x)).T @ R_inv @ (y-self.h(x)))

    def loss_func_jacobian(self, x, y):
        # cal jacobian of loss function
        return jacobian(lambda x: self.loss_func(x, y))(x)
        
    def loss_func_hessian(self, x, y):
        # cal hessian of loss function
        return hessian(lambda x: self.loss_func(x, y))(x)
    
    def loss_func_hessian_diff(self, x, y, epsilon=1e-4):
        n = len(x)
        Hessian = np.zeros((n, n))
        f = self.loss_func
        fx = f(x, y)
        
        for i in range(n):
            for j in range(i, n):
                x_ij = x.copy()
                x_ij[i] += epsilon
                x_ij[j] += epsilon
                fij = f(x_ij, y)
                
                x_i = x.copy()
                x_i[i] += epsilon
                fi = f(x_i, y)
                
                x_j = x.copy()
                x_j[j] += epsilon
                fj = f(x_j, y)
                
                Hessian[i, j] = (fij - fi - fj + fx) / (epsilon**2)
                Hessian[j, i] = Hessian[i, j]
                
        return Hessian
    
    def predict(self):
        # print('----------predict----------')

        # is_positive_semidefinite(self.P)
        sigmas = self.points.sigma_points(self.x, self.P)

        self.sigmas_f = np.zeros((len(sigmas), self.dim_x))
        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = self.f(s)        
        
        self.x, self.P = UT(self.sigmas_f, self.points.Wm, self.points.Wc, self.Q)

        is_positive_semidefinite(self.P)

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
        beta = self.beta
        epsilon = self.epsilon
        
        if n_iterations is None:
            n_iterations = self.n_iterations
        
        # 求初始迭代步的Laplace近似后验估计
        x_hat_prior = self.x.copy()
        P_inv_prior = np.linalg.inv(self.P).copy()
        # x_hat, P_inv = self.update_init(y, x_hat_prior, self.P.copy())
        x_hat, P_inv = x_hat_prior, P_inv_prior
        is_positive_semidefinite(self.P)
        L = np.linalg.cholesky(P_inv)

        for _ in range(n_iterations):
            P = np.linalg.inv(P_inv)
            time1 = time.time()
            # 0.01025 s
            E_hessian = P_inv_prior + cal_mean(lambda x: self.loss_func_hessian_diff(x, y), x_hat, P, self.points)
            time2 = time.time()
            # 2.193e-05 s
            L = (1 - beta / 2) * L + beta / 2 * E_hessian @ np.linalg.inv(L).T
            time3 = time.time()
            # 2.145e-05 s
            P_inv_next = L @ L.T + epsilon * np.eye(self.dim_x)
            P_next = np.linalg.inv(P_inv_next)
            time4 = time.time()
            
            # 0.0026s
            x_hat_next = x_hat - beta * (P_next @ cal_mean(lambda x: self.loss_func_jacobian(x, y), x_hat, P, self.points) - P_next @ P_inv_prior @ (x_hat - x_hat_prior))
            time5 = time.time()
            time_dict = {'E_hess': time2 - time1,
                        'L_update': time3 - time2,
                        'P_cal': time4 - time3,
                        'x_cal': time5 - time4,
                        'one_iter_time': time5 - time1}
            # print(time_dict)
            kld = kl_divergence(x_hat, P, x_hat_next, P_next)
            # print(_, " iteration: ", kld)
            if kld < self.threshold:
                P_inv = P_inv_next.copy()
                x_hat = x_hat_next.copy()
                break

            P_inv = P_inv_next.copy()
            x_hat = x_hat_next.copy()
        
            
        self.x = x_hat
        self.P = np.linalg.inv(P_inv)

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
