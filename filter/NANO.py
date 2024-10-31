import time

import autograd.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from autograd import hessian, jacobian
from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints
from filterpy.kalman import KalmanFilter as KF
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import unscented_transform as UT
from scipy.optimize import minimize

from .utils import (cal_mean, cal_mean_mc, is_positive_semidefinite,
                    kl_divergence)


class NANO:

    lr : float = 1.0 # 0.2
    threshold: float = 1e-4

    def __init__(self, model, loss_type='log_likelihood_loss', init_type='prior',
                 derivate_type='stein', iekf_max_iter=1, n_iterations=10, 
                    delta=5, c=5, beta=1e-4):    
        self.model = model
        self.dim_x = model.dim_x
        self.dim_y = model.dim_y    
        self.x = model.x0
        self.P = model.P0

        self.f = model.f
        self.h = model.h
        self.jac_f = model.jac_f
        self.jac_h = model.jac_h
        self.Q = model.Q
        self.R = model.R
        self._I = np.eye(self.dim_x)

        self.n_iterations = n_iterations
        self.points = MerweScaledSigmaPoints(self.dim_x, alpha=0.1, beta=2.0, kappa=1.0)
        self.x_prior = self.x
        self.P_prior = self.P
        self.x_post = self.x
        self.P_post = self.P

        self.init_type = init_type
        self.derivate_type = derivate_type
        self.iekf_max_iter = iekf_max_iter
        self.delta = delta
        self.c = c
        self.beta = beta

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
    
    def pseudo_huber_loss(self, x, y):
        delta = self.delta
        mse_loss = np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))
        return delta**2 * (np.sqrt(1 + mse_loss / delta**2) - 1)
    
    def weighted_log_likelihood_loss(self, x, y):
        c = self.c
        mse_loss = 0.5 * np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))
        weight = 1/(1 + mse_loss / c**2)
        return weight * mse_loss
    
    def beta_likelihood_loss(self, x, y):
        beta = self.beta
        R_inv = np.linalg.inv(self.R)
        det_R = np.linalg.det(self.R)
        return 1 / ((beta + 1)**1.5*(2*np.pi)**(self.dim_y*beta/2)) * det_R**(beta / 2) \
                - (1 / beta + 1) / ((2 * np.pi)**(beta*self.dim_y/2) * det_R**(beta/2)) * np.exp(-0.5*beta*(y-self.h(x)).T @ R_inv @ (y-self.h(x)))

    def loss_func_jacobian(self, x, y):
        # cal jacobian of loss function
        return jacobian(lambda x: self.loss_func(x, y))(x)
        
    def loss_func_hessian(self, x, y):
        # cal hessian of loss function
        return hessian(lambda x: self.loss_func(x, y))(x)
    
    def loss_func_hessian_diff(self, x, y, epsilon=5e-5):
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
    
    def map_loss(self, x_prior, P_prior, x_posterior, y):
        l1 = 0.5 * (x_posterior - x_prior).T @ np.linalg.inv(P_prior) @ (x_posterior - x_prior) 
        l2 = self.loss_func(x_posterior, y)
        return l1 + l2
            
    def update_init(self, y, x_prior, P_prior):
        # Laplace Approximation
        loss = lambda x_posterior: self.map_loss(x_prior, P_prior, x_posterior, y)
        x_hat_posterior = minimize(loss, x0=x_prior, method='BFGS').x
        P_posterior_inv = hessian(lambda x: self.map_loss(x_prior, P_prior, x, y))(x_hat_posterior)
        return x_hat_posterior, P_posterior_inv
    
    def update_iekf_init(self, y, x_prior, P_prior, max_iter=1):
        # Iterated Extended Kalman Filter (IEKF) for Maximum A Posteriori (MAP)
        x_hat = x_prior
        for i in range(max_iter):
            H = self.jac_h(x_hat)
            hx = self.h(x_hat)
            v = y - hx - H @ (x_prior - x_hat)
            PHT = P_prior @ H.T
            S = H @ PHT + self.R
            K = PHT @ np.linalg.inv(S)
            x_hat = x_prior + K @ v
        
        x_hat_posterior = x_hat
        I_KH = self._I - K @ H
        P_posterior = (I_KH @ P_prior @ I_KH.T) + (K @ self.R @ K.T)
        P_posterior_inv = np.linalg.inv(P_posterior)
        
        return x_hat_posterior, P_posterior_inv
    
    def predict(self, u=0):
        sigmas = self.points.sigma_points(self.x, self.P)

        self.sigmas_f = np.zeros((len(sigmas), self.dim_x))
        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = self.f(s, u)        
        
        self.x, self.P = UT(self.sigmas_f, self.points.Wm, self.points.Wc, self.Q)

        is_positive_semidefinite(self.P)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
    
    def update(self, y, lr = None, n_iterations = None):
        if lr == None:
            lr = self.lr
        if n_iterations is None:
            n_iterations = self.n_iterations
        
        # Initialize the first iteration step
        x_hat_prior = self.x.copy()
        P_inv_prior = np.linalg.inv(self.P).copy()
        if self.init_type == 'prior':
            x_hat, P_inv = x_hat_prior, P_inv_prior
        elif self.init_type == 'laplace':
            x_hat, P_inv = self.update_init(y, x_hat_prior, self.P.copy())
        else: # self.init_type == 'iekf'
            x_hat, P_inv = self.update_iekf_init(y, x_hat_prior, self.P.copy(), self.iekf_max_iter)
        
        is_positive_semidefinite(P_inv)

        for _ in range(n_iterations):          
            P = np.linalg.inv(P_inv)
            is_positive_semidefinite(P)

            if self.derivate_type == 'stein':
                E_hessian = P_inv @ cal_mean(lambda x: np.outer(x-x_hat, x-x_hat)*self.loss_func(x,y), x_hat, P, self.points) @ P_inv \
                            - cal_mean(lambda x: self.loss_func(x, y), x_hat, P, self.points) * P_inv
                P_inv_next = P_inv_prior - lr * E_hessian
                is_positive_semidefinite(P_inv_next)
                P_next = np.linalg.inv(P_inv_next)
                x_hat_next = x_hat - lr*(P_next @ P_inv @ cal_mean(lambda x: (x - x_hat) * self.loss_func(x, y), x_hat, P, self.points) - P_next @ P_inv_prior @ (x_hat - x_hat_prior))

            else: # self.derivate_type == 'direct'
                P_inv_next = P_inv_prior + lr*cal_mean(lambda x: self.loss_func_hessian_diff(x, y), x_hat, P, self.points)
                is_positive_semidefinite(P_inv_next)
                P_next = np.linalg.inv(P_inv_next)
                x_hat_next = x_hat - lr*(P_next @ cal_mean(lambda x: self.loss_func_jacobian(x, y), x_hat, P, self.points) - P_next @ P_inv_prior @ (x_hat - x_hat_prior))

            kld = kl_divergence(x_hat, P, x_hat_next, P_next)
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
