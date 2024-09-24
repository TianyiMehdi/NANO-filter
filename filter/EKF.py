from filterpy.kalman import ExtendedKalmanFilter
import autograd.numpy as np


class EKF(ExtendedKalmanFilter):
    def __init__(self, model):
        super().__init__(dim_x=model.dim_x,
                         dim_z=model.dim_y,
                         )
        
        self.f = model.f
        self.h = model.h
        self.jac_f = model.jac_f
        self.jac_h = model.jac_h

        self.Q = model.Q
        self.R = model.R

        self.x = model.x0
        self.P = model.P0 

    def predict(self, u=0):
        F = self.jac_f(self.x, u)
        self.x = self.f(self.x, u)
        self.P = F @ self.P @ F.T + self.Q
        
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, y):
        H = self.jac_h(self.x)
        hx = self.h(self.x)
        PHT = self.P @ H.T
        self.S = H @ PHT + self.R
        self.K = PHT @ np.linalg.inv(self.S)
        self.x = self.x + self.K @ (y - hx)
        I_KH = self._I - self.K @ H
        self.P = (I_KH @ self.P @ I_KH.T) + (self.K @ self.R @ self.K.T)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()