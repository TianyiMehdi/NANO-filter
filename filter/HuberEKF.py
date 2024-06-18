from filterpy.kalman import ExtendedKalmanFilter
import numpy as np
from scipy.linalg import sqrtm

class HuberEKF(ExtendedKalmanFilter):
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
        F = self.jac_f(self.x)
        self.x = self.f(self.x)
        self.P = F @ self.P @ F.T + self.Q
        
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def _rho_function(self, x, r=1):
        for i in range(len(x)):
            if abs(x[i]) < r:
                x[i] = 0.5 * x[i] ** 2
            else:
                x[i] = r * abs(x[i]) - 0.5 * r * r
        return x
    
    def _phi_function(self, x, r=1):
        for i in range(len(x)):
            if abs(x[i]) < r:
                x[i] = 1
            else:
                x[i] = r * np.sign(x[i]) / x[i] 
        return x
    
    def update(self, y, max_iter=10, epson=1e-2):
        H = self.jac_h(self.x)
        x = self.x.copy()
        n, m = self.dim_z, self.dim_x
        zero_n_m = np.zeros((n, m))
        zero_m_n = np.zeros((m, n))
        SI = np.block([[self.R, zero_n_m],
                    [zero_m_n, self.P]])
        SI_inv_sqrt = np.linalg.inv(np.real(sqrtm(SI)))
        M = SI_inv_sqrt @ np.block([[H], [np.eye(self.dim_x)]])
        P_sqrt = SI_inv_sqrt[self.R.shape[0]:, self.R.shape[0]:]
        R_sqrt = SI_inv_sqrt[:self.R.shape[0], :self.R.shape[0]]

        x_ = 100 * np.ones(self.dim_x)
        
        for i in range(max_iter):
            if(np.sum(abs(x_ - x)) < epson):
                break
            x_ = x
            z = SI_inv_sqrt @ np.block([y, x]) 
            # x = np.linalg.inv(M.T @ M) @ M.T   
            e = z - M @ x
            phi = self._phi_function(e)
            Phi = np.diag(phi)
            hx = self.h(x)
            P_ = P_sqrt @ np.linalg.inv(Phi[self.R.shape[0]:, self.R.shape[0]:]) @ P_sqrt
            R_ = R_sqrt @ np.linalg.inv(Phi[:self.R.shape[0], :self.R.shape[0]]) @ R_sqrt
            PHT = P_ @ H.T
            self.S = H @ PHT + R_
            self.K = PHT @ np.linalg.inv(self.S)
            x = x + self.K @ (y - hx)
            # print(i, x)
        self.x = self.x + self.K @ (y - hx)
        I_KH = self._I - self.K @ H
        self.P = (I_KH @ P_ @ I_KH.T) + (self.K @ self.R @ self.K.T)

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()