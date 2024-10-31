import autograd.numpy as np

class Model:

    def __init__(self, state_outlier_flag=False, 
                measurement_outlier_flag=False, noise_type='Gaussian'):

        self.state_outlier_flag = state_outlier_flag
        self.measurement_outlier_flag = measurement_outlier_flag
        self.noise_type = noise_type
    
    def f(self, x, u=None):
        pass

    def h(self, x):
        pass

    def jac_f(self, x, u=None):
        pass

    def jac_h(self, x):
        pass

    def f_withnoise(self, x, u=None):
        pass

    def h_withnoise(self, x):
        pass
    
    