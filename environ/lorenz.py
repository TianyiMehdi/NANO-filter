import os
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from scipy.stats import halfcauchy, gamma, invgamma
from scipy.linalg import expm
import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy import sin, cos, arctan
# from base import Model
np.random.seed(42)


@dataclass_json
@dataclass
class Lorenz:
    
    dt: float = 0.02

    def __init__(self, noise_type: str = 'Gaussian'):
        self.dim_x = 3
        self.dim_y = 5
        self.P0 = 0.1 * np.eye(self.dim_x)
        self.x0 =  np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.P0)
        
        self.var = np.array([1e-2, 1e-2, 1e-2])
        self.obs_var = np.array([1e-2, 1e-4, 1e-4, 1e-4, 1e-4])
        self.Q = np.diag(self.var)
        self.R = np.diag(self.obs_var)

    def f(self, x):
        x_1 = x[0]
        A_x = np.array([
            [-10, 10, 0],
            [28, -1, -x_1],
            [0, x_1, -8/3]
        ])
        return expm(A_x * self.dt) @ x
    
    def h(self, x):
        rho = np.sqrt(np.sum(np.square(x)))
        r = np.sqrt(np.sum(np.square(x[:2])))
        cos_theta = x[0] / r
        sin_theta = x[1] / r
        cos_phi = r / rho
        sin_phi = x[2] / rho
        return np.array([rho, cos_theta, sin_theta, cos_phi, sin_phi])
    
    def f_withnoise(self, x):
        return self.f(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.Q)
    
    def h_withnoise(self, x):
        return self.h(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_y), cov=self.R)
    
    def jac_f(self, x_hat):
        return jacobian(lambda x: self.f(x))(x_hat)
    
    def jac_h(self, x_hat):
        return jacobian(lambda x: self.h(x))(x_hat)




# def show_sys_img(sys: Lorenz, x0: jax.Array):
#     x = x0
#     states = [x]
#     for _ in range(2000):
#         x = sys.transition(x)
#         states.append(x)
#     states = jax.device_get(jnp.stack(states))
#     fig = plt.figure(figsize=(4, 4), tight_layout=True)
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot3D(states[:, 0], states[:, 1], states[:, 2])
#     plt.savefig('lorenz.png')


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    lorenz = Lorenz()
    # show_sys_img(lorenz, np.random.normal(size=(3,)))