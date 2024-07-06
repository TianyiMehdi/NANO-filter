from dataclasses import dataclass
import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy import sin, cos, arctan, pi, arctan2, sqrt

class BlackHole:
    
    dt : float = 0.02

    def __init__(self, state_outlier_flag=False, 
                measurement_outlier_flag=False, noise_type='Gaussian'):
        # System size
        self.dim_x = 2
        self.dim_y = 1
        self.P0 = np.eye(self.dim_x)
        self.x0 = np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.P0)
        

        self.state_outlier_flag = state_outlier_flag
        self.measurement_outlier_flag = measurement_outlier_flag
        self.noise_type = noise_type
        self.Q = np.diag(np.array([1e-2, 1e-2]))
        self.R = np.diag(np.array([1e-2]))
        
        self.I_last = None

        # Other parameters
        self.M = 1.0
        self.e = 0.5
        self.p = 100
    
    def f(self, x):
        """Transition function for the model."""
        x_ = rk4(self.dxdt, x, 0, self.dt)
        return x_

    def h(self, x):
        """Observation function for the model."""
        # ddI = second_order_fd(self.get_quadrupole(states, 0), self.dt)
        I = self.get_quadrupole(x)
        # if I_last == None:
        state_next1 = self.f(x)
        I_next1 = self.get_quadrupole(state_next1)
        state_next2 = self.f(state_next1)
        I_next2 = self.get_quadrupole(state_next2)
        state_next3 = self.f(state_next2)
        I_next3 = self.get_quadrupole(state_next3)
        ddI = (2 * I - 5 * I_next1 + 4 * I_next2 - I_next3) / self.dt ** 2
        # else:
        #     I_next = self.get_quadrupole(self.transition(states))
        #     ddI = (I_next - 2 * I + I_last) / self.dt ** 2
        # self.I_last = I
        # print(ddI.shape)
        ddIxx, ddIyy, ddIxy = ddI
        return np.array([(ddIxx - ddIyy) * np.sqrt(4 * np.pi / 5)])
    
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
                if prob <= 0.9:
                    cov = self.R  # 95%概率使用R
                else:
                    cov = 100 * self.R  # 5%概率使用100R
            else:
                cov = self.R
            return self.h(x) + np.random.multivariate_normal(mean=np.zeros(self.dim_y), cov=cov)
        else:
            return self.h(x) + np.random.laplace(loc=0, scale=1, size=(self.dim_y, ))

    def jac_f(self, x_hat):
        return jacobian(lambda x: self.f(x))(x_hat)
    
    def jac_h(self, x_hat):
        return jacobian(lambda x: self.h(x))(x_hat)
    
    def dxdt(self, x, t):
        M, e, p = self.M, self.e, self.p
        phi, chi = x
        dphi = (p - 2 - 2 * e * cos(chi)) * (1 + e * cos(chi)) ** 2
        dphi /= M * p ** (3 / 2) * sqrt((p - 2) ** 2 - 4 * e**2)
        dchi = (
            (p - 2 - 2 * e * cos(chi))
            * (1 + e * cos(chi)) ** 2
            * sqrt(p - 6 - 2 * e * cos(chi))
        )
        dchi /= M * p**2 * sqrt((p - 2) ** 2 - 4 * e**2)
        return np.array([dphi, dchi])

    def euclidean_norm(self, chi):
        M, e, p = self.M, self.e, self.p
        r = p * M / (1 + e * cos(chi))
        return r
    
    def convert_to_trajectories(self, x):
        phi, chi = x
        r = self.euclidean_norm(chi)
        r_2 = np.array([r * cos(phi), r * sin(phi)])
        return r_2

    def get_quadrupole(self, x):
        r = self.convert_to_trajectories(x)
        x, y = r
        Ixx = x**2 * self.M
        Iyy = y**2 * self.M
        Ixy = x * y * self.M
        return np.array([Ixx, Iyy, Ixy])

def rk4(f, x, t, dt, stages=4, s=0):
    """Runge-Kutta (explicit, non-adaptive) numerical (S)ODE solvers.

    For ODEs, the order of convergence equals the number of `stages`.

    For SDEs with additive noise (`s>0`), the order of convergence
    (both weak and strong) is 1 for `stages` equal to 1 or 4.
    These correspond to the classic Euler-Maruyama scheme and the Runge-Kutta
    scheme for S-ODEs respectively, see `bib.grudzien2020numerical`
    for a DA-specific discussion on integration schemes and their discretization errors.

    Parameters
    ----------
    f : function
        The time derivative of the dynamical system. Must be of the form `f(t, x)`

    x : ndarray or float
        State vector of the forcing term

    t : float
        Starting time of the integration

    dt : float
        Integration time step.

    stages : int, optional
        The number of stages of the RK method.
        When `stages=1`, this becomes the Euler (-Maruyama) scheme.
        Default: 4.

    s : float
        The diffusion coeffient (std. dev) for models with additive noise.
        Default: 0, yielding deterministic integration.

    Returns
    -------
    ndarray
        State vector at the new time, `t+dt`
    """

    # Draw noise
    if s > 0:
        W = s * np.sqrt(dt) * np.random.randn(*x.shape)
    else:
        W = 0

    # Approximations to Delta x
    if stages >= 1: k1 = dt * f(x,           t)         + W    # noqa
    if stages >= 2: k2 = dt * f(x+k1/2.0,    t+dt/2.0)  + W    # noqa
    if stages == 3: k3 = dt * f(x+k2*2.0-k1, t+dt)      + W    # noqa
    if stages == 4:
                    k3 = dt * f(x+k2/2.0,    t+dt/2.0)  + W    # noqa
                    k4 = dt * f(x+k3,        t+dt)      + W    # noqa

    # Mix proxies
    if    stages == 1: y = x + k1                              # noqa
    elif  stages == 2: y = x + k2                              # noqa
    elif  stages == 3: y = x + (k1 + 4.0*k2 + k3)/6.0          # noqa
    elif  stages == 4: y = x + (k1 + 2.0*(k2 + k3) + k4)/6.0   # noqa
    else:
        raise NotImplementedError

    return y

# if __name__ == "__main__":
#     blackhole = BlackHole()
#     x0 = np.random.normal(size=(2, ))
#     print(blackhole.transition(x0), blackhole.observation(x0))
