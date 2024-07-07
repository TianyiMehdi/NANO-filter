from dataclasses import dataclass
import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy import sin, cos, arctan


@dataclass
class Vehicle:
    """Vehicle config for the slide-slip vehicle model.
    """
    # Vehicle parameters
    m: float = 1500 # mass [kg]
    dt: float = 0.01
    B: float = 14.0
    C: float = 1.43
    D: float = 0.75
    u: float = 20.0
    I_zz: float = 2420.0
    L: float = 2.54
    a: float = 1.14
    b: float = 1.40
    g: float = 9.81

    def __init__(self, state_outlier_flag=False, 
                measurement_outlier_flag=False, noise_type='Gaussian'):
        D = self.D
        g = self.g
        dt = self.dt
        u = self.u
        L = self.L
        m = self.m
        a = self.a
        b = self.b
        I_zz = self.I_zz
        self.A1 = -D*g*dt / (u*L)
        self.A2 = D*m*g*a*b*dt / (I_zz * L)

        self.dim_x = 2
        self.dim_y = 2
        self.P0 = 0.1 * np.eye(self.dim_x)
        self.x0 =  np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.P0)


        self.state_outlier_flag = state_outlier_flag
        self.measurement_outlier_flag = measurement_outlier_flag
        self.noise_type = noise_type
        self.var = np.array([1e-2, 1e-2])
        self.obs_var = np.array([1e-2, 1e-2])
        self.Q = np.diag(self.var)
        self.R = np.diag(self.obs_var)
    
    def f(self, x):
        """Transition function for the slide-slip vehicle model.
        states: (2, )
        returns: (2, )
        """
        A1 = self.A1
        b = self.b
        C = self.C
        B = self.B
        a = self.a
        u = self.u
        b = self.b
        A2 = self.A2
        dt = self.dt

        theta = x[0]
        ut = - theta  # control policy
        # ut = jnp.zeros_like(theta)
        omega = x[1]
        v1 = sin(C*arctan(B*(theta - ut + a*omega/u)))
        v2 = sin(C*arctan(B*(theta - b*omega/u)))
        f1 = A1 * b * cos(ut) * v1 + A1 * a * v2 - omega * dt
        f2 = -A2 * v1 + A2 * v2
        return np.stack([f1, f2])
    
    def h(self, x):
        theta, omega = x
        alpha_1 = theta + theta + omega * self.a / self.u
        alpha_2 = theta - omega * self.b / self.u
        FY1 = - sin(self.C * arctan(self.B * alpha_1))
        FY2 = - sin(self.C * arctan(self.B * alpha_2))
        return np.array([FY1, FY2])
    
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

if __name__ == '__main__':
    x0 = np.random.uniform(-0.05, 0.05, size=(2, ))
    vehicle = Vehicle()
    print(vehicle.f(x0), vehicle.h(x0))
    print(vehicle.jac_f(x0))
    print(vehicle.jac_h(x0))

# # generate dataset
# def transition_numpy(states: np.ndarray, vehicle: Vehicle):
#     """Transition function for the slide-slip vehicle model.
#     states: (num_traj, 2)
#     returns: (num_traj, 2)
#     """
#     A1 = vehicle.A1
#     b = vehicle.b
#     C = vehicle.C
#     B = vehicle.B
#     a = vehicle.a
#     u = vehicle.u
#     b = vehicle.b
#     A2 = vehicle.A2
#     dt = vehicle.dt

#     theta = states[:, 0]
#     ut = - theta  # control policy
#     omega = states[:, 1]
#     v1 = np.sin(C*np.arctan(B*(theta - ut + a*omega/u)))
#     v2 = np.sin(C*np.arctan(B*(theta - b*omega/u)))
#     f1 = A1 * b * np.cos(ut) * v1 + A1 * a * v2 - omega * dt
#     f2 = -A2 * v1 + A2 * v2
#     return np.stack([f1, f2], axis=1)

# def process_noises(size: int, var: np.ndarray) -> np.ndarray:
#     return np.random.multivariate_normal(mean=np.zeros(2), cov=np.diag(var), size=size).astype(np.float32)

# def observation_noises(size: int, obsvar: np.ndarray) -> np.ndarray:
#     return np.random.multivariate_normal(mean=np.zeros(2), cov=np.diag(obsvar), size=size).astype(np.float32)

# def generate_data(num_traj: int, traj_len: int, save_dir: str, mode:str, vehicle: Vehicle, seed: int = 42):
#     np.random.seed(seed)
#     data_dir = Path(save_dir)
#     data_dir.mkdir(parents=True, exist_ok=True)
#     save_path = data_dir / f"{mode}.npz"

#     states = np.zeros((num_traj, traj_len, 2), dtype=np.float32)
#     observations = np.zeros((num_traj, traj_len, 2), dtype=np.float32)
#     states[:, 0, :] = np.random.uniform(-0.05, 0.05, size=(num_traj, 2))
#     for i in tqdm(range(traj_len - 1)):
#         states[:, i+1, :] = vehicle.f(states[:, i, :]) + process_noises(num_traj, vehicle.var)
#     observations = states + observation_noises((num_traj, traj_len), vehicle.obs_logvar)
#     np.savez_compressed(save_path, states=states, observations=observations)


# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--num_traj", type=int, default=30)
#     parser.add_argument("--traj_len", type=int, default=50)
#     parser.add_argument("--save_dir", type=str, default="dataset")
#     parser.add_argument("--mode", type=str, default="vehicle")
#     parser.add_argument("--seed", type=int, default=42)
#     args = parser.parse_args()
#     vehicle = Vehicle()
#     generate_data(**vars(args), vehicle=vehicle)
