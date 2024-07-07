import os
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from itertools import product
from scipy.linalg import expm
from scipy.linalg import LinAlgError, bandwidth
import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy import sin, cos, arctan, sqrt
# from base import Model
np.random.seed(42)


@dataclass_json
@dataclass
class Lorenz:
    
    dt: float = 0.02

    def __init__(self, state_outlier_flag=False, 
                measurement_outlier_flag=False, noise_type='Gaussian'):
        self.dim_x = 3
        self.dim_y = 5
        self.P0 = 0.01 * np.eye(self.dim_x)
        self.x0 =  np.random.multivariate_normal(mean=np.zeros(self.dim_x), cov=self.P0)

        self.state_outlier_flag = state_outlier_flag
        self.measurement_outlier_flag = measurement_outlier_flag
        self.noise_type = noise_type
        
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


def expm(A):
    """Compute the matrix exponential of an array.

    Parameters
    ----------
    A : ndarray
        Input with last two dimensions are square ``(..., n, n)``.

    Returns
    -------
    eA : ndarray
        The resulting matrix exponential with the same shape of ``A``

    Notes
    -----
    Implements the algorithm given in [1], which is essentially a Pade
    approximation with a variable order that is decided based on the array
    data.

    For input with size ``n``, the memory usage is in the worst case in the
    order of ``8*(n**2)``. If the input data is not of single and double
    precision of real and complex dtypes, it is copied to a new array.

    For cases ``n >= 400``, the exact 1-norm computation cost, breaks even with
    1-norm estimation and from that point on the estimation scheme given in
    [2] is used to decide on the approximation order.

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham, (2009), "A New Scaling
           and Squaring Algorithm for the Matrix Exponential", SIAM J. Matrix
           Anal. Appl. 31(3):970-989, :doi:`10.1137/09074721X`

    .. [2] Nicholas J. Higham and Francoise Tisseur (2000), "A Block Algorithm
           for Matrix 1-Norm Estimation, with an Application to 1-Norm
           Pseudospectra." SIAM J. Matrix Anal. Appl. 21(4):1185-1201,
           :doi:`10.1137/S0895479899356080`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import expm, sinm, cosm

    Matrix version of the formula exp(0) = 1:

    >>> expm(np.zeros((3, 2, 2)))
    array([[[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[1., 0.],
            [0., 1.]]])

    Euler's identity (exp(i*theta) = cos(theta) + i*sin(theta))
    applied to a matrix:

    >>> a = np.array([[1.0, 2.0], [-1.0, 3.0]])
    >>> expm(1j*a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])
    >>> cosm(a) + 1j*sinm(a)
    array([[ 0.42645930+1.89217551j, -2.13721484-0.97811252j],
           [ 1.06860742+0.48905626j, -1.71075555+0.91406299j]])

    """
    a = np.asarray(A)
    n = a.shape[-1]
    # Empty array
    if min(*a.shape) == 0:
        return np.empty_like(a)

    # Scalar case
    if a.shape[-2:] == (1, 1):
        return np.exp(a)

    if not np.issubdtype(a.dtype, np.inexact):
        a = a.astype(float)
    elif a.dtype == np.float16:
        a = a.astype(np.float32)

    # Explicit formula for 2x2 case, formula (2.2) in [1]
    # without Kahan's method numerical instabilities can occur.
    if a.shape[-2:] == (2, 2):
        a1, a2, a3, a4 = (a[..., [0], [0]],
                          a[..., [0], [1]],
                          a[..., [1], [0]],
                          a[..., [1], [1]])
        mu = sqrt((a1-a4)**2 + 4*a2*a3)/2.  # csqrt slow but handles neg.vals

        eApD2 = np.exp((a1+a4)/2.)
        AmD2 = (a1 - a4)/2.
        coshMu = np.cosh(mu)
        sinchMu = np.ones_like(coshMu)
        mask = mu != 0
        sinchMu[mask] = np.sinh(mu[mask]) / mu[mask]
        eA = np.empty((a.shape), dtype=mu.dtype)
        eA[..., [0], [0]] = eApD2 * (coshMu + AmD2*sinchMu)
        eA[..., [0], [1]] = eApD2 * a2 * sinchMu
        eA[..., [1], [0]] = eApD2 * a3 * sinchMu
        eA[..., [1], [1]] = eApD2 * (coshMu - AmD2*sinchMu)
        if np.isrealobj(a):
            return eA.real
        return eA

    # larger problem with unspecified stacked dimensions.
    n = a.shape[-1]
    eA = np.empty(a.shape, dtype=a.dtype)
    # working memory to hold intermediate arrays
    Am = np.empty((5, n, n), dtype=a.dtype)

    # Main loop to go through the slices of an ndarray and passing to expm
    for ind in product(*[range(x) for x in a.shape[:-2]]):
        aw = a[ind]

        lu = bandwidth(aw)
        if not any(lu):  # a is diagonal?
            eA[ind] = np.diag(np.exp(np.diag(aw)))
            continue

        # Generic/triangular case; copy the slice into scratch and send.
        # Am will be mutated by pick_pade_structure
        Am[0, :, :] = aw
        m, s = pick_pade_structure(Am)

        if s != 0:  # scaling needed
            Am[:4] *= [[[2**(-s)]], [[4**(-s)]], [[16**(-s)]], [[64**(-s)]]]

        pade_UV_calc(Am, n, m)
        eAw = Am[0]

        if s != 0:  # squaring needed

            if (lu[1] == 0) or (lu[0] == 0):  # lower/upper triangular
                # This branch implements Code Fragment 2.1 of [1]

                diag_aw = np.diag(aw)
                # einsum returns a writable view
                np.einsum('ii->i', eAw)[:] = np.exp(diag_aw * 2**(-s))
                # super/sub diagonal
                sd = np.diag(aw, k=-1 if lu[1] == 0 else 1)

                for i in range(s-1, -1, -1):
                    eAw = eAw @ eAw

                    # diagonal
                    np.einsum('ii->i', eAw)[:] = np.exp(diag_aw * 2.**(-i))
                    exp_sd = _exp_sinch(diag_aw * (2.**(-i))) * (sd * 2**(-i))
                    if lu[1] == 0:  # lower
                        np.einsum('ii->i', eAw[1:, :-1])[:] = exp_sd
                    else:  # upper
                        np.einsum('ii->i', eAw[:-1, 1:])[:] = exp_sd

            else:  # generic
                for _ in range(s):
                    eAw = eAw @ eAw

        # Zero out the entries from np.empty in case of triangular input
        if (lu[0] == 0) or (lu[1] == 0):
            eA[ind] = np.triu(eAw) if lu[0] == 0 else np.tril(eAw)
        else:
            eA[ind] = eAw

    return eA


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