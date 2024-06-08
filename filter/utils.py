import autograd.numpy as np
import warnings

def is_positive_semidefinite(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Matrix is not square')
    
    eigenvalues = np.linalg.eigvals(matrix)
    
    if np.any(eigenvalues < 0):
        print('Eigenvalues: ', eigenvalues)
        warnings.warn('Matrix is not positive semidefinite')
        
            
def cal_mean(func, mean, var, points):
    sigmas = points.sigma_points(mean, var)
    first_eval = func(sigmas[0])
    
    if isinstance(first_eval, np.ndarray):
        sigmas_func = np.zeros((sigmas.shape[0], *first_eval.shape))
    else:
        sigmas_func = np.zeros(sigmas.shape[0])
    
    for i, s in enumerate(sigmas):
        sigmas_func[i] = func(s)

        # if isinstance(sigmas_func[i], np.ndarray) and sigmas_func[i].ndim == 2:
        #     is_positive_semidefinite(sigmas_func[i])
    
    mean_func = np.tensordot(points.Wm, sigmas_func, axes=([0], [0]))
    return mean_func

def cal_mean_mc(func, mean, var, num_samples=100):
    samples = np.random.multivariate_normal(mean, var, num_samples)
    # 计算每个样本在函数 f 上的值
    sample_values = np.apply_along_axis(func, 1, samples)
    # print('1:', sample_values.shape)
    # 计算样本值的平均值
    expectation = np.mean(sample_values, axis=0)
    # print('2:', expectation.shape)
    return expectation