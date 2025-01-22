import numpy as np
from expm import expm

def vanloan_general(trans_mat, path, tim, omega_dict):
    """
    This function performs the van Loan (1978) method for 
    finding the integral of a series of multiplying matrix
    exponentials. Function generalizes to any number of matrices.

    :param trans_mat: Transition rate matrix for the CTMC.
    :type trans_mat: Numpy array of type: float64[:, :].
    :param path: Subpath of omega values followed by the CTMC.
    :type path: Numpy array of type: int64[:, :].
    :param tim: Time at which to evaluate the integral.
    :type tim: float64.
    :param omega_dict: Numba dictionary of combinations of omega values and the possible final states of each as array of booleans.
    :type omega_dict: Numba dictionary of key: Tuple((int64, int64)) and value: np.array of boolean.
    :return: Result of the integral of the series of multiplying matrix exponentials.
    :rtype: Numpy array of type: float64[:, :].
    """
    n = trans_mat.shape[0]
    steps = len(path)
    C_mat = np.zeros((n * steps, n * steps))
    C_mat[0:n, 0:n] = trans_mat

    for idx in range(1, steps):
        C_mat[n * idx : n * (idx + 1), n * idx : n * (idx + 1)] = trans_mat
        sub_om_init = (path[idx - 1, 0], path[idx - 1, 1])
        sub_om_fin = (path[idx, 0], path[idx, 1])

        A_mat = (
            np.diag(omega_dict[sub_om_init].astype(np.float64))
            @ trans_mat
            @ np.diag(omega_dict[sub_om_fin].astype(np.float64))
        )
        C_mat[n * (idx - 1) : n * idx, n * idx : n * (idx + 1)] = A_mat
    result = np.zeros((n, n), dtype=np.float64)
    result = expm(C_mat * (tim))[0:n, -n:]
    return result
