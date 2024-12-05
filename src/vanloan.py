import numpy as np
import numba as nb
from expm import expm


@nb.jit(nopython=True)
def vanloan_general(trans_mat, path, tim, omega_dict):
    n = trans_mat.shape[0]
    steps = len(path)
    C_mat = np.zeros((n * steps, n * steps))
    C_mat[0:n, 0:n] = trans_mat

    for idx in range(1, steps):
        C_mat[n * idx : n * (idx + 1), n * idx : n * (idx + 1)] = trans_mat
        # Extract the previous step as a tuple

        sub_om_init = (path[idx - 1, 0], path[idx - 1, 1])  # Row idx-1, both columns

        # Extract the current step as a tuple
        sub_om_fin = (path[idx, 0], path[idx, 1])  # Row idx, both columns

        A_mat = (
            np.diag(omega_dict[sub_om_init].astype(np.float64))
            @ trans_mat
            @ np.diag(omega_dict[sub_om_fin].astype(np.float64))
        )
        C_mat[n * (idx - 1) : n * idx, n * idx : n * (idx + 1)] = A_mat
    result = np.zeros((n, n), dtype=np.float64)
    result = expm(C_mat * (tim))[0:n, -n:]
    return result
