import numpy as np
from expm import expm


def vanloan_general(trans_mat, path, tim, omega_dict):
    n = trans_mat.shape[0]
    steps = len(path) - 1
    C_mat = np.zeros((n * steps, n * steps))
    C_mat[0:n, 0:n] = trans_mat

    for idx in range(1, steps):
        C_mat[n * idx : n * (idx + 1), n * idx : n * (idx + 1)] = trans_mat
        sub_om_init = path[idx - 1]
        sub_om_fin = path[idx]
        A_mat = (
            np.diag(omega_dict[sub_om_init])
            @ trans_mat
            @ np.diag(omega_dict[sub_om_fin])
        )
        C_mat[n * (idx - 1) : n * idx, n * idx : n * (idx + 1)] = A_mat

    result = expm(C_mat * (tim))[0:n, -n:]
    return result
