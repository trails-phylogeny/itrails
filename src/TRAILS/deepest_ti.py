import numpy as np
import numba as nb


@nb.jit(nopython=True)
def deepest_ti(trans_mat_noabs, omega_dict_noabs, path):
    steps = len(path) - 1
    n = trans_mat_noabs.shape[0]
    if steps == 1:
        C_mat = trans_mat_noabs

    elif steps > 1:
        C_mat = np.zeros((n * steps, n * steps))
        C_mat[0:n, 0:n] = trans_mat_noabs
        for idx in range(1, steps):

            sub_om_init = path[idx - 1]
            sub_om_fin = path[idx]
            A_mat = (
                omega_dict_noabs[sub_om_init][:, np.newaxis]
                * trans_mat_noabs
                * omega_dict_noabs[sub_om_fin][np.newaxis, :]
            )
            C_mat[n * idx : n * (idx + 1), n * idx : n * (idx + 1)] = trans_mat_noabs
            C_mat[n * (idx - 1) : n * idx, n * idx : n * (idx + 1)] = A_mat
    sub_om_init = path[-2]
    sub_om_fin = path[-1]
    A_mat = np.ascontiguousarray(
        omega_dict_noabs[sub_om_init][:, np.newaxis]
        * trans_mat_noabs
        * omega_dict_noabs[sub_om_fin][np.newaxis, :]
    )
    result = (-np.linalg.inv(C_mat))[:n, -n:] @ A_mat
    return result
