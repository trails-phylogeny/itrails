import numpy as np


def deepest_ti(trans_mat_noabs, omega_dict_noabs, path):
    steps = len(path) - 1
    n = trans_mat_noabs.shape[0]
    if steps == 1:
        C_mat = trans_mat_noabs

    elif steps > 1:
        C_mat = np.zeros((n * steps, n * steps))
        C_mat[0:n, 0:n] = trans_mat_noabs
        for idx in range(1, steps):

            sub_om_init = (path[idx - 1, 0], path[idx - 1, 1])
            sub_om_fin = (path[idx, 0], path[idx, 1])
            A_mat = (
                np.diag(omega_dict_noabs[sub_om_init].astype(np.float64))
                @ trans_mat_noabs
                @ np.diag(omega_dict_noabs[sub_om_fin].astype(np.float64))
            )
            C_mat[n * idx : n * (idx + 1), n * idx : n * (idx + 1)] = trans_mat_noabs
            C_mat[n * (idx - 1) : n * idx, n * idx : n * (idx + 1)] = A_mat
    if path.shape[0] < 2:
        print("here", path)
    sub_om_init = (path[-2, 0], path[-2, 1])
    sub_om_fin = (path[-1, 0], path[-1, 1])
    A_mat = np.ascontiguousarray(
        np.diag(omega_dict_noabs[sub_om_init].astype(np.float64))
        @ trans_mat_noabs
        @ np.diag(omega_dict_noabs[sub_om_fin].astype(np.float64))
    )
    result = (-np.linalg.inv(C_mat))[:n, -n:] @ A_mat
    return result
