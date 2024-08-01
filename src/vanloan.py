import numpy as np
from expm import expm
from scipy.sparse import csr_matrix
import time
from trans_mat import wrapper_state_general, get_trans_mat

subpath_1 = [((3, 0), (3, 3)), ((3, 3), (7, 3)), ((7, 3), (7, 7))]
subpath_2 = [((3, 0), (3, 3)), ((3, 3), (3, 7)), ((3, 7), (7, 7))]
subpath_3 = [((3, 0), (7, 0)), ((7, 0), (7, 3)), ((7, 3), (7, 7))]
subpath_4 = [((3, 0), (7, 3)), ((7, 3), (7, 7))]
subpath_5 = [((3, 0), (3, 3)), ((3, 3), (7, 7))]


sub_1 = [((7, 0), (7, 3)), ((7, 3), (7, 7))]

subpath_list = [subpath_1, subpath_2, subpath_3, subpath_4, subpath_5]

tim = 1

transitions_3, omega_dict_3, state_dict_3, omega_nonrev_counts_3 = (
    wrapper_state_general(3)
)
transitions_2, omega_dict_2, state_dict_2, omega_nonrev_counts_2 = (
    wrapper_state_general(2)
)
transitions_1, omega_dict_1, state_dict_1, omega_nonrev_counts_1 = (
    wrapper_state_general(1)
)
coal_A = 0.2
rho_A = 0.3
coal_B = 0.25
rho_B = 0.15
coal_C = 0.35
rho_C = 0.2
coal_AB = 0.40
rho_AB = 0.3
coal_ABC = 0.35
rho_ABC = 0.25

trans_mat_a = get_trans_mat(transitions_1, 1, coal_A, rho_A)
trans_mat_b = get_trans_mat(transitions_1, 1, coal_B, rho_B)
trans_mat_c = get_trans_mat(transitions_1, 1, coal_C, rho_C)
trans_mat_ab = get_trans_mat(transitions_2, 2, coal_AB, rho_AB)
trans_mat_abc = get_trans_mat(transitions_3, 3, coal_ABC, rho_ABC)


def vanloan_general(trans_mat, subpath, tim, omega_dict):
    n = trans_mat.shape[0]
    steps = len(subpath)
    C_mat = np.zeros((n * steps, n * steps))
    C_mat[0:n, 0:n] = trans_mat
    omega_init = subpath[0][0]
    omega_fin = subpath[-1][-1]
    for idx, step in enumerate(subpath[:-1], start=1):
        # note to myself: simplify A_mat
        C_mat[n * idx : n * (idx + 1), n * idx : n * (idx + 1)] = trans_mat
        sub_om_init = step[0]
        sub_om_fin = step[1]
        A_mat = (
            np.diag(omega_dict[sub_om_init])
            @ trans_mat
            @ np.diag(omega_dict[sub_om_fin])
        )
        C_mat[n * (idx - 1) : n * idx, n * idx : n * (idx + 1)] = A_mat

    result = (
        np.diag(omega_dict[omega_init])
        @ (expm(C_mat * (tim))[0:n, -n:])
        @ np.diag(omega_dict[omega_fin])
    )
    return result


vanloan_general(trans_mat_abc, sub_1, tim, omega_dict_3)
results = 0
time0 = time.time()
for subpath in subpath_list:
    result = vanloan_general(trans_mat_abc, subpath, tim, omega_dict_3)
    print(subpath)
    print(result.sum())
    results += result.sum()
    sparse = csr_matrix(result)
    print(sparse)
time1 = time.time()
print(f"Results: {results}")
print(f"Done! Time: {time1- time0}")
