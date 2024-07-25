import numpy as np
from expm import expm
from trans_mat import wrapper_state_general, get_trans_mat


subpath_dm = [((7, 0), (7, 3)), ((7, 3), (7, 7))]
time = 1

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


def vanloan_general(trans_mat, subpath, time, omega_dict):
    """
    This function performs the van Loan (1978) method for
    finding the integral of a series of 2 multiplying matrix
    exponentials.

    Parameters
    ----------
    trans_mat : numeric numpy matrix
        Transition rate matrix.
    tup : tupple
        Tupple of size 2, where the first and second entries
        are lists with the indices of the transition rate
        matrix to go from and to in the first instantaneous
        transition.
    omega_start : list of integers
        List of starting states of the transition rate matrix.
    omega_end : list of integers
        List of ending states of the transition rate matrix.
    time : float
        Upper boundary of the definite integral.
    """
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
        @ (expm(C_mat * (time))[0:n, -n:])
        @ np.diag(omega_dict[omega_fin])
    )
    return result


result = vanloan_general(trans_mat_abc, subpath_dm, time, omega_dict_3)
print(result)
