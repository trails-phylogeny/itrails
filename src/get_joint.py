import numpy as np
from trans_mat import get_trans_mat, wrapper_state_general
from numba import typeof
import numba as nb
from numba.typed import Dict, List
from numba.types import Tuple, int64, float64, boolean
from expm import expm
from numba import njit, jit

from deep_identify import get_all_paths_deep

# from scipy.linalg import expm
from cut_times import cutpoints_AB, cutpoints_ABC, get_times
from combine_states import combine_states_general
import time
from run_mc import run_mc_AB, run_mc_ABC

from scipy.sparse import csr_matrix
import pickle
import cProfile


def get_joint_prob_mat(
    t_A,
    t_B,
    t_AB,
    t_C,
    rho_A,
    rho_B,
    rho_AB,
    rho_C,
    rho_ABC,
    coal_A,
    coal_B,
    coal_AB,
    coal_C,
    coal_ABC,
    n_int_AB,
    n_int_ABC,
    cut_AB="standard",
    cut_ABC="standard",
    # tmp_path="./",
):

    # Get state spaces
    transitions_1, omega_dict_1, state_dict_1, omega_nonrev_counts_1 = (
        wrapper_state_general(1)
    )
    transitions_2, omega_dict_2, state_dict_2, omega_nonrev_counts_2 = (
        wrapper_state_general(2)
    )
    transitions_3, omega_dict_3, state_dict_3, omega_nonrev_counts_3 = (
        wrapper_state_general(3)
    )

    # Get transition matrices
    trans_mat_a = get_trans_mat(transitions_1, 1, coal_A, rho_A)
    trans_mat_b = get_trans_mat(transitions_1, 1, coal_B, rho_B)
    trans_mat_c = get_trans_mat(transitions_1, 1, coal_C, rho_C)
    trans_mat_ab = get_trans_mat(transitions_2, 2, coal_AB, rho_AB)
    trans_mat_abc = get_trans_mat(transitions_3, 3, coal_ABC, rho_ABC)

    # Get initial probabilities of 1seq CTMC
    pi_1seq = np.zeros(2)
    pi_1seq[state_dict_1[(1, 1)]] = 1
    pi_A = pi_B = pi_C = pi_1seq

    # Get final probabilities of 1seq CTMC

    start_placeholder = ((-1, -1, -1), (-1, -1, -1))

    final_A = Dict.empty(
        key_type=nb.types.UniTuple(nb.types.UniTuple(int64, 3), 2),
        value_type=float64[:, :],
    )
    final_B = Dict.empty(
        key_type=nb.types.UniTuple(nb.types.UniTuple(int64, 3), 2),
        value_type=float64[:, :],
    )
    final_C = Dict.empty(
        key_type=nb.types.UniTuple(nb.types.UniTuple(int64, 3), 2),
        value_type=float64[:, :],
    )
    final_A[start_placeholder] = (pi_A @ expm(trans_mat_a * t_A)).reshape(1, -1)
    final_B = {}
    final_B[start_placeholder] = (pi_B @ expm(trans_mat_b * t_B)).reshape(1, -1)
    final_C = {}
    final_C[start_placeholder] = (pi_C @ expm(trans_mat_c * t_C)).reshape(1, -1)

    # Dicts to know position of hidden states in the prob matrix.
    number_dict_A = state_dict_1
    number_dict_B = state_dict_1
    number_dict_C = state_dict_1
    number_dict_AB = state_dict_2
    number_dict_ABC = state_dict_3
    # Combine A and B CTMCs

    pi_AB = combine_states_general(
        number_dict_A,
        number_dict_B,
        number_dict_AB,
        final_A,
        final_B,
        n_int_AB,
        n_int_ABC,
    )

    # Get cutopints and times based on cutopoints.
    if isinstance(cut_AB, str):
        cut_AB = cutpoints_AB(n_int_AB, t_AB, coal_AB)
    times_AB = get_times(cut_AB, list(range(len(cut_AB))))
    inverted_omega_nonrev_counts = Dict.empty(
        key_type=nb.types.int64,  # Key type: integer
        value_type=nb.types.ListType(nb.types.int64),  # Value type: list of integers
    )

    # Populate the dictionary
    inverted_omega_nonrev_counts[0] = List([0])
    inverted_omega_nonrev_counts[1] = List([3])

    final_AB = run_mc_AB(
        trans_mat_ab,
        times_AB,
        omega_dict_2,
        pi_AB,
        n_int_AB,
    )

    pi_ABC = combine_states_general(
        number_dict_AB,
        number_dict_C,
        number_dict_ABC,
        final_AB,
        final_C,
        n_int_AB,
        n_int_ABC,
    )

    if isinstance(cut_ABC, str):
        cut_ABC = cutpoints_ABC(n_int_ABC, coal_ABC)
    times_ABC = get_times(cut_ABC, list(range(len(cut_ABC))))
    inverted_omega_nonrev_counts = Dict.empty(
        key_type=nb.types.int64, value_type=nb.types.ListType(nb.types.int64)
    )

    # Fill the dictionary with values
    inverted_omega_nonrev_counts[0] = List([0])
    inverted_omega_nonrev_counts[1] = List([3, 5, 6])
    inverted_omega_nonrev_counts[2] = List([7])

    final_ABC = run_mc_ABC(
        trans_mat_abc,
        times_ABC,
        omega_dict_3,
        pi_ABC,
        omega_nonrev_counts_3,
        inverted_omega_nonrev_counts,
        n_int_ABC,
    )

    return final_ABC


time0 = time.time()
final_ABC = get_joint_prob_mat(
    t_A=10,
    t_B=10,
    t_AB=20,
    t_C=20,
    rho_A=0.3,
    rho_B=0.4,
    rho_AB=0.6,
    rho_C=0.3,
    rho_ABC=0.4,
    coal_A=0.6,
    coal_B=0.4,
    coal_AB=0.2,
    coal_C=0.5,
    coal_ABC=0.4,
    n_int_AB=3,
    # p_init_A=np.array([1, 0], dtype=np.float64),
    # p_init_B=np.array([1, 0], dtype=np.float64),
    # p_init_C=np.array([0, 1], dtype=np.float64),
    cut_AB="standard",
    cut_ABC="standard",
    n_int_ABC=3,
)
time1 = time.time()

prob = 0
for path, value in final_ABC.items():
    print(path, np.sum(value))
    prob += np.sum(value)

print(prob)
print("Precomputing done!")
print(f"Time precomputing: {time1 - time0}")

time0 = time.time()
get_joint_prob_mat(
    t_A=10,
    t_B=10,
    t_AB=20,
    t_C=20,
    rho_A=0.3,
    rho_B=0.4,
    rho_AB=0.6,
    rho_C=0.3,
    rho_ABC=0.4,
    coal_A=0.6,
    coal_B=0.4,
    coal_AB=0.2,
    coal_C=0.5,
    coal_ABC=0.4,
    n_int_AB=3,
    # p_init_A=np.array([1, 0], dtype=np.float64),
    # p_init_B=np.array([1, 0], dtype=np.float64),
    # p_init_C=np.array([0, 1], dtype=np.float64),
    cut_AB="standard",
    cut_ABC="standard",
    n_int_ABC=3,
)
time1 = time.time()
prob = 0
for path, value in final_ABC.items():
    print(path, np.sum(value))
    prob += np.sum(value)

print(prob)
print(f"Time after precomputing: {time1 - time0}")
