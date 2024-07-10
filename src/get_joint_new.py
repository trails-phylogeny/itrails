import numpy as np
from trans_mat import (
    get_trans_mat,
    wrapper_state_1,
    wrapper_state_2,
    wrapper_state_3,
    wrapper_state_general,
)
from expm import expm
from cut_times import cutpoints_AB, cutpoints_ABC, get_times
from combine_states import combine_states_general
from numba.typed import Dict
from numba.types import Tuple, int64
from numba import jit
import time

# int_tuple = Tuple((int64, int64))
# @jit(nopython=True)
# def get_omegas_numba(omega_nums):
#     omega_dict = Dict.empty(
#         key_type=int_tuple,
#         value_type=int_tuple,
#     )
#
#     for i in range(len(omega_nums)):
#         for j in range(len(omega_nums)):
#             omega_dict[(omega_nums[i], omega_nums[j])] = (omega_nums[i], omega_nums[j])
#
#     return omega_dict


@jit(nopython=True)
def get_final_per_interval_nodict(trans_mat_ab, times, omega_dict, pi_AB):
    # Create dictionary for base case, also every omega will be called -1, -1.
    time_0dict = {}
    all_omegas = (-1, -1)
    mat_0 = np.zeros_like(trans_mat_ab)
    # Base case for time 0, from all omegas to every other possibility.
    exponential_time_0 = expm(trans_mat_ab * times[0])
    time0 = time.time()
    for key, value in omega_dict.items():
        exponential_time_add = mat_0.copy()  # 15x15 0s
        exponential_time_add[:, min(value) : max(value) + 1] = exponential_time_0[
            :, min(value) : max(value) + 1
        ]
        time_0dict[(all_omegas, key)] = exponential_time_add
    # Dictionary to accumulate the results, keys are omega paths as tuples, values are precomputed matrices.
    acc_results = time_0dict

    # Each of the time cuts
    for i in range(1, len(times)):
        exponential_time = expm(trans_mat_ab * times[i])
        each_time_dict = {}
        actual_results = {}
        # Populate a temp dictionary with the every possible slice.
        for key, value in omega_dict.items():
            for key2, value2 in omega_dict.items():
                if key[0] <= key2[0] and key[1] <= key2[1]:
                    exponential_time_add = mat_0.copy()
                    exponential_time_add[
                        min(value) : max(value) + 1, min(value2) : max(value2) + 1
                    ] = exponential_time[
                        min(value) : max(value) + 1, min(value2) : max(value2) + 1
                    ]

                    each_time_dict[(key, key2)] = exponential_time_add
        # Multiply each possible slice with the results that we already had and update the accumulated results dictionary.
        for transition_0, matrix_0 in acc_results.items():
            end_state = transition_0[-1]
            for transition_1, matrix_1 in each_time_dict.items():
                start_state = transition_1[0]
                if start_state == end_state:
                    result = matrix_0 @ matrix_1
                    actual_results[(*transition_0, transition_1[1])] = result
                else:
                    continue
        acc_results = actual_results

    # Create the final_p vector multiplying each slice by pi and populate it with the calculation for each state we end up in.
    # Final prob vector debug is just a sum of every to see if they sum up to 1
    final_prob_vector_debug = np.zeros((trans_mat_ab.shape[0]), dtype=np.float64)
    final_prob_vector = {}
    for trans, prob_slice in acc_results.items():
        pi_slice = pi_AB @ prob_slice
        final_prob_vector[trans] = pi_slice
        for i in range(len(final_prob_vector_debug)):
            final_prob_vector_debug[i] += pi_slice[i]

    time1 = time.time()
    print(time1 - time0)
    return final_prob_vector_debug, acc_results, final_prob_vector


def get_final_per_interval(trans_mat_ab, times, omega_dict, pi_AB):

    accumulated_results = {}
    all_omegas = (-1, -1)
    exponential_time_0 = expm(trans_mat_ab * times[0])
    for key, value in omega_dict.items():
        sliced_mat = exponential_time_0 @ np.diag(value)
        accumulated_results[(all_omegas, key)] = sliced_mat

    for i in range(1, len(times)):
        exponential_time = expm(trans_mat_ab * times[i])
        each_time_dict = {}
        actual_results = {}

        for key, value in omega_dict.items():
            for key2, value2 in omega_dict.items():
                # implement omega dict
                if key[0] <= key2[0] and key[1] <= key2[1]:
                    sliced_mat = np.diag(value) @ exponential_time @ np.diag(value2)
                    each_time_dict[(key, key2)] = sliced_mat

        for transition_0, matrix_0 in accumulated_results.items():
            end_state = transition_0[-1]
            for transition_1, matrix_1 in each_time_dict.items():
                start_state = transition_1[0]
                if start_state == end_state:
                    result = matrix_0 @ matrix_1
                    actual_results[(*transition_0, transition_1[1])] = result
                else:
                    continue
        accumulated_results = actual_results

    final_prob_vector = {}
    for path, matrix in accumulated_results.items():
        pi_slice = pi_AB @ matrix
        final_prob_vector[path] = pi_slice

    return accumulated_results, final_prob_vector


def get_final_per_interval_noif(
    trans_mat_ab, times, omega_dict, pi_AB, omega_nonrev_counts
):

    accumulated_results = {}
    all_omegas = (-1, -1)
    exponential_time_0 = expm(trans_mat_ab * times[0])
    for key, value in omega_dict.items():
        sliced_mat = exponential_time_0 @ np.diag(value)
        accumulated_results[(all_omegas, key)] = sliced_mat

    for i in range(1, len(times)):
        exponential_time = expm(trans_mat_ab * times[i])
        each_time_dict = {}
        actual_results = {}

        for omega_init, value in omega_dict.items():
            for omega_fin, value2 in omega_dict.items():
                if (
                    omega_init[0] == omega_fin[0]
                    or omega_nonrev_counts[omega_init[0]]
                    < omega_nonrev_counts[omega_fin[0]]
                ) and (
                    omega_init[1] == omega_fin[1]
                    or omega_nonrev_counts[omega_init[1]]
                    < omega_nonrev_counts[omega_fin[1]]
                ):
                    sliced_mat = np.diag(value) @ exponential_time @ np.diag(value2)
                    each_time_dict[(omega_init, omega_fin)] = sliced_mat

        for transition_0, matrix_0 in accumulated_results.items():
            end_state = transition_0[-1]
            for transition_1, matrix_1 in each_time_dict.items():
                start_state = transition_1[0]
                if start_state == end_state:
                    result = matrix_0 @ matrix_1
                    actual_results[(*transition_0, transition_1[1])] = result
                else:
                    continue
        accumulated_results = actual_results

    final_prob_vector = {}
    for path, matrix in accumulated_results.items():
        pi_slice = pi_AB @ matrix
        final_prob_vector[path] = pi_slice

    return final_prob_vector


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
    # n_int_ABC,
    cut_AB="standard",
    # cut_ABC="standard",
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
    final_A = {}
    final_A[0] = pi_A @ expm(trans_mat_a * t_A)
    final_B = {}
    final_B[0] = pi_B @ expm(trans_mat_b * t_B)
    final_C = {}
    final_C[0] = pi_C @ expm(trans_mat_c * t_C)

    # Dicts to know position of hidden states in the prob matrix.
    number_dict_A = state_dict_1
    number_dict_B = state_dict_1
    number_dict_C = state_dict_1
    number_dict_AB = state_dict_2
    number_dict_ABC = state_dict_3

    # Combine A and B CTMCs
    pi_AB = combine_states_general(
        number_dict_A, number_dict_B, number_dict_AB, final_A, final_B
    )

    # Get cutopints and times based on cutopoints.
    if isinstance(cut_AB, str):
        cut_AB = cutpoints_AB(n_int_AB, t_AB, coal_AB)
    times_AB = get_times(cut_AB, list(range(len(cut_AB))))

    final_prob_vector = get_final_per_interval_noif(
        trans_mat_ab, times_AB, omega_dict_2, pi_AB, omega_nonrev_counts_2
    )

    return final_prob_vector


final_prob_vector = get_joint_prob_mat(
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
)
