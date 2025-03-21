# Packages
import numba as nb
import numpy as np
import ray

# Functions
from itrails.combine_states import combine_states_wrapper
from itrails.cutpoints import cutpoints_AB, cutpoints_ABC, get_times
from itrails.expm import expm
from itrails.run_markov_chain_AB import run_markov_chain_AB
from itrails.run_markov_chain_ABC import run_markov_chain_ABC
from itrails.trans_mat import get_trans_mat, wrapper_state_general


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
):

    transitions_1, omega_dict_1, state_dict_1, omega_nonrev_counts_1 = (
        wrapper_state_general(1)
    )
    transitions_2, omega_dict_2, state_dict_2, omega_nonrev_counts_2 = (
        wrapper_state_general(2)
    )
    transitions_3, omega_dict_3, state_dict_3, omega_nonrev_counts_3 = (
        wrapper_state_general(3)
    )

    trans_mat_a = get_trans_mat(transitions_1, 1, coal_A, rho_A)
    trans_mat_b = get_trans_mat(transitions_1, 1, coal_B, rho_B)
    trans_mat_c = get_trans_mat(transitions_1, 1, coal_C, rho_C)
    trans_mat_ab = get_trans_mat(transitions_2, 2, coal_AB, rho_AB)
    trans_mat_abc = get_trans_mat(transitions_3, 3, coal_ABC, rho_ABC)

    pi_1seq = np.zeros(2)
    pi_1seq[state_dict_1[(1, 1)]] = 1
    pi_A = pi_B = pi_C = pi_1seq

    start_placeholder = ((-1, -1, -1), (-1, -1, -1))

    final_A = nb.typed.Dict.empty(
        key_type=nb.types.UniTuple(nb.types.UniTuple(nb.types.int64, 3), 2),
        value_type=nb.types.float64[:, :],
    )
    final_B = nb.typed.Dict.empty(
        key_type=nb.types.UniTuple(nb.types.UniTuple(nb.types.int64, 3), 2),
        value_type=nb.types.float64[:, :],
    )
    final_C = nb.typed.Dict.empty(
        key_type=nb.types.UniTuple(nb.types.UniTuple(nb.types.int64, 3), 2),
        value_type=nb.types.float64[:, :],
    )
    final_A[start_placeholder] = (pi_A @ expm(trans_mat_a * t_A)).reshape(1, -1)
    final_B = {}
    final_B[start_placeholder] = (pi_B @ expm(trans_mat_b * t_B)).reshape(1, -1)
    final_C = {}
    final_C[start_placeholder] = (pi_C @ expm(trans_mat_c * t_C)).reshape(1, -1)

    number_dict_A = state_dict_1
    number_dict_B = state_dict_1
    number_dict_C = state_dict_1
    number_dict_AB = state_dict_2
    number_dict_ABC = state_dict_3
    pi_AB = combine_states_wrapper(
        number_dict_A,
        number_dict_B,
        number_dict_AB,
        final_A,
        final_B,
    )

    if isinstance(cut_AB, str):
        cut_AB = cutpoints_AB(n_int_AB, t_AB, coal_AB)
    times_AB = get_times(cut_AB, list(range(len(cut_AB))))
    inverted_omega_nonrev_counts = nb.typed.Dict.empty(
        key_type=nb.types.int64,
        value_type=nb.types.ListType(nb.types.int64),
    )

    inverted_omega_nonrev_counts[0] = nb.typed.List([0])
    inverted_omega_nonrev_counts[1] = nb.typed.List([3])

    print("Before run_MC_AB", flush=True)
    ray.init(
        ignore_reinit_error=True,
        local_mode=True,
    )
    final_AB = run_markov_chain_AB(
        trans_mat_ab,
        times_AB,
        omega_dict_2,
        pi_AB,
        n_int_AB,
    )
    print("After run_MC_AB", flush=True)
    ray.shutdown()
    pi_ABC = combine_states_wrapper(
        number_dict_AB,
        number_dict_C,
        number_dict_ABC,
        final_AB,
        final_C,
    )

    if isinstance(cut_ABC, str):
        cut_ABC = cutpoints_ABC(n_int_ABC, coal_ABC)
    times_ABC = get_times(cut_ABC, list(range(len(cut_ABC))))
    inverted_omega_nonrev_counts = nb.typed.Dict.empty(
        key_type=nb.types.int64, value_type=nb.types.ListType(nb.types.int64)
    )

    inverted_omega_nonrev_counts[0] = nb.typed.List([0])
    inverted_omega_nonrev_counts[1] = nb.typed.List([3, 5, 6])
    inverted_omega_nonrev_counts[2] = nb.typed.List([7])
    # if ray.is_initialized():
    #    ray.shutdown()
    print("Before run_MC_ABC", flush=True)
    ray.init(
        ignore_reinit_error=True,
        local_mode=True,
    )
    final_ABC = run_markov_chain_ABC(
        trans_mat_abc,
        times_ABC,
        omega_dict_3,
        pi_ABC,
        omega_nonrev_counts_3,
        inverted_omega_nonrev_counts,
        n_int_ABC,
        species=3,
        absorbing_state=(7, 7),
    )
    print("After run_MC_ABC", flush=True)
    return final_ABC


""" 
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
    cut_AB="standard",
    cut_ABC="standard",
    n_int_ABC=3,
)


prob = 0
for path, value in final_ABC.items():
    print(path, value)
    prob += value

print(prob)
print("Precomputing done!")


final_ABC = get_joint_prob_mat(
    t_A=240000,
    t_B=240000,
    t_AB=40000,
    t_C=280000,
    rho_A=1e-8,
    rho_B=1e-8,
    rho_AB=1e-8,
    rho_C=1e-8,
    rho_ABC=1e-8,
    coal_A=1 / 1e-8,
    coal_B=1 / 1e-8,
    coal_AB=1 / 1e-8,
    coal_C=1 / 1e-8,
    coal_ABC=1 / 1e-8,
    n_int_AB=3,
    cut_AB="standard",
    cut_ABC="standard",
    n_int_ABC=3,
)

prob = 0
for path, value in final_ABC.items():
    print(path, value)
    prob += value

print(prob)
 """
