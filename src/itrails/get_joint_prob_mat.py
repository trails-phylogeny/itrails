# Packages
import numba as nb
import numpy as np

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
    """
    Compute the joint probability matrix via sequential Markov chain steps and state combination.

    This function calculates the joint probability matrix for a system involving three sets of species
    (or speciation events) using input parameters for time, recombination rates, and coalescent rates.
    It constructs individual transition matrices for separate species or combined states, then sequentially
    simulates the Markov chain process for species pairs (AB) and the combined species (ABC) to produce
    the final joint probability matrix.

    :param t_A: Time (in coalescent units) for species A.
    :type t_A: float
    :param t_B: Time (in coalescent units) for species B.
    :type t_B: float
    :param t_AB: Time (in coalescent units) between speciation events for species A and B.
    :type t_AB: float
    :param t_C: Time (in coalescent units) for species C.
    :type t_C: float
    :param rho_A: Recombination rate for species A.
    :type rho_A: float
    :param rho_B: Recombination rate for species B.
    :type rho_B: float
    :param rho_AB: Recombination rate for the combined species A and B.
    :type rho_AB: float
    :param rho_C: Recombination rate for species C.
    :type rho_C: float
    :param rho_ABC: Recombination rate for the combined species A, B, and C.
    :type rho_ABC: float
    :param coal_A: Coalescent rate for species A.
    :type coal_A: float
    :param coal_B: Coalescent rate for species B.
    :type coal_B: float
    :param coal_AB: Coalescent rate for the combined species A and B.
    :type coal_AB: float
    :param coal_C: Coalescent rate for species C.
    :type coal_C: float
    :param coal_ABC: Coalescent rate for the combined species A, B, and C.
    :type coal_ABC: float
    :param n_int_AB: Number of discretized time intervals for the A-B process.
    :type n_int_AB: int
    :param n_int_ABC: Number of discretized time intervals for the A-B-C process.
    :type n_int_ABC: int
    :param cut_AB: Option for cutpoints in the A-B process; if a string, standard cutpoints are computed.
    :type cut_AB: str or array-like
    :param cut_ABC: Option for cutpoints in the A-B-C process; if a string, standard cutpoints are computed.
    :type cut_ABC: str or array-like

    :return: A numba typed dictionary mapping state tuples to numpy arrays containing the joint probability
             matrices computed via the Markov chain process.
    :rtype: dict
    """

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

    final_AB = run_markov_chain_AB(
        trans_mat_ab,
        times_AB,
        omega_dict_2,
        pi_AB,
        n_int_AB,
    )

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
    return final_ABC
