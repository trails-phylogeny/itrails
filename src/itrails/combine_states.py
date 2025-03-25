import numba as nb
import numpy as np


def combine_states(
    state_dict_1, state_dict_2, state_dict_sum, final_probs_1, final_probs_2
):
    """
    Function that combines two dictionaries of individual states and their final probabilities into a single dictionary of combined states and starting probabilities.

    :param state_dict_1: Dictionary of states and indices for the first CTMC, can be 1 sequence or 2 sequence CTMC.
    :type state_dict_1: Numba Dictionary of Key: Tuple of int64 and Value: int64.
    :param state_dict_2: Dictionary of states and indices for the second CTMC, can only be 1 sequence CTMC.
    :type state_dict_2: Numba Dictionary of Key: Tuple(int64, int64) and Value: int64
    :param state_dict_sum: Dictionary of states and indices for the combined CTMC.
    :type state_dict_sum: Numba Dictionary of Key: Tuple of int64 and Value: int64.
    :param final_probs_1: Array of final probabilities for a single key of the first CTMC.
    :type final_probs_1: Array of type: float64[:, :]
    :param final_probs_2: Array of final probabilities for the only key of the second CTMC.
    :type final_probs_2: Array of type: float64[:, :]
    :return: Array of starting probabilities for each combined state in the combined CTMC.
    :rtype: Array of type: float64[:, :]
    """
    len_array = len(list(state_dict_1.keys())[0]) + len(list(state_dict_2.keys())[0])
    init_comb_dict = {}
    init_comb_all = np.zeros((len(state_dict_sum.keys())), dtype=np.float64)
    for key_1, index_1 in state_dict_1.items():
        left_1 = key_1[: len(key_1) // 2]
        right_1 = key_1[len(key_1) // 2 :]
        for key_2, index_2 in state_dict_2.items():
            left_2 = key_2[: len(key_2) // 2]
            right_2 = key_2[len(key_2) // 2 :]
            comb_state = np.zeros((len_array), dtype=np.int64)
            used_values_1 = {}
            used_values_2 = {}
            current_value = 1
            index = 0
            for value in left_1:
                if value in used_values_1:
                    comb_state[index] = used_values_1[value]
                else:
                    used_values_1[value] = current_value
                    comb_state[index] = current_value
                    current_value += 1
                index += 1

            for value in left_2:
                if value in used_values_2:
                    comb_state[index] = used_values_2[value]
                else:
                    used_values_2[value] = current_value
                    comb_state[index] = current_value
                    current_value += 1
                index += 1

            for value in right_1:
                if value in used_values_1:
                    comb_state[index] = used_values_1[value]
                else:
                    used_values_1[value] = current_value
                    comb_state[index] = current_value
                    current_value += 1
                index += 1

            for value in right_2:
                if value in used_values_2:
                    comb_state[index] = used_values_2[value]
                else:
                    used_values_2[value] = current_value
                    comb_state[index] = current_value
                    current_value += 1
                index += 1

            init_comb_dict[tuple(comb_state)] = (
                final_probs_1[0, index_1] * final_probs_2[0, index_2]
            )
    for state in init_comb_dict.keys():
        index_AB = state_dict_sum[state]
        init_comb_all[index_AB] = init_comb_dict[state]
    return init_comb_all


def combine_states_wrapper(
    state_dict_1,
    state_dict_2,
    state_dict_sum,
    final_probs_1,
    final_probs_2,
):
    """
    Wrapper function that combines dictionaries of states and their final probabilities into a single dictionary of combined states and starting probabilities.

    :param state_dict_1: Dictionary of states and indices for the first CTMC, can be 1 sequence or 2 sequence CTMC.
    :type state_dict_1: Numba Dictionary of Key: Tuple of int64 and Value: int64.
    :param state_dict_2: Dictionary of states and indices for the second CTMC, can only be 1 sequence CTMC.
    :type state_dict_2: Numba Dictionary of Key: Tuple(int64, int64) and Value: int64.
    :param state_dict_sum: Dictionary of states and indices for the combined CTMC.
    :type state_dict_sum: Numba Dictionary of Key: Tuple of int64 and Value: int64.
    :param final_probs_1: Final probability dictionary for the first CTMC.
    :type final_probs_1: Numba Dictionary of Key: UniTuple(nb.types.UniTuple(int64, 3), 2) and Value: float64[:, :].
    :param final_probs_2: Final probability dictionary for the second CTMC.
    :type final_probs_2: Numba Dictionary of Key: UniTuple(nb.types.UniTuple(int64, 3), 2) and Value: float64[:, :].
    :raises NotImplementedError: Not implemented for more than 2 species in state_dict_1 or and more than 1 species in state_dict_2.
    :raises Exception: Fallback if invalid format in final_probs_1 or final_probs_2.
    :return: Dictionary of combined states and starting probabilities for each state.
    :rtype: Numba Dictionary of Key: UniTuple(nb.types.UniTuple(int64, 3), 2) and Value: float64[:, :].
    """
    pi_dict = nb.typed.Dict.empty(
        key_type=nb.types.UniTuple(nb.types.UniTuple(nb.types.int64, 3), 2),
        value_type=nb.types.float64[:, :],
    )

    start_placeholder = ((-1, -1, -1), (-1, -1, -1))

    if len(final_probs_1.keys()) > 1 and len(final_probs_2.keys()) > 1:
        raise NotImplementedError

    elif len(final_probs_1.keys()) > 1 and len(final_probs_2.keys()) == 1:
        prob2 = final_probs_2[start_placeholder]
        for path, prob1 in final_probs_1.items():
            pi_vector_combined = combine_states(
                state_dict_1, state_dict_2, state_dict_sum, prob1, prob2
            )

            pi_dict[path] = pi_vector_combined.reshape(1, -1)

        return pi_dict

    elif len(final_probs_1.keys()) == 1 and len(final_probs_2.keys()) == 1:

        prob1 = final_probs_1[start_placeholder]
        prob2 = final_probs_2[start_placeholder]
        pi_vector_combined = combine_states(
            state_dict_1, state_dict_2, state_dict_sum, prob1, prob2
        )
        pi_dict[start_placeholder] = pi_vector_combined.reshape(1, -1)
        return pi_dict

    else:
        raise Exception("Invalid final_probs_1 or final_probs_2")
