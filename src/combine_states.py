import numpy as np
import numba as nb
from numba.typed import Dict
from numba.types import Tuple, int64, float64, boolean


def combine_states(
    state_dict_1, state_dict_2, state_dict_sum, final_probs_1, final_probs_2
):
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


@nb.jit(nopython=True)
def combine_by_omega(by_omega_1, by_omega_2):
    return (
        by_omega_1[0] if by_omega_1[0] != -1 else by_omega_2[0],
        by_omega_1[1] if by_omega_1[1] != -1 else by_omega_2[1],
    )


def combine_states_general(
    state_dict_1,
    state_dict_2,
    state_dict_sum,
    final_probs_1,
    final_probs_2,
    n_int_AB,
    n_int_ABC,
):
    pi_dict = Dict.empty(
        key_type=nb.types.UniTuple(nb.types.UniTuple(int64, 3), 2),
        value_type=float64[:, :],
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
