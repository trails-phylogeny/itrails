import numpy as np


def combine_states(
    state_dict_1, state_dict_2, state_dict_sum, final_probs_1, final_probs_2
):
    len_array = len(list(state_dict_1.keys())[0]) + len(list(state_dict_2.keys())[0])
    init_comb_dict = {}
    init_comb_all = np.zeros((len(state_dict_sum.keys())), dtype=np.float64)
    tmp_states = []
    for key_1 in state_dict_1.keys():
        index_1 = state_dict_1[key_1]
        for key_2 in state_dict_2.keys():
            index_2 = state_dict_2[key_2]
            counter = 0
            half_1 = len(key_1) // 2
            half_2 = len(key_2) // 2
            comb_state = np.zeros((len_array), dtype=np.int64)

            for i, left_1 in enumerate(key_1[:half_1]):
                comb_state[i] = left_1
                counter += 1

            for i, left_2 in enumerate(key_2[:half_2], start=counter):
                comb_state[i] = max(comb_state[: len_array // 2] + 1)
                counter += 1

            for i, right_1 in enumerate(key_1[half_1:], start=counter):
                comb_state[i] = (
                    right_1
                    if right_1 in key_1[: i + 1]
                    else max(comb_state[: i + 1] + 1)
                )
                counter += 1

            for i, right_2 in enumerate(key_2[half_2:], start=counter):
                comb_state[i] = (
                    max(comb_state[: len_array // 2])
                    if right_2 in key_2[:half_2]
                    else max(comb_state[: i + 1] + 1)
                )
                counter += 1

            init_comb_dict[tuple(comb_state)] = (
                final_probs_1[index_1] * final_probs_2[index_2]
            )

    for state in init_comb_dict.keys():
        index_AB = state_dict_sum[state]
        init_comb_all[index_AB] = init_comb_dict[state]
    return init_comb_all
