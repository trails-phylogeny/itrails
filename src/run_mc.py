import numpy as np
from expm import expm
from combine_states import combine_by_omega
from vanloan import vanloan_general
from vanloan_identify import get_all_paths


def run_mc(
    trans_mat,
    times,
    omega_dict,
    pi_start,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    initial,
):

    if initial == 1:
        accumulated_results = {}
        start = 1
        exponential_time_0 = expm(trans_mat * times[0])
        for key, value in omega_dict.items():
            sliced_mat = (
                pi_start[(-1, -1), (None, None)] @ exponential_time_0 @ np.diag(value)
            )
            accumulated_results[((-1, -1), key), (None, None)] = sliced_mat
    elif initial == 0:
        start = 0
        accumulated_results = pi_start

    for i in range(start, len(times)):
        if times[i] != float("inf"):
            exponential_time = expm(trans_mat * times[i])
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
                        all_paths = get_all_paths(
                            omega_init,
                            omega_fin,
                            omega_nonrev_counts,
                            inverted_omega_nonrev_counts,
                        )
                        if len(all_paths.keys()) == 1:
                            by_omega = (None, None)
                            sliced_mat = (
                                np.diag(value) @ exponential_time @ np.diag(value2)
                            )
                            each_time_dict[(omega_init, omega_fin), (by_omega)] = (
                                sliced_mat
                            )
                        elif len(all_paths.keys()) > 1:

                            for by_omega in all_paths.keys():
                                vl_sum = np.zeros_like(trans_mat)
                                for path in all_paths[by_omega]:
                                    vl_res = vanloan_general(
                                        trans_mat, path, times[i], omega_dict
                                    )
                                    vl_sum += vl_res
                                vl_res_sliced = (
                                    np.diag(value) @ vl_sum @ np.diag(value2)
                                )
                                each_time_dict[(omega_init, omega_fin), (by_omega)] = (
                                    vl_res_sliced
                                )
            for (transition_0, by_omega_0), matrix_0 in accumulated_results.items():
                end_state = transition_0[-1]
                for (transition_1, by_omega_1), matrix_1 in each_time_dict.items():
                    start_state = transition_1[0]
                    if start_state == end_state:
                        updated_omega = combine_by_omega(by_omega_0, by_omega_1)
                        result = matrix_0 @ matrix_1
                        actual_results[
                            (*transition_0, transition_1[1]), (updated_omega)
                        ] = result
                    else:
                        continue
        accumulated_results = actual_results
    final_prob_vector = accumulated_results
    return final_prob_vector
