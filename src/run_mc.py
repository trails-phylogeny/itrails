from scipy.sparse import csr_matrix
import numpy as np
from numba import jit
from expm import expm
from combine_states import combine_by_omega
from vanloan import vanloan_general
from vanloan_identify import get_all_paths_vl
from deep_identify import get_all_paths_deep
from remove_absorbing import remove_absorbing_indices
from deepest_ti import deepest_ti


def run_mc(
    trans_mat,
    times,
    omega_dict,
    pi_start,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    absorbing_state,
    species,
    initial,
):
    accumulated_results = {}
    start = 0
    noabs_mask = np.array(omega_dict[absorbing_state]) == False
    if initial == 1:
        start = 1
        exponential_time_0 = expm(trans_mat * times[0])
        for key, value in omega_dict.items():
            sliced_mat = exponential_time_0 * value[np.newaxis, :]
            accumulated_results[((-1, -1), key), (None, None)] = sliced_mat

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
                        all_paths = get_all_paths_vl(
                            omega_init,
                            omega_fin,
                            omega_nonrev_counts,
                            inverted_omega_nonrev_counts,
                        )
                        if len(all_paths.keys()) == 1:
                            by_omega = (None, None)

                            sliced_mat = (
                                value[:, np.newaxis]
                                * exponential_time
                                * value2[np.newaxis, :]
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

                                vl_sum_slice = (
                                    value[:, np.newaxis]
                                    * vl_sum
                                    * value2[np.newaxis, :]
                                )
                                each_time_dict[(omega_init, omega_fin), (by_omega)] = (
                                    vl_sum_slice
                                )
            if i == 0:
                accumulated_results = each_time_dict
                continue
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

        elif times[i] == float("inf") and i != 0:

            actual_results = {}

            trans_mat_noabs = trans_mat[noabs_mask][:, noabs_mask]
            omega_dict_noabs = remove_absorbing_indices(
                omega_dict=omega_dict, absorbing_key=absorbing_state, species=species
            )
            for (transition_0, by_omega_0), matrix_0 in accumulated_results.items():
                end_state = transition_0[-1]
                all_paths = get_all_paths_deep(
                    end_state,
                    absorbing_state,
                    omega_nonrev_counts,
                    inverted_omega_nonrev_counts,
                )
                if len(all_paths.keys()) > 1:
                    matrix_0_noabs = matrix_0[noabs_mask][:, noabs_mask]
                    for by_omega in all_paths.keys():
                        deep_ti_sum = np.zeros_like(matrix_0_noabs)
                        for path in all_paths[by_omega]:
                            deep_ti = deepest_ti(
                                trans_mat_noabs, omega_dict_noabs, path
                            )

                            deep_ti_sum += deep_ti

                        result = matrix_0_noabs @ deep_ti_sum
                        updated_omega = combine_by_omega(by_omega_0, by_omega)
                        actual_results[(*transition_0, path[-1]), (updated_omega)] = (
                            result
                        )
                else:
                    actual_results[(*transition_0, transition_0[-1]), (by_omega_0)] = (
                        matrix_0
                    )

            accumulated_results = actual_results
        elif times[i] == float("inf") and i == 0:
            actual_results = {}

            trans_mat_noabs = trans_mat[noabs_mask][:, noabs_mask]
            omega_dict_noabs = remove_absorbing_indices(
                omega_dict=omega_dict, absorbing_key=absorbing_state, species=species
            )
            for (transition_0, by_omega_0), matrix_0 in pi_start.items():
                end_state = transition_0[-1]
                all_paths = get_all_paths_deep(
                    end_state,
                    absorbing_state,
                    omega_nonrev_counts,
                    inverted_omega_nonrev_counts,
                )
                if len(all_paths.keys()) > 1:
                    for by_omega in all_paths.keys():
                        deep_ti_sum = np.zeros_like(trans_mat_noabs)
                        for path in all_paths[by_omega]:
                            deep_ti = deepest_ti(
                                trans_mat_noabs, omega_dict_noabs, path
                            )

                            deep_ti_sum += deep_ti

                        result = deep_ti_sum
                        updated_omega = combine_by_omega(by_omega_0, by_omega)
                        actual_results[(end_state, path[-1]), (updated_omega)] = (
                            deep_ti_sum
                        )
                else:
                    actual_results[(end_state, transition_0[-1]), (by_omega_0)] = (
                        matrix_0
                    )

            accumulated_results = actual_results
    final_prob_vector = {}
    if initial == 0 and len(times) == 1:
        for (transition_1, by_omega_1), matrix_1 in accumulated_results.items():
            for (transition_0, by_omega_0), pi in pi_start.items():
                if transition_0[-1] == transition_1[0]:
                    updated_omega = combine_by_omega(by_omega_0, by_omega_1)
                    if pi.shape == matrix_1.shape:
                        final_prob_vector[
                            (*transition_0, *(transition_1[1:])), updated_omega
                        ] = matrix_1
                    elif pi.shape != matrix_1.shape:
                        pi_noabs = pi[noabs_mask]
                        result = pi_noabs @ matrix_1
                        final_prob_vector[
                            (*transition_0, *(transition_1[1:])), updated_omega
                        ] = result

    else:
        for (transition_1, by_omega_1), matrix_1 in accumulated_results.items():
            for (transition_0, by_omega_0), pi in pi_start.items():
                if transition_0[-1] == transition_1[0]:
                    updated_omega = combine_by_omega(by_omega_0, by_omega_1)
                    if pi.shape == matrix_1[0].shape:

                        result = pi @ matrix_1
                        final_prob_vector[
                            (*transition_0, *(transition_1[1:])), updated_omega
                        ] = result
                    elif pi.shape != matrix_1[0].shape:
                        pi_noabs = pi[noabs_mask]
                        result = pi_noabs @ matrix_1
                        final_prob_vector[
                            (*transition_0, *(transition_1[1:])), updated_omega
                        ] = result

    return final_prob_vector
