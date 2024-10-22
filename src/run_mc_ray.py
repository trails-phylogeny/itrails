from scipy.sparse import csr_matrix
import numpy as np

from expm import expm
import ray

# from scipy.linalg import expm
from combine_states import combine_by_omega
from vanloan import vanloan_general
from vanloan_identify import get_all_paths_vl
from deep_identify import get_all_paths_deep
from remove_absorbing import remove_absorbing_indices
from deepest_ti import deepest_ti

ray.init(ignore_reinit_error=True, include_dashboard=True)


@ray.remote
def compute_for_omega_pair_ray(
    omega_init,
    omega_fin,
    exponential_time,
    trans_mat,
    t,
    omega_dict,
    diag_values,
    diag_values_T,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
):
    # Use the arguments directly
    result_dict = {}
    value = omega_dict[omega_init]
    value2 = omega_dict[omega_fin]
    diag_value = diag_values[omega_init]
    diag_value2_T = diag_values_T[omega_fin]

    all_paths = get_all_paths_vl(
        omega_init,
        omega_fin,
        omega_nonrev_counts,
        inverted_omega_nonrev_counts,
    )
    if len(all_paths) == 1:
        by_omega = (None, None)
        sliced_mat = diag_value * exponential_time * diag_value2_T
        result_dict[(omega_init, omega_fin), by_omega] = sliced_mat
    elif len(all_paths) > 1:
        vl_sum = np.zeros_like(trans_mat)
        for by_omega, paths in all_paths.items():
            for path in paths:
                vl_res = vanloan_general(trans_mat, path, t, omega_dict)
                vl_sum += vl_res
            vl_sum_slice = diag_value * vl_sum * diag_value2_T
            result_dict[(omega_init, omega_fin), by_omega] = vl_sum_slice
    return result_dict


@ray.remote
def compute_inf_time_inot0_ray(
    transition_0,
    by_omega_0,
    matrix_0,
    trans_mat_noabs,
    omega_dict_noabs,
    noabs_mask,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    absorbing_state,
):

    actual_results = {}
    end_state = transition_0[-1]

    # Assuming get_all_paths_deep and deepest_ti are serializable or can be modified accordingly
    all_paths = get_all_paths_deep(
        end_state,
        absorbing_state,
        omega_nonrev_counts,
        inverted_omega_nonrev_counts,
    )

    if len(all_paths.keys()) > 1:
        matrix_0_noabs = matrix_0[noabs_mask][:, noabs_mask]
        for by_omega, paths in all_paths.items():
            deep_ti_sum = np.zeros_like(matrix_0_noabs)
            for path in paths:
                deep_ti = deepest_ti(trans_mat_noabs, omega_dict_noabs, path)
                deep_ti_sum += deep_ti
            result = matrix_0_noabs @ deep_ti_sum
            updated_omega = combine_by_omega(by_omega_0, by_omega)
            actual_results[(*transition_0, path[-1]), (updated_omega)] = result
    else:
        actual_results[(*transition_0, transition_0[-1]), (by_omega_0)] = matrix_0
    return actual_results


@ray.remote
def compute_inf_time_i0_ray(
    transition_0,
    by_omega_0,
    pi,
    trans_mat_noabs,
    omega_dict_noabs,
    noabs_mask,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    absorbing_state,
):

    actual_results = {}
    end_state = transition_0[-1]

    all_paths = get_all_paths_deep(
        end_state,
        absorbing_state,
        omega_nonrev_counts,
        inverted_omega_nonrev_counts,
    )

    if len(all_paths.keys()) > 1:
        pi_noabs = pi[noabs_mask]
        for by_omega, paths in all_paths.items():
            deep_ti_sum = np.zeros_like(trans_mat_noabs)
            for path in paths:
                deep_ti = deepest_ti(trans_mat_noabs, omega_dict_noabs, path)
                deep_ti_sum += deep_ti
            result = pi_noabs @ deep_ti_sum
            updated_omega = combine_by_omega(by_omega_0, by_omega)
            actual_results[(end_state, path[-1]), (updated_omega)] = result
    else:
        actual_results[(end_state, transition_0[-1]), (by_omega_0)] = pi
    return actual_results


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
    # Convert Numba arrays to NumPy arrays if necessary
    omega_dict_serializable = {k: np.array(v) for k, v in omega_dict.items()}
    omega_nonrev_counts_serializable = dict(omega_nonrev_counts)
    inverted_omega_nonrev_counts_serializable = dict(inverted_omega_nonrev_counts)
    diag_values = {
        key: value[:, np.newaxis] for key, value in omega_dict_serializable.items()
    }
    diag_values_T = {
        key: value[np.newaxis, :] for key, value in omega_dict_serializable.items()
    }

    expm_cache = {}
    if initial == 1:
        start = 1
        t0 = times[0]
        exponential_time_0 = expm_cache.get(t0)
        if exponential_time_0 is None:
            exponential_time_0 = expm(trans_mat * t0)
            expm_cache[t0] = exponential_time_0
        for key in omega_dict.keys():
            sliced_mat = diag_values[key] * exponential_time_0
            accumulated_results[((-1, -1), key), (None, None)] = sliced_mat

    for i in range(start, len(times)):
        t = times[i]
        if t != float("inf"):
            exponential_time = expm_cache.get(t)
            if exponential_time is None:
                exponential_time = expm(trans_mat * t)
                expm_cache[t] = exponential_time

            omega_pairs = [
                (omega_init, omega_fin)
                for omega_init in omega_dict.keys()
                for omega_fin in omega_dict.keys()
                if (
                    (
                        omega_init[0] == omega_fin[0]
                        or omega_nonrev_counts[omega_init[0]]
                        < omega_nonrev_counts[omega_fin[0]]
                    )
                    and (
                        omega_init[1] == omega_fin[1]
                        or omega_nonrev_counts[omega_init[1]]
                        < omega_nonrev_counts[omega_fin[1]]
                    )
                )
            ]

            # Place large objects into Ray's object store
            trans_mat_ref = ray.put(trans_mat)
            exponential_time_ref = ray.put(exponential_time)
            omega_dict_ref = ray.put(omega_dict_serializable)
            diag_values_ref = ray.put(diag_values)
            diag_values_T_ref = ray.put(diag_values_T)

            # Launch tasks in parallel
            futures = [
                compute_for_omega_pair_ray.remote(
                    omega_init,
                    omega_fin,
                    exponential_time_ref,
                    trans_mat_ref,
                    t,
                    omega_dict_ref,
                    diag_values_ref,
                    diag_values_T_ref,
                    omega_nonrev_counts_serializable,
                    inverted_omega_nonrev_counts_serializable,
                )
                for omega_init, omega_fin in omega_pairs
            ]
            # Collect the results
            results = ray.get(futures)

            # Combine results
            each_time_dict = {}
            for res in results:
                each_time_dict.update(res)

            actual_results = {}
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
            accumulated_results = actual_results

        elif times[i] == float("inf") and i != 0:

            trans_mat_noabs = trans_mat[noabs_mask][:, noabs_mask]
            omega_dict_noabs = remove_absorbing_indices(
                omega_dict=omega_dict, absorbing_key=absorbing_state, species=species
            )
            trans_mat_noabs = np.array(trans_mat_noabs)
            omega_dict_noabs_serializable = {
                k: np.array(v) for k, v in omega_dict_noabs.items()
            }
            noabs_mask = np.array(noabs_mask)

            trans_mat_noabs_ref = ray.put(trans_mat_noabs)
            omega_dict_noabs_ref = ray.put(omega_dict_noabs_serializable)
            noabs_mask_ref = ray.put(noabs_mask)

            futures = [
                compute_inf_time_inot0_ray.remote(
                    transition_0,
                    by_omega_0,
                    matrix_0,
                    trans_mat_noabs_ref,
                    omega_dict_noabs_ref,
                    noabs_mask_ref,
                    omega_nonrev_counts_serializable,
                    inverted_omega_nonrev_counts_serializable,
                    absorbing_state,
                )
                for (transition_0, by_omega_0), matrix_0 in accumulated_results.items()
            ]
            results = ray.get(futures)
            actual_results = {}
            for res in results:
                actual_results.update(res)
            accumulated_results = actual_results

        elif times[i] == float("inf") and i == 0:

            trans_mat_noabs = trans_mat[noabs_mask][:, noabs_mask]
            omega_dict_noabs = remove_absorbing_indices(
                omega_dict=omega_dict, absorbing_key=absorbing_state, species=species
            )
            pi_start_serializable = {k: np.array(v) for k, v in pi_start.items()}
            trans_mat_noabs = np.array(trans_mat_noabs)
            omega_dict_noabs_serializable = {
                k: np.array(v) for k, v in omega_dict_noabs.items()
            }
            noabs_mask = np.array(noabs_mask)
            trans_mat_noabs_ref = ray.put(trans_mat_noabs)
            omega_dict_noabs_ref = ray.put(omega_dict_noabs_serializable)
            noabs_mask_ref = ray.put(noabs_mask)

            futures = [
                compute_inf_time_i0_ray.remote(
                    transition_0,
                    by_omega_0,
                    pi,
                    trans_mat_noabs_ref,
                    omega_dict_noabs_ref,
                    noabs_mask_ref,
                    omega_nonrev_counts_serializable,
                    inverted_omega_nonrev_counts_serializable,
                    absorbing_state,
                )
                for (transition_0, by_omega_0), pi in pi_start_serializable.items()
            ]
            results = ray.get(futures)
            accumulated_results = {}
            for res in results:
                accumulated_results.update(res)
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
