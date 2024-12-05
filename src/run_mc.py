from scipy.sparse import csr_matrix
import numpy as np
from numba import jit
from expm import expm
from combine_states import combine_by_omega
from vanloan import vanloan_general
from vanloan_identify import get_all_paths_vl_jit
from deep_identify import get_all_paths_deep
from remove_absorbing import remove_absorbing_indices
from deepest_ti import deepest_ti
import numba as nb
from numba.typed import Dict
from numba.types import Tuple, int64, float64, boolean, UniTuple
from numba import objmode, typeof


@nb.jit(nopython=True)
def translate_to_omega(key):
    left = key[0]
    right = key[1]
    if left[0] == -1:
        if left[1] == left[2] and left[1] != -1:
            l_omega = 7
        else:
            l_omega = 0
    elif left[0] == 0:
        if left[2] != -1:
            l_omega = 7
        elif left[2] == -1:
            l_omega = 3
    elif left[0] == 1:
        if left[2] != -1:
            l_omega = 7
        elif left[2] == -1:
            l_omega = 3
    elif left[0] == 2:

        if left[2] != -1:
            l_omega = 7
        elif left[2] == -1:
            l_omega = 5
    elif left[0] == 3:
        if left[2] != -1:
            l_omega = 7
        elif left[2] == -1:
            l_omega = 5
    if right[0] == -1:
        if right[1] == right[2] and right[1] != -1:
            r_omega = 7
        else:
            r_omega = 0
    elif right[0] == 0:
        if right[2] != -1:
            r_omega = 7
        elif right[2] == -1:
            r_omega = 3
    elif right[0] == 1:
        if right[2] != -1:
            r_omega = 7
        elif right[2] == -1:
            r_omega = 3
    elif right[0] == 2:
        if right[2] != -1:
            r_omega = 7
        elif right[2] == -1:
            r_omega = 5
    elif right[0] == 3:
        if right[2] != -1:
            r_omega = 7
        elif right[2] == -1:
            r_omega = 6
    return (l_omega, r_omega)


@nb.jit(nopython=True, parallel=True)
def compute_matrices_end_15(
    prob_mats, exponential_time, omega_end_masks, num_combinations
):
    results = np.zeros((num_combinations, 1, 15), dtype=np.float64)
    for i in nb.prange(num_combinations):
        mat_mult_result = prob_mats[i] @ exponential_time
        results[i] = mat_mult_result * omega_end_masks[i]
    return results


@nb.jit(nopython=True, parallel=True)
def compute_matrices_start_end_15(
    prob_mats, exponential_time, omega_start_masks, omega_end_masks, num_combinations
):
    results = np.zeros((num_combinations, 1, 15), dtype=np.float64)
    for i in nb.prange(num_combinations):
        sliced_mat = (
            omega_start_masks[i]  # Shape: (15, 1)
            @ exponential_time  # Shape: (15, 15)
            @ omega_end_masks[i]  # Shape: (1, 15)
        )  # Resulting shape: (15, 15)

        results[i] = prob_mats[i] @ sliced_mat
    return results


@nb.jit(nopython=True, parallel=True)
def compute_matrices_start_end_203(
    prob_mats, exponential_time, omega_start_masks, omega_end_masks, num_combinations
):
    prob_mats = np.ascontiguousarray(prob_mats)
    exponential_time = np.ascontiguousarray(exponential_time)
    omega_start_masks = np.ascontiguousarray(omega_start_masks)
    omega_end_masks = np.ascontiguousarray(omega_end_masks)

    # Pre-allocate result array for the actual number of combinations
    results = np.zeros((num_combinations, 1, 203), dtype=np.float64)
    for i in nb.prange(num_combinations):
        sliced_mat = (
            omega_start_masks[i]  # Shape: (203, 1)
            @ exponential_time[i]  # Shape: (203, 203)
            @ omega_end_masks[i]  # Shape: (1, 203)
        )  # Resulting shape: (203, 203)

        results[i] = prob_mats[i] @ sliced_mat
    return results


@nb.jit(nopython=True, parallel=True)
def vanloan_parallel(
    vl_idx,
    time,
    trans_mat,
    omega_dict,
    vl_keys_acc_array,
    vl_paths_acc_array,
    vl_omega_masks_start,
    vl_omega_masks_end,
    vl_prob_mats,
):
    max_key = 0
    for key_array in vl_keys_acc_array:
        for key in key_array:
            max_key = max(max_key, key[-1])
    results = np.zeros((vl_idx, 9, max_key, 203, 203))

    valid_counts = np.zeros(vl_idx, dtype=np.int64)
    for i in nb.prange(vl_idx):
        key_array = vl_keys_acc_array[i]
        paths_array = vl_paths_acc_array[i]
        valid_length = 0

        for k in range(len(key_array)):
            if key_array[k, -1] == 0:
                break
            valid_length += 1
        valid_counts[i] = valid_length
        for j in nb.prange(valid_length):
            key = key_array[j]
            key_last = key[-1]
            for k in nb.prange(key_last):
                path = paths_array[j][k][1 : paths_array[j][k][0][0] + 1]
                result = vanloan_general(trans_mat, path, time, omega_dict)
                results[i, j, k, :, :] = result
    final_results = np.zeros((vl_idx, 9, 1, 203))
    final_keys = np.zeros((vl_idx, 9, 6), dtype=np.int64)
    # vl_sums = np.zeros((9, 203, 203))
    for i in nb.prange(vl_idx):
        key_array = vl_keys_acc_array[i]
        valid_length = valid_counts[i]
        omega_start_mask = vl_omega_masks_start[i]
        omega_end_mask = vl_omega_masks_end[i]
        all_res = results[i]

        for j in range(valid_length):
            key = key_array[j]
            res_by_key = all_res[j]

            # vl_sums[j, :, :] = results[i, j, :, :, :].sum(axis=0)  # echar ojo
            sliced_mat = (
                omega_start_mask @ res_by_key[:, :, :].sum(axis=0) @ omega_end_mask
            )

            final_results[i, j] = vl_prob_mats[i] @ sliced_mat  # Shape: (1, 203)
            final_keys[i, j] = key[:-1]

    total_valid = valid_counts.sum()
    flattened_results = np.zeros((total_valid, 1, 203))
    flattened_keys = np.zeros((total_valid, 6), dtype=np.int64)

    idx = 0
    for i in range(vl_idx):
        valid_length = valid_counts[i]
        for j in range(valid_length):
            flattened_results[idx] = final_results[i, j]
            flattened_keys[idx] = final_keys[i, j]

            idx += 1

    return flattened_keys, flattened_results, total_valid


@nb.jit(nopython=True)
def run_mc_AB(
    trans_mat,
    times,
    omega_dict,
    prob_dict,
    n_int_AB,
):

    step = 0
    exponential_time_0 = expm(trans_mat * times[step])
    og_keys = list(prob_dict.keys())

    for path in og_keys:
        prob_mats = np.zeros((4, 1, 15), dtype=np.float64)
        omega_masks_end = np.zeros((4, 1, 15), dtype=np.float64)
        keys = np.zeros((4, 6), dtype=np.int64)
        result_idx = 0
        prob_mat = prob_dict[path]
        l_path, r_path = path[0], path[1]
        l_results = np.full((2, 3), -1, dtype=np.int64)
        r_results = np.full((2, 3), -1, dtype=np.int64)
        l_results[0] = l_path
        r_results[0] = r_path
        l_results[1] = (0, step, l_path[2]) if l_path[0] == -1 else l_path
        r_results[1] = (0, step, r_path[2]) if r_path[0] == -1 else r_path
        for l_row in l_results:
            l_tuple = (int(l_row[0]), int(l_row[1]), int(l_row[2]))
            for r_row in r_results:

                r_tuple = (int(r_row[0]), int(r_row[1]), int(r_row[2]))
                if (l_tuple, r_tuple) in og_keys and not (
                    np.array_equal(l_row, l_path) and np.array_equal(r_row, r_path)
                ):
                    continue
                else:
                    new_key = (
                        l_tuple,
                        r_tuple,
                    )

                    omega_end_mask = omega_dict[translate_to_omega(new_key)].reshape(
                        1, 15
                    )

                    new_row = np.array(
                        [l_row[0], l_row[1], l_row[2], r_row[0], r_row[1], r_row[2]],
                        dtype=np.int64,
                    )

                    prob_mats[result_idx] = np.ascontiguousarray(prob_mat)
                    omega_masks_end[result_idx] = np.ascontiguousarray(omega_end_mask)
                    keys[result_idx] = np.ascontiguousarray(new_row)
                    result_idx += 1

        results = compute_matrices_end_15(
            prob_mats, exponential_time_0, omega_masks_end, result_idx
        )

    step += 1
    for i in range(result_idx):
        prob_dict[
            (
                (keys[i][0], keys[i][1], keys[i][2]),
                (
                    keys[i][3],
                    keys[i][4],
                    keys[i][5],
                ),
            )
        ] = results[i]

    for _ in range(step, n_int_AB):
        exponential_time = expm(trans_mat * times[step])
        og_keys = list(prob_dict.keys())

        for path in og_keys:
            prob_mats = np.zeros((4, 1, 15), dtype=np.float64)
            omega_masks_start = np.zeros((4, 15, 15), dtype=np.float64)
            omega_masks_end = np.zeros((4, 15, 15), dtype=np.float64)
            keys = np.zeros((4, 6), dtype=np.int64)
            result_idx = 0
            prob_mat = prob_dict[path]
            l_path, r_path = path[0], path[1]
            l_results = np.full((2, 3), -1, dtype=np.int64)
            r_results = np.full((2, 3), -1, dtype=np.int64)
            l_results[0] = l_path
            r_results[0] = r_path
            l_results[1] = (0, step, l_path[2]) if l_path[0] == -1 else l_path
            r_results[1] = (0, step, r_path[2]) if r_path[0] == -1 else r_path
            for l_row in l_results:
                l_tuple = (int(l_row[0]), int(l_row[1]), int(l_row[2]))
                for r_row in r_results:

                    r_tuple = (int(r_row[0]), int(r_row[1]), int(r_row[2]))
                    if (l_tuple, r_tuple) in og_keys and not (
                        np.array_equal(l_row, l_path) and np.array_equal(r_row, r_path)
                    ):
                        continue
                    else:
                        new_key = (
                            l_tuple,
                            r_tuple,
                        )
                        omega_start = translate_to_omega(path)
                        omega_end = translate_to_omega(new_key)
                        omega_start_mask = omega_dict[omega_start]
                        omega_end_mask = omega_dict[omega_end]

                        new_row = np.array(
                            [
                                l_row[0],
                                l_row[1],
                                l_row[2],
                                r_row[0],
                                r_row[1],
                                r_row[2],
                            ],
                            dtype=np.int64,
                        )

                        prob_mats[result_idx] = np.ascontiguousarray(prob_mat)
                        omega_masks_start[result_idx] = np.diag(omega_start_mask)
                        omega_masks_end[result_idx] = np.diag(omega_end_mask)
                        keys[result_idx] = np.ascontiguousarray(new_row)
                        result_idx += 1

            results = compute_matrices_start_end_15(
                prob_mats,
                exponential_time,
                omega_masks_start,
                omega_masks_end,
                result_idx,
            )

            for i in range(result_idx):
                prob_dict[
                    (
                        (keys[i][0], keys[i][1], keys[i][2]),
                        (
                            keys[i][3],
                            keys[i][4],
                            keys[i][5],
                        ),
                    )
                ] = results[i]
        step += 1

    return prob_dict


@nb.jit(nopython=True)
def run_mc_ABC(
    trans_mat,
    times,
    omega_dict,
    prob_dict,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    n_int_ABC,
):

    # (-1, -1, -1)(-1, -1, -1)
    # 1x203

    for step in range(n_int_ABC - 1):
        exponential_time = expm(trans_mat * times[step])
        og_keys = list(prob_dict.keys())
        for path in og_keys:
            prob_mats = np.zeros((324, 1, 203), dtype=np.float64)
            vl_prob_mats = np.zeros((324, 1, 203), dtype=np.float64)
            omega_masks_start = np.zeros((324, 203, 203), dtype=np.float64)
            omega_masks_end = np.zeros((324, 203, 203), dtype=np.float64)
            vl_omega_masks_start = np.zeros((324, 203, 203), dtype=np.float64)
            vl_omega_masks_end = np.zeros((324, 203, 203), dtype=np.float64)
            exponential_times = np.zeros((324, 203, 203), dtype=np.float64)
            keys = np.zeros((324, 6), dtype=np.int64)
            vl_keys_acc_array = np.zeros((324, 9, 7), dtype=np.int64)
            vl_paths_acc_array = np.zeros((324, 9, 15, 16, 2), dtype=np.int64)
            result_idx = 0
            vl_idx = 0
            prob_mat = prob_dict[path]
            l_path, r_path = path[0], path[1]
            l_results = np.full((6, 3), -1, dtype=np.int64)
            r_results = np.full((6, 3), -1, dtype=np.int64)
            l_results[0] = l_path
            r_results[0] = r_path

            l_results[1] = (l_path[0], step, step) if l_path[0] == -1 else l_path
            r_results[1] = (r_path[0], step, step) if r_path[0] == -1 else r_path

            l_results[2] = (1, step, l_path[2]) if l_path[0] == -1 else l_path
            r_results[2] = (1, step, r_path[2]) if r_path[0] == -1 else r_path

            l_results[3] = (2, step, l_path[2]) if l_path[0] == -1 else l_path
            r_results[3] = (2, step, r_path[2]) if r_path[0] == -1 else r_path

            l_results[4] = (3, step, l_path[2]) if l_path[0] == -1 else l_path
            r_results[4] = (3, step, r_path[2]) if r_path[0] == -1 else r_path

            l_results[5] = (
                (l_path[0], l_path[1], step)
                if l_path[0] != -1 and l_path[2] == -1
                else l_path
            )
            r_results[5] = (
                (r_path[0], r_path[1], step)
                if r_path[0] != -1 and r_path[2] == -1
                else r_path
            )

            for l_row in l_results:
                l_tuple = (int(l_row[0]), int(l_row[1]), int(l_row[2]))
                for r_row in r_results:
                    r_tuple = (int(r_row[0]), int(r_row[1]), int(r_row[2]))
                    if (l_tuple, r_tuple) in og_keys and not (
                        np.array_equal(l_row, l_path) and np.array_equal(r_row, r_path)
                    ):
                        continue
                    else:
                        new_key = (
                            l_tuple,
                            r_tuple,
                        )
                        omega_start = translate_to_omega(path)
                        omega_end = translate_to_omega(new_key)
                        omega_start_mask = omega_dict[omega_start]
                        omega_end_mask = omega_dict[omega_end]

                        if (
                            l_tuple[0] != 0
                            and l_tuple[1] == l_tuple[2]
                            and l_tuple[1] != -1
                        ) or (
                            r_tuple[0] != 0
                            and r_tuple[1] == r_tuple[2]
                            and r_tuple[1] != -1
                        ):
                            omega_start_array = np.array(
                                [omega_start[0], omega_start[1]]
                            )
                            omega_end_array = np.array([omega_end[0], omega_end[1]])
                            (
                                key_array,
                                paths_array,
                            ) = get_all_paths_vl_jit(
                                omega_start_array,
                                omega_end_array,
                                omega_nonrev_counts,
                                inverted_omega_nonrev_counts,
                                l_tuple,
                                r_tuple,
                                l_row,
                                r_row,
                                max_num_keys=10,
                                max_num_subpaths_per_key=20,
                                max_path_length=15,
                                max_total_subpaths=200,
                            )

                            num_keys = key_array.shape[0]
                            num_paths = paths_array.shape[0]

                            vl_keys_acc_array[vl_idx, :num_keys] = key_array[:num_keys]

                            vl_paths_acc_array[vl_idx, :num_paths] = paths_array[
                                :num_paths
                            ]

                            vl_omega_masks_start[vl_idx] = np.diag(omega_start_mask)
                            vl_prob_mats[vl_idx] = prob_mat
                            vl_omega_masks_end[vl_idx] = np.diag(omega_end_mask)
                            vl_idx += 1
                        else:
                            new_row = np.array(
                                [
                                    l_row[0],
                                    l_row[1],
                                    l_row[2],
                                    r_row[0],
                                    r_row[1],
                                    r_row[2],
                                ],
                                dtype=np.int64,
                            )
                            prob_mats[result_idx] = prob_mat

                            omega_masks_start[result_idx] = np.diag(omega_start_mask)
                            omega_masks_end[result_idx] = np.diag(omega_end_mask)
                            keys[result_idx] = new_row
                            exponential_times[result_idx] = exponential_time
                            result_idx += 1

            flattened_keys, flattened_results, total_valid = vanloan_parallel(
                vl_idx,
                times[step],
                trans_mat,
                omega_dict,
                vl_keys_acc_array,
                vl_paths_acc_array,
                vl_omega_masks_start,
                vl_omega_masks_end,
                vl_prob_mats,
            )

            results_novl = compute_matrices_start_end_203(
                prob_mats,
                exponential_times,
                omega_masks_start,
                omega_masks_end,
                result_idx,
            )
            # compute_matrices_start_end_203.parallel_diagnostics(level=4)
            for i in range(result_idx):

                prob_dict[
                    (
                        (keys[i][0], keys[i][1], keys[i][2]),
                        (
                            keys[i][3],
                            keys[i][4],
                            keys[i][5],
                        ),
                    )
                ] = results_novl[i]
            for i in range(total_valid):

                prob_dict[
                    (
                        (
                            flattened_keys[i][0],
                            flattened_keys[i][1],
                            flattened_keys[i][2],
                        ),
                        (
                            flattened_keys[i][3],
                            flattened_keys[i][4],
                            flattened_keys[i][5],
                        ),
                    )
                ] = flattened_results[i]
    # step += 1

    """ og_keys = list(prob_dict.keys())
    noabs_mask = omega_dict[(7, 7)] == False
    trans_mat_noabs = trans_mat[noabs_mask][:, noabs_mask]
    omega_dict_noabs = remove_absorbing_indices(
        omega_dict=omega_dict, absorbing_key=(7, 7), species=3
    )
    for path in og_keys:
        l_path, r_path = path[0], path[1]
        if (l_path[2] == -1 and l_path[2] == l_path[1]) or (
            r_path[2] == -1 and r_path[2] == r_path[1]
        ):
            prob_mat_sliced = prob_dict[path][:, noabs_mask]
            omega_end = translate_to_omega(path)
            all_paths = get_all_paths_deep(
                omega_end,
                (7, 7),
                omega_nonrev_counts,
                inverted_omega_nonrev_counts,
            )
            for by_omega in all_paths.keys():

                new_omega_l = (
                    1
                    if by_omega[0] == 3
                    else (
                        2 if by_omega[0] == 5 else 3 if by_omega[0] == 6 else l_tuple[0]
                    )
                )
                new_omega_r = (
                    1
                    if by_omega[1] == 3
                    else (
                        2 if by_omega[1] == 5 else 3 if by_omega[1] == 6 else r_tuple[0]
                    )
                )
                l_tuple = (
                    (l_path[0], l_path[1], l_path[2])
                    if not (l_path[2] == -1 and l_path[2] == l_path[1])
                    else (new_omega_l, step, step)
                )
                r_tuple = (
                    (r_path[0], r_path[1], r_path[2])
                    if not (r_path[2] == -1 and r_path[2] == r_path[1])
                    else (new_omega_r, step, step)
                )
                new_key = (l_tuple, r_tuple)

                deep_ti_sum = np.zeros_like(trans_mat_noabs)
                for deep_ti_path in all_paths[by_omega]:
                    deep_ti = deep_ti = deepest_ti(
                        trans_mat_noabs, omega_dict_noabs, deep_ti_path
                    )

                    deep_ti_sum += deep_ti

                prob_dict[new_key] = prob_mat_sliced @ deep_ti_sum
            del prob_dict[path]

    og_keys = list(prob_dict.keys())
    for path in og_keys:
        l_path, r_path = path[0], path[1]
        if l_path[2] == -1 and r_path[2] == -1:
            prob_mat = prob_dict[path]
            l_tuple = (l_path[0], l_path[1], step)
            r_tuple = (r_path[0], r_path[1], step)
            new_key = (l_tuple, r_tuple)
            prob_dict[new_key] = prob_mat
            del prob_dict[path]
        elif l_path[2] == -1:
            prob_mat = prob_dict[path]
            l_tuple = (l_path[0], l_path[1], step)
            r_tuple = (r_path[0], r_path[1], r_path[2])
            new_key = (l_tuple, r_tuple)
            prob_dict[new_key] = prob_mat
            del prob_dict[path]
        elif r_path[2] == -1:
            prob_mat = prob_dict[path]
            l_tuple = (l_path[0], l_path[1], l_path[2])
            r_tuple = (r_path[0], r_path[1], step)
            new_key = (l_tuple, r_tuple)
            prob_dict[new_key] = prob_mat
            del prob_dict[path] """

    return prob_dict


""" 
def run_mc(
    trans_mat,
    times,
    omega_dict,
    pi_start,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    absorbing_state,
    species,
    start,
    key_type,
    stage,
    n_int_AB,
    n_int_ABC,
):
    step = start
    noabs_mask = omega_dict[absorbing_state] == False
    if start == 0:
        step += 1
        exponential_time_0 = expm(trans_mat * times[0])
        original_keys = list(pi_start.keys())
        for key, value in omega_dict.items():
            sliced_mat = exponential_time_0 * value[np.newaxis, :]
            for og_key in original_keys:
                og_value = pi_start[og_key]
                with objmode(new_key="UniTuple(UniTuple(int64, 2), 8)"):
                    new_key = list(og_key)
                    new_key[1] = key

                    new_key = tuple(new_key)
                pi_start[new_key] = og_value @ sliced_mat

        for og_key in original_keys:
            del pi_start[og_key]

    for i in range(step, len(times) + start):

        if times[i - start] != float("inf"):
            original_keys = list(pi_start.keys())
            exponential_time = expm(trans_mat * times[i - start])
            for path in original_keys:
                value = pi_start[path]
                omega_init = path[i]
                omega_value = omega_dict[omega_init]
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
                            # if len(all_possible_vl[(omega_init, omega_fin)].keys()) == 1:

                            sliced_mat = (
                                omega_value[:, np.newaxis]
                                * exponential_time
                                * value2[np.newaxis, :]
                            )
                            updated_path = nb.types.UniTuple(
                                nb.types.UniTuple(int64, 2), len(path)
                            )
                            # updated_path = list(path)
                            # updated_path[i + 1] = omega_fin

                            updated_path = nb.types.UniTuple(
                                nb.types.int64, len(updated_path)
                            )
                            for idx in range(len(path)):
                                if idx == i + 1:
                                    updated_path[idx] = key  # Update the specific index
                                else:
                                    updated_path[idx] = path[idx]
                            pi_start[updated_path] = np.asanyarray(value @ sliced_mat)

                        elif len(all_paths.keys()) > 1:
                            # elif len(all_possible_vl[(omega_init, omega_fin)].keys()) > 1:
                            # all_paths = all_possible_vl[(omega_init, omega_fin)]
                            for by_omega in all_paths.keys():
                                vl_sum = np.zeros_like(trans_mat)
                                for vl_path in all_paths[by_omega]:
                                    vl_res = vanloan_general(
                                        trans_mat, vl_path, times[i - start], omega_dict
                                    )

                                    vl_sum += vl_res

                                vl_sum_slice = (
                                    omega_value[:, np.newaxis]
                                    * vl_sum
                                    * value2[np.newaxis, :]
                                )
                                updated_path = nb.types.UniTuple(
                                    nb.types.int64, len(updated_path)
                                )
                                for idx in range(len(path)):
                                    if idx == i + 1:
                                        updated_path[idx] = (
                                            key  # Update the specific index
                                        )
                                    elif idx == len(path) - 1:
                                        updated_path[idx] = by_omega
                                    else:
                                        updated_path[idx] = path[idx]
                                # updated_path = list(path)
                                # updated_path[i + 1] = omega_fin
                                # updated_path[-1] = by_omega
                                # updated_path = nb.types.UniTuple(
                                #    nb.types.int64, len(updated_path)
                                # )(updated_path)
                                pi_start[updated_path] = value @ vl_sum_slice
            for path in original_keys:
                del pi_start[path]

        elif times[i - start] == float("inf"):
            original_keys = list(pi_start.keys())
            trans_mat_noabs = trans_mat[noabs_mask][:, noabs_mask]
            omega_dict_noabs = remove_absorbing_indices(
                omega_dict=omega_dict, absorbing_key=absorbing_state, species=species
            )
            for path in original_keys:
                matrix_0 = pi_start[path]
                end_state = path[i]
                by_omega_0 = path[-1]
                all_paths = get_all_paths_deep(
                    end_state,
                    absorbing_state,
                    omega_nonrev_counts,
                    inverted_omega_nonrev_counts,
                )
                if len(all_paths.keys()) > 1:
                    # if len(all_possible_deep[(end_state, absorbing_state)].keys()) > 1:
                    matrix_0_noabs = matrix_0[:, noabs_mask]
                    # all_paths = all_possible_deep[(end_state, absorbing_state)]
                    for by_omega in all_paths.keys():
                        deep_ti_sum = np.zeros_like(trans_mat_noabs)
                        for deep_path in all_paths[by_omega]:

                            deep_ti = deepest_ti(
                                trans_mat_noabs, omega_dict_noabs, deep_path
                            )

                            deep_ti_sum += deep_ti
                        updated_omega = combine_by_omega(by_omega_0, by_omega)
                        updated_path = nb.types.UniTuple(
                            nb.types.int64, len(updated_path)
                        )
                        for idx in range(len(path)):
                            if idx == i + 1:
                                updated_path[idx] = (
                                    absorbing_state  # Update the specific index
                                )
                            elif idx == len(path) - 1:
                                updated_path[idx] = updated_omega
                            else:
                                updated_path[idx] = path[idx]
                        # updated_path = list(path)
                        # updated_path[i + 1] = absorbing_state
                        # updated_path[-1] = updated_omega
                        # updated_path = nb.types.UniTuple(
                        #    nb.types.int64, len(updated_path)
                        # )(updated_path)
                        result = matrix_0_noabs @ deep_ti_sum

                        pi_start[updated_path] = result
                elif len(all_paths.keys()) == 1:
                    # elif len(all_possible_deep[(end_state, absorbing_state)].keys()) == 1:
                    updated_path = nb.types.UniTuple(nb.types.int64, len(updated_path))
                    for idx in range(len(path)):
                        if idx == i + 1:
                            updated_path[idx] = absorbing_state
                        else:
                            updated_path[idx] = path[idx]
                    # updated_path = list(path)
                    # updated_path[i + 1] = absorbing_state
                    # updated_path = nb.types.UniTuple(nb.types.int64, len(updated_path))(
                    #    updated_path
                    # )
                    pi_start[updated_path] = matrix_0
            for path in original_keys:
                del pi_start[path]

    return pi_start
 """
