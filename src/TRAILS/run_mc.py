import pickle

import numpy as np
import ray
from deepest_ti import deep_identify_wrapper, deepest_ti
from expm import expm
from helper_omegas import remove_absorbing_indices, translate_to_omega
from vanloan import vanloan, vanloan_identify_wrapper


@ray.remote
def compute_matrix_end(prob_mat, exponential_time, omega_end_mask):
    """
    Helper function that computes the first matrix multiplication, then slices the matrix to get the columns corresponding to the end state.

    :param prob_mat: Matrix of probabilities
    :type prob_mat: Numpy array
    :param exponential_time: Exponential matrix of transition matrix multiplied by time.
    :type exponential_time: Numpy array
    :param omega_end_mask: Vector of booleans that masks the columns of the matrix.
    :type omega_end_mask: Numpy array of booleans.
    :return: Sliced matrix
    :rtype: Numpy array
    """
    mat_mult_result = prob_mat @ exponential_time
    result = mat_mult_result * omega_end_mask
    return result


@ray.remote
def compute_matrix_start_end(
    prob_mat, exponential_time, omega_start_mask, omega_end_mask
):
    """
    Helper function that computes all matrix multiplications but the first, then slices the matrix to get the rows corresponding to the starting state and columns corresponding to the end state.

    :param prob_mat: Matrix of probabilities
    :type prob_mat: Numpy array
    :param exponential_time: Exponential matrix of transition matrix multiplied by time.
    :type exponential_time: Numpy array
    :param omega_start_mask: Vector of booleans that masks the rows of the matrix.
    :type omega_start_mask: Numpy array of booleans.
    :param omega_end_mask: Vector of booleans that masks the columns of the matrix.
    :type omega_end_mask: Numpy array of booleans.
    :return: Sliced matrix
    :rtype: Numpy array
    """
    sliced_mat = (omega_start_mask) @ (exponential_time) @ (omega_end_mask)

    result = (prob_mat) @ (sliced_mat)
    return result


def compute_matrices_end(
    prob_mats, exponential_time, omega_end_masks, num_combinations
):
    """
    Wrapper that parallelizes the computation of all the possibilities within the first step.

    :param prob_mats: 3D array of probabilities, number in the first dimension is the number of combinations and contains a 2D array of probabilities.
    :type prob_mats: Numpy array
    :param exponential_time: Exponential matrix of transition matrix multiplied by time.
    :type exponential_time: Numpy array
    :param omega_end_masks: 3D array to mask the matrix based on final omega, number in the first dimension is the number of combinations and contains a 2D array of booleans.
    :type omega_end_masks: Numpy array
    :param num_combinations: Total number of operations.
    :type num_combinations: int
    :return: Array of results
    :rtype: Numpy array
    """
    results = []
    for i in range(num_combinations):
        result = compute_matrix_end.remote(
            prob_mats[i], exponential_time, omega_end_masks[i]
        )
        results.append(result)
    results = ray.get(results)
    return np.array(results)


def compute_matrices_start_end(
    prob_mats, exponential_time, omega_start_masks, omega_end_masks, num_combinations
):
    """
    Wrapper that parallelizes the computation of all the possibilities after the first step.

    :param prob_mats: 3D array of probabilities, number in the first dimension is the number of combinations and contains a 2D array of probabilities.
    :type prob_mats: Numpy array
    :param exponential_time: Exponential matrix of transition matrix multiplied by time.
    :type exponential_time: Numpy array
    :param omega_start_masks: 3D array to mask the matrix based on initial omega, number in the first dimension is the number of combinations and contains a 2D array of booleans.
    :type omega_start_masks: Numpy array
    :param omega_end_masks: 3D array to mask the matrix based on final omega, number in the first dimension is the number of combinations and contains a 2D array of booleans.
    :type omega_end_masks: Numpy array
    :param num_combinations: Total number of operations.
    :type num_combinations: int
    :return: Array of results
    :rtype: Numpy array
    """
    results = []
    for i in range(num_combinations):
        result = compute_matrix_start_end.remote(
            prob_mats[i], exponential_time, omega_start_masks[i], omega_end_masks[i]
        )
        results.append(result)
    results = ray.get(results)
    return np.array(results)



@ray.remote
def vanloan_worker_inner(
    idx_i,
    idx_j,
    trans_mat,
    omega_dict_serialized,
    key,
    time,
    paths_array_j,
    omega_start_mask,
    omega_end_mask,
    prob_mat,
):
    """
    Ray worker function for every combination of subpaths when multiple coalescents happen in a same time interval (Van Loan).

    :param idx_i: Marker of the path index within all possible Van Loan paths.
    :type idx_i: int
    :param idx_j: Marker of the subpath index within all possible subpaths.
    :type idx_j: int
    :param trans_mat: Transition matrix.
    :type trans_mat: Numpy array
    :param omega_dict_serialized: Serialized dictionary of omega indices (key) and vector of booleans where each key has the states (value).
    :type omega_dict_serialized: Serialized dictionary
    :param key: Key that represents the current state.
    :type key: Numpy array
    :param time: End time of the interval.
    :type time: float
    :param paths_array_j: Array of possible transitions within a subpath.
    :type paths_array_j: Numpy array
    :param omega_start_mask: 2D array to mask the matrix based on initial omega, array of booleans.
    :type omega_start_mask: Numpy array
    :param omega_end_mask: 2D array to mask the matrix based on final omega, array of booleans.
    :type omega_end_mask: Numpy array
    :param prob_mat: Array of probabilities for each state at the start of the time interval.
    :type prob_mat: Numpy array
    :return: I and J indices, Probability matrix summed over every possible transitions for a subpath, Updated Key.
    :rtype: Tuple(int, int, Numpy array, Numpy array)
    """
    # Deserialize omega_dict
    omega_dict = pickle.loads(omega_dict_serialized)

    key_last = key[-1]
    results = np.zeros((key_last, 203, 203))

    # Compute intermediate results
    for k in range(key_last):
        path = paths_array_j[k][1 : paths_array_j[k][0][0] + 1]
        results[k] = vanloan(trans_mat, path, time, omega_dict)

    # Summing results and computing sliced matrix
    sliced_mat = omega_start_mask @ results.sum(axis=0) @ omega_end_mask
    final_result = prob_mat @ sliced_mat
    final_key = key[:-1]

    return idx_i, idx_j, final_result, final_key


def vanloan_parallel_inner(
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
    """
    Wrapper function that parallelizes the computation of all the possibilities where multiple coalescents happen in a same time interval starting from a certain state (Van Loan).

    :param vl_idx: Total number of cases to parallelize.
    :type vl_idx: int
    :param time: End time of the interval.
    :type time: float
    :param trans_mat: Transition matrix.
    :type trans_mat: Numpy array
    :param omega_dict: Dictionary of omega indices (key) and vector of booleans where each key has the states (value).
    :type omega_dict: Numba typed dictionary
    :param vl_keys_acc_array: Array that accumulates, in each index of the first dimension, the collection of keys for each case.
    :type vl_keys_acc_array: Numpy array
    :param vl_paths_acc_array: Array that accumulates, in each index of the first dimension, the collection of paths for each case.
    :type vl_paths_acc_array: Numpy array
    :param vl_omega_masks_start: 3D array to mask the matrix based on initial omega, number in the first dimension is each case and contains a 2D array of booleans.
    :type vl_omega_masks_start: Numpy array
    :param vl_omega_masks_end: 3D array to mask the matrix based on final omega, number in the first dimension is each case and contains a 2D array of booleans.
    :type vl_omega_masks_end: Numpy array
    :param vl_prob_mats: Array of starting probabilities for each case at the start of the time interval.
    :type vl_prob_mats: Numpy array
    :return: Array of keys, Array of probability matrices summed over every possible transitions for a subpath, Total number of valid cases.
    :rtype: Tuple(Numpy array, Numpy array, int)
    """
    omega_dict_python = dict(omega_dict)

    omega_dict_serialized = pickle.dumps(omega_dict_python)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Submit tasks for each valid (i, j)
    tasks = []
    for i in range(vl_idx):
        key_array = vl_keys_acc_array[i]
        paths_array = vl_paths_acc_array[i]
        omega_start_mask = vl_omega_masks_start[i]
        omega_end_mask = vl_omega_masks_end[i]
        prob_mat = vl_prob_mats[i]

        for j, key in enumerate(key_array):
            if key[-1] == 0:  # Stop at invalid keys
                break
            task = vanloan_worker_inner.remote(
                i,
                j,
                trans_mat,
                omega_dict_serialized,
                key,
                time,
                paths_array[j],
                omega_start_mask,
                omega_end_mask,
                prob_mat,
            )
            tasks.append(task)

    # Gather results
    results = ray.get(tasks)

    # Combine results
    total_valid = len(results)
    flattened_results = np.zeros((total_valid, 1, 203))
    flattened_keys = np.zeros((total_valid, 6), dtype=np.int64)

    for idx, (i, j, final_result, final_key) in enumerate(results):
        flattened_results[idx] = final_result
        flattened_keys[idx] = final_key

    return flattened_keys, flattened_results, total_valid


@ray.remote
def deepest_worker_inner(
    idx_i,
    idx_j,
    trans_mat_noabs,
    omega_dict_noabs_serialized,  # Serialized dictionary
    key,
    paths_array_j,
    acc_prob_mat_noabs,
    path_lengths_j,
):
    """
    Ray worker function for every combination of subpaths when multiple coalescents happen in the last time interval (Deepest TI).

    :param idx_i: Marker of the path index within all possible Deepest TI paths.
    :type idx_i: int
    :param idx_j: Marker of the subpath index within all possible subpaths.
    :type idx_j: int
    :param trans_mat_noabs: Transition matrix without absorbing states.
    :type trans_mat_noabs: Numpy array
    :param omega_dict_noabs_serialized: Serialized dictionary of omega indices (key) and vector of booleans where each key has the states (value), lacks absorbing states.
    :type omega_dict_noabs_serialized: Serialized dictionary
    :param paths_array_j: Array of possible transitions within a subpath.
    :type paths_array_j: Numpy array
    :param acc_prob_mat_noabs: Array of probabilities for each state at the start of the time interval without absorbing states.
    :type acc_prob_mat_noabs: Numpy array
    :param path_lengths_j: Array of lengths of each subpath.
    :type path_lengths_j: Numpy array
    :return: I and J indices, Probability matrix summed over every possible transitions for a subpath, Updated Key.
    :rtype: Tuple(int, int, Numpy array, Numpy array)
    """
    # Deserialize omega_dict
    omega_dict_noabs = pickle.loads(omega_dict_noabs_serialized)

    num_subpaths = len(path_lengths_j)
    results = np.zeros((num_subpaths, 201, 201))

    # Compute intermediate results
    for k in range(num_subpaths):
        path = paths_array_j[k][: path_lengths_j[k]]
        results[k] = deepest_ti(trans_mat_noabs, omega_dict_noabs, path)

    # Summing results and computing sliced matrix
    deep_ti_sum = results.sum(axis=0)
    final_result = acc_prob_mat_noabs @ deep_ti_sum

    return idx_i, idx_j, final_result, key


def deepest_parallel_inner(
    deepest_idx,
    trans_mat_noabs,
    omega_dict_noabs,
    deepest_keys_acc_array,
    deepest_paths_acc_array,
    deepest_path_lengths_array,
    acc_prob_mats_noabs,
):
    """
    Wrapper function that parallelizes the computation of all the possibilities where multiple coalescents happen in the last time interval (Deepest TI).

    :param deepest_idx: Total number of cases to parallelize.
    :type deepest_idx: int
    :param trans_mat_noabs: Transition matrix without absorbing states.
    :type trans_mat_noabs: Numpy array
    :param omega_dict_noabs: Dictionary of omega indices (key) and vector of booleans where each key has the states (value), lacks absorbing states.
    :type omega_dict_noabs: Numba typed dictionary
    :param deepest_keys_acc_array: Array that accumulates, in each index of the first dimension, the collection of keys for each case.
    :type deepest_keys_acc_array: Numpy array
    :param deepest_paths_acc_array: Array that accumulates, in each index of the first dimension, the collection of paths for each case.
    :type deepest_paths_acc_array: Numpy array
    :param deepest_path_lengths_array: Array that accumulates, in each index of the first dimension, the lengths of each path for each case.
    :type deepest_path_lengths_array: Numpy array
    :param acc_prob_mats_noabs: Array of starting probabilities for each case at the start of the time interval without absorbing states.
    :type acc_prob_mats_noabs: Numpy array
    :return: Array of keys, Array of probability matrices summed over every possible transitions for a subpath, Total number of valid cases.
    :rtype: Tuple(Numpy array, Numpy array, int)
    """
    omega_dict_python = dict(omega_dict_noabs)

    omega_dict_noabs_serialized = pickle.dumps(omega_dict_python)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    tasks = []
    for i in range(deepest_idx):
        key_array = deepest_keys_acc_array[i]
        paths_array = deepest_paths_acc_array[i]
        acc_prob_mat_noabs = acc_prob_mats_noabs[i]

        for j, key in enumerate(key_array):
            path_lengths_j = deepest_path_lengths_array[i][j]
            if all(x == 0 for x in key):
                break
            task = deepest_worker_inner.remote(
                i,
                j,
                trans_mat_noabs,
                omega_dict_noabs_serialized,
                key,
                paths_array[j][: np.count_nonzero(path_lengths_j)],
                acc_prob_mat_noabs,
                path_lengths_j[: np.count_nonzero(path_lengths_j)],
            )
            tasks.append(task)

    # Gather results
    results = ray.get(tasks)

    # Combine results
    total_valid = len(results)
    flattened_results = np.zeros((total_valid, 1, 201))
    flattened_keys = np.zeros((total_valid, 6), dtype=np.int64)

    for idx, (i, j, final_result, final_key) in enumerate(results):
        flattened_results[idx] = final_result
        flattened_keys[idx] = final_key

    return flattened_keys, flattened_results, total_valid


def run_mc_AB(
    trans_mat,
    times,
    omega_dict,
    prob_dict,
    n_int_AB,
):
    """
    Function that runs the Discrete Time Markov chain for species A and B.

    :param trans_mat: Transition matrix.
    :type trans_mat: Numpy array
    :param times: Array of cut times.
    :type times: Numpy array
    :param omega_dict: Dictionary of omega indices (key) and vector of booleans where each key has the states (value).
    :type omega_dict: Numba typed dictionary
    :param prob_dict: Dictionary of each path (keys) and probabilities for each state at the start of the first time interval (values).
    :type prob_dict: Numba typed dictionary
    :param n_int_AB: Number of intervals for species A and B.
    :type n_int_AB: int
    :return: Updated dictionary of each path (keys) and probabilities for each state at the end of the last time interval (values).
    :rtype: Numba typed dictionary
    """
    step = 0
    exponential_time_0 = expm(trans_mat * times[step])
    exponential_time_0 = exponential_time_0.copy()
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

        results = compute_matrices_end(
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
        exponential_time = exponential_time.copy()

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

            results = compute_matrices_start_end(
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


def run_mc_ABC(
    trans_mat,
    times,
    omega_dict,
    prob_dict,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    n_int_ABC,
    species,
    absorbing_state=(7, 7),
):
    """
    Function that runs the Discrete Time Markov chain for species A, B, and C.

    :param trans_mat: Transition matrix.
    :type trans_mat: Numpy array
    :param times: Array of cut times, last timepoint is infinity.
    :type times: Numpy array
    :param omega_dict: Dictionary of omega indices (key) and vector of booleans where each key has the states (value).
    :type omega_dict: Numba typed dictionary
    :param prob_dict: Dictionary of each path (keys) and probabilities for each state at the start of the first time interval (values).
    :type prob_dict: Numba typed dictionary
    :param omega_nonrev_counts: Dictionary of omega indices (key) and number of non-reversible transitions (value).
    :type omega_nonrev_counts: Numba typed dictionary
    :param inverted_omega_nonrev_counts: Dictionary of omega indices (key) and number of non-reversible transitions (value), inverted.
    :type inverted_omega_nonrev_counts: Numba typed dictionary
    :param n_int_ABC: Number of intervals for species A, B, and C.
    :type n_int_ABC: int
    :param species: Number of species.
    :type species: int
    :param absorbing_state: State where all coalecents have happened, defaults to (7, 7)
    :type absorbing_state: tuple, optional
    :return: Updated dictionary of each path (keys) and probabilities for each state at the end of the Markov chain (time equals inf)(values).
    :rtype: Numba typed dictionary
    """

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
            keys = np.zeros((324, 6), dtype=np.int64)
            vl_keys_acc_array = np.zeros((324, 9, 7), dtype=np.int64)
            vl_paths_acc_array = np.zeros((324, 9, 15, 16, 2), dtype=np.int64)
            l_results = np.full((6, 3), -1, dtype=np.int64)
            r_results = np.full((6, 3), -1, dtype=np.int64)
            result_idx = 0
            vl_idx = 0

            prob_mat = prob_dict[path]
            l_path, r_path = path[0], path[1]

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
                            ) = vanloan_identify_wrapper(
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
                            result_idx += 1

            flattened_keys, flattened_results, total_valid = vanloan_parallel_inner(
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

            results_novl = compute_matrices_start_end(
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
    og_keys = list(prob_dict.keys())
    noabs_mask = not omega_dict[absorbing_state]
    trans_mat_noabs = trans_mat[noabs_mask][:, noabs_mask]
    omega_dict_noabs = remove_absorbing_indices(
        omega_dict=omega_dict, absorbing_key=absorbing_state, species=species
    )
    prob_dict_sum = {}
    for path in og_keys:
        (l_path, r_path) = path
        acc_prob_mats_noabs = np.zeros((324, 1, 201), dtype=np.float64)
        deepest_keys_acc_array = np.zeros((324, 9, 6), dtype=np.int64)
        deepest_paths_acc_array = np.zeros((324, 9, 15, 16, 2), dtype=np.int64)
        deepest_path_lengths_array = np.zeros((324, 9, 9), dtype=np.int64)
        deepest_idx = 0
        prob_mat = prob_dict[path]

        # Case 1: ((Number != -1, Number != -1, Number != -1), (Number != -1, Number != -1, Number != -1))
        if all(x != -1 for x in l_path) and all(x != -1 for x in r_path):
            prob_dict_sum[path] = np.sum(prob_mat)
            continue

        # Case 2: ((Number != -1, Number != -1, Number != -1), (Number != -1, Number != -1, -1))
        elif (
            all(x != -1 for x in l_path)
            and r_path[2] == -1
            and all(x != -1 for x in r_path[:2])
        ):
            new_key = (l_path, (r_path[0], r_path[1], n_int_ABC - 1))
            prob_dict_sum[new_key] = np.sum(prob_dict[path])
            prob_dict[new_key] = prob_dict.pop(path)
            continue

        # Case 3: ((Number != -1, Number != -1, Number != -1), (-1, -1, -1))
        elif all(x != -1 for x in l_path) and all(x == -1 for x in r_path):
            l_tuple = (int(l_path[0]), int(l_path[1]), int(l_path[2]))
            r_tuple = (int(r_path[0]), n_int_ABC - 1, n_int_ABC - 1)
            new_key = (
                l_tuple,
                r_tuple,
            )
            l_row = [l_tuple[0], l_tuple[1], l_tuple[2]]
            r_row = [r_tuple[0], r_tuple[1], r_tuple[2]]
            new_path = new_key
            omega_start = translate_to_omega(path)

            (keys_array, paths_array, path_lengths_array, max_subpaths) = (
                deep_identify_wrapper(
                    omega_start,
                    absorbing_state,
                    omega_nonrev_counts,
                    inverted_omega_nonrev_counts,
                    new_path,
                )
            )
            num_keys = keys_array.shape[0]
            num_paths = paths_array.shape[0]
            keys_per_path = paths_array.shape[1]
            subpaths_per_path = path_lengths_array.shape[1]
            deepest_keys_acc_array[deepest_idx, :num_keys] = keys_array[:num_keys]
            deepest_paths_acc_array[
                deepest_idx, :num_paths, :keys_per_path, :max_subpaths
            ] = paths_array
            deepest_path_lengths_array[deepest_idx, :num_keys, :subpaths_per_path] = (
                path_lengths_array
            )
            acc_prob_mats_noabs[deepest_idx] = prob_mat[:, noabs_mask]
            deepest_idx += 1
            prob_dict.pop(path)

        # Case 4: ((Number != -1, Number != -1, -1), (Number != -1, Number != -1, Number != -1))
        elif (
            l_path[2] == -1
            and all(x != -1 for x in l_path[:2])
            and all(x != -1 for x in r_path)
        ):
            new_key = ((l_path[0], l_path[1], n_int_ABC - 1), r_path)
            prob_dict_sum[new_key] = np.sum(prob_dict[path])
            prob_dict[new_key] = prob_dict.pop(path)
            continue

        # Case 5: ((Number != -1, Number != -1, -1), (Number != -1, Number != -1, -1))
        elif (
            l_path[2] == -1
            and all(x != -1 for x in l_path[:2])
            and r_path[2] == -1
            and all(x != -1 for x in r_path[:2])
        ):
            new_key = (
                (l_path[0], l_path[1], n_int_ABC - 1),
                (r_path[0], r_path[1], n_int_ABC - 1),
            )
            prob_dict_sum[new_key] = np.sum(prob_dict[path])
            prob_dict[new_key] = prob_dict.pop(path)
            continue

        # Case 6: ((Number != -1, Number != -1, -1), (-1, -1, -1))
        elif (
            l_path[2] == -1
            and all(x != -1 for x in l_path[:2])
            and all(x == -1 for x in r_path)
        ):
            l_tuple = (int(l_path[0]), int(l_path[1]), n_int_ABC - 1)
            r_tuple = (int(r_path[0]), n_int_ABC - 1, n_int_ABC - 1)
            new_key = (
                l_tuple,
                r_tuple,
            )
            l_row = [l_tuple[0], l_tuple[1], l_tuple[2]]
            r_row = [r_tuple[0], r_tuple[1], r_tuple[2]]
            new_path = new_key
            omega_start = translate_to_omega(path)

            (keys_array, paths_array, path_lengths_array, max_subpaths) = (
                deep_identify_wrapper(
                    omega_start,
                    absorbing_state,
                    omega_nonrev_counts,
                    inverted_omega_nonrev_counts,
                    new_path,
                )
            )
            num_keys = keys_array.shape[0]
            num_paths = paths_array.shape[0]
            keys_per_path = paths_array.shape[1]
            subpaths_per_path = path_lengths_array.shape[1]
            deepest_keys_acc_array[deepest_idx, :num_keys] = keys_array[:num_keys]
            deepest_paths_acc_array[
                deepest_idx, :num_paths, :keys_per_path, :max_subpaths
            ] = paths_array  # [:num_paths]
            deepest_path_lengths_array[deepest_idx, :num_keys, :subpaths_per_path] = (
                path_lengths_array
            )
            acc_prob_mats_noabs[deepest_idx] = prob_mat[:, noabs_mask]
            deepest_idx += 1
            prob_dict.pop(path)

        # Case 7: ((-1, -1, -1), (Number != -1, Number != -1, Number != -1))
        elif all(x == -1 for x in l_path) and all(x != -1 for x in r_path):
            l_tuple = (int(l_path[0]), n_int_ABC - 1, n_int_ABC - 1)
            r_tuple = (int(r_path[0]), int(r_path[1]), int(r_path[2]))
            new_key = (
                l_tuple,
                r_tuple,
            )
            l_row = [l_tuple[0], l_tuple[1], l_tuple[2]]
            r_row = [r_tuple[0], r_tuple[1], r_tuple[2]]
            new_path = new_key
            omega_start = translate_to_omega(path)

            (keys_array, paths_array, path_lengths_array, max_subpaths) = (
                deep_identify_wrapper(
                    omega_start,
                    absorbing_state,
                    omega_nonrev_counts,
                    inverted_omega_nonrev_counts,
                    new_path,
                )
            )
            num_keys = keys_array.shape[0]
            num_paths = paths_array.shape[0]
            keys_per_path = paths_array.shape[1]
            subpaths_per_path = path_lengths_array.shape[1]
            deepest_keys_acc_array[deepest_idx, :num_keys] = keys_array[:num_keys]
            deepest_paths_acc_array[
                deepest_idx, :num_paths, :keys_per_path, :max_subpaths
            ] = paths_array
            deepest_path_lengths_array[deepest_idx, :num_keys, :subpaths_per_path] = (
                path_lengths_array
            )
            acc_prob_mats_noabs[deepest_idx] = prob_mat[:, noabs_mask]
            deepest_idx += 1
            prob_dict.pop(path)

        # Case 8: ((-1, -1, -1), (Number != -1, Number != -1, -1))
        elif (
            all(x == -1 for x in l_path)
            and r_path[2] == -1
            and all(x != -1 for x in r_path[:2])
        ):
            l_tuple = (int(l_path[0]), n_int_ABC - 1, n_int_ABC - 1)
            r_tuple = (int(r_path[0]), int(r_path[1]), n_int_ABC - 1)
            new_key = (
                l_tuple,
                r_tuple,
            )
            l_row = [l_tuple[0], l_tuple[1], l_tuple[2]]
            r_row = [r_tuple[0], r_tuple[1], r_tuple[2]]
            new_path = new_key
            omega_start = translate_to_omega(path)

            (keys_array, paths_array, path_lengths_array, max_subpaths) = (
                deep_identify_wrapper(
                    omega_start,
                    absorbing_state,
                    omega_nonrev_counts,
                    inverted_omega_nonrev_counts,
                    new_path,
                )
            )
            num_keys = keys_array.shape[0]
            num_paths = paths_array.shape[0]
            keys_per_path = paths_array.shape[1]
            subpaths_per_path = path_lengths_array.shape[1]
            deepest_keys_acc_array[deepest_idx, :num_keys] = keys_array[:num_keys]
            deepest_paths_acc_array[
                deepest_idx, :num_paths, :keys_per_path, :max_subpaths
            ] = paths_array
            deepest_path_lengths_array[deepest_idx, :num_keys, :subpaths_per_path] = (
                path_lengths_array
            )
            acc_prob_mats_noabs[deepest_idx] = prob_mat[:, noabs_mask]
            deepest_idx += 1
            prob_dict.pop(path)

        # Case 9: ((-1, -1, -1), (-1, -1, -1))
        elif all(x == -1 for x in l_path) and all(x == -1 for x in r_path):
            l_tuple = (int(l_path[0]), n_int_ABC - 1, n_int_ABC - 1)
            r_tuple = (int(r_path[0]), n_int_ABC - 1, n_int_ABC - 1)
            new_key = (
                l_tuple,
                r_tuple,
            )
            l_row = [l_tuple[0], l_tuple[1], l_tuple[2]]
            r_row = [r_tuple[0], r_tuple[1], r_tuple[2]]
            new_path = new_key
            omega_start = translate_to_omega(path)

            (keys_array, paths_array, path_lengths_array, max_subpaths) = (
                deep_identify_wrapper(
                    omega_start,
                    absorbing_state,
                    omega_nonrev_counts,
                    inverted_omega_nonrev_counts,
                    new_path,
                )
            )
            num_keys = keys_array.shape[0]
            num_paths = paths_array.shape[0]
            keys_per_path = paths_array.shape[1]
            subpaths_per_path = path_lengths_array.shape[1]
            deepest_keys_acc_array[deepest_idx, :num_keys] = keys_array[:num_keys]
            deepest_paths_acc_array[
                deepest_idx, :num_paths, :keys_per_path, :max_subpaths
            ] = paths_array
            deepest_path_lengths_array[deepest_idx, :num_keys, :subpaths_per_path] = (
                path_lengths_array
            )
            acc_prob_mats_noabs[deepest_idx] = prob_mat[:, noabs_mask]
            deepest_idx += 1
            prob_dict.pop(path)

        flattened_keys, flattened_results, total_valid = deepest_parallel_inner(
            deepest_idx,
            trans_mat_noabs,
            omega_dict_noabs,
            deepest_keys_acc_array,
            deepest_paths_acc_array,
            deepest_path_lengths_array,
            acc_prob_mats_noabs,
        )

        for i in range(total_valid):
            prob_dict_sum[
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
            ] = np.sum(flattened_results[i])
    return prob_dict_sum
