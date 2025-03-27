import numpy as np
from joblib import Parallel, delayed

import itrails.ncpu as ncpu
from itrails.expm import expm
from itrails.helper_omegas import translate_to_omega


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


def compute_matrices_end_wrapper(
    prob_mats,
    exponential_time,
    omega_end_masks,
    num_combinations,
):
    """Parallel wrapper that computes the sliced matrix result for each combination by invoking compute_matrix_end on each set of inputs; for every index from 0 to num_combinations-1, it multiplies the corresponding probability matrix (from prob_mats) with the result of slicing the exponential_time matrix using the corresponding omega_end_mask, and aggregates all results into a single numpy array; the computations are performed in parallel using joblib.Parallel with ncpu.N_CPU workers.

    :param prob_mats: List or array of probability matrices where each element is a numpy array.
    :type prob_mats: list or np.ndarray.
    :param exponential_time: Numpy array representing the matrix computed by applying the exponential function to the transition matrix multiplied by time.
    :type exponential_time: np.ndarray.
    :param omega_end_masks: List or array of boolean vectors used to mask the columns of the exponential_time matrix for each combination.
    :type omega_end_masks: list or np.ndarray of bool.
    :param num_combinations: Total number of combinations (iterations) over which to compute the sliced matrix.
    :type num_combinations: int.
    :return: Numpy array containing the resulting matrices computed for each combination.
    :rtype: np.ndarray."""
    results = Parallel(n_jobs=ncpu.N_CPU)(
        delayed(compute_matrix_end)(prob_mats[i], exponential_time, omega_end_masks[i])
        for i in range(num_combinations)
    )
    return np.array(results)


def compute_matrices_start_end_wrapper(
    prob_mats,
    exponential_time,
    omega_start_masks,
    omega_end_masks,
    num_combinations,
):
    """Parallel wrapper that computes the sliced matrix result for each combination by invoking compute_matrix_start_end on each set of inputs; for every index from 0 to num_combinations-1, it multiplies the corresponding probability matrix (from prob_mats) with the result of slicing the exponential_time matrix using the corresponding omega_start_mask and omega_end_mask, and aggregates all results into a single numpy array; the computations are performed in parallel using joblib.Parallel with ncpu.N_CPU workers.

    :param prob_mats: List or array of probability matrices where each element is a numpy array.
    :type prob_mats: list or np.ndarray.
    :param exponential_time: Numpy array representing the matrix computed by applying the exponential function to the transition matrix multiplied by time.
    :type exponential_time: np.ndarray.
    :param omega_start_masks: List or array of boolean vectors used to mask the rows of the exponential_time matrix for each combination.
    :type omega_start_masks: list or np.ndarray of bool.
    :param omega_end_masks: List or array of boolean vectors used to mask the columns of the exponential_time matrix for each combination.
    :type omega_end_masks: list or np.ndarray of bool.
    :param num_combinations: Total number of combinations (iterations) over which to compute the sliced matrix.
    :type num_combinations: int.
    :return: Numpy array containing the resulting matrices computed for each combination.
    :rtype: np.ndarray."""
    results = Parallel(n_jobs=ncpu.N_CPU)(
        delayed(compute_matrix_start_end)(
            prob_mats[i], exponential_time, omega_start_masks[i], omega_end_masks[i]
        )
        for i in range(num_combinations)
    )
    return np.array(results)


def run_markov_chain_AB(
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

        results = compute_matrices_end_wrapper(
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

            results = compute_matrices_start_end_wrapper(
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
