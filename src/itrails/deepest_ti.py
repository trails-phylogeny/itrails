import numpy as np


def deep_identify(
    current,
    absorbing_state,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    path,
    all_paths_dict,
    by_l=-1,
    by_r=-1,
):
    """
    Recursive function that identifies, in each iteration, the possible paths that can be taken from the current state to the final state. The function is called recursively until the final state is reached. The function stores the paths in the paths_array and all_paths_array arrays.

    :param current: Current omega state in the recursion
    :type current: Tuple of int
    :param absorbing_state: Absorbing state
    :type absorbing_state: Tuple of int
    :param omega_nonrev_counts: Dictionary containing the number of non-reversible coalescents (value) for each omega state (key)
    :type omega_nonrev_counts: Numba typed Dict
    :param inverted_omega_nonrev_counts: Dictionary containing the omega states (value) for each number of non-reversible coalescents (key)
    :type inverted_omega_nonrev_counts: Numba typed Dict
    :param path: Current path in the recursion
    :type path: Numpy array
    :param all_paths_dict: Dictionary that recursively gets filled up with all the paths
    :type all_paths_dict: Numpy array
    :param by_l: Current omega left subpath, defaults to -1 (initial placeholder value)
    :type by_l: int, optional
    :param by_r: Current omega right subpath, defaults to -1 (initial placeholder value)
    :type by_r: int, optional
    """
    # Calculate differences
    diff_l = omega_nonrev_counts[absorbing_state[0]] - omega_nonrev_counts[current[0]]
    diff_r = omega_nonrev_counts[absorbing_state[1]] - omega_nonrev_counts[current[1]]

    # Termination condition
    if diff_l <= 1 and diff_r <= 1:
        key = (by_l, by_r)
        if key not in all_paths_dict:
            all_paths_dict[key] = []
        # Create a copy of the current path
        path_copy = [tuple(p) for p in path]
        all_paths_dict[key].append(path_copy)
        return

    start_l = omega_nonrev_counts[current[0]]
    start_r = omega_nonrev_counts[current[1]]
    end_l = omega_nonrev_counts[absorbing_state[0]]
    end_r = omega_nonrev_counts[absorbing_state[1]]

    # Explore next states for left
    if start_l < end_l:
        next_states_l = inverted_omega_nonrev_counts[start_l + 1]
        for left in next_states_l:
            new_state = (left, current[1])
            new_by_l = (
                by_l
                if by_l != -1
                else (
                    left
                    if omega_nonrev_counts[left] == 1 and start_l + 1 != end_l
                    else -1
                )
            )
            path.append(new_state)
            deep_identify(
                new_state,
                absorbing_state,
                omega_nonrev_counts,
                inverted_omega_nonrev_counts,
                path,
                all_paths_dict,
                new_by_l,
                by_r,
            )
            path.pop()

    # Explore next states for right
    if start_r < end_r:
        next_states_r = inverted_omega_nonrev_counts[start_r + 1]
        for right in next_states_r:
            new_state = (current[0], right)
            new_by_r = (
                by_r
                if by_r != -1
                else (
                    right
                    if omega_nonrev_counts[right] == 1 and start_r + 1 != end_r
                    else -1
                )
            )
            path.append(new_state)
            deep_identify(
                new_state,
                absorbing_state,
                omega_nonrev_counts,
                inverted_omega_nonrev_counts,
                path,
                all_paths_dict,
                by_l,
                new_by_r,
            )
            path.pop()

    # Explore next states for both left and right
    if start_l < end_l and start_r < end_r:
        next_states_l = inverted_omega_nonrev_counts[start_l + 1]
        next_states_r = inverted_omega_nonrev_counts[start_r + 1]
        for left in next_states_l:
            for right in next_states_r:
                if omega_nonrev_counts[right] > start_r:
                    new_state = (left, right)
                    new_by_l = (
                        by_l
                        if by_l != -1
                        else (
                            left
                            if omega_nonrev_counts[left] == 1 and start_l + 1 != end_l
                            else -1
                        )
                    )
                    new_by_r = (
                        by_r
                        if by_r != -1
                        else (
                            right
                            if omega_nonrev_counts[right] == 1 and start_r + 1 != end_r
                            else -1
                        )
                    )
                    path.append(new_state)
                    deep_identify(
                        new_state,
                        absorbing_state,
                        omega_nonrev_counts,
                        inverted_omega_nonrev_counts,
                        path,
                        all_paths_dict,
                        new_by_l,
                        new_by_r,
                    )
                    path.pop()


def deep_identify_wrapper(
    omega_init,
    absorbing_state,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    path_to_convert,
):
    """
    Wrapper function for the deep_identify function. This function initializes the arrays and dictionaries needed for the recursion and calls the deep_identify function. In the end it returns the transformed keys and the paths.

    :param omega_init: Initial omega state
    :type omega_init: Tuple of int
    :param absorbing_state: Absorbing state
    :type absorbing_state: Tuple of int
    :param omega_nonrev_counts: Dictionary containing the number of non-reversible coalescents (value) for each omega state (key)
    :type omega_nonrev_counts: Numba typed Dict
    :param inverted_omega_nonrev_counts: Dictionary containing the omega states (value) for each number of non-reversible coalescents (key)
    :type inverted_omega_nonrev_counts: Numba typed Dict
    :return: Resulting keys and paths
    :rtype: Tuple(Array, Array, Array, int)
    """

    all_paths_dict = {}
    path = [omega_init]
    deep_identify(
        omega_init,
        absorbing_state,
        omega_nonrev_counts,
        inverted_omega_nonrev_counts,
        path,
        all_paths_dict,
    )

    # Convert dictionary to keys_array and paths_array
    keys_array = np.array(list(all_paths_dict.keys()))
    keys_array_final = np.zeros((len(keys_array), 6))
    path_to_convert_array = np.array(path_to_convert)
    flattened_array = np.hstack(path_to_convert_array.ravel())
    coded_dict = {3: 1, 5: 2, 6: 3}
    for i, key in enumerate(keys_array):
        keys_array_final[i] = flattened_array
        if key[0] != -1 and keys_array_final[i][0] == -1:
            keys_array_final[i][0] = coded_dict[key[0]]
        if key[1] != -1 and keys_array_final[i][3] == -1:
            keys_array_final[i][3] = coded_dict[key[1]]

    # Find the maximum number of paths and subpaths for padding
    max_paths = max(len(paths) for paths in all_paths_dict.values())
    max_subpaths = max(
        max(len(subpath) for subpath in paths) for paths in all_paths_dict.values()
    )

    # Initialize paths_array with zeros
    paths_array = np.zeros((len(all_paths_dict), max_paths, max_subpaths, 2))

    # Initialize path_lengths_array to store the length of each path
    path_lengths_array = np.zeros((len(all_paths_dict), max_paths))

    # Fill paths_array and path_lengths_array
    for i, (key, paths) in enumerate(all_paths_dict.items()):
        for j, subpath in enumerate(paths):
            path_lengths_array[i, j] = len(subpath)  # Store the length of the path
            for k, point in enumerate(subpath):
                paths_array[i, j, k] = point

    return keys_array_final, paths_array, path_lengths_array, max_subpaths


def deepest_ti(trans_mat_noabs, omega_dict_noabs, path):
    """
    This function calculated the integral of matrix exponentials with an infinite time limit.

    :param trans_mat_noabs: Transition matrix without absorbing states
    :type trans_mat_noabs: Numpy array
    :param omega_dict_noabs: Omega dictionary without absorbing states
    :type omega_dict_noabs: Numpy array
    :param path: Path of omega states
    :type path: Numpy array
    :return: Result of the integral of the series of multiplying matrix exponentials.
    :rtype: Numpy array
    """
    steps = len(path) - 1
    n = trans_mat_noabs.shape[0]
    if steps == 1:
        C_mat = trans_mat_noabs

    elif steps > 1:
        C_mat = np.zeros((n * steps, n * steps))
        C_mat[0:n, 0:n] = trans_mat_noabs
        for idx in range(1, steps):

            sub_om_init = (path[idx - 1, 0], path[idx - 1, 1])
            sub_om_fin = (path[idx, 0], path[idx, 1])
            A_mat = (
                np.diag(omega_dict_noabs[sub_om_init].astype(np.float64))
                @ trans_mat_noabs
                @ np.diag(omega_dict_noabs[sub_om_fin].astype(np.float64))
            )
            C_mat[n * idx : n * (idx + 1), n * idx : n * (idx + 1)] = trans_mat_noabs
            C_mat[n * (idx - 1) : n * idx, n * idx : n * (idx + 1)] = A_mat

    sub_om_init = (path[-2, 0], path[-2, 1])
    sub_om_fin = (path[-1, 0], path[-1, 1])
    A_mat = np.ascontiguousarray(
        np.diag(omega_dict_noabs[sub_om_init].astype(np.float64))
        @ trans_mat_noabs
        @ np.diag(omega_dict_noabs[sub_om_fin].astype(np.float64))
    )
    result = (-np.linalg.inv(C_mat))[:n, -n:] @ A_mat
    return result
