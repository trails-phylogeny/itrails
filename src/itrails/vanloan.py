import numpy as np

from itrails.expm import expm


def vanloan_identify(
    current,
    omega_fin,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    path,
    depth,
    key_indices,
    key_array,
    paths_array,
    subpath_counts,
    num_keys_array,
    max_num_keys,
    max_num_subpaths_per_key,
    max_path_length,
    all_paths_array,
    path_lengths,
    path_key_indices,
    total_subpaths_array,
    max_total_subpaths,
    by_l=-1,
    by_r=-1,
):
    """
    Recursive function that identifies, in each iteration, the possible paths that can be taken from the current state to the final state. The function is called recursively until the final state is reached. The function stores the paths in the paths_array and all_paths_array arrays.

    :param current: Current omega state in the recursion
    :type current: Tuple of int
    :param omega_fin: Omega state to reach
    :type omega_fin: Tuple of int
    :param omega_nonrev_counts: Dictionary containing the number of non-reversible coalescents (value) for each omega state (key)
    :type omega_nonrev_counts: Numba typed Dict
    :param inverted_omega_nonrev_counts: Dictionary containing the omega states (value) for each number of non-reversible coalescents (key)
    :type inverted_omega_nonrev_counts: Numba typed Dict
    :param path: Current path in the recursion
    :type path: Numpy array
    :param depth: Current depth in the recursion
    :type depth: int
    :param key_indices: Helper array to store the indices of the omega subpath being taken
    :type key_indices: Numpy array
    :param key_array: Array to store current omega state of each step
    :type key_array: Numpy array
    :param paths_array: Array storing every subpath taken
    :type paths_array: Numpy array
    :param subpath_counts: Array storing the number of subpaths taken for each complete path
    :type subpath_counts: Numpy array
    :param num_keys_array: Array storing the number of keys
    :type num_keys_array: Numpy array
    :param max_num_keys: Upper limit for the number of keys
    :type max_num_keys: int
    :param max_num_subpaths_per_key: Upper limit for the number of subpaths per key
    :type max_num_subpaths_per_key: int
    :param max_path_length: Upper limit for the path length
    :type max_path_length: int
    :param all_paths_array: Array storing all possible subpaths
    :type all_paths_array: Numpy array
    :param path_lengths: Array storing the length of each path
    :type path_lengths: Numpy array
    :param path_key_indices: Array storing the key index for each path
    :type path_key_indices: Numpy array
    :param total_subpaths_array: Array storing the total number of subpaths
    :type total_subpaths_array: Numpy array
    :param max_total_subpaths: Upper limit for the total number of subpaths
    :type max_total_subpaths: int
    :param by_l: Current omega left subpath, defaults to -1 (initial placeholder value)
    :type by_l: int, optional
    :param by_r: Current omega right subpath, defaults to -1 (initial placeholder value)
    :type by_r: int, optional
    :return: None
    :rtype: None
    """
    if current[0] == omega_fin[0] and current[1] == omega_fin[1]:
        num_keys = num_keys_array[0]
        by_l_val = by_l
        by_r_val = by_r

        key_found = False
        for k in range(num_keys):
            if key_indices[k, 0] == by_l_val and key_indices[k, 1] == by_r_val:
                key_idx = k
                key_found = True
                break
        if not key_found:
            if num_keys >= max_num_keys:
                return
            key_idx = num_keys
            key_indices[key_idx, 0] = by_l_val
            key_indices[key_idx, 1] = by_r_val
            key_array[key_idx, 0] = by_l_val
            key_array[key_idx, 1] = by_r_val
            key_array[key_idx, 2] = 0
            subpath_counts[key_idx] = 0
            num_keys_array[0] += 1

        subpath_idx = subpath_counts[key_idx]
        if subpath_idx >= max_num_subpaths_per_key:
            return

        path_length = depth
        for i in range(path_length):
            paths_array[key_idx, subpath_idx, i, 0] = path[i, 0]
            paths_array[key_idx, subpath_idx, i, 1] = path[i, 1]

        subpath_counts[key_idx] += 1
        key_array[key_idx, 2] = subpath_counts[key_idx]

        total_subpaths = total_subpaths_array[0]
        if total_subpaths >= max_total_subpaths:
            return
        for i in range(path_length):
            all_paths_array[total_subpaths, i, 0] = path[i, 0]
            all_paths_array[total_subpaths, i, 1] = path[i, 1]
        path_lengths[total_subpaths] = path_length
        path_key_indices[total_subpaths] = key_idx
        total_subpaths_array[0] += 1

        return

    start_l = omega_nonrev_counts[current[0]]
    start_r = omega_nonrev_counts[current[1]]
    end_l = omega_nonrev_counts[omega_fin[0]]
    end_r = omega_nonrev_counts[omega_fin[1]]

    if start_l < end_l:
        next_states_l = inverted_omega_nonrev_counts[start_l + 1]
        for i in range(len(next_states_l)):
            left = next_states_l[i]
            new_state = (left, current[1])
            new_by_l = by_l
            if by_l == -1:
                if omega_nonrev_counts[left] == 1 and start_l + 1 != end_l:
                    new_by_l = left

            if depth >= max_path_length:
                return
            path[depth, 0] = new_state[0]
            path[depth, 1] = new_state[1]

            vanloan_identify(
                new_state,
                omega_fin,
                omega_nonrev_counts,
                inverted_omega_nonrev_counts,
                path,
                depth + 1,
                key_indices,
                key_array,
                paths_array,
                subpath_counts,
                num_keys_array,
                max_num_keys,
                max_num_subpaths_per_key,
                max_path_length,
                all_paths_array,
                path_lengths,
                path_key_indices,
                total_subpaths_array,
                max_total_subpaths,
                new_by_l,
                by_r,
            )

    if start_r < end_r:
        next_states_r = inverted_omega_nonrev_counts[start_r + 1]
        for i in range(len(next_states_r)):
            right = next_states_r[i]
            new_state = (current[0], right)
            new_by_r = by_r
            if by_r == -1:
                if omega_nonrev_counts[right] == 1 and start_r + 1 != end_r:
                    new_by_r = right

            if depth >= max_path_length:
                return
            path[depth, 0] = new_state[0]
            path[depth, 1] = new_state[1]

            vanloan_identify(
                new_state,
                omega_fin,
                omega_nonrev_counts,
                inverted_omega_nonrev_counts,
                path,
                depth + 1,
                key_indices,
                key_array,
                paths_array,
                subpath_counts,
                num_keys_array,
                max_num_keys,
                max_num_subpaths_per_key,
                max_path_length,
                all_paths_array,
                path_lengths,
                path_key_indices,
                total_subpaths_array,
                max_total_subpaths,
                by_l,
                new_by_r,
            )

    if start_l < end_l and start_r < end_r:
        next_states_l = inverted_omega_nonrev_counts[start_l + 1]
        next_states_r = inverted_omega_nonrev_counts[start_r + 1]
        for i in range(len(next_states_l)):
            left = next_states_l[i]
            for j in range(len(next_states_r)):
                right = next_states_r[j]
                if omega_nonrev_counts[right] > start_r:
                    new_state = (left, right)
                    new_by_l = by_l
                    new_by_r = by_r
                    if by_l == -1:
                        if omega_nonrev_counts[left] == 1 and start_l + 1 != end_l:
                            new_by_l = left
                    if by_r == -1:
                        if omega_nonrev_counts[right] == 1 and start_r + 1 != end_r:
                            new_by_r = right

                    if depth >= max_path_length:
                        return
                    path[depth, 0] = new_state[0]
                    path[depth, 1] = new_state[1]

                    vanloan_identify(
                        new_state,
                        omega_fin,
                        omega_nonrev_counts,
                        inverted_omega_nonrev_counts,
                        path,
                        depth + 1,
                        key_indices,
                        key_array,
                        paths_array,
                        subpath_counts,
                        num_keys_array,
                        max_num_keys,
                        max_num_subpaths_per_key,
                        max_path_length,
                        all_paths_array,
                        path_lengths,
                        path_key_indices,
                        total_subpaths_array,
                        max_total_subpaths,
                        new_by_l,
                        new_by_r,
                    )


def vanloan_identify_wrapper(
    omega_init,
    omega_fin,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    l_tuple,
    r_tuple,
    l_row,
    r_row,
    max_num_keys,
    max_num_subpaths_per_key,
    max_path_length,
    max_total_subpaths,
):
    """
    Wrapper function for the vanloan_identify function. This function initializes the arrays and dictionaries needed for the recursion and calls the vanloan_identify function. In the end it returns the transformed keys and the paths.

    :param omega_init: Initial omega state
    :type omega_init: Tuple of int
    :param omega_fin: End omega state
    :type omega_fin: Tuple of int
    :param omega_nonrev_counts: Dictionary containing the number of non-reversible coalescents (value) for each omega state (key)
    :type omega_nonrev_counts: Numba typed Dict
    :param inverted_omega_nonrev_counts: Dictionary containing the omega states (value) for each number of non-reversible coalescents (key)
    :type inverted_omega_nonrev_counts: Numba typed Dict
    :param l_tuple: Tuple containing the left omega state
    :type l_tuple: Numba typed Tuple
    :param r_tuple: Tuple containing the right omega state
    :type r_tuple: Numba typed Tuple
    :param l_row: Resulting left state
    :type l_row: Tuple of int
    :param r_row: Resulting right state
    :type r_row: Tuple of int
    :param max_num_keys: Upper limit for the number of keys
    :type max_num_keys: int
    :param max_num_subpaths_per_key: Upper limit for the number of subpaths per key
    :type max_num_subpaths_per_key: int
    :param max_path_length: Upper limit for the path length
    :type max_path_length: int
    :param max_total_subpaths: Upper limit for the total number of subpaths
    :type max_total_subpaths: int
    :return: Resulting keys and paths
    :rtype: Tuple(Array, Array)
    """
    key_indices = np.full((max_num_keys, 2), -1, dtype=np.int64)
    key_array = np.zeros((max_num_keys, 3), dtype=np.int64)
    paths_array = np.zeros(
        (max_num_keys, max_num_subpaths_per_key, max_path_length, 2), dtype=np.int64
    )
    subpath_counts = np.zeros(max_num_keys, dtype=np.int64)
    num_keys_array = np.zeros(1, dtype=np.int64)

    all_paths_array = np.zeros((max_total_subpaths, max_path_length, 2), dtype=np.int64)
    path_lengths = np.zeros(max_total_subpaths, dtype=np.int64)
    path_key_indices = np.zeros(max_total_subpaths, dtype=np.int64)
    total_subpaths_array = np.zeros(1, dtype=np.int64)

    path = np.zeros((max_path_length, 2), dtype=np.int64)
    path[0, 0] = omega_init[0]
    path[0, 1] = omega_init[1]
    depth = 1

    vanloan_identify(
        current=omega_init,
        omega_fin=omega_fin,
        omega_nonrev_counts=omega_nonrev_counts,
        inverted_omega_nonrev_counts=inverted_omega_nonrev_counts,
        path=path,
        depth=depth,
        key_indices=key_indices,
        key_array=key_array,
        paths_array=paths_array,
        subpath_counts=subpath_counts,
        num_keys_array=num_keys_array,
        max_num_keys=max_num_keys,
        max_num_subpaths_per_key=max_num_subpaths_per_key,
        max_path_length=max_path_length,
        all_paths_array=all_paths_array,
        path_lengths=path_lengths,
        path_key_indices=path_key_indices,
        total_subpaths_array=total_subpaths_array,
        max_total_subpaths=max_total_subpaths,
        by_l=-1,
        by_r=-1,
    )

    num_keys = num_keys_array[0]
    total_subpaths = total_subpaths_array[0]

    key_array = key_array[:num_keys]
    paths_array = paths_array[:num_keys]
    subpath_counts = subpath_counts[:num_keys]
    all_paths_array = all_paths_array[:total_subpaths]
    path_lengths = path_lengths[:total_subpaths]
    path_key_indices = path_key_indices[:total_subpaths]

    res_paths = np.full((num_keys, 15, max_path_length + 1, 2), -1, dtype=np.int64)

    for i in range(len(key_array)):
        local_subpaths = 0
        for k, j in enumerate(path_key_indices):
            if j == i:
                res_paths[i, local_subpaths, 0] = path_lengths[k]
                res_paths[i, local_subpaths, 1 : path_lengths[k] + 1] = paths_array[
                    i, local_subpaths, : path_lengths[k]
                ]
                local_subpaths += 1
    transformed_keys = np.zeros((max_num_keys, 7), dtype=np.int64)
    tot = 0
    for i in key_array:

        new_omega_l = (
            1 if i[0] == 3 else (2 if i[0] == 5 else 3 if i[0] == 6 else l_tuple[0])
        )
        new_omega_r = (
            1 if i[1] == 3 else (2 if i[1] == 5 else 3 if i[1] == 6 else r_tuple[0])
        )
        new_row = np.array(
            [
                new_omega_l,
                int(l_row[1]),
                int(l_row[2]),
                new_omega_r,
                int(r_row[1]),
                int(r_row[2]),
                i[2],
            ],
            dtype=np.int64,
        )
        transformed_keys[tot] = new_row
        tot += 1
    return (
        transformed_keys[:tot],
        res_paths,
    )


def vanloan(trans_mat, path, tim, omega_dict):
    """
    This function performs the van Loan (1978) method for finding the integral of a product of matrix exponentials. Function generalizes to any number of matrices.

    :param trans_mat: Transition rate matrix for the CTMC.
    :type trans_mat: Numpy array of type: float64[:, :].
    :param path: Subpath of omega values followed by the CTMC.
    :type path: Numpy array of type: int64[:, :].
    :param tim: Time at which to evaluate the integral.
    :type tim: float64.
    :param omega_dict: Numba dictionary of combinations of omega values and the possible final states of each as array of booleans.
    :type omega_dict: Numba dictionary of key: Tuple((int64, int64)) and value: np.array of boolean.
    :return: Result of the integral of the series of multiplying matrix exponentials.
    :rtype: Numpy array of type: float64[:, :].
    """
    n = trans_mat.shape[0]
    steps = len(path)
    C_mat = np.zeros((n * steps, n * steps))
    C_mat[0:n, 0:n] = trans_mat

    for idx in range(1, steps):
        C_mat[n * idx : n * (idx + 1), n * idx : n * (idx + 1)] = trans_mat
        sub_om_init = (path[idx - 1, 0], path[idx - 1, 1])
        sub_om_fin = (path[idx, 0], path[idx, 1])

        A_mat = (
            np.diag(omega_dict[sub_om_init].astype(np.float64))
            @ trans_mat
            @ np.diag(omega_dict[sub_om_fin].astype(np.float64))
        )
        C_mat[n * (idx - 1) : n * idx, n * idx : n * (idx + 1)] = A_mat
    result = np.zeros((n, n), dtype=np.float64)
    result = expm(C_mat * (tim))[0:n, -n:]
    return result
