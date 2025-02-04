import numba as nb
import numpy as np


@nb.jit(nopython=True)
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

    :param current: _description_
    :type current: _type_
    :param omega_fin: _description_
    :type omega_fin: _type_
    :param omega_nonrev_counts: _description_
    :type omega_nonrev_counts: _type_
    :param inverted_omega_nonrev_counts: _description_
    :type inverted_omega_nonrev_counts: _type_
    :param path: _description_
    :type path: _type_
    :param depth: _description_
    :type depth: _type_
    :param key_indices: _description_
    :type key_indices: _type_
    :param key_array: _description_
    :type key_array: _type_
    :param paths_array: _description_
    :type paths_array: _type_
    :param subpath_counts: _description_
    :type subpath_counts: _type_
    :param num_keys_array: _description_
    :type num_keys_array: _type_
    :param max_num_keys: _description_
    :type max_num_keys: _type_
    :param max_num_subpaths_per_key: _description_
    :type max_num_subpaths_per_key: _type_
    :param max_path_length: _description_
    :type max_path_length: _type_
    :param all_paths_array: _description_
    :type all_paths_array: _type_
    :param path_lengths: _description_
    :type path_lengths: _type_
    :param path_key_indices: _description_
    :type path_key_indices: _type_
    :param total_subpaths_array: _description_
    :type total_subpaths_array: _type_
    :param max_total_subpaths: _description_
    :type max_total_subpaths: _type_
    :param by_l: _description_, defaults to -1
    :type by_l: int, optional
    :param by_r: _description_, defaults to -1
    :type by_r: int, optional
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
            l = next_states_l[i]
            new_state = (l, current[1])
            new_by_l = by_l
            if by_l == -1:
                if omega_nonrev_counts[l] == 1 and start_l + 1 != end_l:
                    new_by_l = l

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
            r = next_states_r[i]
            new_state = (current[0], r)
            new_by_r = by_r
            if by_r == -1:
                if omega_nonrev_counts[r] == 1 and start_r + 1 != end_r:
                    new_by_r = r

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
            l = next_states_l[i]
            for j in range(len(next_states_r)):
                r = next_states_r[j]
                if omega_nonrev_counts[r] > start_r:
                    new_state = (l, r)
                    new_by_l = by_l
                    new_by_r = by_r
                    if by_l == -1:
                        if omega_nonrev_counts[l] == 1 and start_l + 1 != end_l:
                            new_by_l = l
                    if by_r == -1:
                        if omega_nonrev_counts[r] == 1 and start_r + 1 != end_r:
                            new_by_r = r

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


@nb.jit(nopython=True)
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
