import numpy as np


def generate_paths_deep(
    current,
    absorbing_state,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    path,
    all_paths_dict,
    by_l=-1,
    by_r=-1,
):
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
        for l in next_states_l:
            new_state = (l, current[1])
            new_by_l = (
                by_l
                if by_l != -1
                else (l if omega_nonrev_counts[l] == 1 and start_l + 1 != end_l else -1)
            )
            path.append(new_state)
            generate_paths_deep(
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
        for r in next_states_r:
            new_state = (current[0], r)
            new_by_r = (
                by_r
                if by_r != -1
                else (r if omega_nonrev_counts[r] == 1 and start_r + 1 != end_r else -1)
            )
            path.append(new_state)
            generate_paths_deep(
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
        for l in next_states_l:
            for r in next_states_r:
                if omega_nonrev_counts[r] > start_r:
                    new_state = (l, r)
                    new_by_l = (
                        by_l
                        if by_l != -1
                        else (
                            l
                            if omega_nonrev_counts[l] == 1 and start_l + 1 != end_l
                            else -1
                        )
                    )
                    new_by_r = (
                        by_r
                        if by_r != -1
                        else (
                            r
                            if omega_nonrev_counts[r] == 1 and start_r + 1 != end_r
                            else -1
                        )
                    )
                    path.append(new_state)
                    generate_paths_deep(
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


def get_all_paths_deep(
    omega_init,
    absorbing_state,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    path_to_convert,
):
    all_paths_dict = {}
    path = [omega_init]
    generate_paths_deep(
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
    paths_array = np.zeros(
        (len(all_paths_dict), max_paths, max_subpaths, 2)    )

    # Initialize path_lengths_array to store the length of each path
    path_lengths_array = np.zeros((len(all_paths_dict), max_paths))

    # Fill paths_array and path_lengths_array
    for i, (key, paths) in enumerate(all_paths_dict.items()):
        for j, subpath in enumerate(paths):
            path_lengths_array[i, j] = len(subpath)  # Store the length of the path
            for k, point in enumerate(subpath):
                paths_array[i, j, k] = point

    return keys_array_final, paths_array, path_lengths_array, max_subpaths


"""
# Prepare omega_nonrev_counts and inverted_omega_nonrev_counts
omega_nonrev_counts = {}
inverted_omega_nonrev_counts = {}

# Example data (populate with your actual data)
# omega_nonrev_counts[node] = count
omega_nonrev_counts[0] = 0
omega_nonrev_counts[3] = 1
omega_nonrev_counts[5] = 1
omega_nonrev_counts[6] = 1
omega_nonrev_counts[7] = 2

# inverted_omega_nonrev_counts[count] = list of nodes with that count
inverted_omega_nonrev_counts[0] = [0]
inverted_omega_nonrev_counts[1] = [3, 5, 6]
inverted_omega_nonrev_counts[2] = [7]

# Initial and absorbing states
omega_init = (0, 0)
absorbing_state = (7, 7)

path_to_convert = ((-1, 2, 2), (-1, 2, 2))


# Get all paths
keys_array, paths_array, path_lengths_array = get_all_paths_deep(
    omega_init,
    absorbing_state,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    path_to_convert,
)

print("keys_array:")
print(keys_array)
print("\npaths_array:")
print(paths_array)
print("\npath_lengths_array:")
print(path_lengths_array)
 """
