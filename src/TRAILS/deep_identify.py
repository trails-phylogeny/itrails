import numba as nb

# Define Numba nb.types
tuple_key_type = nb.types.UniTuple(nb.types.int64, 2)  # (nb.types.int64, nb.types.int64)
path_type = nb.types.ListType(tuple_key_type)  # nb.typed.List of (nb.types.int64, nb.types.int64)
paths_list_type = nb.types.ListType(
    path_type
)  # nb.typed.List of paths (each path is a nb.typed.List of tuples)
key_type = nb.types.UniTuple(nb.types.int64, 2)
value_type = paths_list_type


@nb.jit(nopython=True)
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
            all_paths_dict[key] = nb.typed.List.empty_list(path_type)
        # Create a copy of the current path
        path_copy = nb.typed.List.empty_list(tuple_key_type)
        for p in path:
            path_copy.append(p)
        all_paths_dict[key].append(path_copy)
        return

    start_l = omega_nonrev_counts[current[0]]
    start_r = omega_nonrev_counts[current[1]]
    end_l = omega_nonrev_counts[absorbing_state[0]]
    end_r = omega_nonrev_counts[absorbing_state[1]]

    # Explore next states for left
    if start_l < end_l:
        next_states_l = inverted_omega_nonrev_counts[start_l + 1]
        for i in range(len(next_states_l)):
            l = next_states_l[i]
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
        for i in range(len(next_states_r)):
            r = next_states_r[i]
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
        for i in range(len(next_states_l)):
            l = next_states_l[i]
            for j in range(len(next_states_r)):
                r = next_states_r[j]
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


@nb.jit(nopython=True)
def get_all_paths_deep(
    omega_init, absorbing_state, omega_nonrev_counts, inverted_omega_nonrev_counts
):
    all_paths_dict = nb.typed.Dict.empty(key_type=key_type, value_type=paths_list_type)
    path = nb.typed.List.empty_list(tuple_key_type)
    path.append(omega_init)
    generate_paths_deep(
        omega_init,
        absorbing_state,
        omega_nonrev_counts,
        inverted_omega_nonrev_counts,
        path,
        all_paths_dict,
    )
    return all_paths_dict


""" 
# Prepare omega_nonrev_counts and inverted_omega_nonrev_counts
omega_nonrev_counts = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.int64)
inverted_omega_nonrev_counts = nb.typed.Dict.empty(
    key_type=nb.types.int64, value_type=nb.types.ListType(nb.types.int64)
)

# Example data (populate with your actual data)
# omega_nonrev_counts[node] = count
omega_nonrev_counts[0] = 0
omega_nonrev_counts[3] = 1
omega_nonrev_counts[5] = 1
omega_nonrev_counts[6] = 1
omega_nonrev_counts[7] = 2

# inverted_omega_nonrev_counts[count] = nb.typed.List of nodes with that count
inverted_omega_nonrev_counts[0] = nb.typed.List.empty_list(nb.types.int64)
inverted_omega_nonrev_counts[0].append(0)

inverted_omega_nonrev_counts[1] = nb.typed.List.empty_list(nb.types.int64)
inverted_omega_nonrev_counts[1].append(3)
inverted_omega_nonrev_counts[1].append(5)
inverted_omega_nonrev_counts[1].append(6)

inverted_omega_nonrev_counts[2] = nb.typed.List.empty_list(nb.types.int64)
inverted_omega_nonrev_counts[2].append(7)

# Initial and absorbing states
omega_init = (0, 0)
absorbing_state = (7, 7)

# Get all paths
all_paths_dict = get_all_paths_deep(
    omega_init, absorbing_state, omega_nonrev_counts, inverted_omega_nonrev_counts
)


# Function to print the results outside of Numba's njit
def print_all_paths(all_paths_dict):
    for key in all_paths_dict:
        print(f"Key: {key}")
        paths_list = all_paths_dict[key]
        for path in paths_list:
            print("Path:", [tuple(p) for p in path])
        print()


# Print the results
print_all_paths(all_paths_dict)
 """
