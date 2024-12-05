from numba import njit, types
from numba.typed import List, Dict

# Define the element types
path_element_type = types.UniTuple(types.int64, 2)
path_list_type = types.ListType(path_element_type)
result_element_type = types.Tuple((types.int64, types.int64, path_list_type))


@njit
def generate_paths_vl_jit(
    current,
    omega_fin,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    path,
    all_paths_list,
    by_l=-1,
    by_r=-1,
):
    if current == omega_fin:
        # Copy the path
        path_copy = List.empty_list(path_element_type)
        for item in path:
            path_copy.append(item)
        # Append the result as a tuple (by_l, by_r, path_copy)
        all_paths_list.append((by_l, by_r, path_copy))
        return

    start_l = omega_nonrev_counts[current[0]]
    start_r = omega_nonrev_counts[current[1]]
    end_l = omega_nonrev_counts[omega_fin[0]]
    end_r = omega_nonrev_counts[omega_fin[1]]

    if start_l < end_l:
        next_states = inverted_omega_nonrev_counts[start_l + 1]
        for l in next_states:
            new_state = (l, current[1])
            new_by_l = (
                by_l
                if by_l != -1
                else (l if omega_nonrev_counts[l] == 1 and start_l + 1 != end_l else -1)
            )
            # Create a new path
            new_path = List.empty_list(path_element_type)
            for item in path:
                new_path.append(item)
            new_path.append(new_state)
            generate_paths_vl_jit(
                new_state,
                omega_fin,
                omega_nonrev_counts,
                inverted_omega_nonrev_counts,
                new_path,
                all_paths_list,
                new_by_l,
                by_r,
            )

    if start_r < end_r:
        next_states = inverted_omega_nonrev_counts[start_r + 1]
        for r in next_states:
            new_state = (current[0], r)
            new_by_r = (
                by_r
                if by_r != -1
                else (r if omega_nonrev_counts[r] == 1 and start_r + 1 != end_r else -1)
            )
            # Create a new path
            new_path = List.empty_list(path_element_type)
            for item in path:
                new_path.append(item)
            new_path.append(new_state)
            generate_paths_vl_jit(
                new_state,
                omega_fin,
                omega_nonrev_counts,
                inverted_omega_nonrev_counts,
                new_path,
                all_paths_list,
                by_l,
                new_by_r,
            )

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
                    # Create a new path
                    new_path = List.empty_list(path_element_type)
                    for item in path:
                        new_path.append(item)
                    new_path.append(new_state)
                    generate_paths_vl_jit(
                        new_state,
                        omega_fin,
                        omega_nonrev_counts,
                        inverted_omega_nonrev_counts,
                        new_path,
                        all_paths_list,
                        new_by_l,
                        new_by_r,
                    )


@njit
def get_all_paths_vl_jit(
    omega_init, omega_fin, omega_nonrev_counts, inverted_omega_nonrev_counts
):
    # Initialize the result list with specified type
    all_paths_list = List.empty_list(result_element_type)
    # Initialize the path with the starting state
    path = List.empty_list(path_element_type)
    path.append(omega_init)
    generate_paths_vl_jit(
        omega_init,
        omega_fin,
        omega_nonrev_counts,
        inverted_omega_nonrev_counts,
        path,
        all_paths_list,
    )
    return all_paths_list


# Example usage
from numba import types
from numba.typed import Dict, List

# Initialize your Numba typed dictionaries
omega_nonrev_counts = Dict.empty(
    key_type=types.int64,
    value_type=types.int64,
)
omega_nonrev_counts[0] = 0
omega_nonrev_counts[3] = 1
omega_nonrev_counts[5] = 1
omega_nonrev_counts[6] = 1
omega_nonrev_counts[7] = 2

inverted_omega_nonrev_counts = Dict.empty(
    key_type=types.int64,
    value_type=types.ListType(types.int64),
)
inverted_omega_nonrev_counts[0] = List([0])
inverted_omega_nonrev_counts[1] = List([3, 5, 6])
inverted_omega_nonrev_counts[2] = List([7])

omega_init = (0, 0)
omega_fin = (3, 7)

# Call the Numba function
all_paths_list = get_all_paths_vl_jit(
    omega_init,
    omega_fin,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
)


# Now process the list to build the desired dictionary
def build_all_paths_dict(all_paths_list):
    all_paths_dict = {}
    for by_l, by_r, path in all_paths_list:
        key = (by_l, by_r)
        # Convert the Numba typed list to a Python list
        path_py = [tuple(state) for state in path]
        if key not in all_paths_dict:
            all_paths_dict[key] = []
        all_paths_dict[key].append(path_py)
    return all_paths_dict


# Build the dictionary
all_paths_dict = build_all_paths_dict(all_paths_list)

# Now you can use all_paths_dict as in your original code
total = 0
for key, value in all_paths_dict.items():
    print(key, value)
    total += len(value)
print("Total number of paths:", total)
