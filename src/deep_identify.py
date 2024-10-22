def generate_paths_deep(
    current,
    absorbing_state,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    path,
    all_paths_dict,
    by_l=None,
    by_r=None,
):
    # Adjust the termination condition:
    diff_l = omega_nonrev_counts[absorbing_state[0]] - omega_nonrev_counts[current[0]]
    diff_r = omega_nonrev_counts[absorbing_state[1]] - omega_nonrev_counts[current[1]]

    # If the sum of differences is 2, this is a valid termination point.
    if diff_l <= 1 and diff_r <= 1:
        key = (by_l, by_r)
        if key not in all_paths_dict:
            all_paths_dict[key] = []
        all_paths_dict[key].append(path[:])
        return

    # Proceed with the original path generation logic:
    start_l = omega_nonrev_counts[current[0]]
    start_r = omega_nonrev_counts[current[1]]
    end_l = omega_nonrev_counts[absorbing_state[0]]
    end_r = omega_nonrev_counts[absorbing_state[1]]

    if start_l < end_l:
        for l in inverted_omega_nonrev_counts[start_l + 1]:
            new_state = (l, current[1])
            new_by_l = (
                by_l
                if by_l is not None
                else (
                    l if omega_nonrev_counts[l] == 1 and start_l + 1 != end_l else None
                )
            )
            generate_paths_deep(
                new_state,
                absorbing_state,
                omega_nonrev_counts,
                inverted_omega_nonrev_counts,
                path + [new_state],
                all_paths_dict,
                new_by_l,
                by_r,
            )

    if start_r < end_r:
        for r in inverted_omega_nonrev_counts[start_r + 1]:
            new_state = (current[0], r)
            new_by_r = (
                by_r
                if by_r is not None
                else (
                    r if omega_nonrev_counts[r] == 1 and start_r + 1 != end_r else None
                )
            )
            generate_paths_deep(
                new_state,
                absorbing_state,
                omega_nonrev_counts,
                inverted_omega_nonrev_counts,
                path + [new_state],
                all_paths_dict,
                by_l,
                new_by_r,
            )

    if start_l < end_l and start_r < end_r:
        for l in inverted_omega_nonrev_counts[start_l + 1]:
            for r in inverted_omega_nonrev_counts[start_r + 1]:
                if omega_nonrev_counts[r] > start_r:
                    new_state = (l, r)
                    new_by_l = (
                        by_l
                        if by_l is not None
                        else (
                            l
                            if omega_nonrev_counts[l] == 1 and start_l + 1 != end_l
                            else None
                        )
                    )
                    new_by_r = (
                        by_r
                        if by_r is not None
                        else (
                            r
                            if omega_nonrev_counts[r] == 1 and start_r + 1 != end_r
                            else None
                        )
                    )
                    generate_paths_deep(
                        new_state,
                        absorbing_state,
                        omega_nonrev_counts,
                        inverted_omega_nonrev_counts,
                        path + [new_state],
                        all_paths_dict,
                        new_by_l,
                        new_by_r,
                    )


# Wrapper function to get all paths
def get_all_paths_deep(
    omega_init, absorbing_state, omega_nonrev_counts, inverted_omega_nonrev_counts
):
    all_paths_dict = {}
    generate_paths_deep(
        omega_init,
        absorbing_state,
        omega_nonrev_counts,
        inverted_omega_nonrev_counts,
        [omega_init],
        all_paths_dict,
    )
    return all_paths_dict
