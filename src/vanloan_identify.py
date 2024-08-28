from trans_mat import wrapper_state_general
import numpy as np


def generate_paths(
    current,
    omega_fin,
    omega_nonrev_counts,
    inverted_omega_nonrev_counts,
    path,
    all_paths_dict,
    by_l=None,
    by_r=None,
):
    if current == omega_fin:
        key = (by_l, by_r)
        if key not in all_paths_dict:
            all_paths_dict[key] = []
        all_paths_dict[key].append(path[:])
        return

    start_l = omega_nonrev_counts[current[0]]
    start_r = omega_nonrev_counts[current[1]]
    end_l = omega_nonrev_counts[omega_fin[0]]
    end_r = omega_nonrev_counts[omega_fin[1]]

    if start_l < end_l:
        for l in inverted_omega_nonrev_counts[start_l + 1]:
            new_state = (l, current[1])
            new_by_l = (
                by_l
                if by_l is not None
                else (l if omega_nonrev_counts[l] == 1 else None)
            )
            generate_paths(
                new_state,
                omega_fin,
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
                else (r if omega_nonrev_counts[r] == 1 else None)
            )
            generate_paths(
                new_state,
                omega_fin,
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
                        else (l if omega_nonrev_counts[l] == 1 else None)
                    )
                    new_by_r = (
                        by_r
                        if by_r is not None
                        else (r if omega_nonrev_counts[r] == 1 else None)
                    )
                    generate_paths(
                        new_state,
                        omega_fin,
                        omega_nonrev_counts,
                        inverted_omega_nonrev_counts,
                        path + [new_state],
                        all_paths_dict,
                        new_by_l,
                        new_by_r,
                    )


def get_all_paths(
    omega_init, omega_fin, omega_nonrev_counts, inverted_omega_nonrev_counts
):
    all_paths_dict = {}
    generate_paths(
        omega_init,
        omega_fin,
        omega_nonrev_counts,
        inverted_omega_nonrev_counts,
        [omega_init],
        all_paths_dict,
    )
    return all_paths_dict


omega_nonrev_counts = {0: 0, 3: 1, 5: 1, 6: 1, 7: 2}
inverted_omega_nonrev_counts = {0: [0], 1: [3, 5, 6], 2: [7]}
omega_init = (0, 0)
omega_fin = (7, 7)

all_paths = get_all_paths(
    omega_init, omega_fin, omega_nonrev_counts, inverted_omega_nonrev_counts
)
print(all_paths)
