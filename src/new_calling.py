import numba as nb
from numba.typed import Dict, List
from numba.types import Tuple, int64, float64, boolean, UniTuple
import numpy as np
from vanloan_identify import get_all_paths_vl

""" 
final_A = Dict.empty(
    key_type=nb.types.UniTuple(nb.types.UniTuple(int64, 3), 2),
    value_type=float64[:, :],
)

final_A[((-1, -1, -1), (-1, -1, -1))] = np.array([[1.0, 0.0, 0.0]])

stage = "AB"
n_int_AB = 3
n_int_ABC = 3
if stage =="AB":
    for i in range(1, n_int_AB + 1):

        for key, value in final_A.items():
            if key[0][1] == -1 and key[1][1] == -1:
                new_key = ((key[0][0], i, key[0][2]), (key[0][0], i, key[1][2]))
                new_value = value
            print(value)
            print("\n")

elif stage == "ABC": """


@nb.jit(nopython=True)
def translate_to_omega(key):
    left = key[0]
    right = key[1]
    if left[0] == -1:
        if left[1] == left[2] and left[1] != -1:
            l_omega = 7
        else:
            l_omega = 0
    elif left[0] == 0:
        if left[2] != -1:
            l_omega = 7
        elif left[2] == -1:
            l_omega = 3
    elif left[0] == 1:
        if left[2] != -1:
            l_omega = 7
        elif left[2] == -1:
            l_omega = 3
    elif left[0] == 2:

        if left[2] != -1:
            l_omega = 7
        elif left[2] == -1:
            l_omega = 5
    elif left[0] == 3:
        if left[2] != -1:
            l_omega = 7
        elif left[2] == -1:
            l_omega = 5
    if right[0] == -1:
        if right[1] == right[2] and right[1] != -1:
            r_omega = 7
        else:
            r_omega = 0
    elif right[0] == 0:
        if right[2] != -1:
            r_omega = 7
        elif right[2] == -1:
            r_omega = 3
    elif right[0] == 1:
        if right[2] != -1:
            r_omega = 7
        elif right[2] == -1:
            r_omega = 3
    elif right[0] == 2:
        if right[2] != -1:
            r_omega = 7
        elif right[2] == -1:
            r_omega = 5
    elif right[0] == 3:
        if right[2] != -1:
            r_omega = 7
        elif right[2] == -1:
            r_omega = 6
    return (l_omega, r_omega)


from trans_mat import get_trans_mat, wrapper_state_general

transitions_3, omega_dict, state_dict_3, omega_nonrev_counts = wrapper_state_general(3)
inverted_omega_nonrev_counts = Dict.empty(
    key_type=nb.types.int64, value_type=nb.types.ListType(nb.types.int64)
)

# Fill the dictionary with values
inverted_omega_nonrev_counts[0] = List([0])
inverted_omega_nonrev_counts[1] = List([3, 5, 6])
inverted_omega_nonrev_counts[2] = List([7])


# print(translate_to_omega(((0, 3, 2), (-1, -1, -1))))
prob_dict = Dict.empty(
    key_type=nb.types.UniTuple(nb.types.UniTuple(int64, 3), 2), value_type=float64[:, :]
)
prob_dict[((-1, -1, -1), (-1, -1, -1))] = np.array([[1.0, 0.0, 0.0]])

step = 1
og_keys = list(prob_dict.keys())
for path in og_keys:
    prob_mat = prob_dict[path]
    l_path, r_path = path[0], path[1]
    l_results = np.full((6, 3), -1, dtype=np.int64)
    r_results = np.full((6, 3), -1, dtype=np.int64)
    l_results[0] = l_path
    r_results[0] = r_path

    l_results[1] = (l_path[0], step, step) if l_path[0] == -1 else l_path
    r_results[1] = (l_path[0], step, step) if r_path[0] == -1 else r_path

    # l_results[2] = (l_path[0], step, step) if l_path[0] == -1 else l_path
    # r_results[2] = (l_path[0], step, step) if r_path[0] == -1 else r_path
    #
    # l_results[3] = (l_path[0], step, step) if l_path[0] == -1 else l_path
    # r_results[3] = (l_path[0], step, step) if r_path[0] == -1 else r_path

    l_results[2] = (1, step, l_path[2]) if l_path[0] == -1 else l_path
    r_results[2] = (1, step, r_path[2]) if r_path[0] == -1 else r_path

    l_results[3] = (2, step, l_path[2]) if l_path[0] == -1 else l_path
    r_results[3] = (2, step, r_path[2]) if r_path[0] == -1 else r_path

    l_results[4] = (3, step, l_path[2]) if l_path[0] == -1 else l_path
    r_results[4] = (3, step, r_path[2]) if r_path[0] == -1 else r_path

    l_results[5] = (
        (l_path[0], l_path[1], step) if l_path[0] != -1 and l_path[3] == -1 else l_path
    )
    r_results[5] = (
        (r_path[0], r_path[1], step) if r_path[0] != -1 and r_path[3] == -1 else r_path
    )

    for l_row in l_results:
        l_tuple = (int(l_row[0]), int(l_row[1]), int(l_row[2]))
        for r_row in r_results:

            r_tuple = (int(r_row[0]), int(r_row[1]), int(r_row[2]))
            if (l_tuple, r_tuple) in prob_dict and not (
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
                print(f"{path} {omega_start} {new_key} {omega_end}")
                omega_start_mask = omega_dict[omega_start]
                omega_end_mask = omega_dict[omega_end]
                if (
                    l_tuple[0] != 0 and l_tuple[1] == l_tuple[2] and l_tuple[1] != -1
                ) or (
                    r_tuple[0] != 0 and r_tuple[1] == r_tuple[2] and r_tuple[1] != -1
                ):

                    all_paths = get_all_paths_vl(
                        omega_start,
                        omega_end,
                        omega_nonrev_counts,
                        inverted_omega_nonrev_counts,
                    )

                    for by_omega in all_paths.keys():
                        new_omega_l = (
                            1
                            if by_omega[0] == 3
                            else (
                                2 if by_omega[0] == 5 else 3 if by_omega[0] == 6 else -1
                            )
                        )
                        new_omega_r = (
                            1
                            if by_omega[1] == 3
                            else (
                                2 if by_omega[1] == 5 else 3 if by_omega[1] == 6 else -1
                            )
                        )
                        new_key = (
                            (new_omega_l, int(l_row[1]), int(l_row[2])),
                            (new_omega_r, int(r_row[1]), int(r_row[2])),
                        )
                        print(f"{new_key} vanloan")

                else:
                    print(f"{new_key} no vanloan")
# VL cuando el primero no es 0, y 2 y 3 != 0 y != nintABC
