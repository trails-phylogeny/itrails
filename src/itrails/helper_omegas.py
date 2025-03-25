import numba as nb
import numpy as np

from itrails.trans_mat import bell_numbers


@nb.jit(nopython=True)
def combine_by_omega(by_omega_1, by_omega_2):
    """
    Function that updates an existing subpath based on the omega values, keeps the original value if it is not -1 (already a subpath), adds the new value if it is -1 (crates a subpath).

    :param by_omega_1: Original subpath.
    :type by_omega_1: Tuple of int64.
    :param by_omega_2: New subpath.
    :type by_omega_2: Tuple of int64.
    :return: Updated subpath.
    :rtype: Tuple of int64.
    """
    return (
        by_omega_1[0] if by_omega_1[0] != -1 else by_omega_2[0],
        by_omega_1[1] if by_omega_1[1] != -1 else by_omega_2[1],
    )


@nb.jit(nopython=True)
def translate_to_omega(key):
    """Function that translates a key to its omega values.

    :param key: Key to translate.
    :type key: Tuple of Tuple of int64.
    :return: Omega values.
    :rtype: Tuple of int64.
    """
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


@nb.jit(nopython=True)
def remove_absorbing_indices(
    omega_dict,
    absorbing_key,
    species,
    tuple_omegas=nb.types.Tuple((nb.types.int64, nb.types.int64)),
):
    """
    Function that removes the absorbing states from the omega dictionary.

    :param omega_dict: Dictionary of omega indices (key) and vector of booleans where each key has the states (value).
    :type omega_dict: Numba typed dictionary.
    :param absorbing_key: Key of the absorbing states.
    :type absorbing_key: Tuple of int64.
    :param species: Number of species.
    :type species: int64.
    :param tuple_omegas: Type of omega, defaults to nb.types.Tuple((nb.types.int64, nb.types.int64))
    :type tuple_omegas: Numba type, optional.
    :return: Dictionary without the absorbing states.
    :rtype: Numba typed dictionary.
    """
    absorbing_indices = np.where(omega_dict[absorbing_key])[0]
    total_states = bell_numbers(2 * species) - 2
    omega_dict_noabs = nb.typed.Dict.empty(
        key_type=tuple_omegas,
        value_type=np.zeros(total_states, dtype=nb.types.boolean),
    )

    for key, value in omega_dict.items():
        if key != absorbing_key:
            new_array = np.delete(value, absorbing_indices)
            omega_dict_noabs[key] = new_array

    return omega_dict_noabs
