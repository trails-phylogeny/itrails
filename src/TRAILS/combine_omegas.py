import numba as nb

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