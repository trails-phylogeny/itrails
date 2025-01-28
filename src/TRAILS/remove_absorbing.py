import numpy as np
import numba as nb
from trans_mat import bell_numbers


@nb.jit(nopython=True)
def remove_absorbing_indices(
    omega_dict, absorbing_key, species, tuple_omegas=nb.types.Tuple((nb.types.int64, nb.types.int64))
):
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
