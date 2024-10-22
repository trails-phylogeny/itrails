import numpy as np
from numba import jit
from numba.types import Tuple, int64, boolean
from numba.typed import Dict
from trans_mat import bell_numbers


@jit(nopython=True)
def remove_absorbing_indices(
    omega_dict, absorbing_key, species, tuple_omegas=Tuple((int64, int64))
):
    # Get the indices of the True values in the absorbing state
    absorbing_indices = np.where(omega_dict[absorbing_key])[0]
    total_states = bell_numbers(2 * species) - 2
    # Create a new dictionary to store the result
    omega_dict_noabs = Dict.empty(
        key_type=tuple_omegas,
        value_type=np.zeros(total_states, dtype=boolean),
    )

    for key, value in omega_dict.items():
        if key != absorbing_key:
            # Remove the entries at the absorbing indices from the current array
            new_array = np.delete(value, absorbing_indices)
            omega_dict_noabs[key] = new_array

    return omega_dict_noabs
