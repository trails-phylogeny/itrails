import itertools as it

import numba as nb
import numpy as np


@nb.jit(nopython=True)
def bell_numbers(n):
    """Given a number 'n', returns the n'th Bell Number used for initializing the matrices with the correct number of rows.

    :param n: number for which the Bell number is returned.
    :type n: int.
    :return: the n'th Bell Number.
    :rtype: int."""
    dp = [1] + [0] * n
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = dp[i - 1]
        for j in range(1, i + 1):
            temp = dp[j]
            dp[j] = prev + dp[j - 1]
            prev = temp
    return dp[0]


def partition(collection):
    """Generator that creates all set partitions from a list of consecutive numbers; for a given list, it yields nested lists representing all possible partitions of its elements into at most len(collection) subsets.

    :param collection: list of integers ranging from 1 to n.
    :type collection: list[int].
    :return: generator yielding lists of lists representing the partitions.
    :rtype: generator."""
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
        yield [[first]] + smaller


def set_partitions(species):
    """Returns the set partitions for a given number of species in the CTMC by using the bell_numbers and partition generator; the resulting numpy array reformats each partition so that each number represents one nucleotide position (e.g., partition [[1,2,3],[4,5,6]] becomes [1,1,1,2,2,2]).

    :param species: number of species for which partitions are generated.
    :type species: int.
    :return: numpy array with reformatted set partitions.
    :rtype: np.array."""
    num_rows = bell_numbers(2 * species)
    num_cols = 2 * species
    state_array = np.zeros((num_rows, num_cols), dtype=int)
    for n, p in enumerate(partition(list(range(1, 2 * species + 1))), 1):
        for j, subsublist in enumerate(sorted(p)):
            for value in subsublist:
                state_array[n - 1][value - 1] = j + 1
    return state_array


@nb.jit(nopython=True)
def translate_to_minimum(array):
    """Reformats a set partition so that its values are renumbered consecutively starting from 1; for example, converts [1,2,2,2,4,4] to [1,2,2,2,3,3].

    :param array: set partition to reformat.
    :type array: np.array of int with shape (1, n_species*2).
    :return: reformatted set partition with consecutive minimum values.
    :rtype: np.array of int with shape (1, n_species*2)."""
    unique_values = np.unique(array)
    value_map = {value: i + 1 for i, value in enumerate(np.sort(unique_values))}
    translated_array = np.array([value_map[value] for value in array])
    return translated_array


def find_revcoal_recomb(state_array, species, state_dict):
    """Given a state array and number of species, returns all possible reversible coalescences (and recombinations) for each state; produces an array with (4*species + 3) columns where the first 2*species columns represent the state before the event and the next 2*species columns represent the state after the event.

    :param state_array: set partition array for the given number of species.
    :type state_array: np.array of int with shape (Bell number, n_species*2).
    :param species: number of species. :type species: int.
    :param state_dict: dictionary mapping each state tuple to its corresponding number.
    :type state_dict: dict.
    :return: array with reversible coalescences and recombinations.
    :rtype: np.array."""
    rev_coal_count = 0
    rev_coals = np.empty((0, species * 4 + 3), dtype=int)
    for row in state_array:
        l_nucl = row[:species]
        r_nucl = row[species:]
        l_set = set(l_nucl)
        r_set = set(r_nucl)
        if l_set != r_set:
            l_diff = l_set.difference(r_set)
            r_diff = r_set.difference(l_set)
            for i in r_diff:
                for j in l_diff:
                    rev_coal_count += 1
                    rev_coal_state = np.zeros(species * 2, dtype=int)
                    rev_coal_state[:species] = l_nucl
                    for k, val in enumerate(r_nucl):
                        if val == i:
                            rev_coal_state[species + k] = j
                        else:
                            rev_coal_state[species + k] = val
                    ordered_rev_coal = translate_to_minimum(rev_coal_state)
                    new_coal_rec = np.empty((2, species * 4 + 3), dtype=int)
                    new_coal_rec[0, : species * 2] = row
                    new_coal_rec[0, species * 2 : species * 4] = ordered_rev_coal
                    new_coal_rec[0, species * 4] = state_dict[tuple(row)]
                    new_coal_rec[0, species * 4 + 1] = state_dict[
                        tuple(ordered_rev_coal)
                    ]
                    new_coal_rec[0, species * 4 + 2] = 1
                    new_coal_rec[1, : species * 2] = ordered_rev_coal
                    new_coal_rec[1, species * 2 : species * 4] = row
                    new_coal_rec[1, species * 4] = state_dict[tuple(ordered_rev_coal)]
                    new_coal_rec[1, species * 4 + 1] = state_dict[tuple(row)]
                    new_coal_rec[1, species * 4 + 2] = 2
                    rev_coals = np.vstack((rev_coals, new_coal_rec))
    return rev_coals


def find_norevcoal(state_array, species, state_dict):
    """Given a state array and number of species, returns all possible non-reversible coalescences for each state; produces an array with (4*species + 3) columns where the first 2*species columns represent the state before the event and the next 2*species columns represent the state after the event.

    :param state_array: set partition array for the given number of species.
    :type state_array: np.array of int with shape (Bell number, n_species*2).
    :param species: number of species.
    :type species: int.
    :param state_dict: dictionary mapping each state tuple to its corresponding number.
    :type state_dict: dict.
    :return: array with non-reversible coalescences.
    :rtype: np.array."""
    norevcoal_count = 0
    norev_coals = np.empty((0, species * 4 + 3), dtype=int)
    for row in state_array:
        l_nucl = row[:species]
        r_nucl = row[species:]
        l_set = set(l_nucl)
        r_set = set(r_nucl)
        norevcoal_state = np.zeros(species * 2, dtype=int)
        changed_pairs = []
        if len(l_set) > 1:
            for i, num1 in enumerate(l_nucl):
                for j, num2 in enumerate(l_nucl):
                    if i < j and num1 != num2:
                        pair = sorted((num1, num2))
                        if pair not in changed_pairs:
                            norevcoal_count += 1
                            change = min(num1, num2)
                            for k, val in enumerate(row):
                                if val == num1 or val == num2:
                                    norevcoal_state[k] = change
                                else:
                                    norevcoal_state[k] = val
                            changed_pairs.append(pair)
                            ordered_norev_coal = translate_to_minimum(norevcoal_state)
                            new_norev_coal = np.empty((1, species * 4 + 3), dtype=int)
                            new_norev_coal[:, : species * 2] = row
                            new_norev_coal[:, species * 2 : species * 4] = (
                                ordered_norev_coal
                            )
                            new_norev_coal[:, species * 4] = state_dict[tuple(row)]
                            new_norev_coal[:, species * 4 + 1] = state_dict[
                                tuple(ordered_norev_coal)
                            ]
                            new_norev_coal[:, species * 4 + 2] = 1
                            norev_coals = np.vstack((norev_coals, new_norev_coal))
        if len(r_set) > 1:
            for i, num1 in enumerate(r_nucl):
                for j, num2 in enumerate(r_nucl):
                    if i < j and num1 != num2:
                        pair = sorted((num1, num2))
                        if pair not in changed_pairs:
                            norevcoal_count += 1
                            change = min(num1, num2)
                            for k, val in enumerate(row):
                                if val == num1 or val == num2:
                                    norevcoal_state[k] = change
                                else:
                                    norevcoal_state[k] = val
                            changed_pairs.append(pair)
                            ordered_norev_coal = translate_to_minimum(norevcoal_state)
                            new_norev_coal = np.empty((1, species * 4 + 3), dtype=int)
                            new_norev_coal[:, : species * 2] = row
                            new_norev_coal[:, species * 2 : species * 4] = (
                                ordered_norev_coal
                            )
                            new_norev_coal[:, species * 4] = state_dict[tuple(row)]
                            new_norev_coal[:, species * 4 + 1] = state_dict[
                                tuple(ordered_norev_coal)
                            ]
                            new_norev_coal[:, species * 4 + 2] = 1
                            norev_coals = np.vstack((norev_coals, new_norev_coal))
    return norev_coals


@nb.jit(nopython=True)
def number_array_1(
    state_array,
    species,
    mss,
    tuple_omegas=nb.types.Tuple((nb.types.int64, nb.types.int64)),
    tuple_states=nb.types.Tuple((nb.types.int64, nb.types.int64)),
):
    """For the 1-species CTMC, generates two dictionaries: an omega dictionary that tracks the location of each coalescence state (keys are tuples of minimum increasing substring sums) and a state dictionary mapping each state tuple to its index.

    :param state_array: array with every set partition for 1 species.
    :type state_array: np.array of int with shape (Bell number, n_species*2).
    :param species: number of species.
    :type species: int.
    :param mss: list of minimum increasing substring sums.
    :type mss: list[int].
    :param tuple_omegas: nb.types.Tuple for the omega dictionary keys.
    :type tuple_omegas: nb.types.Tuple.
    :param tuple_states: nb.types.Tuple for the state dictionary keys.
    :type tuple_states: nb.types.Tuple.
    :return: omega dictionary and state dictionary.
    :rtype: tuple."""
    total_states = bell_numbers(2 * species)
    state_dict = nb.typed.Dict.empty(key_type=tuple_states, value_type=nb.types.int64)
    omega_dict = nb.typed.Dict.empty(
        key_type=tuple_omegas, value_type=np.zeros(total_states, dtype=nb.types.boolean)
    )
    max_index = 2 * species + 1
    for i, row in enumerate(state_array):
        state_tuple = (row[0], row[1])
        l_nucl = row[:species]
        l_nucl_counts = nb.typed.Dict.empty(
            key_type=nb.types.int64, value_type=nb.types.int64[:]
        )
        index_tracker = np.zeros((max_index, len(l_nucl)), dtype=nb.types.int64) - 1
        counts = np.zeros(max_index, dtype=np.int32)
        for index, number in enumerate(l_nucl):
            current_count = counts[number]
            index_tracker[number, current_count] = index
            counts[number] += 1
        number_entries = np.sum(counts > 0)
        results_numbers = np.zeros(number_entries, dtype=nb.types.int64)
        results_indices = []
        result_index = 0
        for number in range(max_index):
            if counts[number] > 0:
                results_numbers[result_index] = number
                results_indices.append(index_tracker[number, : counts[number]])
                result_index += 1
        for number, indices in zip(results_numbers, results_indices):
            l_nucl_counts[number] = indices
        r_nucl = row[species:]
        r_nucl_counts = nb.typed.Dict.empty(
            key_type=nb.types.int64, value_type=nb.types.int64[:]
        )
        index_tracker = np.zeros((max_index, len(r_nucl)), dtype=nb.types.int64) - 1
        counts = np.zeros(max_index, dtype=nb.types.int64)
        for index, number in enumerate(r_nucl):
            current_count = counts[number]
            index_tracker[number, current_count] = index
            counts[number] += 1
        number_entries = np.sum(counts > 0)
        results_numbers = np.zeros(number_entries, dtype=nb.types.int64)
        results_indices = []
        result_index = 0
        for number in range(max_index):
            if counts[number] > 0:
                results_numbers[result_index] = number
                results_indices.append(index_tracker[number, : counts[number]])
                result_index += 1
        for number, indices in zip(results_numbers, results_indices):
            r_nucl_counts[number] = indices
        l_omega = 0
        r_omega = 0
        for j in set(l_nucl):
            if len(l_nucl_counts[j]) > 1:
                for k in l_nucl_counts[j]:
                    l_omega += mss[k]
        for j in set(r_nucl):
            if len(r_nucl_counts[j]) > 1:
                for k in r_nucl_counts[j]:
                    r_omega += mss[k]
        state_dict[state_tuple] = i
        if (l_omega, r_omega) not in omega_dict:
            omega_dict[(l_omega, r_omega)] = np.zeros(
                total_states, dtype=nb.types.boolean
            )
            omega_dict[(l_omega, r_omega)][i] = True
        else:
            omega_dict[(l_omega, r_omega)][i] = True
    return omega_dict, state_dict


@nb.jit(nopython=True)
def number_array_2(
    state_array,
    species,
    mss,
    tuple_omegas=nb.types.Tuple((nb.types.int64, nb.types.int64)),
    tuple_states=nb.types.Tuple(
        (nb.types.int64, nb.types.int64, nb.types.int64, nb.types.int64)
    ),
):
    """For the 2-species CTMC, generates two dictionaries: an omega dictionary tracking the location of each coalescence state (keys are tuples of minimum increasing substring sums) and a state dictionary mapping each state tuple to its index.

    :param state_array: array with every set partition for 2 species.
    :type state_array: np.array of int with shape (Bell number, n_species*2).
    :param species: number of species.
    :type species: int.
    :param mss: list of minimum increasing substring sums.
    :type mss: list[int].
    :param tuple_omegas: nb.types.Tuple for omega dictionary keys.
    :type tuple_omegas: nb.types.Tuple.
    :param tuple_states: nb.types.Tuple for state dictionary keys.
    :type tuple_states: nb.types.Tuple.
    :return: omega dictionary and state dictionary.
    :rtype: tuple."""
    total_states = bell_numbers(2 * species)
    state_dict = nb.typed.Dict.empty(key_type=tuple_states, value_type=nb.types.int64)
    omega_dict = nb.typed.Dict.empty(
        key_type=tuple_omegas, value_type=np.zeros(total_states, dtype=nb.types.boolean)
    )
    max_index = 2 * species + 1
    for i, row in enumerate(state_array):
        state_tuple = (row[0], row[1], row[2], row[3])
        l_nucl = row[:species]
        l_nucl_counts = nb.typed.Dict.empty(
            key_type=nb.types.int64, value_type=nb.types.int64[:]
        )
        index_tracker = np.zeros((max_index, len(l_nucl)), dtype=nb.types.int64) - 1
        counts = np.zeros(max_index, dtype=np.int32)
        for index, number in enumerate(l_nucl):
            current_count = counts[number]
            index_tracker[number, current_count] = index
            counts[number] += 1
        number_entries = np.sum(counts > 0)
        results_numbers = np.zeros(number_entries, dtype=nb.types.int64)
        results_indices = []
        result_index = 0
        for number in range(max_index):
            if counts[number] > 0:
                results_numbers[result_index] = number
                results_indices.append(index_tracker[number, : counts[number]])
                result_index += 1
        for number, indices in zip(results_numbers, results_indices):
            l_nucl_counts[number] = indices
        r_nucl = row[species:]
        r_nucl_counts = nb.typed.Dict.empty(
            key_type=nb.types.int64, value_type=nb.types.int64[:]
        )
        index_tracker = np.zeros((max_index, len(r_nucl)), dtype=nb.types.int64) - 1
        counts = np.zeros(max_index, dtype=nb.types.int64)
        for index, number in enumerate(r_nucl):
            current_count = counts[number]
            index_tracker[number, current_count] = index
            counts[number] += 1
        number_entries = np.sum(counts > 0)
        results_numbers = np.zeros(number_entries, dtype=nb.types.int64)
        results_indices = []
        result_index = 0
        for number in range(max_index):
            if counts[number] > 0:
                results_numbers[result_index] = number
                results_indices.append(index_tracker[number, : counts[number]])
                result_index += 1
        for number, indices in zip(results_numbers, results_indices):
            r_nucl_counts[number] = indices
        l_omega = 0
        r_omega = 0
        for j in set(l_nucl):
            if len(l_nucl_counts[j]) > 1:
                for k in l_nucl_counts[j]:
                    l_omega += mss[k]
        for j in set(r_nucl):
            if len(r_nucl_counts[j]) > 1:
                for k in r_nucl_counts[j]:
                    r_omega += mss[k]
        state_dict[state_tuple] = i
        if (l_omega, r_omega) not in omega_dict:
            omega_dict[(l_omega, r_omega)] = np.zeros(
                total_states, dtype=nb.types.boolean
            )
            omega_dict[(l_omega, r_omega)][i] = True
        else:
            omega_dict[(l_omega, r_omega)][i] = True
    return omega_dict, state_dict


@nb.jit(nopython=True)
def number_array_3(
    state_array,
    species,
    mss,
    tuple_omegas=nb.types.Tuple((nb.types.int64, nb.types.int64)),
    tuple_states=nb.types.Tuple(
        (
            nb.types.int64,
            nb.types.int64,
            nb.types.int64,
            nb.types.int64,
            nb.types.int64,
            nb.types.int64,
        )
    ),
):
    """For the 3-species CTMC, generates two dictionaries: an omega dictionary tracking the location of each coalescence state (keys are tuples of minimum increasing substring sums) and a state dictionary mapping each state tuple to its index.

    :param state_array: array with every set partition for 3 species.
    :type state_array: np.array of int with shape (Bell number, n_species*2).
    :param species: number of species.
    :type species: int.
    :param mss: list of minimum increasing substring sums.
    :type mss: list[int].
    :param tuple_omegas: nb.types.Tuple for omega dictionary keys.
    :type tuple_omegas: nb.types.Tuple.
    :param tuple_states: nb.types.Tuple for state dictionary keys.
    :type tuple_states: nb.types.Tuple.
    :return: omega dictionary and state dictionary.
    :rtype: tuple."""
    total_states = bell_numbers(2 * species)
    state_dict = nb.typed.Dict.empty(key_type=tuple_states, value_type=nb.types.int64)
    omega_dict = nb.typed.Dict.empty(
        key_type=tuple_omegas, value_type=np.zeros(total_states, dtype=nb.types.boolean)
    )
    max_index = 2 * species + 1
    for i, row in enumerate(state_array):
        state_tuple = (row[0], row[1], row[2], row[3], row[4], row[5])
        l_nucl = row[:species]
        l_nucl_counts = nb.typed.Dict.empty(
            key_type=nb.types.int64, value_type=nb.types.int64[:]
        )
        index_tracker = np.zeros((max_index, len(l_nucl)), dtype=nb.types.int64) - 1
        counts = np.zeros(max_index, dtype=np.int32)
        for index, number in enumerate(l_nucl):
            current_count = counts[number]
            index_tracker[number, current_count] = index
            counts[number] += 1
        number_entries = np.sum(counts > 0)
        results_numbers = np.zeros(number_entries, dtype=nb.types.int64)
        results_indices = []
        result_index = 0
        for number in range(max_index):
            if counts[number] > 0:
                results_numbers[result_index] = number
                results_indices.append(index_tracker[number, : counts[number]])
                result_index += 1
        for number, indices in zip(results_numbers, results_indices):
            l_nucl_counts[number] = indices
        r_nucl = row[species:]
        r_nucl_counts = nb.typed.Dict.empty(
            key_type=nb.types.int64, value_type=nb.types.int64[:]
        )
        index_tracker = np.zeros((max_index, len(r_nucl)), dtype=nb.types.int64) - 1
        counts = np.zeros(max_index, dtype=nb.types.int64)
        for index, number in enumerate(r_nucl):
            current_count = counts[number]
            index_tracker[number, current_count] = index
            counts[number] += 1
        number_entries = np.sum(counts > 0)
        results_numbers = np.zeros(number_entries, dtype=nb.types.int64)
        results_indices = []
        result_index = 0
        for number in range(max_index):
            if counts[number] > 0:
                results_numbers[result_index] = number
                results_indices.append(index_tracker[number, : counts[number]])
                result_index += 1
        for number, indices in zip(results_numbers, results_indices):
            r_nucl_counts[number] = indices
        l_omega = 0
        r_omega = 0
        for j in set(l_nucl):
            if len(l_nucl_counts[j]) > 1:
                for k in l_nucl_counts[j]:
                    l_omega += mss[k]
        for j in set(r_nucl):
            if len(r_nucl_counts[j]) > 1:
                for k in r_nucl_counts[j]:
                    r_omega += mss[k]
        state_dict[state_tuple] = i
        if (l_omega, r_omega) not in omega_dict:
            omega_dict[(l_omega, r_omega)] = np.zeros(
                total_states, dtype=nb.types.boolean
            )
            omega_dict[(l_omega, r_omega)][i] = True
        else:
            omega_dict[(l_omega, r_omega)][i] = True
    return omega_dict, state_dict


def get_trans_mat(transition_mat, species, coal, rho):
    """Computes the CTMC transition probability matrix given a transition matrix, number of species, coal rate, and recombination rate; the matrix size is determined by the Bell number for 2*species and the diagonal is set to the negative sum of the off-diagonals.

    :param transition_mat: array representing the transition events.
    :type transition_mat: np.array.
    :param species: number of species.
    :type species: int.
    :param coal: rate for non-reversible coalescence.
    :type coal: float.
    :param rho: rate for reversible events (recombination).
    :type rho: float.
    :return: transition probability matrix.
    :rtype: np.array of float64."""
    mat_size = bell_numbers(2 * species)
    trans_prob_array = np.zeros((mat_size, mat_size), dtype=np.float64)
    for row in transition_mat:
        x_value = row[species * 4 + 1]
        y_value = row[species * 4]
        trans_prob_array[y_value, x_value] = rho if row[species * 4 + 2] == 2 else coal
    for i in range(mat_size):
        trans_prob_array[i, i] = -sum(trans_prob_array[i])
    return trans_prob_array


def get_omega_nonrev_counts(species):
    """Computes and returns a dictionary mapping each omega value (sum of selected mss values) to its corresponding non-reversible coalescence count (subset size minus one).

    :param species: number of species.
    :type species: int.
    :return: dictionary mapping omega values to non-reversible counts.
    :rtype: nb.typed.Dict."""
    omega_nonrev_counts = nb.typed.Dict.empty(
        key_type=nb.types.int64, value_type=nb.types.int64
    )
    omega_nonrev_counts[0] = 0
    mss = [2**i for i in range(species)]
    for size in range(2, len(mss) + 1):
        for subset in it.combinations(mss, size):
            omega_nonrev_counts[sum(subset)] = len(subset) - 1
    return omega_nonrev_counts


def wrapper_state_1():
    """Wrapper function that returns the transition matrix, omega dictionary, state dictionary, and omega non-reversible counts for 1 species.

    :return: tuple containing transition matrix, omega dictionary, state dictionary, and omega non-reversible counts for 1 species.
    :rtype: tuple."""
    species = 1
    mss = [2**i for i in range(species)]
    state_array_1 = set_partitions(species)
    omega_dict_1, state_dict_1 = number_array_1(state_array_1, species, mss)
    omega_nonrev_counts_1 = get_omega_nonrev_counts(species)
    coal_and_rec_1 = find_revcoal_recomb(state_array_1, species, state_dict_1)
    norev_coals_1 = find_norevcoal(state_array_1, species, state_dict_1)
    transitions_1 = np.vstack((coal_and_rec_1, norev_coals_1))
    return transitions_1, omega_dict_1, state_dict_1, omega_nonrev_counts_1


def wrapper_state_2():
    """Wrapper function that returns the transition matrix, omega dictionary, state dictionary, and omega non-reversible counts for 2 species.

    :return: tuple containing transition matrix, omega dictionary, state dictionary, and omega non-reversible counts for 2 species.
    :rtype: tuple."""
    species = 2
    mss = [2**i for i in range(species)]
    state_array_2 = set_partitions(species)
    omega_dict_2, state_dict_2 = number_array_2(state_array_2, species, mss)
    omega_nonrev_counts_2 = get_omega_nonrev_counts(species)
    coal_and_rec_2 = find_revcoal_recomb(state_array_2, species, state_dict_2)
    norev_coals_2 = find_norevcoal(state_array_2, species, state_dict_2)
    transitions_2 = np.vstack((coal_and_rec_2, norev_coals_2))
    return transitions_2, omega_dict_2, state_dict_2, omega_nonrev_counts_2


def wrapper_state_3():
    """Wrapper function that returns the transition matrix, omega dictionary, state dictionary, and omega non-reversible counts for 3 species.

    :return: tuple containing transition matrix, omega dictionary, state dictionary, and omega non-reversible counts for 3 species.
    :rtype: tuple."""
    species = 3
    mss = [2**i for i in range(species)]
    state_array_3 = set_partitions(species)
    omega_dict_3, state_dict_3 = number_array_3(state_array_3, species, mss)
    omega_nonrev_counts_3 = get_omega_nonrev_counts(species)
    coal_and_rec_3 = find_revcoal_recomb(state_array_3, species, state_dict_3)
    norev_coals_3 = find_norevcoal(state_array_3, species, state_dict_3)
    transitions_3 = np.vstack((coal_and_rec_3, norev_coals_3))
    return transitions_3, omega_dict_3, state_dict_3, omega_nonrev_counts_3


def wrapper_state_general(species):
    """Wrapper function that returns the transition matrix, omega dictionary, state dictionary, and omega non-reversible counts for n species (where n must be 1, 2, or 3).

    :param species: number of species (must be 1, 2, or 3).
    :type species: int.
    :return: tuple containing transition matrix, omega dictionary, state dictionary, and omega non-reversible counts.
    :rtype: tuple."""
    mss = [2**i for i in range(species)]
    state_array = set_partitions(species)
    if species == 1:
        omega_dict, state_dict = number_array_1(state_array, species, mss)
    elif species == 2:
        omega_dict, state_dict = number_array_2(state_array, species, mss)
    elif species == 3:
        omega_dict, state_dict = number_array_3(state_array, species, mss)
    else:
        raise ValueError("Species must be 1, 2 or 3")
    omega_nonrev_counts = get_omega_nonrev_counts(species)
    coal_and_rec = find_revcoal_recomb(state_array, species, state_dict)
    norev_coals = find_norevcoal(state_array, species, state_dict)
    transitions = np.vstack((coal_and_rec, norev_coals))
    return transitions, omega_dict, state_dict, omega_nonrev_counts
