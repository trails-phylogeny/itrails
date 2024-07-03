import numpy as np
import numba
from numba import jit
from numba.typed import Dict
from numba.types import Tuple, int64, float64, boolean
from itertools import combinations


@jit(nopython=True)
def bell_numbers(n):
    """
    Given a number 'n', this function returns the n'th Bell Number,
    used for initializing the matrices with the correct number of rows.

    :param n: Number for which the Bell number is returned.
    :type n: int
    :return: The n'th Bell Number.
    :rtype: int
    """
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
    """
    Generator that creates all the set partitions. From a list of
    consecutive numbers ranging from 1 to the number of species,
    it generates nested lists with all the possible partitions of
    the n elements in the list in, at most, n subsets.

    The set partitions do have format such as [1,2,3,4,5,6] or
    [[1,2,3][4,5,6]].

    :param collection: List that ranges from 1 to double the number of species for the CTMC.
    :type collection: list[int]
    :return: Nested lists with all the possible partitions of the n elements in the list in, at most, n subsets.
    :rtype: list[list[int]]
    """
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
        yield [[first]] + smaller


def set_partitions(species):
    """
    Function that gets the set partitions for a certain number
    of species in the CTMC. It has nested the bell_numbers
    function and nested_list_partition generator. It returns a
    numpy array with reformatted set partitions.

    Each number in the reformatted set partitions represents one
    position for the nucleotides. In the sense that partition
    [[1,2,3][4,5,6]] is converted into 1,1,1,2,2,2.

    :param species: Number of species for which we want to generate the partitions.
    :type species: int
    :return: Array with the set partitions for the given number of species.
    :rtype: np.array
    """
    num_rows = bell_numbers(2 * species)
    num_cols = 2 * species
    state_array = np.zeros((num_rows, num_cols), dtype=int)
    for n, p in enumerate(partition(list(range(1, 2 * species + 1))), 1):
        for j, subsublist in enumerate(sorted(p)):
            for value in subsublist:
                state_array[n - 1][value - 1] = j + 1
    return state_array


# @jit(nopython=True)
def translate_to_minimum(array):
    """
    Function that reformats the set partitions so that each of
    them always have the minimum values as they appear. After
    functions find_rec_revcoal and find_norevcoal set partitions
    are not ensured to mantain the consecutive nature.

    For example, the set partition 1,2,2,2,3,3 could appear as
    1,2,2,2,4,4. This function ensures that those partitions are
    reformatted as they should.

    :param array: Set partition to reformat.
    :type array: np.array of int and shape (1, n_species*2)
    :return: Set partition with the minimum values as they appear.
    :rtype: np.array of int and shape (1, n_species*2)
    """
    unique_values = np.unique(array)
    value_map = {value: i + 1 for i, value in enumerate(np.sort(unique_values))}
    translated_array = np.array([value_map[value] for value in array])
    return translated_array


def find_revcoal_recomb(state_array, species, state_dict):
    """
    Function that takes the state array and number of
    species as arguments, then returns all the possible
    reversible coalescences for each state, because of
    the symmetry of reversible coalescences and recombinations
    it finds both.

    It generates an array of species * 4 columns
    in which the first species * 2 are the states before a
    reversible coalescence/after recombination and the last
    species * 2 columns are the states after a reversible
    coalescence/before a recombination.

    :param state_array: Set partition of said number of species.
    :type state_array: np.array of int and shape (n'th bell_number, n_species*2)
    :param species: Number of species.
    :type species: int
    :param state_dict: Dictionary with tuple of each state as keys and corresponding number as values.
    :type state_dict: dict
    :return: Array with the reversible coalescences and recombinations.
    :rtype: np.array
    """
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
    """
    Function that takes the state array and number of
    species as arguments, then returns all the possible
    non reversible coalescences for each state.

    It generates an array of species * 4 columns
    in which the first species * 2 are the states
    before a non reversible coalescence and the last
    species * 2 columns are the states after a non
    reversible coalescence/before a recombination.

    :param state_array: Set partition of said number of species.
    :type state_array: np.array of int and shape (n'th bell_number, n_species*2)
    :param species: Number of species.
    :type species: int
    :param state_dict: Dictionary with tuple of each state as keys and corresponding number as values.
    :type state_dict: dict[]
    :return: Array with the non reversible coalescences.
    :rtype: np.array
    """
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


# def sums_mss(mss):
#     all_sums = [0]
#     for r in range(2, len(mss) + 1):
#         for combo in combinations(mss, r):
#             all_sums.append(sum(combo))
#     return all_sums


@jit(nopython=True)
def number_array_1(
    state_array,
    species,
    mss,
    tuple_omegas=Tuple((int64, int64)),
    tuple_states=Tuple((int64, int64)),
):
    """
    Function that, for the 1 species CTMC, generates two dictionaries.
    Omega dict will keep track of where in the transition matrix each
    coalescence state is located (keys are tuples of minimum incresing
    substring sums, values are integers).
    State dict will keep track of which number does each state take in
    the matrix (keys are tuples of states, values are integers).

    :param state_array: Array with every set partition of 1 species.
    :type state_array: np.array of int and shape (n'th bell_number, n_species*2)
    :param species: Number of species.
    :type species: int
    :param mss: List with the minimum increasing substring sums.
    :type mss: list[int]
    :param tuple_omegas: Tuple type for the omega dictionary, defined for the typed dictionary.
    :type tuple_omegas: Tuple
    :param tuple_states: Tuple type for states, defined for the typed dictionary.
    :type tuple_states: Tuple
    """
    total_states = bell_numbers(2 * species)
    state_dict = Dict.empty(
        key_type=tuple_states,
        value_type=int64,
    )
    omega_dict = Dict.empty(
        key_type=tuple_omegas,
        value_type=np.zeros(total_states, dtype=boolean),
    )

    max_index = 2 * species + 1
    for i, row in enumerate(state_array):
        state_tuple = (row[0], row[1])
        l_nucl = row[:species]
        l_nucl_counts = Dict.empty(
            key_type=int64,
            value_type=int64[:],
        )
        index_tracker = np.zeros((max_index, len(l_nucl)), dtype=int64) - 1

        counts = np.zeros(max_index, dtype=np.int32)
        for index, number in enumerate((l_nucl)):
            current_count = counts[number]
            index_tracker[number, current_count] = index
            counts[number] += 1
        number_entries = np.sum(counts > 0)
        results_numbers = np.zeros(number_entries, dtype=int64)
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
        r_nucl_counts = Dict.empty(
            key_type=int64,
            value_type=int64[:],
        )
        index_tracker = np.zeros((max_index, len(r_nucl)), dtype=int64) - 1
        counts = np.zeros(max_index, dtype=int64)
        for index, number in enumerate((r_nucl)):
            current_count = counts[number]
            index_tracker[number, current_count] = index
            counts[number] += 1

        number_entries = np.sum(counts > 0)
        results_numbers = np.zeros(number_entries, dtype=int64)
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
                    l_omega += mss[k - 1]

        for j in set(r_nucl):
            if len(r_nucl_counts[j]) > 1:
                for k in r_nucl_counts[j]:
                    r_omega += mss[k - 1]

        state_dict[state_tuple] = i

        if (l_omega, r_omega) not in omega_dict:
            omega_dict[(l_omega, r_omega)] = np.zeros(total_states, dtype=boolean)
            omega_dict[(l_omega, r_omega)][i] = True
        else:
            omega_dict[(l_omega, r_omega)][i] = True

    return omega_dict, state_dict


@jit(nopython=True)
def number_array_2(
    state_array,
    species,
    mss,
    tuple_omegas=Tuple((int64, int64)),
    tuple_states=Tuple((int64, int64, int64, int64)),
):
    """
    Function that, for the 2 species CTMC, generates two dictionaries.
    Omega dict will keep track of where in the transition matrix each
    coalescence state is located (keys are tuples of minimum incresing
    substring sums, values are integers).
    State dict will keep track of which number does each state take in
    the matrix (keys are tuples of states, values are integers).

    :param state_array: Array with every set partition of 2 species.
    :type state_array: np.array of int and shape (n'th bell_number, n_species*2)
    :param species: Number of species.
    :type species: int
    :param mss: List with the minimum increasing substring sums.
    :type mss: list[int]
    :param tuple_omegas: Tuple type for the omega dictionary, defined for the typed dictionary.
    :type tuple_omegas: Tuple
    :param tuple_states: Tuple type for states, defined for the typed dictionary.
    :type tuple_states: Tuple
    """
    total_states = bell_numbers(2 * species)
    state_dict = Dict.empty(
        key_type=tuple_states,
        value_type=int64,
    )
    omega_dict = Dict.empty(
        key_type=tuple_omegas,
        value_type=np.zeros(total_states, dtype=boolean),
    )

    max_index = 2 * species + 1
    for i, row in enumerate(state_array):
        state_tuple = (row[0], row[1], row[2], row[3])
        l_nucl = row[:species]
        l_nucl_counts = Dict.empty(
            key_type=int64,
            value_type=int64[:],
        )
        index_tracker = np.zeros((max_index, len(l_nucl)), dtype=int64) - 1

        counts = np.zeros(max_index, dtype=np.int32)
        for index, number in enumerate((l_nucl)):
            current_count = counts[number]
            index_tracker[number, current_count] = index
            counts[number] += 1
        number_entries = np.sum(counts > 0)
        results_numbers = np.zeros(number_entries, dtype=int64)
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
        r_nucl_counts = Dict.empty(
            key_type=int64,
            value_type=int64[:],
        )
        index_tracker = np.zeros((max_index, len(r_nucl)), dtype=int64) - 1
        counts = np.zeros(max_index, dtype=int64)
        for index, number in enumerate((r_nucl)):
            current_count = counts[number]
            index_tracker[number, current_count] = index
            counts[number] += 1

        number_entries = np.sum(counts > 0)
        results_numbers = np.zeros(number_entries, dtype=int64)
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
                    l_omega += mss[k - 1]

        for j in set(r_nucl):
            if len(r_nucl_counts[j]) > 1:
                for k in r_nucl_counts[j]:
                    r_omega += mss[k - 1]

        state_dict[state_tuple] = i

        if (l_omega, r_omega) not in omega_dict:
            omega_dict[(l_omega, r_omega)] = np.zeros(total_states, dtype=boolean)
            omega_dict[(l_omega, r_omega)][i] = True
        else:
            omega_dict[(l_omega, r_omega)][i] = True

    return omega_dict, state_dict


@jit(nopython=True)
def number_array_3(
    state_array,
    species,
    mss,
    tuple_omegas=Tuple((int64, int64)),
    tuple_states=Tuple((int64, int64, int64, int64, int64, int64)),
):
    """
    Function that, for the 3 species CTMC, generates two dictionaries.
    Omega dict will keep track of where in the transition matrix each
    coalescence state is located (keys are tuples of minimum incresing
    substring sums, values are integers).
    State dict will keep track of which number does each state take in
    the matrix (keys are tuples of states, values are integers).

    :param state_array: Array with every set partition of 3 species.
    :type state_array: np.array of int and shape (n'th bell_number, n_species*2)
    :param species: Number of species.
    :type species: int
    :param mss: List with the minimum increasing substring sums.
    :type mss: list[int]
    :param tuple_omegas: Tuple type for the omega dictionary, defined for the typed dictionary.
    :type tuple_omegas: Tuple
    :param tuple_states: Tuple type for states, defined for the typed dictionary.
    :type tuple_states: Tuple
    """
    total_states = bell_numbers(2 * species)
    state_dict = Dict.empty(
        key_type=tuple_states,
        value_type=int64,
    )
    omega_dict = Dict.empty(
        key_type=tuple_omegas,
        value_type=np.zeros(total_states, dtype=boolean),
    )

    max_index = 2 * species + 1
    for i, row in enumerate(state_array):
        state_tuple = (row[0], row[1], row[2], row[3], row[4], row[5])
        l_nucl = row[:species]
        l_nucl_counts = Dict.empty(
            key_type=int64,
            value_type=int64[:],
        )
        index_tracker = np.zeros((max_index, len(l_nucl)), dtype=int64) - 1

        counts = np.zeros(max_index, dtype=np.int32)
        for index, number in enumerate((l_nucl)):
            current_count = counts[number]
            index_tracker[number, current_count] = index
            counts[number] += 1
        number_entries = np.sum(counts > 0)
        results_numbers = np.zeros(number_entries, dtype=int64)
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
        r_nucl_counts = Dict.empty(
            key_type=int64,
            value_type=int64[:],
        )
        index_tracker = np.zeros((max_index, len(r_nucl)), dtype=int64) - 1
        counts = np.zeros(max_index, dtype=int64)
        for index, number in enumerate((r_nucl)):
            current_count = counts[number]
            index_tracker[number, current_count] = index
            counts[number] += 1

        number_entries = np.sum(counts > 0)
        results_numbers = np.zeros(number_entries, dtype=int64)
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
                    l_omega += mss[k - 1]

        for j in set(r_nucl):
            if len(r_nucl_counts[j]) > 1:
                for k in r_nucl_counts[j]:
                    r_omega += mss[k - 1]

        state_dict[state_tuple] = i

        if (l_omega, r_omega) not in omega_dict:
            omega_dict[(l_omega, r_omega)] = np.zeros(total_states, dtype=boolean)
            omega_dict[(l_omega, r_omega)][i] = True
        else:
            omega_dict[(l_omega, r_omega)][i] = True

    return omega_dict, state_dict


def get_trans_mat(transition_mat, species, coal, rho):
    mat_size = bell_numbers(2 * species)
    trans_prob_array = np.zeros((mat_size, mat_size), dtype=np.float64)

    for row in transition_mat:
        x_value = row[species * 4 + 1]
        y_value = row[species * 4]
        trans_prob_array[y_value, x_value] = rho if row[species * 4 + 2] == 2 else coal

    for i in range(mat_size):
        trans_prob_array[i, i] = -sum(trans_prob_array[i])
    return trans_prob_array


def wrapper_state_1():
    species = 1
    mss = [2**i for i in range(species)]
    state_array_1 = set_partitions(species)
    omega_dict_1, state_dict_1 = number_array_1(state_array_1, species, mss)
    coal_and_rec_1 = find_revcoal_recomb(state_array_1, species, state_dict_1)
    norev_coals_1 = find_norevcoal(state_array_1, species, state_dict_1)
    transitions_1 = np.vstack((coal_and_rec_1, norev_coals_1))
    return transitions_1, omega_dict_1, state_dict_1


def wrapper_state_2():
    species = 2
    mss = [2**i for i in range(species)]
    state_array_2 = set_partitions(species)
    omega_dict_2, state_dict_2 = number_array_2(state_array_2, species, mss)
    coal_and_rec_2 = find_revcoal_recomb(state_array_2, species, state_dict_2)
    norev_coals_2 = find_norevcoal(state_array_2, species, state_dict_2)
    transitions_2 = np.vstack((coal_and_rec_2, norev_coals_2))
    return transitions_2, omega_dict_2, state_dict_2


def wrapper_state_3():
    species = 3
    mss = [2**i for i in range(species)]
    state_array_3 = set_partitions(species)
    omega_dict_3, state_dict_3 = number_array_3(state_array_3, species, mss)
    coal_and_rec_3 = find_revcoal_recomb(state_array_3, species, state_dict_3)
    norev_coals_3 = find_norevcoal(state_array_3, species, state_dict_3)
    transitions_3 = np.vstack((coal_and_rec_3, norev_coals_3))
    return transitions_3, omega_dict_3, state_dict_3
