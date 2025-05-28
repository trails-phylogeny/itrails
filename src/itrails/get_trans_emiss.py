import numpy as np

from itrails.cutpoints import cutpoints_AB, cutpoints_ABC
from itrails.get_emission_prob_mat import get_emission_prob_mat
from itrails.get_joint_prob_mat import get_joint_prob_mat


def trans_emiss_calc(
    t_A,
    t_B,
    t_C,
    t_2,
    t_upper,
    t_out,
    N_AB,
    N_ABC,
    r,
    n_int_AB,
    n_int_ABC,
    cut_AB,
    cut_ABC,
):
    """
    Calculate the emission and transition probabilities given a set of parameters.

    :param t_A: Time in generations from the present to the first speciation event for species A (times mutation rate)
    :type t_A: numeric
    :param t_B: Time in generations from the present to the first speciation event for species B (times mutation rate)
    :type t_B: numeric
    :param t_C: Time in generations from the present to the second speciation event for species C (times mutation rate)
    :type t_C: numeric
    :param t_2: Time in generations from the first speciation event to the second speciation event (times mutation rate)
    :type t_2: numeric
    :param t_upper: Time in generations between the end of the second-to-last interval and the third speciation event (times mutation rate)
    :type t_upper: numeric
    :param t_out: Time in generations from the present to the third speciation event for species D, plus the divergence between the ancestor of D and the ancestor of A, B, and C at the time of the third speciation event (times mutation rate)
    :type t_out: numeric
    :param N_AB: Effective population size between speciation events (times mutation rate)
    :type N_AB: numeric
    :param N_ABC: Effective population size in deep coalescence, before the second speciation event (times mutation rate)
    :type N_ABC: numeric
    :param r: Recombination rate per site per generation (divided by mutation rate)
    :type r: numeric
    :param n_int_AB: Number of discretized time intervals between speciation events
    :type n_int_AB: int
    :param n_int_ABC: Number of discretized time intervals in deep coalescent
    :type n_int_ABC: int
    :param cut_AB: Cutpoints in the A-B process.
    :type cut_AB: Array-like
    :param cut_ABC: Cutpoints in the A-B-C process.
    :type cut_ABC: Array-like

    :return: A tuple containing:
             - **a**: Transition probability matrix.
             - **b**: Emission probability matrix.
             - **pi**: Vector of starting probabilities of the hidden states.
             - **hidden_names**: Mapping from indices to hidden state names.
             - **observed_names**: Mapping from indices to observed names.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, dict, dict)
    """

    # Reference Ne (for normalization)
    N_ref = N_ABC
    # Speciation times (in coalescent units, i.e. number of generations / N_ref)
    t_A = t_A / N_ref
    t_B = t_B / N_ref
    t_AB = t_2 / N_ref
    t_C = t_C / N_ref
    t_upper = t_upper / N_ref
    t_out = t_out / N_ref
    # Recombination rates (r = rec. rate per site per generation)
    rho_A = N_ref * r
    rho_B = N_ref * r
    rho_AB = N_ref * r
    rho_C = N_ref * r
    rho_ABC = N_ref * r
    # Coalescent rates
    coal_A = N_ref / N_AB
    coal_B = N_ref / N_AB
    coal_AB = N_ref / N_AB
    coal_C = N_ref / N_AB
    coal_ABC = N_ref / N_ABC
    # Mutation rates (mu = mut. rate per site per generation)
    mu_A = N_ref * (4 / 3)
    mu_B = N_ref * (4 / 3)
    mu_C = N_ref * (4 / 3)
    mu_D = N_ref * (4 / 3)
    mu_AB = N_ref * (4 / 3)
    mu_ABC = N_ref * (4 / 3)

    if isinstance(cut_AB, str):
        if cut_AB == "standard":
            cut_AB = cutpoints_AB(n_int_AB, t_AB, coal_AB)
    if isinstance(cut_ABC, str):
        if cut_ABC == "standard":
            cut_ABC = cutpoints_ABC(n_int_ABC, coal_ABC)

    tr_dict = get_joint_prob_mat(
        t_A,
        t_B,
        t_AB,
        t_C,
        rho_A,
        rho_B,
        rho_AB,
        rho_C,
        rho_ABC,
        coal_A,
        coal_B,
        coal_AB,
        coal_C,
        coal_ABC,
        n_int_AB,
        n_int_ABC,
        cut_AB,
        cut_ABC,
    )

    hidden_states, emission_dicts = get_emission_prob_mat(
        t_A,
        t_B,
        t_AB,
        t_C,
        t_upper,
        t_out,
        rho_A,
        rho_B,
        rho_AB,
        rho_C,
        rho_ABC,
        coal_A,
        coal_B,
        coal_AB,
        coal_C,
        coal_ABC,
        n_int_AB,
        n_int_ABC,
        mu_A,
        mu_B,
        mu_C,
        mu_D,
        mu_AB,
        mu_ABC,
        cut_AB,
        cut_ABC,
    )

    observed_keys = list(emission_dicts[0].keys())
    b = np.array([[em[key] for key in observed_keys] for em in emission_dicts])
    sorted_indices = sorted(range(len(hidden_states)), key=lambda i: hidden_states[i])
    hidden_list_sorted = [hidden_states[i] for i in sorted_indices]
    emission_array_sorted = b[sorted_indices]
    sorted_hidden_dict = dict(enumerate(hidden_list_sorted))
    observed_names = {i: key for i, key in enumerate(observed_keys)}

    state_to_index = {v: k for k, v in sorted_hidden_dict.items()}
    n_states = len(hidden_states)

    transition_matrix = np.zeros((n_states, n_states))

    for (from_state, to_state), prob in tr_dict.items():
        i = state_to_index[from_state]
        j = state_to_index[to_state]
        transition_matrix[i, j] = prob

    pi = transition_matrix.sum(axis=1)

    a = transition_matrix / pi[:, None]

    return a, emission_array_sorted, pi, sorted_hidden_dict, observed_names
