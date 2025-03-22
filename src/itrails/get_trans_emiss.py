import numpy as np

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
    cut_AB="standard",
    cut_ABC="standard",
):
    """
    This function calculates the emission and transition probabilities
    given a certain set of parameters.

    Parameters
    ----------
    t_A : numeric
        Time in generations from present to the first speciation event for species A
        (times mutation rate)
    t_B : numeric
        Time in generations from present to the first speciation event for species B
        (times mutation rate)
    t_C : numeric
        Time in generations from present to the second speciation event for species C
        (times mutation rate)
    t_2 : numeric
        Time in generations from the first speciation event to the second speciation event
        (times mutation rate)
    t_upper : numeric
        Time in generations between the end of the second-to-last interval and the third
        speciation event (times mutation rate)
    t_out : numeric
        Time in generations from present to the third speciation event for species D, plus
        the divergence between the ancestor of D and the ancestor of A, B and C at the time
        of the third speciation event (times mutation rate)
    N_AB : numeric
        Effective population size between speciation events (times mutation rate)
    N_ABC : numeric
        Effective population size in deep coalescence, before the second speciation event
        (times mutation rate)
    r : numeric
        Recombination rate per site per generation (divided by mutation rate)
    n_int_AB : integer
        Number of discretized time intervals between speciation events
    n_int_ABC : integer
        Number of discretized time intervals in deep coalescent
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
    # Convert dictionary to DataFrame

    # Get all unique states
    unique_states = sorted(set(state for pair in tr_dict.keys() for state in pair))

    # Create mapping from states to indices
    state_to_index = {state: i for i, state in enumerate(unique_states)}
    # index_to_state = {i: state for state, i in state_to_index.items()}  # Reverse mapping
    hidden_names = {
        i: state for i, state in enumerate(unique_states)
    }  # Equivalent to index_to_state
    # Initialize an empty transition matrix
    n_states = len(unique_states)
    transition_matrix = np.zeros((n_states, n_states))

    # Fill the matrix with probabilities
    for (from_state, to_state), prob in tr_dict.items():
        from_idx = state_to_index[from_state]
        to_idx = state_to_index[to_state]
        transition_matrix[from_idx, to_idx] = prob

    pi = transition_matrix.sum(axis=1)

    # Avoid division by zero
    a = np.divide(transition_matrix, pi, where=pi != 0)

    # Get emissions using the modified function (which now returns lists)
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
    # Sort emissions by hidden state (assuming hidden_states can be compared)
    sorted_data = sorted(zip(hidden_states, emission_dicts), key=lambda x: x[0])
    sorted_states, sorted_emissions = zip(*sorted_data)
    hidden_names = {i: state for i, state in enumerate(sorted_states)}
    # Assume all emission dictionaries have the same keys.
    observed_keys = sorted(list(sorted_emissions[0].keys()))
    observed_names = {i: key for i, key in enumerate(observed_keys)}
    # Build emission matrix 'b': each row corresponds to a hidden state and columns follow the order in observed_keys.
    b = np.array([[em[key] for key in observed_keys] for em in sorted_emissions])

    return a, b, pi, hidden_names, observed_names
