import multiprocessing as mp
import os
import sys
import time

import numpy as np
import yaml
from numba import njit
from numba.typed import List
from ray.util.multiprocessing import Pool
from scipy.optimize import minimize

from itrails.cutpoints import cutpoints_ABC
from itrails.get_emission_prob_mat import (
    get_emission_prob_mat,
)
from itrails.get_joint_prob_mat import get_joint_prob_mat
from itrails.read_data import (
    get_idx_state,
    get_idx_state_new_method,
)


def forward_loglik_par(a, b, pi, V, order):
    """
    Log-likelihood (parallelized)

    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : numpy array
        Vector of observed states, as integer indices
    """
    order = List(order)
    return forward_loglik(a, b, pi, V, order)


def loglik_wrapper_par(a, b, pi, V_lst):
    """
    Log-likelihood wrapper (parallelized)

    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : list of numpy arrays
        List of vectors of observed states, as integer indices
    """
    order = list()
    for i in range(624 + 1):
        order.append(get_idx_state(i))
    pool_lst = []
    for i in range(len(V_lst)):
        pool_lst.append((a, b, pi, V_lst[i], order))
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = mp.cpu_count()
    pool = Pool(ncpus)
    res = pool.starmap_async(forward_loglik_par, pool_lst)
    acc = 0
    for i in res.get():
        acc += i
    return acc


def loglik_wrapper_par_new_method(a, b, pi, V_lst):
    """
    Log-likelihood wrapper (parallelized)

    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : list of numpy arrays
        List of vectors of observed states, as integer indices
    """
    order = list()
    for i in range(125):
        order.append(get_idx_state_new_method(i))
    pool_lst = []
    for i in range(len(V_lst)):
        pool_lst.append((a, b, pi, V_lst[i], order))
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = mp.cpu_count()
    pool = Pool(ncpus)
    res = pool.starmap_async(forward_loglik_par, pool_lst)
    acc = 0
    for i in res.get():
        acc += i
    return acc


def loglik_wrapper(a, b, pi, V_lst):
    """
    Log-likelihood wrapper.

    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : list of numpy arrays
        List of vectors of observed states, as integer indices
    """
    order = List()
    for i in range(624 + 1):
        order.append(get_idx_state(i))
    acc = 0
    prev_time = time.time()
    events_count = len(V_lst)
    for i in range(len(V_lst)):
        acc += forward_loglik(a, b, pi, V_lst[i], order)
        if (time.time() - prev_time) > 1:
            # print('{}%'.format(round(100*(i/events_count), 3)), end = '\r')
            prev_time = time.time()
    return acc


def loglik_wrapper_new_method(a, b, pi, V_lst):
    """
    Log-likelihood wrapper.

    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : list of numpy arrays
        List of vectors of observed states, as integer indices
    """
    order = List()
    for i in range(125):
        order.append(get_idx_state_new_method(i))
    acc = 0
    prev_time = time.time()
    events_count = len(V_lst)
    for i in range(len(V_lst)):
        acc += forward_loglik(a, b, pi, V_lst[i], order)
        if (time.time() - prev_time) > 1:
            # print('{}%'.format(round(100*(i/events_count), 3)), end = '\r')
            prev_time = time.time()
    return acc


@njit
def forward_loglik(a, b, pi, V, order):
    """
    Log-likelihood.

    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : numpy array
        Vector of observed states, as integer indices
    """
    alpha = forward(a, b, pi, V, order)
    x = alpha[-1, :].max()
    return np.log(np.exp(alpha[len(V) - 1] - x).sum()) + x


@njit
def forward(a, b, pi, V, order):
    """
    Forward algorithm, that allows for missing data.

    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : numpy array
        Vector of observed states, as integer indices
    """
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = np.log(pi * b[:, order[V[0]]].sum(axis=1))
    for t in range(1, V.shape[0]):
        x = alpha[t - 1, :].max()
        alpha[t, :] = (
            np.log((np.exp(alpha[t - 1] - x) @ a) * b[:, order[V[t]]].sum(axis=1)) + x
        )
    return alpha


@njit
def backward(a, b, V, order):
    """
    Backward algorithm.

    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    V : numpy array
        Vector of observed states, as integer indices
    """
    beta = np.zeros((V.shape[0], a.shape[0]))
    beta[V.shape[0] - 1] = np.zeros((a.shape[0]))
    for t in range(V.shape[0] - 2, -1, -1):
        x = beta[t + 1, :].max()
        beta[t, :] = (
            np.log((np.exp(beta[t + 1] - x) * b[:, order[V[t + 1]]].sum(axis=1)) @ a)
            + x
        )
    return beta


def post_prob(a, b, pi, V, order):
    """
    Posterior probabilities.

    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : numpy array
        Vector of observed states, as integer indices
    """
    alpha = forward(a, b, pi, V, order)
    beta = backward(a, b, V, order)
    post_prob = alpha + beta
    max_row = post_prob.max(1).reshape(-1, 1)
    post_prob = np.exp(post_prob - max_row) / np.exp(post_prob - max_row).sum(
        1
    ).reshape(-1, 1)
    return post_prob


def post_prob_wrapper(a, b, pi, V_lst):
    """
    Posterior probability wrapper.

    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : list of numpy arrays
        List of vectors of observed states, as integer indices
    """
    res_lst = []
    order = List()
    for i in range(624 + 1):
        order.append(get_idx_state(i))
    prev_time = time.time()
    events_count = len(V_lst)
    for i in range(len(V_lst)):
        res_lst.append(post_prob(a, b, pi, V_lst[i], order))
    return res_lst


@njit
def viterbi(a, b, pi, V):
    """
    Viterbi path

    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : numpy array
        Vector of observed states
    """
    T = V.shape[0]
    M = a.shape[0]
    omega = np.zeros((T, M))
    omega[0, :] = np.log(pi * b[:, V[0]])
    prev = np.zeros((T - 1, M))
    for t in range(1, T):
        for j in range(M):
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])
            prev[t - 1, j] = np.argmax(probability)
            omega[t, j] = np.max(probability)
    S = np.zeros(T)
    last_state = np.argmax(omega[T - 1, :])
    S[0] = last_state
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
    S = np.flip(S)
    return S


def write_list(lst, res_name):
    """
    This function appends a list to a csv file.

    Parameters
    ----------
    lst : list
        List of values to append
    res_name : str
        File name to append to
    """
    with open(f"{res_name}", "a") as f:
        for i in range(len(lst)):
            f.write(str(lst[i]))
            if i != (len(lst) - 1):
                f.write(",")
        f.write("\n")


def update_best_model(
    best_model_yaml, optim_variables, current_optim_params, current_result, iteration
):
    """
    Updates a YAML file that stores the best model information.

    This function checks whether the current result (typically a -log likelihood value)
    is better (i.e. lower) than the one stored in the YAML file. If it is better or if
    no best result has been stored yet, the function updates (or creates) the file with
    the following four main fields:

    - ``fixed_parameters``: A nested dictionary of the fixed parameters (written only once).
    - ``optimized_parameters``: A dictionary with keys corresponding to the optimized
      parameters (from ``optim_variables``) and their current values (from ``current_optim_params``).
    - ``results``: The best (lowest) -log likelihood value of the model.
    - ``settings``: A copy of the settings from the original model configuration.

    :param best_model_yaml: Path to the YAML file (e.g., "best_model.yaml").
    :type best_model_yaml: str
    :param fixed_params: Dictionary of fixed parameters.
    :type fixed_params: dict
    :param optim_variables: List of names for optimized parameters.
    :type optim_variables: list
    :param current_optim_params: List of current optimized parameter values.
                                 The order should correspond to the order of ``optim_variables``.
    :type current_optim_params: list
    :param current_result: The current -log likelihood value to compare against the stored value.
                           Lower values are considered better.
    :type current_result: float
    :param settings: Settings from the original model configuration.
    :type settings: dict
    """
    # Attempt to load existing best model information (if any)
    if os.path.exists(best_model_yaml):
        with open(best_model_yaml, "r") as f:
            try:
                best_model_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"Error loading best model file: {e}")
                sys.exit(1)
    else:
        raise FileNotFoundError(f"Best model file not found: {best_model_yaml}")

    # Retrieve the stored best result, if available
    mu = float(best_model_data["fixed_parameters"]["mu"])
    prev_loglik = best_model_data["results"]["log_likelihood"]
    update_flag = False

    # Update if no stored result exists or if the current result is better (i.e. lower)
    if prev_loglik is None or current_result > prev_loglik:
        update_flag = True

    if update_flag:
        # Build the optimized_parameters dictionary
        optim_dict = {
            vari: current_optim_params[i] for i, vari in enumerate(optim_variables)
        }
        for param, value in optim_dict.items():
            if param == "r":
                optim_dict[param] = float(value) * mu
            else:
                optim_dict[param] = float(value) / mu

        # Update the best model data dictionary with the four main fields
        best_model_data["optimized_parameters"] = optim_dict
        best_model_data["results"]["log_likelihood"] = current_result
        best_model_data["results"]["iteration"] = iteration

        with open(best_model_yaml, "w") as f:
            yaml.dump(best_model_data, f)


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
    tmp_path="./",
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


def optimization_wrapper(arg_lst, optimized_params, case, d, V_lst, res_name, info):
    best_model_yaml = os.path.join(res_name, "best_model.yaml")

    d_copy = d.copy()

    for i, param in enumerate(optimized_params):
        d_copy[param] = arg_lst[i]
    cut_ABC = cutpoints_ABC(d_copy["n_int_ABC"], 1)
    if case == frozenset(["t_A", "t_B", "t_C"]):

        t_out = (
            (
                (((d_copy["t_A"] + d_copy["t_B"]) / 2 + d_copy["t_2"]) + d_copy["t_C"])
                / 2
                + cut_ABC[d_copy["n_int_ABC"] - 1] * d_copy["N_ABC"]
                + d_copy["t_upper"]
                + 2 * d_copy["N_ABC"]
            )
            if "t_out" not in d_copy
            else d_copy["t_out"]
        )
        d_copy["t_out"] = t_out

    elif case == frozenset(["t_1", "t_A"]):
        t_B = d_copy["t_1"]
        t_C = d_copy["t_1"] + d_copy["t_2"]
        t_out = (
            d_copy["t_1"]
            + d_copy["t_2"]
            + cut_ABC[d_copy["n_int_ABC"] - 1] * d_copy["N_ABC"]
            + d_copy["t_upper"]
            + 2 * d_copy["N_ABC"]
            if "t_out" not in d_copy
            else d_copy["t_out"]
        )
        d_copy["t_B"] = t_B
        d_copy["t_C"] = t_C
        d_copy["t_out"] = t_out
        d_copy.pop("t_1")
    elif case == frozenset(["t_1", "t_B"]):
        t_A = d_copy["t_1"]
        t_C = d_copy["t_1"] + d_copy["t_2"]
        t_out = (
            d_copy["t_1"]
            + d_copy["t_2"]
            + cut_ABC[d_copy["n_int_ABC"] - 1] * d_copy["N_ABC"]
            + d_copy["t_upper"]
            + 2 * d_copy["N_ABC"]
            if "t_out" not in d_copy
            else d_copy["t_out"]
        )
        d_copy["t_A"] = t_A
        d_copy["t_C"] = t_C
        d_copy["t_out"] = t_out
        d_copy.pop("t_1")
    elif case == frozenset(["t_1", "t_C"]):
        t_A = d_copy["t_1"]
        t_B = d_copy["t_1"]
        t_out = (
            d_copy["t_1"]
            + d_copy["t_2"]
            + cut_ABC[d_copy["n_int_ABC"] - 1] * d_copy["N_ABC"]
            + d_copy["t_upper"]
            + 2 * d_copy["N_ABC"]
            if "t_out" not in d_copy
            else d_copy["t_out"]
        )
        d_copy["t_A"] = t_A
        d_copy["t_B"] = t_B
        d_copy["t_out"] = t_out
        d_copy.pop("t_1")
    elif case == frozenset(["t_A", "t_B"]):
        t_C = (d_copy["t_A"] + d_copy["t_B"]) / 2 + d_copy["t_2"]
        t_out = (
            (
                (((d_copy["t_A"] + d_copy["t_B"]) / 2 + d_copy["t_2"]) + t_C) / 2
                + cut_ABC[d_copy["n_int_ABC"] - 1] * d_copy["N_ABC"]
                + d_copy["t_upper"]
                + 2 * d_copy["N_ABC"]
            )
            if "t_out" not in d_copy
            else d_copy["t_out"]
        )
        d_copy["t_C"] = t_C
        d_copy["t_out"] = t_out
    elif case == frozenset(["t_A", "t_C"]):
        t_B = (d_copy["t_A"] + d_copy["t_C"] - d_copy["t_2"]) / 2
        t_out = (
            (
                (((d_copy["t_A"] + t_B) / 2 + d_copy["t_2"]) + d_copy["t_C"]) / 2
                + cut_ABC[d_copy["n_int_ABC"] - 1] * d_copy["N_ABC"]
                + d_copy["t_upper"]
                + 2 * d_copy["N_ABC"]
            )
            if "t_out" not in d_copy
            else d_copy["t_out"]
        )
        d_copy["t_B"] = t_B
        d_copy["t_out"] = t_out
    elif case == frozenset(["t_B", "t_C"]):
        t_A = (d_copy["t_B"] + d_copy["t_C"] - d_copy["t_2"]) / 2
        t_out = (
            (
                (((t_A + d_copy["t_B"]) / 2 + d_copy["t_2"]) + d_copy["t_C"]) / 2
                + cut_ABC[d_copy["n_int_ABC"] - 1] * d_copy["N_ABC"]
                + d_copy["t_upper"]
                + 2 * d_copy["N_ABC"]
            )
            if "t_out" not in d_copy
            else d_copy["t_out"]
        )
        d_copy["t_A"] = t_A
        d_copy["t_out"] = t_out
    elif case == frozenset(["t_1"]):
        t_A = t_B = d_copy["t_1"]
        t_C = d_copy["t_1"] + d_copy["t_2"]
        t_out = (
            d_copy["t_1"]
            + d_copy["t_2"]
            + cut_ABC[d_copy["n_int_ABC"] - 1] * d_copy["N_ABC"]
            + d_copy["t_upper"]
            + 2 * d_copy["N_ABC"]
            if "t_out" not in d_copy
            else d_copy["t_out"]
        )
        d_copy["t_A"] = t_A
        d_copy["t_B"] = t_B
        d_copy["t_C"] = t_C
        d_copy["t_out"] = t_out
        d_copy.pop("t_1")

    # Calculate transition and emission probabilities
    a, b, pi, hidden_names, observed_names = trans_emiss_calc(
        d_copy["t_A"],
        d_copy["t_B"],
        d_copy["t_C"],
        d_copy["t_2"],
        d_copy["t_upper"],
        d_copy["t_out"],
        d_copy["N_AB"],
        d_copy["N_ABC"],
        d_copy["r"],
        d_copy["n_int_AB"],
        d_copy["n_int_ABC"],
        "standard",
        "standard",
        info["tmp_path"],
    )
    # Save indices for hidden and observed states
    # if info["Nfeval"] == 0:
    #    pd.DataFrame(
    #        {"idx": list(hidden_names.keys()), "hidden": list(hidden_names.values())}
    #    ).to_csv("hidden_states.csv", index=False)
    #    pd.DataFrame(
    #        {
    #            "idx": list(observed_names.keys()),
    #            "observed": list(observed_names.values()),
    #        }
    #    ).to_csv("observed_states.csv", index=False)
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = mp.cpu_count()
    # Calculate log-likelihood
    if len(V_lst) >= ncpus:
        loglik = loglik_wrapper_par(a, b, pi, V_lst)
    else:
        loglik = loglik_wrapper(a, b, pi, V_lst)
    # Write parameter estimates, likelihood and time
    write_list(
        [info["Nfeval"]] + arg_lst.tolist() + [loglik, time.time() - info["time"]],
        os.path.join(res_name, "optimization_history.csv"),
    )
    # Update best model
    update_best_model(
        best_model_yaml,
        optimized_params,
        arg_lst,
        loglik,
        info["Nfeval"],
    )
    # Update optimization cycle
    info["Nfeval"] += 1
    return -loglik


def optimizer(
    optim_variables,
    optim_list,
    bounds,
    fixed_params,
    V_lst,
    res_name,
    case,
    method="Nelder-Mead",
    header=True,
    tmp_path="./",
):
    """
    Optimization function.

    Parameters
    ----------
    optim_params : dictionary
        Dictionary containing the initial values for the
        parameters to be optimized, and their optimization
        bounds. The structure of the dictionary should be
        as follows:
            dct['variable'] = [initial_value, lower_bound, upper_bound]
        The dictionary must contain either 6 (t_1, t_2, t_upper, N_AB, N_ABC, r)
        or 9 (t_A, t_B, t_C, t_2, t_upper, t_out, N_AB, N_ABC, r) entries,
        in that specific order.
    fixed params : dictionary
        Dictionary containing the values for the fixed parameters.
        The dictionary must contain entries n_int_AB and n_int_ABC (in no particular order).
    V_lst : list of numpy arrays
        List of arrays of integers corresponding to the the observed states.
    res_name : str
        Location and name of the file where the results should be
        saved (in csv format).
    """
    optimization_history = os.path.join(res_name, "optimization_history.csv")
    if header:
        write_list(
            ["n_eval"] + list(optim_variables) + ["loglik", "time"],
            optimization_history,
        )
    options = {"maxiter": 10000, "disp": True}
    # if method in ['L-BFGS-B', 'TNC']:
    #     if len(optim_params) == 6:
    #         options['eps'] = np.array([10, 1, 10, 1, 1, 1e-9])
    #     elif len(optim_params) == 9:
    #         options['eps'] = np.array([10, 10, 10, 1, 10, 10, 1, 1, 1e-9])
    d_copy = fixed_params.copy()
    res = minimize(
        optimization_wrapper,
        x0=optim_list,
        args=(
            optim_variables,
            case,
            d_copy,
            V_lst,
            res_name,
            {"Nfeval": 0, "time": time.time(), "tmp_path": tmp_path},
        ),
        method=method,
        bounds=bounds,
        options=options,
    )
