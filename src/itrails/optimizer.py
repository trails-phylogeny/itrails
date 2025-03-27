import multiprocessing as mp
import os
import time

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from numba.typed import List
from scipy.optimize import minimize

import itrails.ncpu as ncpu
from itrails.cutpoints import cutpoints_ABC
from itrails.get_trans_emiss import trans_emiss_calc
from itrails.read_data import (
    get_idx_state,
    get_idx_state_new_method,
)
from itrails.yaml_helpers import update_best_model


def forward_loglik_par(a, b, pi, V, order):
    """Computes the log-likelihood in parallel by converting the provided order into a List and then calling forward_loglik.

    :param a: Transition probability matrix.
    :type a: numpy array.
    :param b: Emission probability matrix.
    :type b: numpy array.
    :param pi: Vector of starting probabilities of the hidden states.
    :type pi: numpy array.
    :param V: Vector of observed states represented as integer indices.
    :type V: numpy array.
    :param order: List of indices mapping observed states to emission probabilities.
    :type order: list.
    :return: Log-likelihood value.
    :rtype: float."""
    order = List(order)
    return forward_loglik(a, b, pi, V, order)


def loglik_wrapper_par(a, b, pi, V_lst):
    """Parallel log-likelihood wrapper that builds an order list using get_idx_state, then computes the log-likelihood for each observed state vector in V_lst in parallel using joblib, and returns the sum of the results.

    :param a: Transition probability matrix.
    :type a: numpy array.
    :param b: Emission probability matrix.
    :type b: numpy array.
    :param pi: Vector of starting probabilities for the hidden states.
    :type pi: numpy array.
    :param V_lst: List of observed state vectors, where each vector is a numpy array of integer indices.
    :type V_lst: list[np.ndarray].
    :return: Sum of log-likelihood values over all observed sequences in V_lst.
    :rtype: float."""
    # Build the 'order' list using get_idx_state (assuming it's defined at top-level).
    order = [get_idx_state(i) for i in range(624 + 1)]
    # Build the list of argument tuples.
    pool_lst = [(a, b, pi, V, order) for V in V_lst]
    # Use ncpus from your global configuration module.
    ncpus = ncpu.N_CPU
    # Run forward_loglik_par in parallel over all argument tuples.
    results = Parallel(n_jobs=ncpus)(
        delayed(forward_loglik_par)(*args) for args in pool_lst
    )
    # Sum up all the results and return the total.
    acc = sum(results)
    return acc


def loglik_wrapper_par_new_method(a, b, pi, V_lst):
    """Parallel log-likelihood wrapper using joblib that builds an order list via get_idx_state_new_method, computes the log-likelihood for each observed state vector in V_lst in parallel, and returns the sum of the computed values.

    :param a: Transition probability matrix.
    :type a: numpy array.
    :param b: Emission probability matrix.
    :type b: numpy array.
    :param pi: Vector of starting probabilities for the hidden states.
    :type pi: numpy array.
    :param V_lst: List of observed state vectors as numpy arrays of integer indices.
    :type V_lst: list[np.ndarray].
    :return: Total log-likelihood value summed over all observed sequences in V_lst.
    :rtype: float."""
    order = [get_idx_state_new_method(i) for i in range(125)]
    pool_args = [(a, b, pi, V, order) for V in V_lst]
    # Determine number of CPUs to use.
    ncpus = ncpu.N_CPU
    # Use joblib's Parallel to run forward_loglik_par in parallel.
    results = Parallel(n_jobs=ncpus)(
        delayed(forward_loglik_par)(*args) for args in pool_args
    )
    # Sum up the results.
    return sum(results)


def loglik_wrapper(a, b, pi, V_lst):
    """Sequential log-likelihood wrapper that builds an order list using get_idx_state, iteratively computes the log-likelihood for each observed state vector in V_lst, and returns the total sum.

    :param a: Transition probability matrix.
    :type a: numpy array.
    :param b: Emission probability matrix.
    :type b: numpy array.
    :param pi: Vector of starting probabilities of the hidden states.
    :type pi: numpy array.
    :param V_lst: List of observed state vectors (as numpy arrays of integer indices).
    :type V_lst: list[np.ndarray].
    :return: Sum of log-likelihood values over all observed sequences in V_lst.
    :rtype: float."""
    order = List()
    for i in range(624 + 1):
        order.append(get_idx_state(i))
    acc = 0
    prev_time = time.time()
    events_count = len(V_lst)
    for i in range(len(V_lst)):
        acc += forward_loglik(a, b, pi, V_lst[i], order)
        if (time.time() - prev_time) > 1:
            prev_time = time.time()
    return acc


def loglik_wrapper_new_method(a, b, pi, V_lst):
    """Sequential log-likelihood wrapper that builds an order list using get_idx_state_new_method, computes the log-likelihood for each observed state vector in V_lst sequentially, and returns the total sum.

    :param a: Transition probability matrix.
    :type a: numpy array.
    :param b: Emission probability matrix.
    :type b: numpy array.
    :param pi: Vector of starting probabilities of the hidden states.
    :type pi: numpy array.
    :param V_lst: List of observed state vectors as numpy arrays of integer indices.
    :type V_lst: list[np.ndarray].
    :return: Total log-likelihood value summed over all observed sequences in V_lst.
    :rtype: float."""
    order = List()
    for i in range(125):
        order.append(get_idx_state_new_method(i))
    acc = 0
    prev_time = time.time()
    events_count = len(V_lst)
    for i in range(len(V_lst)):
        acc += forward_loglik(a, b, pi, V_lst[i], order)
        if (time.time() - prev_time) > 1:
            prev_time = time.time()
    return acc


@njit
def forward_loglik(a, b, pi, V, order):
    """Computes the log-likelihood for a given observed state sequence by running the forward algorithm and applying log-sum-exp for numerical stability.

    :param a: Transition probability matrix.
    :type a: numpy array.
    :param b: Emission probability matrix.
    :type b: numpy array.
    :param pi: Vector of starting probabilities of the hidden states.
    :type pi: numpy array.
    :param V: Vector of observed states (as integer indices).
    :type V: numpy array.
    :param order: List of indices mapping observed states to emission probabilities.
    :type order: list. :return: Log-likelihood value.
    :rtype: float."""
    alpha = forward(a, b, pi, V, order)
    x = alpha[-1, :].max()
    return np.log(np.exp(alpha[len(V) - 1] - x).sum()) + x


@njit
def forward(a, b, pi, V, order):
    """Executes the forward algorithm for Hidden Markov Models allowing for missing data by computing the log-scaled alpha values.

    :param a: Transition probability matrix.
    :type a: numpy array.
    :param b: Emission probability matrix.
    :type b: numpy array.
    :param pi: Vector of starting probabilities of the hidden states.
    :type pi: numpy array.
    :param V: Vector of observed states (as integer indices).
    :type V: numpy array.
    :param order: List of indices mapping observed states to emission probabilities.
    :type order: list.
    :return: Alpha matrix with log probabilities for each time step and hidden state.
    :rtype: numpy array."""
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
    """Performs the backward algorithm for Hidden Markov Models by computing the beta values in log space;
    :param a: Transition probability matrix.

    :type a: numpy array.
    :param b: Emission probability matrix.
    :type b: numpy array.
    :param V: Vector of observed states (as integer indices).
    :type V: numpy array.
    :param order: List of indices mapping observed states to emission probabilities.
    :type order: list.
    :return: Beta matrix with log probabilities for each time step and hidden state.
    :rtype: numpy array."""
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
    """Computes the posterior probabilities of the hidden states for a given observed sequence by combining the forward and backward algorithms and normalizing the result.

    :param a: Transition probability matrix.
    :type a: numpy array.
    :param b: Emission probability matrix.
    :type b: numpy array.
    :param pi: Vector of starting probabilities of the hidden states.
    :type pi: numpy array.
    :param V: Vector of observed states (as integer indices).
    :type V: numpy array.
    :param order: List of indices mapping observed states to emission probabilities.
    :type order: list.
    :return: Posterior probability matrix with probabilities for each time step and hidden state.
    :rtype: numpy array."""
    alpha = forward(a, b, pi, V, order)
    beta = backward(a, b, V, order)
    post_prob = alpha + beta
    max_row = post_prob.max(1).reshape(-1, 1)
    post_prob = np.exp(post_prob - max_row) / np.exp(post_prob - max_row).sum(
        1
    ).reshape(-1, 1)
    return post_prob


def post_prob_wrapper(a, b, pi, V_lst):
    """Wrapper function that computes the posterior probabilities for a list of observed state sequences by building an order list using get_idx_state and iteratively applying post_prob to each sequence.

    :param a: Transition probability matrix.
    :type a: numpy array.
    :param b: Emission probability matrix.
    :type b: numpy array.
    :param pi: Vector of starting probabilities of the hidden states.
    :type pi: numpy array.
    :param V_lst: List of observed state vectors (as numpy arrays of integer indices).
    :type V_lst: list[np.ndarray].
    :return: List of posterior probability matrices corresponding to each observed sequence in V_lst.
    :rtype: list."""
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
def viterbi_old(a, b, pi, V, order):
    """Computes the Viterbi path using an iterative approach in log space by performing dynamic programming over the given observed sequence.

    :param a: Transition probability matrix.
    :type a: numpy array.
    :param b: Emission probability matrix.
    :type b: numpy array.
    :param pi: Vector of starting probabilities of the hidden states.
    :type pi: numpy array.
    :param V: Vector of observed states.
    :type V: numpy array.
    :param order: List of indices mapping observed states to emission probabilities.
    :type order: list.
    :return: Viterbi path as an array of state indices.
    :rtype: numpy array."""
    T = V.shape[0]
    M = a.shape[0]
    omega = np.zeros((T, M))
    omega[0, :] = np.log(pi * b[:, order[V[0]]].sum(axis=1))
    prev = np.zeros((T - 1, M))
    for t in range(1, T):
        for j in range(M):
            probability = (
                omega[t - 1] + np.log(a[:, j]) + np.log(b[j, order[V[t]]].sum())
            )
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


def viterbi(a, b, pi, V, order):
    """Computes the Viterbi path by performing the forward pass in log space with dynamic programming and returning both the omega matrix and the backtracking pointer matrix.

    :param a: Transition probability matrix.
    :type a: numpy array.
    :param b: Emission probability matrix.
    :type b: numpy array.
    :param pi: Vector of starting probabilities of the hidden states.
    :type pi: numpy array.
    :param V: Vector of observed states (as integer indices).
    :type V: numpy array.
    :param order: List of indices mapping observed states to emission probabilities.
    :type order: list.
    :return: Tuple containing the omega matrix and the backtracking pointer matrix.
    :rtype: tuple(numpy array, numpy array)."""
    T = V.shape[0]
    M = a.shape[0]
    omega = np.zeros((T, M))
    omega[0, :] = np.log(pi * b[:, order[V[0]]].sum(axis=1))
    prev = np.zeros((T - 1, M))
    for t in range(1, T):
        probability_matrix = (
            omega[t - 1][:, np.newaxis]
            + np.log(a)
            + np.log(b[:, order[V[t]]].sum(axis=1))
        )
        prev[t - 1, :] = np.argmax(probability_matrix, axis=0)
        omega[t, :] = np.max(probability_matrix, axis=0)
    return omega, prev


def backtrack_viterbi(omega, prev):
    """Reconstructs the optimal Viterbi path by backtracking through the pointer matrix obtained from the viterbi function.

    :param omega: Omega matrix containing the log probabilities for each time step and state.
    :type omega: numpy array.
    :param prev: Backtracking pointer matrix from the Viterbi algorithm.
    :type prev: numpy array. :return: Optimal Viterbi path as an array of state indices.
    :rtype: numpy array."""
    T = omega.shape[0]
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


def viterbi_wrapper(a, b, pi, V_lst):
    """Wrapper for the Viterbi algorithm that builds an order list using get_idx_state, applies the viterbi and backtrack_viterbi functions to each observed state vector in V_lst, and returns a list of Viterbi paths.

    :param a: Transition probability matrix.
    :type a: numpy array.
    :param b: Emission probability matrix.
    :type b: numpy array.
    :param pi: Vector of starting probabilities of the hidden states.
    :type pi: numpy array.
    :param V_lst: List of observed state vectors (as numpy arrays of integer indices).
    :type V_lst: list[np.ndarray].
    :return: List of Viterbi paths corresponding to each observed sequence in V_lst.
    :rtype: list."""
    res_lst = []
    order = List()
    for i in range(624 + 1):
        order.append(get_idx_state(i))
    for i in range(len(V_lst)):
        (omega, prev) = viterbi(a, b, pi, V_lst[i], order)
        res_lst.append(backtrack_viterbi(omega, prev))
    return res_lst


def write_list(lst, res_name):
    """Appends the elements of a list as a comma-separated line to a CSV file with the given file name.

    :param lst: List of values to append.
    :type lst: list.
    :param res_name: File name (or path) to which the list should be appended.
    :type res_name: str.
    :return: None."""
    with open(f"{res_name}", "a") as f:
        for i in range(len(lst)):
            f.write(str(lst[i]))
            if i != (len(lst) - 1):
                f.write(",")
        f.write("\n")


def optimization_wrapper(arg_lst, optimized_params, case, d, V_lst, res_name, info):
    """Objective function for the optimizer that updates a copy of the fixed parameter dictionary with the current optimized values from arg_lst, computes additional derived parameters based on the specified case (which determines how time parameters are combined), calculates the transition probability, emission, and initial state probability matrices via trans_emiss_calc, evaluates the log-likelihood for the observed data V_lst using either a parallel or sequential log-likelihood wrapper depending on available CPUs, logs the evaluation to an optimization history CSV file, updates the best model if the current log-likelihood improves upon the previous best, increments the evaluation count, and returns the negative log-likelihood value (to be minimized).

    :param arg_lst: Array of parameter values to be optimized that will update the fixed parameter dictionary.
    :type arg_lst: numpy array.
    :param optimized_params: List of parameter names (keys in d) that are subject to optimization. :type optimized_params: list[str].
    :param case: A frozenset specifying the combination of time parameters provided (e.g., frozenset(["t_A", "t_B", "t_C"]) or frozenset(["t_1", "t_A"]), etc.). :type case: frozenset.
    :param d: Dictionary of fixed parameter values.
    :type d: dict.
    :param V_lst: List of observed state arrays, where each array is a numpy array of integer indices representing observed states.
    :type V_lst: list[np.ndarray].
    :param res_name: Directory path where result files (e.g., best_model.yaml and optimization_history.csv) will be saved.
    :type res_name: str.
    :param info: Dictionary containing optimization metadata, including "Nfeval" (the number of evaluations so far) and "time" (the start time of the optimization run).
    :type info: dict.
    :return: Negative log-likelihood value (to be minimized by the optimizer).
    :rtype: float."""
    output_dir, output_prefix = os.path.split(res_name)
    best_model_yaml = os.path.join(output_dir, f"{output_prefix}_best_model.yaml")
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
    )

    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = mp.cpu_count()

    if len(V_lst) >= ncpus:
        loglik = loglik_wrapper_par(a, b, pi, V_lst)
    else:
        loglik = loglik_wrapper(a, b, pi, V_lst)

    write_list(
        [info["Nfeval"]] + arg_lst.tolist() + [loglik, time.time() - info["time"]],
        os.path.join(output_dir, f"{output_prefix}_optimization_history.csv"),
    )

    update_best_model(
        best_model_yaml,
        optimized_params,
        arg_lst,
        loglik,
        info["Nfeval"],
    )

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
):
    """
    Optimization function.

    :param optim_params: Dictionary containing the initial values for the parameters to be optimized, and their optimization bounds. The structure of the dictionary should be as follows: ``dct['variable'] = [initial_value, lower_bound, upper_bound]``. The dictionary must contain either 6 entries (``t_1, t_2, t_upper, N_AB, N_ABC, r``) or 9 entries (``t_A, t_B, t_C, t_2, t_upper, t_out, N_AB, N_ABC, r``) in that specific order.
    :type optim_params: dict
    :param fixed_params: Dictionary containing the values for the fixed parameters. The dictionary must include the entries ``n_int_AB`` and ``n_int_ABC`` (in no particular order).
    :type fixed_params: dict
    :param V_lst: List of numpy arrays of integers corresponding to the observed states.
    :type V_lst: list
    :param res_name: File path and name where the results should be saved (in CSV format).
    :type res_name: str
    :return: None. This function updates the results on each iteration of the minimizer.
    :rtype: None
    """
    output_dir, output_prefix = os.path.split(res_name)
    optimization_history = os.path.join(
        output_dir, f"{output_prefix}_optimization_history.csv"
    )
    if header:
        write_list(
            ["n_eval"] + list(optim_variables) + ["loglik", "time"],
            optimization_history,
        )
    options = {"maxiter": 10000, "disp": True}

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
            {"Nfeval": 0, "time": time.time()},
        ),
        method=method,
        bounds=bounds,
        options=options,
    )
