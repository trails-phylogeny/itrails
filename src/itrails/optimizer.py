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
    Log-likelihood wrapper (parallelized) using joblib.

    Parameters
    ----------
    a : numpy array
        Transition probability matrix.
    b : numpy array
        Emission probability matrix.
    pi : numpy array
        Vector of starting probabilities for the hidden states.
    V_lst : list of numpy arrays
        List of observed states (as integer indices).

    Returns
    -------
    acc : float
        The sum of log-likelihood values over all items in V_lst.
    """
    # Build the 'order' list using get_idx_state (assuming it's defined at top-level).
    order = [get_idx_state(i) for i in range(624 + 1)]

    # Build the list of argument tuples
    pool_lst = [(a, b, pi, V, order) for V in V_lst]

    # Use ncpus from your global configuration module
    ncpus = ncpu.N_CPU

    # Run forward_loglik_par in parallel over all argument tuples
    results = Parallel(n_jobs=ncpus)(
        delayed(forward_loglik_par)(*args) for args in pool_lst
    )

    # Sum up all the results and return the total
    acc = sum(results)
    return acc


""" 
def loglik_wrapper_par(a, b, pi, V_lst):
    
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
 """


def loglik_wrapper_par_new_method(a, b, pi, V_lst):
    """
    Log-likelihood wrapper (parallelized) using joblib.

    Parameters:
      a : numpy array
          Transition probability matrix
      b : numpy array
          Emission probability matrix
      pi : numpy array
          Vector of starting probabilities of the hidden states
      V_lst : list of numpy arrays
          List of observed states (as integer indices)

    Returns:
      The sum of the log-likelihood values computed for each V in V_lst.
    """
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


""" 
def loglik_wrapper_par_new_method(a, b, pi, V_lst):
    
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
 """


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
def viterbi_old(a, b, pi, V, order):
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
    for i in range(len(V_lst)):
        (omega, prev) = viterbi(a, b, pi, V_lst[i], order)
        res_lst.append(backtrack_viterbi(omega, prev))
    return res_lst


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
            {"Nfeval": 0, "time": time.time()},
        ),
        method=method,
        bounds=bounds,
        options=options,
    )
