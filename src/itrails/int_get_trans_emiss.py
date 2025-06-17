import numpy as np
import pandas as pd

from itrails.cutpoints import cutpoints_AB, cutpoints_ABC
from itrails.int_get_emission_prob_mat import get_emission_prob_mat_introgression
from itrails.int_get_joint_prob_mat import get_joint_prob_mat_introgression


def trans_emiss_calc_introgression(
    t_A,
    t_B,
    t_C,
    t_2,
    t_upper,
    t_out,
    t_m,
    N_AB,
    N_BC,
    N_ABC,
    r,
    m,
    n_int_AB,
    n_int_ABC,
    cut_AB,
    cut_ABC,
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
        Time in generations from present to the migration event for species B
        (times mutation rate)
    t_C : numeric
        Time in generations from present to the migration event for species C
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
    t_m : numeric
        Time in generagions from admixture time until first speciation time
    N_AB : numeric
        Effective population size between speciation events (times mutation rate) for AB
    N_BC : numeric
        Effective population size between speciation events (times mutation rate) for BC
    N_ABC : numeric
        Effective population size in deep coalescence, before the second speciation event
        (times mutation rate)
    r : numeric
        Recombination rate per site per generation (divided by mutation rate)
    m : numeric
        Migration rate (admixture proportion)
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
    t_m = t_m / N_ref
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
    coal_BC = N_ref / N_BC
    coal_C = N_ref / N_BC
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

    cut_AB = np.asarray(cut_AB, dtype=float)
    cut_ABC = np.asarray(cut_ABC, dtype=float)

    tr = get_joint_prob_mat_introgression(
        t_A,
        t_B,
        t_AB,
        t_C,
        t_m,
        rho_A,
        rho_B,
        rho_AB,
        rho_C,
        rho_ABC,
        coal_A,
        coal_B,
        coal_AB,
        coal_BC,
        coal_C,
        coal_ABC,
        m,
        n_int_AB,
        n_int_ABC,
        cut_AB,
        cut_ABC,
        tmp_path,
    )
    tr = pd.DataFrame(tr, columns=["From", "To", "Prob"]).pivot(
        index=["From"], columns=["To"], values=["Prob"]
    )
    tr.columns = tr.columns.droplevel()
    hidden_names = list(tr.columns)
    hidden_names = dict(zip(range(len(hidden_names)), hidden_names))
    arr = np.array(tr).astype(np.float64)
    pi = arr.sum(axis=1)
    a = arr / pi[:, None]

    em = get_emission_prob_mat_introgression(
        t_A,
        t_B,
        t_AB,
        t_C,
        t_upper,
        t_out,
        t_m,
        rho_A,
        rho_B,
        rho_AB,
        rho_C,
        rho_ABC,
        coal_A,
        coal_B,
        coal_AB,
        coal_BC,
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
    em.hidden_state = em.hidden_state.astype("category")
    em.hidden_state.cat.set_categories(hidden_names)
    em = em.sort_values(["hidden_state"])
    em = em.iloc[:, 1:]
    observed_names = list(em.columns)
    observed_names = dict(zip(range(len(observed_names)), observed_names))
    b = np.array(em)

    return a, b, pi, hidden_names, observed_names
