import numba as nb
import numpy as np
from scipy.special import comb

from itrails.cutpoints import cutpoints_AB, cutpoints_ABC
from itrails.expm import expm


def rate_mat_JC69(mu):
    """
    Return the rate matrix for the JC69 model.
    The JC69 model assumes equal base frequencies and equal substitution rates.

    :param mu: Mutation rate.
    :type mu: numeric
    :return: A 4x4 numpy array representing the JC69 rate matrix.
    :rtype: numpy.ndarray
    """
    return np.full((4, 4), mu / 4) - np.diag([mu, mu, mu, mu])


def p_b_given_a(t, Q):
    """
    Calculate the probability of observing a nucleotide b given a starting nucleotide a, a set of time intervals, and corresponding rate matrices.

    For each pair of nucleotides (a, b), the function computes the probability by first summing the products of each time interval with its corresponding rate matrix, and then applying the matrix exponential.

    :param t: Total time intervals (list or array-like) over which the process is considered.
    :type t: numeric or list of numeric
    :param Q: A list or array of 4x4 rate matrices for a substitution model.
    :type Q: list or numpy.ndarray
    :return: A list of tuples of the form (starting nucleotide, ending nucleotide, probability).
    :rtype: list
    """
    nt = ["A", "G", "C", "T"]
    mat = np.zeros((4, 4))
    for i in range(len(t)):
        mat = mat + t[i] * Q[i]
    P = expm(mat)
    result = []
    for aa in range(4):
        for bb in range(4):
            result.append((nt[aa], nt[bb], P[aa, bb]))
    return result


@nb.jit(nopython=True)
def JC69_analytical_integral(aa, bb, cc, dd, t, mu, k):
    """
    Calculate the integrated probability for observing nucleotides bb, cc, and dd given a starting nucleotide aa, a mutation rate mu, and a coalescent rate k. This corresponds to computing: P(b = bb, c = cc, d = dd | a = aa, mu, t) by integrating the coalescent process over the interval t.

    :param aa: Nucleotide at position a (integer or string).
    :type aa: int or str
    :param bb: Nucleotide at position b (integer or string).
    :type bb: int or str
    :param cc: Nucleotide at position c (integer or string).
    :type cc: int or str
    :param dd: Nucleotide at position d (integer or string).
    :type dd: int or str
    :param t: Total time of the interval (from positions a/b/c to d).
    :type t: numeric
    :param mu: Mutation rate for the JC69 model.
    :type mu: numeric
    :param k: Coalescent rate.
    :type k: numeric
    :return: The integrated probability.
    :rtype: numeric
    """
    alpha = 3 / 4 if aa == dd else -1 / 4
    beta = 3 / 4 if dd == bb else -1 / 4
    gamma = 3 / 4 if dd == cc else -1 / 4

    res = (
        k
        * (
            ((-1 + np.exp(k * t)) * (np.exp(mu * t) + 16 * (alpha + beta) * gamma))
            / (np.exp((k + mu) * t) * k)
            + 4
            * (
                gamma / (np.exp(k * t) * (-k + mu))
                + (alpha + beta) / (k + mu)
                - (alpha + beta) / (np.exp((k + mu) * t) * (k + mu))
                + (4 * alpha * beta) / (k + 2 * mu)
                + (gamma * ((k - mu) ** (-1) + (16 * alpha * beta) / (k + mu)))
                / np.exp(mu * t)
                + (4 * alpha * beta * ((-4 * gamma) / (k + mu) - (k + 2 * mu) ** (-1)))
                / np.exp((k + 2 * mu) * t)
            )
        )
    ) / (64 * (1 - np.exp(-(k * t))))

    return res


def p_b_c_given_a_JC69_analytical(t, mu, k):
    """
    Compute the probability P(b, c | a) for all combinations of nucleotides under the JC69 model.

    :param t: Total time of the interval (from positions a/b/c to d).
    :type t: numeric
    :param mu: Mutation rate for the JC69 model.
    :type mu: numeric
    :param k: Coalescent rate.
    :type k: numeric
    :return: A list of tuples (a, b, c, probability) where a, b, and c are nucleotide letters.
    :rtype: list
    """
    nt = ["A", "G", "C", "T"]
    result = []
    for aa in range(4):
        for bb in range(4):
            for cc in range(4):
                cumsum = 0.0
                for dd in range(4):
                    cumsum += JC69_analytical_integral(aa, bb, cc, dd, t, mu, k)
                result.append((nt[aa], nt[bb], nt[cc], cumsum))
    return result


@nb.jit(nopython=True)
def JC69_analytical_integral_double(aa, bb, cc, dd, ee, ff, t, mu):
    """
    Calculate the integrated probability for observing nucleotides bb, cc, dd, ee, and ff  given a starting nucleotide aa, using the JC69 model. This computes P(b = bb, c = cc, d = dd, e = ee, f = ff | a = aa, mu, t) by integrating over the coalescent process for two coalescent events. Note: The coalescent rate is 1 for two sequences and 3 for three sequences.

    :param aa: Nucleotide at position a (integer or string).
    :type aa: int or str
    :param bb: Nucleotide at position b (integer or string).
    :type bb: int or str
    :param cc: Nucleotide at position c (integer or string).
    :type cc: int or str
    :param dd: Nucleotide at position d (integer or string).
    :type dd: int or str
    :param ee: Nucleotide at the first coalescent event (integer or string).
    :type ee: int or str
    :param ff: Nucleotide at the second coalescent event (integer or string).
    :type ff: int or str
    :param t: Total time of the interval.
    :type t: numeric
    :param mu: Mutation rate for the JC69 model.
    :type mu: numeric
    :return: The integrated probability.
    :rtype: numeric
    """

    alpha = 3 / 4 if aa == ee else -1 / 4
    beta = 3 / 4 if ee == bb else -1 / 4
    gamma = 3 / 4 if ee == ff else -1 / 4
    delta = 3 / 4 if ff == cc else -1 / 4
    epsilon = 3 / 4 if ff == dd else -1 / 4

    res = (
        3
        * (
            (-2 * delta * (-2 - 8 * gamma + mu)) / (-6 + mu + mu**2)
            - (32 * alpha * beta * delta * (2 + mu + 8 * gamma * (1 + mu)))
            / (3 * (1 + mu) ** 2 * (2 + mu))
            - (32 * alpha * beta * epsilon * (2 + mu + 8 * gamma * (1 + mu)))
            / (np.exp(mu * t) * (1 + mu) * (2 + mu) * (3 + mu))
            - (
                8
                * alpha
                * beta
                * (1 + (16 * delta * epsilon) / np.exp(mu * t))
                * (2 + mu + 8 * gamma * (1 + mu))
            )
            / ((1 + mu) * (2 + mu) * (3 + 2 * mu))
            + (
                16
                * delta
                * gamma
                * (
                    (-1 + 2 * beta * (-2 + mu)) * (2 + mu)
                    + 2 * alpha * (-2 + mu) * (2 + 8 * beta + mu)
                )
            )
            / ((-2 + mu) * (2 + mu) * (1 + 2 * mu))
            - (
                4
                * (alpha + beta)
                * (1 + 2 * gamma * (2 + mu))
                * (
                    (3 + 2 * mu) * (3 * np.exp(mu * t) + 4 * epsilon * (3 + mu))
                    + 12
                    * delta
                    * (np.exp(mu * t) * (3 + mu) + 4 * epsilon * (3 + 2 * mu))
                )
            )
            / (3 * np.exp(mu * t) * (2 + mu) * (3 + mu) * (3 + 2 * mu))
            - (
                2
                * epsilon
                * (
                    (2 + 8 * gamma - mu) / ((-3 + mu) * (-2 + mu))
                    + (
                        (1 + mu) * (2 + 8 * beta + mu)
                        + 8 * alpha * (1 + mu + 2 * beta * (2 + mu))
                    )
                    / ((-1 + mu) * (1 + mu) * (2 + mu))
                )
            )
            / np.exp(mu * t)
            - (
                -16 * delta * epsilon * (2 + 8 * gamma - mu) * (2 + 3 * mu + mu**2)
                + np.exp(mu * t) * (-2 - 8 * gamma + mu) * (2 + 3 * mu + mu**2)
                - 3
                * np.exp(mu * t)
                * (-2 + mu)
                * (
                    (1 + mu) * (2 + 8 * beta + mu)
                    + 8 * alpha * (1 + mu + 2 * beta * (2 + mu))
                )
                - 48
                * epsilon
                * (
                    2
                    * gamma
                    * (1 + mu)
                    * (
                        (-1 + 2 * beta * (-2 + mu)) * (2 + mu)
                        + 2 * alpha * (-2 + mu) * (2 + 8 * beta + mu)
                    )
                    + delta
                    * (-2 + mu)
                    * (
                        (1 + mu) * (2 + 8 * beta + mu)
                        + 8 * alpha * (1 + mu + 2 * beta * (2 + mu))
                    )
                )
            )
            / (6 * np.exp(mu * t) * (-2 + mu) * (1 + mu) * (2 + mu))
            + (
                2
                * (
                    2
                    * np.exp(mu * t)
                    * gamma
                    * (1 + mu)
                    * (
                        (-1 + 2 * beta * (-2 + mu)) * (2 + mu)
                        + 2 * alpha * (-2 + mu) * (2 + 8 * beta + mu)
                    )
                    + delta
                    * (
                        32
                        * epsilon
                        * gamma
                        * (1 + mu)
                        * (
                            (-1 + 2 * beta * (-2 + mu)) * (2 + mu)
                            + 2 * alpha * (-2 + mu) * (2 + 8 * beta + mu)
                        )
                        + np.exp(mu * t)
                        * (-2 + mu)
                        * (
                            (1 + mu) * (2 + 8 * beta + mu)
                            + 8 * alpha * (1 + mu + 2 * beta * (2 + mu))
                        )
                    )
                )
            )
            / (np.exp(mu * t) * (1 + mu) ** 2 * (-4 + mu**2))
            + (
                (32 * alpha * beta * delta * (2 + mu + 8 * gamma * (1 + mu)))
                / (3 * (1 + mu) ** 2 * (2 + mu))
                + (
                    32
                    * alpha
                    * beta
                    * np.exp(mu * t)
                    * epsilon
                    * (2 + mu + 8 * gamma * (1 + mu))
                )
                / ((1 + mu) * (2 + mu) * (3 + mu))
                + (
                    8
                    * alpha
                    * beta
                    * np.exp(mu * t)
                    * (1 + (16 * delta * epsilon) / np.exp(mu * t))
                    * (2 + mu + 8 * gamma * (1 + mu))
                )
                / ((1 + mu) * (2 + mu) * (3 + 2 * mu))
                + (
                    4
                    * (alpha + beta)
                    * (1 + 2 * gamma * (2 + mu))
                    * (
                        (3 + 2 * mu)
                        * (
                            3 * np.exp(2 * mu * t)
                            + 4 * np.exp(2 * mu * t) * epsilon * (3 + mu)
                        )
                        + 12
                        * delta
                        * (
                            np.exp(mu * t) * (3 + mu)
                            + 4 * np.exp(mu * t) * epsilon * (3 + 2 * mu)
                        )
                    )
                )
                / (3 * (2 + mu) * (3 + mu) * (3 + 2 * mu))
                + np.exp(2 * (1 + mu) * t)
                * (
                    (2 * delta * (-2 - 8 * gamma + mu))
                    / (np.exp(2 * t) * (-6 + mu + mu**2))
                    - (
                        16
                        * delta
                        * gamma
                        * (
                            (-1 + 2 * beta * (-2 + mu)) * (2 + mu)
                            + 2 * alpha * (-2 + mu) * (2 + 8 * beta + mu)
                        )
                    )
                    / (np.exp(mu * t) * (-2 + mu) * (2 + mu) * (1 + 2 * mu))
                    + 2
                    * np.exp(mu * t)
                    * epsilon
                    * (
                        (2 + 8 * gamma - mu) / (np.exp(2 * t) * (-3 + mu) * (-2 + mu))
                        + (
                            (1 + mu) * (2 + 8 * beta + mu)
                            + 8 * alpha * (1 + mu + 2 * beta * (2 + mu))
                        )
                        / ((-1 + mu) * (1 + mu) * (2 + mu))
                    )
                    + (
                        -16
                        * delta
                        * epsilon
                        * (2 + 8 * gamma - mu)
                        * (2 + 3 * mu + mu**2)
                        + np.exp(mu * t) * (-2 - 8 * gamma + mu) * (2 + 3 * mu + mu**2)
                        - 3
                        * np.exp(2 * t + mu * t)
                        * (-2 + mu)
                        * (
                            (1 + mu) * (2 + 8 * beta + mu)
                            + 8 * alpha * (1 + mu + 2 * beta * (2 + mu))
                        )
                        - 48
                        * np.exp(2 * t)
                        * epsilon
                        * (
                            2
                            * gamma
                            * (1 + mu)
                            * (
                                (-1 + 2 * beta * (-2 + mu)) * (2 + mu)
                                + 2 * alpha * (-2 + mu) * (2 + 8 * beta + mu)
                            )
                            + delta
                            * (-2 + mu)
                            * (
                                (1 + mu) * (2 + 8 * beta + mu)
                                + 8 * alpha * (1 + mu + 2 * beta * (2 + mu))
                            )
                        )
                    )
                    / (6 * np.exp(2 * t) * (-2 + mu) * (1 + mu) * (2 + mu))
                    - (
                        2
                        * (
                            2
                            * np.exp(mu * t)
                            * gamma
                            * (1 + mu)
                            * (
                                (-1 + 2 * beta * (-2 + mu)) * (2 + mu)
                                + 2 * alpha * (-2 + mu) * (2 + 8 * beta + mu)
                            )
                            + delta
                            * (
                                32
                                * epsilon
                                * gamma
                                * (1 + mu)
                                * (
                                    (-1 + 2 * beta * (-2 + mu)) * (2 + mu)
                                    + 2 * alpha * (-2 + mu) * (2 + 8 * beta + mu)
                                )
                                + np.exp(mu * t)
                                * (-2 + mu)
                                * (
                                    (1 + mu) * (2 + 8 * beta + mu)
                                    + 8 * alpha * (1 + mu + 2 * beta * (2 + mu))
                                )
                            )
                        )
                    )
                    / (np.exp(mu * t) * (1 + mu) ** 2 * (-4 + mu**2))
                )
            )
            / np.exp(3 * (1 + mu) * t)
        )
    ) / (1024 * (1 + 0.5 / np.exp(3 * t) - 1.5 / np.exp(t)))
    return res


def p_b_c_d_given_a_JC69_analytical(t, mu):
    """
    Compute the probability P(b, c, d | a) for all nucleotide combinations under the JC69 model.

    :param t: Total time of the interval (from positions a/b/c to d).
    :type t: numeric
    :param mu: Mutation rate for the JC69 model.
    :type mu: numeric
    :return: A list of tuples (a, b, c, d, probability) where a, b, c, and d are nucleotide letters.
    :rtype: list
    """
    nt = ["A", "G", "C", "T"]
    result = []
    for aa in range(4):
        for bb in range(4):
            for cc in range(4):
                for dd in range(4):
                    cumsum = 0.0
                    for ee in range(4):
                        for ff in range(4):
                            cumsum += JC69_analytical_integral_double(
                                aa, bb, cc, dd, ee, ff, t, mu
                            )
                    result.append((nt[aa], nt[bb], nt[cc], nt[dd], cumsum))
    return result


def b_c_d_given_a_to_dict_a_b_c_d(data):
    """
    Convert a list of tuples (a, b, c, d, probability) into a nested dictionary. The resulting dictionary is structured as: dct[a][b][c][d] = probability

    :param data: List of tuples (a, b, c, d, probability), where a, b, c, d are nucleotide letters.
    :type data: list
    :return: Nested dictionary mapping nucleotide combinations to probability.
    :rtype: dict
    """
    dct = {}
    for a, b, c, d, prob in data:
        if a not in dct:
            dct[a] = {}
        if b not in dct[a]:
            dct[a][b] = {}
        if c not in dct[a][b]:
            dct[a][b][c] = {}
        dct[a][b][c][d] = prob
    return dct


def b_c_given_a_to_dict_a_b_c(data):
    """
    Convert a list of tuples (a, b, c, probability) into a nested dictionary. The resulting dictionary is structured as: dct[a][b][c] = probability

    :param data: List of tuples (a, b, c, probability).
    :type data: list
    :return: Nested dictionary mapping nucleotide combinations to probability.
    :rtype: dict
    """
    dct = {}
    for a, b, c, prob in data:
        if a not in dct:
            dct[a] = {}
        if b not in dct[a]:
            dct[a][b] = {}
        dct[a][b][c] = prob
    return dct


def b_given_a_to_dict_a_b(data):
    """
    Convert a list of tuples (a, b, probability) into a nested dictionary. The resulting dictionary is structured as: dct[a][b] = probability

    :param data: List of tuples (a, b, probability).
    :type data: list
    :return: Dictionary mapping nucleotide 'a' to a dictionary mapping 'b' to probability.
    :rtype: dict
    """
    dct = {}
    for a, b, prob in data:
        if a not in dct:
            dct[a] = {}
        dct[a][b] = prob
    return dct


def calc_emissions_single_JC69(
    a0_a1_t_vec,
    b0_b1_t_vec,
    a1b1_ab0_t,
    ab0_ab1_t_vec,
    ab1c1_abc0_t,
    c0_c1_t_vec,
    d0_abc0_t_vec,
    a0_a1_mu_vec,
    b0_b1_mu_vec,
    a1b1_ab0_mu,
    ab0_ab1_mu_vec,
    ab1c1_abc0_mu,
    c0_c1_mu_vec,
    d0_abc0_mu_vec,
    coal_rate_1,
    coal_rate_2,
):
    """
    Compute the emission probabilities for a hidden state containing two coalescent events occurring at different time intervals.

    :param a0_a1_t_vec: List of time intervals for mutation from a0 to a1.
    :type a0_a1_t_vec: list of numeric
    :param b0_b1_t_vec: List of time intervals for mutation from b0 to b1.
    :type b0_b1_t_vec: list of numeric
    :param c0_c1_t_vec: List of time intervals for mutation from c0 to c1.
    :type c0_c1_t_vec: list of numeric
    :param d0_abc0_t_vec: List of time intervals for mutation from d0 to abc0.
    :type d0_abc0_t_vec: list of numeric
    :param ab0_ab1_t_vec: List of time intervals for mutation from ab0 to ab1.
    :type ab0_ab1_t_vec: list of numeric
    :param a1b1_ab0_t: Time interval for the first coalescent event.
    :type a1b1_ab0_t: numeric
    :param ab1c1_abc0_t: Time interval for the second coalescent event.
    :type ab1c1_abc0_t: numeric
    :param a0_a1_mu_vec: Mutation rates for the intervals in a0_a1_t_vec.
    :type a0_a1_mu_vec: list of numeric
    :param b0_b1_mu_vec: Mutation rates for the intervals in b0_b1_t_vec.
    :type b0_b1_mu_vec: list of numeric
    :param c0_c1_mu_vec: Mutation rates for the intervals in c0_c1_t_vec.
    :type c0_c1_mu_vec: list of numeric
    :param d0_abc0_mu_vec: Mutation rates for the intervals in d0_abc0_t_vec.
    :type d0_abc0_mu_vec: list of numeric
    :param ab0_ab1_mu_vec: Mutation rates for the intervals in ab0_ab1_t_vec.
    :type ab0_ab1_mu_vec: list of numeric
    :param a1b1_ab0_mu: Mutation rate for the first coalescent interval.
    :type a1b1_ab0_mu: numeric
    :param ab1c1_abc0_mu: Mutation rate for the second coalescent interval.
    :type ab1c1_abc0_mu: numeric
    :param coal_rate_1: Coalescent rate for the first coalescent event.
    :type coal_rate_1: numeric
    :param coal_rate_2: Coalescent rate for the second coalescent event.
    :type coal_rate_2: numeric
    :return: A dictionary mapping concatenated nucleotide strings to their emission probabilities.
    :rtype: dict
    """

    # a0 to a1
    Q_vec = [rate_mat_JC69(i) for i in a0_a1_mu_vec]
    df_a = p_b_given_a(t=a0_a1_t_vec, Q=Q_vec)
    df_a = b_given_a_to_dict_a_b(df_a)
    # df_a[a0][a1]

    # b1 to b0
    Q_vec = [rate_mat_JC69(i) for i in list(reversed(b0_b1_mu_vec))]
    df_b = p_b_given_a(t=list(reversed(b0_b1_t_vec)), Q=Q_vec)
    df_b = b_given_a_to_dict_a_b(df_b)
    # df_b[b1][b0]

    # c1 to c0
    Q_vec = [rate_mat_JC69(i) for i in list(reversed(c0_c1_mu_vec))]
    df_c = p_b_given_a(t=list(reversed(c0_c1_t_vec)), Q=Q_vec)
    df_c = b_given_a_to_dict_a_b(df_c)
    # df_c[c1][c0]

    # abc0 to d0
    Q_vec = [rate_mat_JC69(i) for i in list(reversed(d0_abc0_mu_vec))]
    df_d = p_b_given_a(t=list(reversed(d0_abc0_t_vec)), Q=Q_vec)
    df_d = b_given_a_to_dict_a_b(df_d)
    # df_d[abc0][d0]

    # ab0 to ab1
    Q_vec = [rate_mat_JC69(i) for i in ab0_ab1_mu_vec]
    df_ab = p_b_given_a(t=ab0_ab1_t_vec, Q=Q_vec)
    df_ab = b_given_a_to_dict_a_b(df_ab)
    # df_ab[ab0][ab1]

    # First coalescent
    df_first = p_b_c_given_a_JC69_analytical(
        t=a1b1_ab0_t, mu=a1b1_ab0_mu, k=coal_rate_1
    )
    df_first = b_c_given_a_to_dict_a_b_c(df_first)
    # df_first[a1][b1][ab0]

    # Second coalescent
    df_second = p_b_c_given_a_JC69_analytical(
        t=ab1c1_abc0_t, mu=ab1c1_abc0_mu, k=coal_rate_2
    )
    df_second = b_c_given_a_to_dict_a_b_c(df_second)
    # df_second[ab1][c1][abc0]

    emissions = {}
    for a0 in ["A", "C", "T", "G"]:
        for b0 in ["A", "C", "T", "G"]:
            for c0 in ["A", "C", "T", "G"]:
                for d0 in ["A", "C", "T", "G"]:
                    acc = 0
                    for a1 in ["A", "C", "T", "G"]:
                        for b1 in ["A", "C", "T", "G"]:
                            for c1 in ["A", "C", "T", "G"]:
                                for ab0 in ["A", "C", "T", "G"]:
                                    for ab1 in ["A", "C", "T", "G"]:
                                        for abc0 in ["A", "C", "T", "G"]:
                                            res = 1
                                            res = res * df_a[a0][a1]
                                            res = res * df_b[b1][b0]
                                            res = res * df_first[a1][b1][ab0]
                                            res = res * df_ab[ab0][ab1]
                                            res = res * df_second[ab1][c1][abc0]
                                            res = res * df_c[c1][c0]
                                            res = res * df_d[abc0][d0]
                                            acc += res
                    emissions[a0 + b0 + c0 + d0] = acc / 4

    return emissions


def calc_emissions_double_JC69(
    a0_a1_t_vec,
    b0_b1_t_vec,
    c0_c1_t_vec,
    a1b1c1_abc0_t,
    d0_abc0_t_vec,
    a0_a1_mu_vec,
    b0_b1_mu_vec,
    c0_c1_mu_vec,
    a1b1c1_abc0_mu,
    d0_abc0_mu_vec,
):
    """
    Compute the emission probabilities for a hidden state containing two coalescent events
    occurring in the same time interval.

    :param a0_a1_t_vec: List of time intervals for mutation from a0 to a1.
    :type a0_a1_t_vec: list of numeric
    :param b0_b1_t_vec: List of time intervals for mutation from b0 to b1.
    :type b0_b1_t_vec: list of numeric
    :param c0_c1_t_vec: List of time intervals for mutation from c0 to c1.
    :type c0_c1_t_vec: list of numeric
    :param d0_abc0_t_vec: List of time intervals for mutation from d0 to abc0.
    :type d0_abc0_t_vec: list of numeric
    :param a1b1c1_abc0_t: Time interval for the coalescent events.
    :type a1b1c1_abc0_t: numeric
    :param a0_a1_mu_vec: Mutation rates for intervals in a0_a1_t_vec.
    :type a0_a1_mu_vec: list of numeric
    :param b0_b1_mu_vec: Mutation rates for intervals in b0_b1_t_vec.
    :type b0_b1_mu_vec: list of numeric
    :param c0_c1_mu_vec: Mutation rates for intervals in c0_c1_t_vec.
    :type c0_c1_mu_vec: list of numeric
    :param a1b1c1_abc0_mu: Mutation rate for the coalescent interval.
    :type a1b1c1_abc0_mu: numeric
    :param d0_abc0_mu_vec: Mutation rates for intervals in d0_abc0_t_vec.
    :type d0_abc0_mu_vec: list of numeric
    :return: A dictionary mapping concatenated nucleotide strings to their emission probabilities.
    :rtype: dict
    """

    # a0 to a1
    Q_vec = [rate_mat_JC69(i) for i in a0_a1_mu_vec]
    df_a = p_b_given_a(t=a0_a1_t_vec, Q=Q_vec)
    df_a = b_given_a_to_dict_a_b(df_a)
    # df_a[a0][a1]

    # b1 to b0
    Q_vec = [rate_mat_JC69(i) for i in list(reversed(b0_b1_mu_vec))]
    df_b = p_b_given_a(t=list(reversed(b0_b1_t_vec)), Q=Q_vec)
    df_b = b_given_a_to_dict_a_b(df_b)
    # df_b[b1][b0]

    # c1 to c0
    Q_vec = [rate_mat_JC69(i) for i in list(reversed(c0_c1_mu_vec))]
    df_c = p_b_given_a(t=list(reversed(c0_c1_t_vec)), Q=Q_vec)
    df_c = b_given_a_to_dict_a_b(df_c)
    # df_c[c1][c0]

    # abc0 to d0
    Q_vec = [rate_mat_JC69(i) for i in list(reversed(d0_abc0_mu_vec))]
    df_d = p_b_given_a(t=list(reversed(d0_abc0_t_vec)), Q=Q_vec)
    df_d = b_given_a_to_dict_a_b(df_d)
    # df_d[abc0][d0]

    # Double coalescent
    df_double = p_b_c_d_given_a_JC69_analytical(t=a1b1c1_abc0_t, mu=a1b1c1_abc0_mu)
    df_double = b_c_d_given_a_to_dict_a_b_c_d(df_double)
    # df_double[a1][b1][c1][abc0]

    emissions = {}
    for a0 in ["A", "C", "T", "G"]:
        for b0 in ["A", "C", "T", "G"]:
            for c0 in ["A", "C", "T", "G"]:
                for d0 in ["A", "C", "T", "G"]:
                    acc = 0
                    for a1 in ["A", "C", "T", "G"]:
                        for b1 in ["A", "C", "T", "G"]:
                            for c1 in ["A", "C", "T", "G"]:
                                for abc0 in ["A", "C", "T", "G"]:
                                    res = 1
                                    res = res * df_a[a0][a1]
                                    res = res * df_b[b1][b0]
                                    res = res * df_c[c1][c0]
                                    res = res * df_double[a1][b1][c1][abc0]
                                    res = res * df_d[abc0][d0]
                                    acc += res
                    emissions[a0 + b0 + c0 + d0] = acc / 4
    return emissions


def get_emission_prob_mat(
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
    cut_AB="standard",
    cut_ABC="standard",
):
    """
    Compute the emission probabilities for all hidden states given a set of population genetics parameters.

    :param t_A: Time between present and the first speciation event for species A.
    :type t_A: numeric
    :param t_B: Time between present and the first speciation event for species B (should equal t_A).
    :type t_B: numeric
    :param t_AB: Time between speciation events.
    :type t_AB: numeric
    :param t_C: Time between present and the second speciation event for species C (should equal t_A + t_AB).
    :type t_C: numeric
    :param t_upper: Time between the last ABC interval and the third speciation event.
    :type t_upper: numeric
    :param t_out: Time from present to the third speciation event for species D (includes divergence time).
    :type t_out: numeric
    :param rho_A: Recombination rate for species A.
    :type rho_A: numeric
    :param rho_B: Recombination rate for species B.
    :type rho_B: numeric
    :param rho_AB: Recombination rate for the AB interval.
    :type rho_AB: numeric
    :param rho_C: Recombination rate for species C.
    :type rho_C: numeric
    :param rho_ABC: Recombination rate for the ABC interval.
    :type rho_ABC: numeric
    :param coal_A: Coalescent rate for species A.
    :type coal_A: numeric
    :param coal_B: Coalescent rate for species B.
    :type coal_B: numeric
    :param coal_AB: Coalescent rate for the AB interval.
    :type coal_AB: numeric
    :param coal_C: Coalescent rate for species C.
    :type coal_C: numeric
    :param coal_ABC: Coalescent rate for the ABC interval.
    :type coal_ABC: numeric
    :param n_int_AB: Number of intervals in the AB portion of the tree.
    :type n_int_AB: int
    :param n_int_ABC: Number of intervals in the ABC portion of the tree.
    :type n_int_ABC: int
    :param mu_A: Mutation rate for species A.
    :type mu_A: numeric
    :param mu_B: Mutation rate for species B.
    :type mu_B: numeric
    :param mu_C: Mutation rate for species C.
    :type mu_C: numeric
    :param mu_D: Mutation rate for species D.
    :type mu_D: numeric
    :param mu_AB: Mutation rate for the AB interval.
    :type mu_AB: numeric
    :param mu_ABC: Mutation rate for the ABC interval.
    :type mu_ABC: numeric
    :param cut_AB: Option for cutpoints in the AB interval; if "standard", default cutpoints are computed.
    :type cut_AB: str or array-like
    :param cut_ABC: Option for cutpoints in the ABC interval; if "standard", default cutpoints are computed.
    :type cut_ABC: str or array-like
    :return: A tuple containing:
             - A list of state identifiers (tuples).
             - A list of corresponding emission probability dictionaries.
    :rtype: tuple(list, list)
    """
    n_markov_states = (
        n_int_AB * n_int_ABC + n_int_ABC * 3 + 3 * comb(n_int_ABC, 2, exact=True)
    )
    if cut_AB == "standard":
        cut_AB = cutpoints_AB(n_int_AB, t_AB, coal_AB)
    if cut_ABC == "standard":
        cut_ABC = cutpoints_ABC(n_int_ABC, coal_ABC)
    probs = np.empty((n_markov_states), dtype=object)
    states = np.empty((n_markov_states), dtype=object)

    # Deep coalescence, two single coalescents
    acc = 0
    for i in range(n_int_ABC):
        for j in range(i + 1, n_int_ABC):

            a0_a1_t_vec = [t_A, t_AB, cut_ABC[i]]
            a0_a1_mu_vec = [mu_A, mu_AB, mu_ABC]
            b0_b1_t_vec = [t_B, t_AB, cut_ABC[i]]
            b0_b1_mu_vec = [mu_B, mu_AB, mu_ABC]
            c0_c1_t_vec = [t_C, cut_ABC[i]]
            c0_c1_mu_vec = [mu_C, mu_ABC]
            ab0_ab1_t_vec = [cut_ABC[j] - cut_ABC[i + 1]]
            ab0_ab1_mu_vec = [mu_ABC]
            a1b1_ab0_t = cut_ABC[i + 1] - cut_ABC[i]
            a1b1_ab0_mu = mu_ABC
            ab1c1_abc0_t = (
                (cut_ABC[j + 1] - cut_ABC[j]) if j != (n_int_ABC - 1) else t_upper
            )
            ab1c1_abc0_mu = mu_ABC
            add = (
                t_upper + cut_ABC[n_int_ABC - 1] - cut_ABC[j + 1]
                if j != (n_int_ABC - 1)
                else 0
            )
            # d0_abc0_t_vec = [t_A+t_AB+cut_ABC[n_int_ABC-1]+t_upper]+[t_peak+add]
            d0_abc0_t_vec = [t_out] + [add]
            # d0_abc0_mu_vec = [mu_D, mu_ABC]
            d0_abc0_mu_vec = [mu_D, mu_ABC]

            # V1 states
            emissions = calc_emissions_single_JC69(
                a0_a1_t_vec,
                b0_b1_t_vec,
                a1b1_ab0_t,
                ab0_ab1_t_vec,
                ab1c1_abc0_t,
                c0_c1_t_vec,
                d0_abc0_t_vec,
                a0_a1_mu_vec,
                b0_b1_mu_vec,
                a1b1_ab0_mu,
                ab0_ab1_mu_vec,
                ab1c1_abc0_mu,
                c0_c1_mu_vec,
                d0_abc0_mu_vec,
                coal_ABC,
                coal_ABC,
            )
            states[acc] = (1, i, j)
            probs[acc] = emissions
            acc += 1

            # V2 states
            emissions = calc_emissions_single_JC69(
                a0_a1_t_vec,
                c0_c1_t_vec,
                a1b1_ab0_t,
                ab0_ab1_t_vec,
                ab1c1_abc0_t,
                b0_b1_t_vec,
                d0_abc0_t_vec,
                a0_a1_mu_vec,
                c0_c1_mu_vec,
                a1b1_ab0_mu,
                ab0_ab1_mu_vec,
                ab1c1_abc0_mu,
                b0_b1_mu_vec,
                d0_abc0_mu_vec,
                coal_ABC,
                coal_ABC,
            )
            new_emissions = {}
            for k in list(emissions.keys()):
                new_emissions[k[0] + k[2] + k[1] + k[3]] = emissions[k]
            states[acc] = (2, i, j)
            probs[acc] = new_emissions
            acc += 1

            # V3 states
            emissions = calc_emissions_single_JC69(
                b0_b1_t_vec,
                c0_c1_t_vec,
                a1b1_ab0_t,
                ab0_ab1_t_vec,
                ab1c1_abc0_t,
                a0_a1_t_vec,
                d0_abc0_t_vec,
                b0_b1_mu_vec,
                c0_c1_mu_vec,
                a1b1_ab0_mu,
                ab0_ab1_mu_vec,
                ab1c1_abc0_mu,
                a0_a1_mu_vec,
                d0_abc0_mu_vec,
                coal_ABC,
                coal_ABC,
            )
            new_emissions = {}
            for k in list(emissions.keys()):
                new_emissions[k[2] + k[0] + k[1] + k[3]] = emissions[k]
            states[acc] = (3, i, j)
            probs[acc] = new_emissions
            acc += 1

    # Deep coalescence, one double coalescent
    for i in range(n_int_ABC):

        a0_a1_t_vec = [t_A, t_AB, cut_ABC[i]]
        a0_a1_mu_vec = [mu_A, mu_AB, mu_ABC]
        b0_b1_t_vec = [t_B, t_AB, cut_ABC[i]]
        b0_b1_mu_vec = [mu_B, mu_AB, mu_ABC]
        c0_c1_t_vec = [t_C, cut_ABC[i]]
        c0_c1_mu_vec = [mu_C, mu_ABC]
        a1b1c1_abc0_t = (
            (cut_ABC[i + 1] - cut_ABC[i]) if i != (n_int_ABC - 1) else t_upper
        )
        a1b1c1_abc0_mu = mu_ABC
        add = (
            t_upper + cut_ABC[n_int_ABC - 1] - cut_ABC[i + 1]
            if i != (n_int_ABC - 1)
            else 0
        )
        # d0_abc0_t_vec = [t_A+t_AB+cut_ABC[n_int_ABC-1]+t_upper]+[t_peak+add]
        d0_abc0_t_vec = [t_out] + [add]
        # d0_abc0_mu_vec = [mu_D, mu_ABC]
        d0_abc0_mu_vec = [mu_D, mu_ABC]

        # V1 states
        emissions = calc_emissions_double_JC69(
            a0_a1_t_vec,
            b0_b1_t_vec,
            c0_c1_t_vec,
            a1b1c1_abc0_t,
            d0_abc0_t_vec,
            a0_a1_mu_vec,
            b0_b1_mu_vec,
            c0_c1_mu_vec,
            a1b1c1_abc0_mu,
            d0_abc0_mu_vec,
        )
        markov = (1, i, i)
        states[acc] = markov
        probs[acc] = emissions
        acc += 1

        # V2 states
        emissions = calc_emissions_double_JC69(
            a0_a1_t_vec,
            c0_c1_t_vec,
            b0_b1_t_vec,
            a1b1c1_abc0_t,
            d0_abc0_t_vec,
            a0_a1_mu_vec,
            c0_c1_mu_vec,
            b0_b1_mu_vec,
            a1b1c1_abc0_mu,
            d0_abc0_mu_vec,
        )
        new_emissions = {}
        for k in list(emissions.keys()):
            new_emissions[k[0] + k[2] + k[1] + k[3]] = emissions[k]
        markov = (2, i, i)
        states[acc] = markov
        probs[acc] = new_emissions
        acc += 1

        # V3 states
        emissions = calc_emissions_double_JC69(
            b0_b1_t_vec,
            c0_c1_t_vec,
            a0_a1_t_vec,
            a1b1c1_abc0_t,
            d0_abc0_t_vec,
            b0_b1_mu_vec,
            c0_c1_mu_vec,
            a0_a1_mu_vec,
            a1b1c1_abc0_mu,
            d0_abc0_mu_vec,
        )
        new_emissions = {}
        for k in list(emissions.keys()):
            new_emissions[k[2] + k[0] + k[1] + k[3]] = emissions[k]
        markov = (3, i, i)
        states[acc] = markov
        probs[acc] = new_emissions
        acc += 1

    # V0 states
    for i in range(n_int_AB):
        for j in range(n_int_ABC):
            a0_a1_t_vec = [t_A, cut_AB[i]]
            a0_a1_mu_vec = [mu_A, mu_AB]
            b0_b1_t_vec = [t_B, cut_AB[i]]
            b0_b1_mu_vec = [mu_B, mu_AB]
            c0_c1_t_vec = [t_C, cut_ABC[j]]
            c0_c1_mu_vec = [mu_C, mu_ABC]
            ab0_ab1_t_vec = [t_AB - cut_AB[i + 1], cut_ABC[j]]
            ab0_ab1_mu_vec = [mu_AB, mu_ABC]
            a1b1_ab0_t = cut_AB[i + 1] - cut_AB[i]
            a1b1_ab0_mu = mu_AB
            ab1c1_abc0_t = (
                cut_ABC[j + 1] - cut_ABC[j] if j != (n_int_ABC - 1) else t_upper
            )
            ab1c1_abc0_mu = mu_ABC
            add = (
                t_upper + cut_ABC[n_int_ABC - 1] - cut_ABC[j + 1]
                if j != (n_int_ABC - 1)
                else 0
            )
            # d0_abc0_t_vec = [t_A+t_AB+cut_ABC[n_int_ABC-1]+t_upper]+[t_peak+add]
            d0_abc0_t_vec = [t_out] + [add]
            # d0_abc0_mu_vec = [mu_D, mu_ABC]
            d0_abc0_mu_vec = [mu_D, mu_ABC]

            emissions = calc_emissions_single_JC69(
                a0_a1_t_vec,
                b0_b1_t_vec,
                a1b1_ab0_t,
                ab0_ab1_t_vec,
                ab1c1_abc0_t,
                c0_c1_t_vec,
                d0_abc0_t_vec,
                a0_a1_mu_vec,
                b0_b1_mu_vec,
                a1b1_ab0_mu,
                ab0_ab1_mu_vec,
                ab1c1_abc0_mu,
                c0_c1_mu_vec,
                d0_abc0_mu_vec,
                coal_AB,
                coal_ABC,
            )
            states[acc] = (0, i, j)
            probs[acc] = emissions
            acc += 1

    # Instead of building a pandas DataFrame, return the two lists.
    # Each element of 'probs' is an emission dictionary for a hidden state.
    # Each element of 'states' is the corresponding hidden state identifier (e.g., a tuple).
    return list(states), list(probs)


def get_emission_prob_mat_introgression(
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
    cut_AB="standard",
    cut_ABC="standard",
):
    """
    Compute the emission probabilities for all hidden states in an introgression model given a set of population genetics parameters.

    :param t_A: Time between present and the first speciation event for species A.
    :type t_A: numeric
    :param t_B: Time between present and the migration event for species A.
    :type t_B: numeric
    :param t_AB: Time between speciation events.
    :type t_AB: numeric
    :param t_C: Time between present and the migration event for species C.
    :type t_C: numeric
    :param t_upper: Time between the last ABC interval and the third speciation event.
    :type t_upper: numeric
    :param t_out: Time from present to the third speciation event for species D (includes divergence time).
    :type t_out: numeric
    :param t_m: Additional time parameter associated with migration.
    :type t_m: numeric
    :param rho_A: Recombination rate for species A.
    :type rho_A: numeric
    :param rho_B: Recombination rate for species B.
    :type rho_B: numeric
    :param rho_AB: Recombination rate for the AB interval.
    :type rho_AB: numeric
    :param rho_C: Recombination rate for species C.
    :type rho_C: numeric
    :param rho_ABC: Recombination rate for the ABC interval.
    :type rho_ABC: numeric
    :param coal_A: Coalescent rate for species A.
    :type coal_A: numeric
    :param coal_B: Coalescent rate for species B.
    :type coal_B: numeric
    :param coal_AB: Coalescent rate for the AB interval.
    :type coal_AB: numeric
    :param coal_BC: Coalescent rate for the BC interval.
    :type coal_BC: numeric
    :param coal_C: Coalescent rate for species C.
    :type coal_C: numeric
    :param coal_ABC: Coalescent rate for the ABC interval.
    :type coal_ABC: numeric
    :param n_int_AB: Number of intervals in the AB part of the tree.
    :type n_int_AB: int
    :param n_int_ABC: Number of intervals in the ABC part of the tree.
    :type n_int_ABC: int
    :param mu_A: Mutation rate for species A.
    :type mu_A: numeric
    :param mu_B: Mutation rate for species B.
    :type mu_B: numeric
    :param mu_C: Mutation rate for species C.
    :type mu_C: numeric
    :param mu_D: Mutation rate for species D.
    :type mu_D: numeric
    :param mu_AB: Mutation rate for the AB interval.
    :type mu_AB: numeric
    :param mu_ABC: Mutation rate for the ABC interval.
    :type mu_ABC: numeric
    :param cut_AB: Option for cutpoints in the AB interval; if "standard", default cutpoints are computed.
    :type cut_AB: str or array-like
    :param cut_ABC: Option for cutpoints in the ABC interval; if "standard", default cutpoints are computed.
    :type cut_ABC: str or array-like
    :return: A tuple containing:
             - A list of state identifiers (tuples).
             - A list of corresponding emission probability dictionaries.
    :rtype: tuple(list, list)
    """
    n_markov_states = (
        2 * n_int_AB * n_int_ABC + n_int_ABC * 3 + 3 * comb(n_int_ABC, 2, exact=True)
    )
    if cut_AB == "standard":
        cut_AB = cutpoints_AB(n_int_AB, t_AB, coal_AB)
    cut_BC = np.concatenate([[0], (cut_AB[1:] + t_m)])
    if cut_ABC == "standard":
        cut_ABC = cutpoints_ABC(n_int_ABC, coal_ABC)
    probs = np.empty((n_markov_states), dtype=object)
    states = np.empty((n_markov_states), dtype=object)

    # Deep coalescence, two single coalescents
    acc = 0
    for i in range(n_int_ABC):
        for j in range(i + 1, n_int_ABC):

            a0_a1_t_vec = [t_A, t_AB, cut_ABC[i]]
            a0_a1_mu_vec = [mu_A, mu_AB, mu_ABC]
            b0_b1_t_vec = [t_B + t_m, t_AB, cut_ABC[i]]
            b0_b1_mu_vec = [mu_B, mu_AB, mu_ABC]
            c0_c1_t_vec = [t_C + t_m + t_AB, cut_ABC[i]]
            c0_c1_mu_vec = [mu_C, mu_ABC]
            ab0_ab1_t_vec = [cut_ABC[j] - cut_ABC[i + 1]]
            ab0_ab1_mu_vec = [mu_ABC]
            a1b1_ab0_t = cut_ABC[i + 1] - cut_ABC[i]
            a1b1_ab0_mu = mu_ABC
            ab1c1_abc0_t = (
                (cut_ABC[j + 1] - cut_ABC[j]) if j != (n_int_ABC - 1) else t_upper
            )
            ab1c1_abc0_mu = mu_ABC
            add = (
                t_upper + cut_ABC[n_int_ABC - 1] - cut_ABC[j + 1]
                if j != (n_int_ABC - 1)
                else 0
            )
            # d0_abc0_t_vec = [t_A+t_AB+cut_ABC[n_int_ABC-1]+t_upper]+[t_peak+add]
            d0_abc0_t_vec = [t_out] + [add]
            # d0_abc0_mu_vec = [mu_D, mu_ABC]
            d0_abc0_mu_vec = [mu_D, mu_ABC]

            # V1 states
            emissions = calc_emissions_single_JC69(
                a0_a1_t_vec,
                b0_b1_t_vec,
                a1b1_ab0_t,
                ab0_ab1_t_vec,
                ab1c1_abc0_t,
                c0_c1_t_vec,
                d0_abc0_t_vec,
                a0_a1_mu_vec,
                b0_b1_mu_vec,
                a1b1_ab0_mu,
                ab0_ab1_mu_vec,
                ab1c1_abc0_mu,
                c0_c1_mu_vec,
                d0_abc0_mu_vec,
                coal_ABC,
                coal_ABC,
            )
            states[acc] = (1, i, j)
            probs[acc] = emissions
            acc += 1

            # V2 states
            emissions = calc_emissions_single_JC69(
                a0_a1_t_vec,
                c0_c1_t_vec,
                a1b1_ab0_t,
                ab0_ab1_t_vec,
                ab1c1_abc0_t,
                b0_b1_t_vec,
                d0_abc0_t_vec,
                a0_a1_mu_vec,
                c0_c1_mu_vec,
                a1b1_ab0_mu,
                ab0_ab1_mu_vec,
                ab1c1_abc0_mu,
                b0_b1_mu_vec,
                d0_abc0_mu_vec,
                coal_ABC,
                coal_ABC,
            )
            new_emissions = {}
            for k in list(emissions.keys()):
                new_emissions[k[0] + k[2] + k[1] + k[3]] = emissions[k]
            states[acc] = (2, i, j)
            probs[acc] = new_emissions
            acc += 1

            # V3 states
            emissions = calc_emissions_single_JC69(
                b0_b1_t_vec,
                c0_c1_t_vec,
                a1b1_ab0_t,
                ab0_ab1_t_vec,
                ab1c1_abc0_t,
                a0_a1_t_vec,
                d0_abc0_t_vec,
                b0_b1_mu_vec,
                c0_c1_mu_vec,
                a1b1_ab0_mu,
                ab0_ab1_mu_vec,
                ab1c1_abc0_mu,
                a0_a1_mu_vec,
                d0_abc0_mu_vec,
                coal_ABC,
                coal_ABC,
            )
            new_emissions = {}
            for k in list(emissions.keys()):
                new_emissions[k[2] + k[0] + k[1] + k[3]] = emissions[k]
            states[acc] = (3, i, j)
            probs[acc] = new_emissions
            acc += 1

    # Deep coalescence, one double coalescent
    for i in range(n_int_ABC):

        a0_a1_t_vec = [t_A, t_AB, cut_ABC[i]]
        a0_a1_mu_vec = [mu_A, mu_AB, mu_ABC]
        b0_b1_t_vec = [t_B + t_m, t_AB, cut_ABC[i]]
        b0_b1_mu_vec = [mu_B, mu_AB, mu_ABC]
        c0_c1_t_vec = [t_C + t_m + t_AB, cut_ABC[i]]
        c0_c1_mu_vec = [mu_C, mu_ABC]
        a1b1c1_abc0_t = (
            (cut_ABC[i + 1] - cut_ABC[i]) if i != (n_int_ABC - 1) else t_upper
        )
        a1b1c1_abc0_mu = mu_ABC
        add = (
            t_upper + cut_ABC[n_int_ABC - 1] - cut_ABC[i + 1]
            if i != (n_int_ABC - 1)
            else 0
        )
        # d0_abc0_t_vec = [t_A+t_AB+cut_ABC[n_int_ABC-1]+t_upper]+[t_peak+add]
        d0_abc0_t_vec = [t_out] + [add]
        # d0_abc0_mu_vec = [mu_D, mu_ABC]
        d0_abc0_mu_vec = [mu_D, mu_ABC]

        # V1 states
        emissions = calc_emissions_double_JC69(
            a0_a1_t_vec,
            b0_b1_t_vec,
            c0_c1_t_vec,
            a1b1c1_abc0_t,
            d0_abc0_t_vec,
            a0_a1_mu_vec,
            b0_b1_mu_vec,
            c0_c1_mu_vec,
            a1b1c1_abc0_mu,
            d0_abc0_mu_vec,
        )
        markov = (1, i, i)
        states[acc] = markov
        probs[acc] = emissions
        acc += 1

        # V2 states
        emissions = calc_emissions_double_JC69(
            a0_a1_t_vec,
            c0_c1_t_vec,
            b0_b1_t_vec,
            a1b1c1_abc0_t,
            d0_abc0_t_vec,
            a0_a1_mu_vec,
            c0_c1_mu_vec,
            b0_b1_mu_vec,
            a1b1c1_abc0_mu,
            d0_abc0_mu_vec,
        )
        new_emissions = {}
        for k in list(emissions.keys()):
            new_emissions[k[0] + k[2] + k[1] + k[3]] = emissions[k]
        markov = (2, i, i)
        states[acc] = markov
        probs[acc] = new_emissions
        acc += 1

        # V3 states
        emissions = calc_emissions_double_JC69(
            b0_b1_t_vec,
            c0_c1_t_vec,
            a0_a1_t_vec,
            a1b1c1_abc0_t,
            d0_abc0_t_vec,
            b0_b1_mu_vec,
            c0_c1_mu_vec,
            a0_a1_mu_vec,
            a1b1c1_abc0_mu,
            d0_abc0_mu_vec,
        )
        new_emissions = {}
        for k in list(emissions.keys()):
            new_emissions[k[2] + k[0] + k[1] + k[3]] = emissions[k]
        markov = (3, i, i)
        states[acc] = markov
        probs[acc] = new_emissions
        acc += 1

    # V0 states
    for i in range(n_int_AB):
        for j in range(n_int_ABC):
            a0_a1_t_vec = [t_A, cut_AB[i]]
            a0_a1_mu_vec = [mu_A, mu_AB]
            b0_b1_t_vec = [t_B + t_m, cut_AB[i]]
            b0_b1_mu_vec = [mu_B, mu_AB]
            c0_c1_t_vec = [t_C + t_m + t_AB, cut_ABC[j]]
            c0_c1_mu_vec = [mu_C, mu_ABC]
            ab0_ab1_t_vec = [t_AB - cut_AB[i + 1], cut_ABC[j]]
            ab0_ab1_mu_vec = [mu_AB, mu_ABC]
            a1b1_ab0_t = cut_AB[i + 1] - cut_AB[i]
            a1b1_ab0_mu = mu_AB
            ab1c1_abc0_t = (
                cut_ABC[j + 1] - cut_ABC[j] if j != (n_int_ABC - 1) else t_upper
            )
            ab1c1_abc0_mu = mu_ABC
            add = (
                t_upper + cut_ABC[n_int_ABC - 1] - cut_ABC[j + 1]
                if j != (n_int_ABC - 1)
                else 0
            )
            # d0_abc0_t_vec = [t_A+t_AB+cut_ABC[n_int_ABC-1]+t_upper]+[t_peak+add]
            d0_abc0_t_vec = [t_out] + [add]
            # d0_abc0_mu_vec = [mu_D, mu_ABC]
            d0_abc0_mu_vec = [mu_D, mu_ABC]

            emissions = calc_emissions_single_JC69(
                a0_a1_t_vec,
                b0_b1_t_vec,
                a1b1_ab0_t,
                ab0_ab1_t_vec,
                ab1c1_abc0_t,
                c0_c1_t_vec,
                d0_abc0_t_vec,
                a0_a1_mu_vec,
                b0_b1_mu_vec,
                a1b1_ab0_mu,
                ab0_ab1_mu_vec,
                ab1c1_abc0_mu,
                c0_c1_mu_vec,
                d0_abc0_mu_vec,
                coal_AB,
                coal_ABC,
            )
            states[acc] = (0, i, j)
            probs[acc] = emissions
            acc += 1

    # Introgression states
    for i in range(n_int_AB):
        for j in range(n_int_ABC):
            a0_a1_t_vec = [t_B, cut_BC[i]]
            a0_a1_mu_vec = [mu_B, mu_AB]
            b0_b1_t_vec = [t_C, cut_BC[i]]
            b0_b1_mu_vec = [mu_C, mu_AB]
            c0_c1_t_vec = [t_A, t_AB, cut_ABC[j]]
            c0_c1_mu_vec = [mu_A, mu_AB, mu_ABC]
            ab0_ab1_t_vec = [t_AB + t_m - cut_BC[i + 1], cut_ABC[j]]
            ab0_ab1_mu_vec = [mu_AB, mu_ABC]
            a1b1_ab0_t = cut_BC[i + 1] - cut_BC[i]
            a1b1_ab0_mu = mu_AB
            ab1c1_abc0_t = (
                cut_ABC[j + 1] - cut_ABC[j] if j != (n_int_ABC - 1) else t_upper
            )
            ab1c1_abc0_mu = mu_ABC
            add = (
                t_upper + cut_ABC[n_int_ABC - 1] - cut_ABC[j + 1]
                if j != (n_int_ABC - 1)
                else 0
            )
            # d0_abc0_t_vec = [t_A+t_AB+cut_ABC[n_int_ABC-1]+t_upper]+[t_peak+add]
            d0_abc0_t_vec = [t_out] + [add]
            # d0_abc0_mu_vec = [mu_D, mu_ABC]
            d0_abc0_mu_vec = [mu_D, mu_ABC]

            emissions = calc_emissions_single_JC69(
                a0_a1_t_vec,
                b0_b1_t_vec,
                a1b1_ab0_t,
                ab0_ab1_t_vec,
                ab1c1_abc0_t,
                c0_c1_t_vec,
                d0_abc0_t_vec,
                a0_a1_mu_vec,
                b0_b1_mu_vec,
                a1b1_ab0_mu,
                ab0_ab1_mu_vec,
                ab1c1_abc0_mu,
                c0_c1_mu_vec,
                d0_abc0_mu_vec,
                coal_BC,
                coal_ABC,
            )

            new_emissions = {}
            for k in list(emissions.keys()):
                new_emissions[k[2] + k[0] + k[1] + k[3]] = emissions[k]
            markov = (4, i, j)
            states[acc] = markov
            probs[acc] = new_emissions
            acc += 1
    return list(states), list(probs)
