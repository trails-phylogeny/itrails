import numpy as np
from scipy.stats import expon, truncexpon


def cutpoints_AB(n_int_AB, t_AB, coal_AB):
    """
    This function returns the cutpoints for the
    intervals for the two-sequence CTMC. The cutpoints
    will be defined by the quantiles of a truncated
    exponential distribution.

    :param n_int_AB: Number of intervals in the two-sequence CTMC.
    :type n_int_AB: int
    :param t_AB: Total time interval of the two-sequence CTMC
    :type t_AB: float
    :param coal_AB: coalescent rate of the two-sequence CTMC.
    :type coal_AB: float
    :return: cut_AB
    :rtype: np.array
    """
    quantiles_AB = np.array(list(range(n_int_AB + 1))) / n_int_AB
    lower, upper, scale = 0, t_AB, 1 / coal_AB
    cut_AB = truncexpon.ppf(
        quantiles_AB, b=(upper - lower) / scale, loc=lower, scale=scale
    )
    return cut_AB


def cutpoints_ABC(n_int_ABC, coal_ABC):
    """
    This function returns the cutpoints for the
    intervals for the three-sequence CTMC. The cutpoints
    will be defined by the quantiles of an exponential
    distribution.

    :param n_int_ABC: Number of intervals in the three-sequence CTMC.
    :type n_int_ABC: int
    :param coal_ABC: coalescent rate of the three-sequence CTMC.
    :type coal_ABC: float
    :return: cut_ABC
    :rtype: np.array
    """
    quantiles_AB = np.array(list(range(n_int_ABC + 1))) / n_int_ABC
    cut_ABC = expon.ppf(quantiles_AB, scale=1 / coal_ABC)
    return cut_ABC


def get_times(cut, intervals):
    """
    Returns a list of time differences for each specified interval.

    This function computes the duration for each interval defined by the given indices.
    It does so by subtracting the earlier cutpoint from the subsequent one based on the indices
    provided in the intervals list.

    :param cut: List of ordered cutpoints.
    :type cut: list[float]
    :param intervals: Ordered indices of cutpoints that define the intervals.
    :type intervals: list[int]
    :return: List of time differences between consecutive cutpoints as defined by intervals.
    :rtype: list[float]
    """
    return [
        cut[intervals[i + 1]] - cut[intervals[i]] for i in range(len(intervals) - 1)
    ]
