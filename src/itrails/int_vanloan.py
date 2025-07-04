import numpy as np
from scipy.linalg import expm


def instant_mat(tup_0, tup_1, trans_mat):
    """
    This function returns the same transition rate matrix
    as supplied, except only the entries from tup_0 to
    tup_1 are kept, and all of the others are zeroed. This
    function is useful to calculate the instantaneous
    transition rate matrix for the van Loan method, as well
    as for the integrals when the time interval goes to
    infinity.

    Parameters
    ----------
    tup_0 : list of integers
        These are the indices of the transition rate matrix
        corresponding to the starting states.
    tup_1 : list of integers
        These are the indices of the transition rate matrix
        corresponding to the end states.
    trans_mat : numeric numpy matrix
        Transition rate matrix.
    """
    n = trans_mat.shape[0]
    A_mat = np.zeros((n, n))
    for a in tup_0:
        for b in tup_1:
            A_mat[a, b] = trans_mat[a, b]
    return A_mat


def vanloan_1(trans_mat, tup, omega_start, omega_end, time):
    """
    This function performs the van Loan (1978) method for
    finding the integral of a series of 2 multiplying matrix
    exponentials.

    Parameters
    ----------
    trans_mat : numeric numpy matrix
        Transition rate matrix.
    tup : tupple
        Tupple of size 2, where the first and second entries
        are lists with the indices of the transition rate
        matrix to go from and to in the first instantaneous
        transition.
    omega_start : list of integers
        List of starting states of the transition rate matrix.
    omega_end : list of integers
        List of ending states of the transition rate matrix.
    time : float
        Upper boundary of the definite integral.
    """
    n = trans_mat.shape[0]
    C_mat = np.zeros((n * 2, n * 2))
    C_mat[0:n, 0:n] = trans_mat
    C_mat[0:n, n : (n * 2)] = instant_mat(tup[0], tup[1], trans_mat)
    C_mat[n : (n * 2), n : (n * 2)] = trans_mat
    res_test = (expm(C_mat * (time))[0:n, -n:])[omega_start][:, omega_end]
    return res_test


def vanloan_2(trans_mat, tup, omega_start, omega_end, time):
    """
    This function performs the van Loan (1978) method for
    finding the integral of a series of 3 multiplying matrix
    exponentials.

    Parameters
    ----------
    trans_mat : numeric numpy matrix
        Transition rate matrix.
    tup : tupple
        Tupple of size 3, where the first and second entries
        are lists with the indices of the transition rate
        matrix to go from and to in the first instantaneous
        transition. Similarly, the third entry is the indices
        to go to from the indices of the second entry.
    omega_start : list of integers
        List of starting states of the transition rate matrix.
    omega_end : list of integers
        List of ending states of the transition rate matrix.
    time : float
        Upper boundary of the definite integral.
    """
    n = trans_mat.shape[0]
    C_mat = np.zeros((n * 3, n * 3))
    C_mat[0:n, 0:n] = trans_mat
    C_mat[0:n, n : (n * 2)] = instant_mat(tup[0], tup[1], trans_mat)
    C_mat[n : (n * 2), n : (n * 2)] = trans_mat
    C_mat[n : (n * 2), (n * 2) : (n * 3)] = instant_mat(tup[1], tup[2], trans_mat)
    C_mat[(n * 2) : (n * 3), (n * 2) : (n * 3)] = trans_mat
    res_test = (expm(C_mat * (time))[0:n, -n:])[omega_start][:, omega_end]
    return res_test


def vanloan_3(trans_mat, tup, omega_start, omega_end, time):
    """
    This function performs the van Loan (1978) method for
    finding the integral of a series of 4 multiplying matrix
    exponentials.

    Parameters
    ----------
    trans_mat : numeric numpy matrix
        Transition rate matrix.
    tup : tupple
        Tupple of size 4, where the first and second entries
        are lists with the indices of the transition rate
        matrix to go from and to in the first instantaneous
        transition. Similarly, the third entry is the indices
        to go to from the indices of the second entry and so on.
    omega_start : list of integers
        List of starting states of the transition rate matrix.
    omega_end : list of integers
        List of ending states of the transition rate matrix.
    time : float
        Upper boundary of the definite integral.
    """
    n = trans_mat.shape[0]
    C_mat = np.zeros((n * 4, n * 4))
    C_mat[0:n, 0:n] = trans_mat
    C_mat[0:n, n : (n * 2)] = instant_mat(tup[0], tup[1], trans_mat)
    C_mat[n : (n * 2), n : (n * 2)] = trans_mat
    C_mat[n : (n * 2), (n * 2) : (n * 3)] = instant_mat(tup[1], tup[2], trans_mat)
    C_mat[(n * 2) : (n * 3), (n * 2) : (n * 3)] = trans_mat
    C_mat[(n * 2) : (n * 3), (n * 3) : (n * 4)] = instant_mat(tup[2], tup[3], trans_mat)
    C_mat[(n * 3) : (n * 4), (n * 3) : (n * 4)] = trans_mat
    res_test = (expm(C_mat * (time))[0:n, -n:])[omega_start][:, omega_end]
    return res_test
