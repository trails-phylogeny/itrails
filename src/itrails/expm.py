from __future__ import division, print_function

import math

import numba as nb
import numpy as np


@nb.jit(nopython=True, parallel=False, fastmath=True)
def expm(A):
    """
    Calculates matrix exponential of a square matrix A.
    Adapted from https://github.com/michael-hartmann/expm/blob/master/python/expm.py
    Algorithm 10.20 from unctions of Matrices: Theory and Computation, Nicholas J. Higham, 2008

    :param A: square matrix
    :type A: np.array
    :return: matrix exponential of A
    :rtype: np.array
    """
    theta3 = 1.5e-2
    theta5 = 2.5e-1
    theta7 = 9.5e-1
    theta9 = 2.1e0
    theta13 = 5.4e0
    # calculate the norm of A
    norm = np.linalg.norm(A, ord=1)

    if norm < theta3:
        dtype = A.dtype
        dim, dim = A.shape
        b = [120, 60, 12, 1]

        U = b[1] * np.eye(dim, dtype=dtype)
        V = b[0] * np.eye(dim, dtype=dtype)

        A2 = A @ A
        A2n = np.eye(dim, dtype=dtype)

        # evaluate (10.33)
        for i in range(1, 3 // 2 + 1):
            A2n = A2n @ A2
            U += b[2 * i + 1] * A2n
            V += b[2 * i] * A2n

        U = A @ U
        return np.linalg.solve(V - U, V + U)

    elif norm < theta5:
        dtype = A.dtype
        dim, dim = A.shape
        b = [30240, 15120, 3360, 420, 30, 1]

        U = b[1] * np.eye(dim, dtype=dtype)
        V = b[0] * np.eye(dim, dtype=dtype)

        A2 = A @ A
        A2n = np.eye(dim, dtype=dtype)

        # evaluate (10.33)
        for i in range(1, 5 // 2 + 1):
            A2n = A2n @ A2
            U += b[2 * i + 1] * A2n
            V += b[2 * i] * A2n

        U = A @ U
        return np.linalg.solve(V - U, V + U)

    elif norm < theta7:
        dtype = A.dtype
        dim, dim = A.shape
        b = [17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1]

        U = b[1] * np.eye(dim, dtype=dtype)
        V = b[0] * np.eye(dim, dtype=dtype)

        A2 = A @ A
        A2n = np.eye(dim, dtype=dtype)

        # evaluate (10.33)
        for i in range(1, 7 // 2 + 1):
            A2n = A2n @ A2
            U += b[2 * i + 1] * A2n
            V += b[2 * i] * A2n

        U = A @ U
        return np.linalg.solve(V - U, V + U)

    elif norm < theta9:
        dtype = A.dtype
        dim, dim = A.shape
        b = [
            17643225600,
            8821612800,
            2075673600,
            302702400,
            30270240,
            2162160,
            110880,
            3960,
            90,
            1,
        ]

        U = b[1] * np.eye(dim, dtype=dtype)
        V = b[0] * np.eye(dim, dtype=dtype)

        A2 = A @ A
        A2n = np.eye(dim, dtype=dtype)

        # evaluate (10.33)
        for i in range(1, 9 // 2 + 1):
            A2n = A2n @ A2
            U += b[2 * i + 1] * A2n
            V += b[2 * i] * A2n

        U = A @ U
        return np.linalg.solve(V - U, V + U)

    else:

        # algorithm 10.20, from line 7
        dim, dim = A.shape
        b = [
            64764752532480000,
            32382376266240000,
            7771770303897600,
            1187353796428800,
            129060195264000,
            10559470521600,
            670442572800,
            33522128640,
            1323241920,
            40840800,
            960960,
            16380,
            182,
            1,
        ]

        s = max(0, int(math.ceil(math.log(norm / theta13) / math.log(2))))
        if s > 0:
            A /= 2**s

        Id = np.eye(dim)
        A2 = A @ A
        A4 = A2 @ A2
        A6 = A2 @ A4

        U = A @ (
            (A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2))
            + b[7] * A6
            + b[5] * A4
            + b[3] * A2
            + b[1] * Id
        )

        V = (
            A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2)
            + b[6] * A6
            + b[4] * A4
            + b[2] * A2
            + b[0] * Id
        )
        # added this
        r13 = np.ascontiguousarray(np.linalg.solve(V - U, V + U))
        return np.linalg.matrix_power(r13, 2**s)
