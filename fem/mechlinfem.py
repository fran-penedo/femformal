import numpy as np
import femformal.system as s

import logging
logger = logging.getLogger('FEMFORMAL')

def mechlinfem(xpart, rho, E, g, f_nodal):
    # Number of equations
    n = xpart.shape[0] - 2

    ls = np.diff(xpart)

    M = np.diag([2 * ls[0] + 2 * ls[1] + l[0]] + \
                [l[i - 1] + 2 * l[i] + 2 * l[i + 1] + l[i]
                 for i in range(1, n - 2)] + \
                [2 * ls[n - 1] + 2 * ls[n] + l[n - 1]]) * rho / 6.0
    offd = [-1.0 / ls[i] for i in range(1, n)]
    K = (np.diag([1.0 / ls[i - 1] + 1.0 / ls[i] for i in range(1, n + 1)]) +
        np.diag(offd, 1) +
        np.diag(offd, -1)) * E
    F = np.r_[g[0] / ls[0], [0 for i in range(n - 2)], g[1] / ls[n]]
    F.shape = (n, 1)
    F = F + f_nodal

    zeros = np.zeros((n, n))
    ident = np.identity((n,n))
    Maug = np.asarray(np.bmat([[ident, zeros],[zeros, M]]))
    Kaug = np.asarray(np.bmat([[zeros, ident],[K, zeros]]))
    Faug = np.vstack([np.zeros((n, 1)), F])

    A = np.linalg.solve(Maug, -Kaug)
    b = np.linalg.solve(Maug, Faug)
    C = np.empty(shape=(0,0))
    system = s.System(A, b, C)

    return system

def init_state(d0, dd0, xpart, g):
    return [g[0]] + [d0[x] for x in xpart[1:-1]] + [g[1]] + [dd0 for x in xpart]
