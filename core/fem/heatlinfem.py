import numpy as np
from .. import system as s

import logging
logger = logging.getLogger('FEMFORMAL')

def heatlinfem(N, L, T, dt):
    n = N
    l = L / n

    M = np.diag([5.0] + [6.0 for i in range(n - 3)] + [5.0]) * l / 6
    K = (np.diag([2.0 for i in range(n - 1)]) +
        np.diag([-1.0 for i in range(n - 2)], 1) +
        np.diag([-1.0 for i in range(n - 2)], -1)) / l
    F = np.r_[T[0], [0 for i in range(n - 3)], T[1]] / l
    # F.shape = (n - 1, 1)

    xpart = np.linspace(0, L, N + 1)
    system = s.FOSystem(M, K, F, xpart, dt)
    tpart = [np.arange(5, 115, 10.0).tolist() for i in range(n-1)]

    return system

