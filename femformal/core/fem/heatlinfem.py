import logging

import numpy as np

from .. import system as sys


logger = logging.getLogger(__name__)

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
    system = sys.FOSystem(M, K, F, xpart, dt)
    tpart = [np.arange(5, 115, 10.0).tolist() for i in range(n-1)]

    return system

def heatlinfem_mix(xpart, rho, E, g, f_nodal, dt):
    # n_g = len([x for x in g if x is not None])
    gg = [x if x is not None else 0.0 for x in g]
    # Number of equations for n_g = 2
    n = xpart.shape[0] - 2
    xmids = (xpart[:-1] + xpart[1:]) / 2.0
    ls = np.diff(xpart)

    try:
        Ev = [E(x) for x in xmids]
    except TypeError:
        Ev = [E for x in xmids]

    try:
        rhov = [rho(x) for x in xmids]
    except:
        rhov = [rho for x in xmids]

    Mdiag = [rhov[0] * ls[0] if g[0] is None else 0.0] + \
        [rhov[i] * ls[i] + rhov[i + 1] * ls[i + 1] for i in range(n)] + \
        [rhov[-1] * ls[-1] if g[1] is None else 0.0]

    Koffd = [-Ev[0] / ls[0] if g[0] is None else 0.0] + \
        [-Ev[i] / ls[i] for i in range(1, n)] + \
        [-Ev[-1] / ls[-1] if g[1] is None else 0.0]
    Kdiag = [Ev[0] / ls[0] if g[0] is None else 1.0] + \
        [Ev[i - 1] / ls[i - 1] + Ev[i] / ls[i] for i in range(1, n + 1)] + \
        [Ev[-1] / ls[-1] if g[1] is None else 1.0]
    F = np.r_[0.0 if g[0] is None else g[0],
              Ev[0] * gg[0] / ls[0], [0 for i in range(n - 2)],
              Ev[-1] * gg[1] / ls[n],
              0.0 if g[-1] is None else g[-1]]

    M = np.diag(Mdiag) / 2.0
    K = np.diag(Kdiag) + np.diag(Koffd, 1) + np.diag(Koffd, -1)
    F = F + f_nodal

    sosys = sys.FOSystem(M, K, F, xpart, dt)

    return sosys

def state(u, xpart, g):
    d0 = [u(x) for x in xpart]
    if g[0] is not None:
        d0[0] = g[0]
    if g[1] is not None:
        d0[-1] = g[1]
    return d0
