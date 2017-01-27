import numpy as np
import femformal.system as s

import logging
logger = logging.getLogger('FEMFORMAL')

def mechlinfem(xpart, rho, E, g, f_nodal):
    # n_g = len([x for x in g if x is not None])
    gg = [x if x is not None else 0.0 for x in g]
    # Number of equations for n_g = 2
    n = xpart.shape[0] - 2

    ls = np.diff(xpart)

    Mdiag = [ls[0] if g[0] is None else 0.0] + \
        [ls[i] + ls[i + 1] for i in range(n)] + \
        [ls[-1] if g[1] is None else 0.0]

    Koffd = [-1.0 / ls[0] if g[0] is None else 0.0] + \
        [-1.0 / ls[i] for i in range(1, n)] + \
        [-1.0/ ls[-1] if g[1] is None else 0.0]
    Kdiag = [1.0 / ls[0] if g[0] is None else 1.0 / E] + \
        [1.0 / ls[i - 1] + 1.0 / ls[i] for i in range(1, n + 1)] + \
        [1.0 / ls[-1] if g[1] is None else 1.0 / E]
    F = np.r_[0.0 if g[0] is None else g[0],
              E * gg[0] / ls[0], [0 for i in range(n - 2)], E * gg[1] / ls[n],
              0.0 if g[-1] is None else g[-1]]

    M = np.diag(Mdiag) * rho / 2.0
    K = (np.diag(Kdiag) +
        np.diag(Koffd, 1) +
        np.diag(Koffd, -1)) * E
    F = F + f_nodal

    print M.shape
    print K.shape
    print F.shape
    sosys = s.SOSystem(M, K, F)
    # system = sosys.to_fosystem()

    return sosys

def state(u, du, xpart, g):
    d0 = [u(x) for x in xpart]
    v0 = [du(x) for x in xpart]
    if g[0] is not None:
        d0[0] = g[0]
        v0[0] = 0.0
    if g[1] is not None:
        d0[-1] = g[1]
        v0[-1] = 0.0
    return d0, v0
