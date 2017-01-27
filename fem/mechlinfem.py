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

    Mdiag = [ls[i] + ls[i + 1] for i in range(n)]

    Koffd = [-1.0 / ls[i] for i in
            range(1 - (1 if g[0] is None else 0), n + (1 if g[1] is None else 0))]
    Kdiag = [1.0 / ls[i - 1] + 1.0 / ls[i] for i in range(1, n + 1)]

    F = np.r_[gg[0] / ls[0], [0 for i in range(n - 2)], gg[1] / ls[n]]
    if g[0] is None:
        Mdiag = [ls[0]] + Mdiag
        Kdiag = [1.0 / ls[0]] + Kdiag
        F = np.r_[0.0, F]
    if g[1] is None:
        Mdiag = Mdiag + [ls[-1]]
        Kdiag = Kdiag + [1.0 / ls[-1]]
        F = np.r_[F, 0.0]

    M = np.diag(Mdiag) * rho / 2.0
    K = (np.diag(Kdiag) +
        np.diag(Koffd, 1) +
        np.diag(Koffd, -1)) * E
    F = F + f_nodal[(0 if g[0] is None else 1):(len(f_nodal) if g[1] is None else -1)]

    print M.shape
    print K.shape
    print F.shape
    sosys = s.SOSystem(M, K, F)
    system = sosys.to_fosystem()

    return system, sosys

def aug_state(u, du, xpart, g):
    gg = [x if x is not None else 0.0 for x in g]
    return [gg[0]] + [u(x) for x in xpart[1:-1]] + [gg[1]] + [du(x) for x in xpart]
