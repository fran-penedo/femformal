import numpy as np
from .. import system as s

import logging
logger = logging.getLogger('FEMFORMAL')

def mechlinfem(xpart, rho, E, g, f_nodal, dt):
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
    for i in [0, -1]:
        if g[i] is not None:
            Ev[i] = lambda x, u: ls[i]

    try:
        rhov = [rho(x) for x in xmids]
    except:
        rhov = [rho for x in xmids]

    Mdiag = [rhov[0] * ls[0] if g[0] is None else 0.0] + \
        [rhov[i] * ls[i] + rhov[i + 1] * ls[i + 1] for i in range(n)] + \
        [rhov[-1] * ls[-1] if g[1] is None else 0.0]

    # Koffd = [-Ev[0] / ls[0] if g[0] is None else 0.0] + \
    #     [-Ev[i] / ls[i] for i in range(1, n)] + \
    #     [-Ev[-1] / ls[-1] if g[1] is None else 0.0]
    # Kdiag = [Ev[0] / ls[0] if g[0] is None else 1.0] + \
    #     [Ev[i - 1] / ls[i - 1] + Ev[i] / ls[i] for i in range(1, n + 1)] + \
    #     [Ev[-1] / ls[-1] if g[1] is None else 1.0]

    Ks = [np.array([[1, -1], [-1, 1]]) / l for l in ls]
    K = [(lambda x, u: Ev(x, u) * k) for k in Ks]

    F = np.r_[0.0 if g[0] is None else g[0],
              Ev[0] * gg[0] / ls[0], [0 for i in range(n - 2)],
              Ev[-1] * gg[1] / ls[n],
              0.0 if g[-1] is None else g[-1]]

    M = np.diag(Mdiag) / 2.0
    F = F + f_nodal

    sosys = s.HybridSOSystem(M, K, F, xpart, dt)

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
