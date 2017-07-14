import logging

import numpy as np

from .. import system as sys


logger = logging.getLogger(__name__)

def mechnlfem(xpart, rho, E, g, f_nodal, dt):
    # n_g = len([x for x in g if x is not None])
    gg = [x if x is not None else 0.0 for x in g]
    # Number of equations for n_g = 2
    n = xpart.shape[0] - 2
    xmids = (xpart[:-1] + xpart[1:]) / 2.0
    ls = np.diff(xpart)

    Ev = [E for x in xmids]

    for i in range(len(Ev)):
        Ev[i].p = xpart[i:i+2]

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
    K = [sys.HybridParameter(Ev[i].invariants,
                             [Ks[i] * v for v in Ev[i].values])
         for i in range(len(Ks))]
    # K = [lambda u, i=i: Ev[i](u) * Ks[i]
    #       for i in range(len(Ks))]
    if g[0] is not None:
        K[0].values = [np.diag([1.0, v / ls[0]]) for v in Ev[0].values]
        # lambda u: np.diag([1.0, Ev[0](u) / ls[0]])
    if g[1] is not None:
        K[-1].values = [np.diag([v / ls[-1], 1.0]) for v in Ev[-1].values]
        # K[-1] = lambda u: np.diag([Ev[-1](u) / ls[-1], 1.0])

    if np.all(np.isclose(gg, 0)):
        F = np.r_[0.0 if g[0] is None else g[0],
                [0 for i in range(n)],
                0.0 if g[-1] is None else g[-1]]
    else:
        F = np.r_[0.0 if g[0] is None else g[0],
                Ev[0] * gg[0] / ls[0], [0 for i in range(n - 2)],
                Ev[-1] * gg[1] / ls[n],
                0.0 if g[-1] is None else g[-1]]

    M = np.diag(Mdiag) / 2.0
    F = F + f_nodal

    sosys = sys.HybridSOSystem(M, K, F, xpart, dt)

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
