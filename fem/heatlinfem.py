import numpy as np
import femformal.system as s
import femformal.util as u
import femformal.logic as logic
import femmilp.system_milp as sysmilp

import logging
logger = logging.getLogger('FEMFORMAL')

def heatlinfem(N, L, T):
    n = N
    l = L / n

    M = np.diag([5.0] + [6.0 for i in range(n - 3)] + [5.0]) * l / 6
    K = (np.diag([2.0 for i in range(n - 1)]) +
        np.diag([-1.0 for i in range(n - 2)], 1) +
        np.diag([-1.0 for i in range(n - 2)], -1)) / l
    F = np.r_[T[0], [0 for i in range(n - 3)], T[1]] / l
    F.shape = (n - 1, 1)

    A = np.linalg.solve(M, -K)
    b = np.linalg.solve(M, F)
    C = np.empty(shape=(0,0))
    system = s.System(A, b, C)
    tpart = [np.arange(5, 115, 10.0).tolist() for i in range(n-1)]
    xpart = np.linspace(0, L, N + 1)

    return system, xpart, tpart

def diag(n, m, i):
    d = np.hstack([j * np.ones((n - 1) / m, dtype=int) for j in range(m)])
    d = np.hstack([d, (m - 1) * np.ones((n - 1) % m, dtype=int)])

    d = d + i
    if i > 0:
        d[d > m - 1] = m - 1
    elif i < 0:
        d[d < 0] = 0

    return d

def build_cs(N, L, T, dt, d0, cregions, cspec, pset=None, discretize_system=True, cstrue=None, eps=None, eta=None, nu=None):
    system, xpart, partition = heatlinfem(N, L, T)
    if discretize_system:
        dsystem = s.cont_to_disc(system, dt)
    else:
        dsystem = None

    if cspec is not None:
        regions = {label: logic.ap_cont_to_disc(pred, xpart)
                for label, pred in cregions.items()}
        dspec = logic.subst_spec_labels_disc(cspec, regions)
        spec = sysmilp.stl_parser().parseString(dspec)[0]
        if discretize_system:
            sysmilp.scale_time(spec, dt)
        t0, tt = spec.shorizon(), spec.horizon()
        md = 0.0
        me = [0.0 for i in range(len(xpart) - 1)]
        mn = [0.0 for i in range(len(xpart) - 1)]
        if cstrue is not None:
            md = max_diff(dsystem, dt, xpart, t0, tt, T, cstrue)
        else:
            if eps is not None:
                md = eps
            if eta is not None:
                me = eta
            if nu is not None:
                mn = nu
        kd = lambda i, isnode, dmu: md
        ke = lambda i, isnode, dmu: (me[i] / 2.0) + dmu * (xpart[1] - xpart[0]) / 2.0
        kn = lambda i, isnode, dmu: (mn[i] if isnode else ((mn[i] + mn[i+1]) / 2.0))
        sysmilp.perturb(spec, kd)
        sysmilp.perturb(spec, ke)
        sysmilp.perturb(spec, kn)
    else:
        spec = None
        regions = None

    rh_N = 2

    return CaseStudy({
        'system': system,
        'dsystem': dsystem,
        'xpart': xpart,
        'T': T,
        'dt': dt,
        'd0': d0,
        'pset': pset,
        'regions': regions,
        'spec': spec,
        'rh_N': rh_N
    })

def max_diff(sys, dt, xpart, t0, tt, xl, xr, T, cstrue):
    mdiff = 0.0
    logger.debug("Starting max_diff")
    for i in range(50):
        if i % 10 == 0:
            logger.debug("Iteration: {}, mdiff = {}".format(i, mdiff))
        a = (np.random.rand() * 4 - 2) * abs(T[1] - T[0]) / xpart[-1]
        b = np.random.rand() * abs(T[1] - T[0])
        x0 = [T[0]] + [min(max(a * x + b, T[0]), T[1]) for x in xpart[1:-1]] + [T[1]]
        y0 = [T[0]] + [min(max(a * x + b, T[0]), T[1]) for x in cstrue.xpart[1:-1]] + [T[1]]
        diff = s.sys_max_diff(
            sys, cstrue.system, dt, cstrue.dt, xpart, cstrue.xpart,
            x0, y0, t0, tt, xl, xr)
        # logger.debug(diff)
        mdiff = max(mdiff, diff)

    logger.debug("mdiff = {}".format(mdiff))
    return mdiff

def max_xdiff(sys, dt, xpart, t0, tt, T, n=50):
    print T
    mdiff = np.zeros((len(xpart) - 1,))
    logger.debug("Starting max_xdiff")
    for i in range(n):
        a = (np.random.rand() * 4 - 2) * abs(T[1] - T[0]) / xpart[-1]
        b = np.random.rand() * abs(T[1] - T[0])
        x0 = [T[0]] + [min(max(a * x + b, T[0]), T[1]) for x in xpart[1:-1]] + [T[1]]
        print x0
        dx = s.sys_max_xdiff(sys, dt, xpart, x0, t0, tt)
        mdiff = np.max([mdiff, dx], axis=0)
    logger.debug("mdiff = {}".format(mdiff))
    return mdiff

def max_tdiff(sys, dt, xpart, t0, tt, T, n=50):
    mdiff = np.zeros((len(xpart),))
    logger.debug("Starting max_tdiff")
    for i in range(n):
        a = (np.random.rand() * 4 - 2) * abs(T[1] - T[0]) / xpart[-1]
        b = np.random.rand() * abs(T[1] - T[0])
        x0 = [T[0]] + [min(max(a * x + b, T[0]), T[1]) for x in xpart[1:-1]] + [T[1]]
        dx = s.sys_max_tdiff(sys, dt, xpart, x0, t0, tt)
        mdiff = np.max([mdiff, dx], axis=0)
    logger.debug("mdiff = {}".format(mdiff))
    return mdiff


class CaseStudy(object):
    def __init__(self, dic):
        copy = dic.copy()
        self.system = copy.pop('system', None)
        self.dsystem = copy.pop('dsystem', None)
        self.xpart = copy.pop('xpart', None)
        self.T = copy.pop('T', 0)
        self.dt = copy.pop('dt', 0)
        self.d0 = copy.pop('d0', None)
        self.pset = copy.pop('pset', None)
        self.regions = copy.pop('regions', None)
        self.spec = copy.pop('spec', None)
        self.rh_N = copy.pop('rh_N', None)

        if len(copy) > 0:
            raise Exception('Undefined parameters in CaseStudy: {}'.format(copy))
