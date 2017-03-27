import numpy as np
from .. import system as s
from .. import logic as logic
# from .. import femmilp.system_milp as sysmilp

import logging
logger = logging.getLogger('FEMFORMAL')

def build_cs(system, d0, g, cregions, cspec, fdt_mult=1,
             pset=None, f=None, discretize_system=True, cstrue=None,
             eps=None, eta=None, nu=None):
    dt = system.dt
    xpart = system.xpart

    if discretize_system:
        dsystem = s.cont_to_disc(system, dt)
        dsystem.dt = dt
        dsystem.xpart = xpart
    else:
        dsystem = None

    if cspec is not None:
        regions = {label: logic.ap_cont_to_disc(pred, xpart)
                for label, pred in cregions.items()}
        dspec = logic.subst_spec_labels_disc(cspec, regions)
        try:
            spec = logic.stl_parser(fdt_mult).parseString(dspec)[0]
        except Exception as e:
            logger.exception("Error while parsing specification:\n{}\n".format(dspec))
            raise e

        # if discretize_system:
        logic.scale_time(spec, dt * fdt_mult)
        md = 0.0
        me = [0.0 for i in range(len(xpart) - 1)]
        mn = [0.0 for i in range(len(xpart) - 1)]
        if eps is not None:
            md = eps
        if eta is not None:
            me = eta
        if nu is not None:
            mn = nu
        kd = lambda i, isnode, dmu: md
        ke = lambda i, isnode, dmu: (me[i] / 2.0) + dmu * (xpart[i+1] - xpart[i]) / 2.0
        kn = lambda i, isnode, dmu: fdt_mult * (mn[i] if isnode else
                                                ((mn[i] + mn[i+1]) / 2.0))
        logic.perturb(spec, kd)
        logic.perturb(spec, ke)
        logic.perturb(spec, kn)
    else:
        spec = None
        regions = None

    rh_N = 2

    return CaseStudy({
        'system': system,
        'dsystem': dsystem,
        'xpart': xpart,
        'g': g,
        'dt': dt,
        'd0': d0,
        'pset': pset,
        'f': f,
        'regions': regions,
        'spec': spec,
        'rh_N': rh_N
    })


def lin_sample(bounds, g, xpart_x, xpart_y=None):
    a = (np.random.rand() * 4 - 2) * abs(bounds[1] - bounds[0]) / xpart_x[-1]
    b = np.random.rand() * abs(bounds[1] - bounds[0])
    x0 = [g[0]] + [min(max(a * x + b, bounds[0]), bounds[1])
                   for x in xpart_x[1:-1]] + [g[1]]
    if xpart_y is not None:
        y0 = [g[0]] + [min(max(a * x + b, bounds[0]), bounds[1])
                       for x in xpart_y[1:-1]] + [g[1]]
        return x0, y0
    else:
        return x0

def so_lin_sample(bounds, g, xpart_x, xpart_y=None):
    a = (np.random.rand()) * abs(bounds[1] - bounds[0]) / xpart_x[-1]
    x0 = [a*x for x in xpart_x]
    vx0 = [0.0 for x in xpart_x]
    if g[0] is not None:
        x0[0] = g[0]
    if g[-1] is not None:
        x0[-1] = g[-1]

    if xpart_y is not None:
        y0 = [a*x for x in xpart_y]
        vy0 = [0.0 for x in xpart_y]
        if g[0]:
            y0[0] = g[0]
        if g[-1]:
            y0[-1] = g[-1]
        return [x0, vx0], [y0, vy0]
    else:
        return [x0, vx0]

def id_sample(bounds, g, xpart_x, xpart_y=None):
    u0, v0 = bounds
    x0 = [u0(x) for x in xpart_x]
    vx0 = [v0(x) for x in xpart_x]

    if xpart_y is not None:
        y0 = [u0(x) for x in xpart_y]
        vy0 = [v0(x) for x in xpart_y]
        return [x0, vx0], [y0, vy0]
    else:
        return [x0, vx0]


def max_diff(sys, g, tlims, xlims, sys_true,
             bounds, sample=None, n=50, log=True):
    if sample is None:
        sample_ic = lin_sample
        sample_f = None
    else:
        sample_ic, sample_f = sample

    bounds_ic, bounds_f = bounds
    sys_x = sys
    sys_y = sys_true
    mdiff = 0.0
    if log:
        logger.debug("Starting max_diff")
    for i in range(n):
        if log and i % 10 == 0:
            logger.debug("Iteration: {}, mdiff = {}".format(i, mdiff))

        if sample_f is not None:
            f_nodal_x, f_nodal_y = sample_f(bounds_f, g, sys.xpart, sys_true.xpart)
            sys_x, sys_y = sys.copy(), sys_true.copy()
            sys_x.F = sys_x.F + f_nodal_x
            sys_y.F = sys_y.F + f_nodal_y
        x0, y0 = sample_ic(bounds_ic, g, sys_x.xpart, sys_y.xpart)

        diff = s.sys_max_diff(sys_x, sys_y, x0, y0, tlims, xlims, plot=False)
        mdiff = max(mdiff, diff)

    if log:
        logger.debug("mdiff = {}".format(mdiff))
    return mdiff

def max_xdiff(sys, g, tlims, bounds, sample=None, n=50, log=True):
    if sample is None:
        sample_ic = lin_sample
        sample_f = None
    else:
        sample_ic, sample_f = sample

    bounds_ic, bounds_f = bounds
    sys_x = sys
    mdiff = np.zeros((len(sys.xpart) - 1,))
    if log:
        logger.debug("Starting max_xdiff")
    for i in range(n):
        if log and i % 10 == 0:
            logger.debug("Iteration: {}, mdiff = {}".format(i, mdiff))
        if sample_f is not None:
            f_nodal = sample_f(bounds_f, g, sys.xpart)
            sys_x = sys.copy()
            sys_x.F = sys_x.F + f_nodal
        x0 = sample_ic(bounds_ic, g, sys_x.xpart)
        dx = s.sys_max_xdiff(sys_x, x0, tlims[0], tlims[1])
        mdiff = np.max([mdiff, dx], axis=0)

    if log:
        logger.debug("mdiff = {}".format(mdiff))
    return mdiff

def max_tdiff(sys, dt, xpart, g, tlims, bounds, sample=None, n=50, log=True):
    if sample is None:
        sample_ic = lin_sample
        sample_f = None
    else:
        sample_ic, sample_f = sample

    bounds_ic, bounds_f = bounds
    sys_x = sys
    mdiff = np.zeros((len(sys.xpart),))
    if log:
        logger.debug("Starting max_tdiff")
    for i in range(n):
        if log and i % 10 == 0:
            logger.debug("Iteration: {}, mdiff = {}".format(i, mdiff))
        if sample_f is not None:
            f_nodal = sample_f(bounds_f, g, sys.xpart)
            sys_x = sys.copy()
            sys_x.F = sys_x.F + f_nodal
        x0 = sample_ic(bounds_ic, g, sys_x.xpart)
        dx = s.sys_max_tdiff(sys_x, x0, tlims[0], tlims[1])
        mdiff = np.max([mdiff, dx], axis=0)

    if log:
        logger.debug("mdiff = {}".format(mdiff))
    return mdiff

def max_der_diff(sys, g, tlims, bounds, sample, n=50, log=True):
    if sample is None:
        sample_ic = lin_sample
        sample_f = None
    else:
        sample_ic, sample_f = sample

    bounds_ic, bounds_f = bounds
    sys_x = sys
    mdiff_x = np.zeros((len(sys.xpart) - 1,))
    mdiff_t = np.zeros((len(sys.xpart),))
    if log:
        logger.debug("Starting max_der_diff")
    for i in range(n):
        if log and i % 10 == 0:
            logger.debug("Iteration: {}, mdiff_x = {}".format(i, mdiff_x))
            logger.debug("Iteration: {}, mdiff_t = {}".format(i, mdiff_t))
        if sample_f is not None:
            f_nodal = sample_f(bounds_f, g, sys.xpart)
            sys_x = sys.copy()
            sys_x.F = sys_x.F + f_nodal
        x0 = sample_ic(bounds_ic, g, sys_x.xpart)
        dx, dtx = s.sosys_max_der_diff(sys_x, x0, tlims)
        mdiff_x = np.max([mdiff_x, dx], axis=0)
        mdiff_t = np.max([mdiff_t, dtx], axis=0)

    if log:
        logger.debug("mdiff_x = {}".format(mdiff_x))
        logger.debug("mdiff_t = {}".format(mdiff_t))
    return mdiff_x, mdiff_t


class CaseStudy(object):
    def __init__(self, dic):
        copy = dic.copy()
        self.system = copy.pop('system', None)
        self.dsystem = copy.pop('dsystem', None)
        self.xpart = copy.pop('xpart', None)
        self.g = copy.pop('g', 0)
        self.dt = copy.pop('dt', 0)
        self.d0 = copy.pop('d0', None)
        self.pset = copy.pop('pset', None)
        self.f = copy.pop('f', None)
        self.regions = copy.pop('regions', None)
        self.spec = copy.pop('spec', None)
        self.rh_N = copy.pop('rh_N', None)
        self.thunk = copy.pop('thunk', None)

        if len(copy) > 0:
            raise Exception('Undefined parameters in CaseStudy: {}'.format(copy))
