import numpy as np
import femformal.system as s
import femformal.logic as logic
import femmilp.system_milp as sysmilp

import logging
logger = logging.getLogger('FEMFORMAL')

def build_cs(system, xpart, dt, d0, g, cregions, cspec,
             pset=None, f=None, discretize_system=True, cstrue=None,
             eps=None, eta=None, nu=None):
    if discretize_system:
        dsystem = s.cont_to_disc(system, dt)
    else:
        dsystem = None

    if cspec is not None:
        regions = {label: logic.ap_cont_to_disc(pred, xpart)
                for label, pred in cregions.items()}
        dspec = logic.subst_spec_labels_disc(cspec, regions)
        spec = sysmilp.stl_parser().parseString(dspec)[0]
        # if discretize_system:
        sysmilp.scale_time(spec, dt)
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



def max_diff(sys, dt, xpart, g, tlims, xlims, sys_true, dt_true, xpart_true,
             bounds, samplef=lin_sample, n=50, log=True):
    mdiff = 0.0
    if log:
        logger.debug("Starting max_diff")
    for i in range(n):
        if log and i % 10 == 0:
            logger.debug("Iteration: {}, mdiff = {}".format(i, mdiff))
        x0, y0 = samplef(bounds, g, xpart, xpart_true)
        diff = s.sys_max_diff(
            sys, sys_true, dt, dt_true, xpart, xpart_true,
            x0, y0, tlims, xlims)
        mdiff = max(mdiff, diff)

    if log:
        logger.debug("mdiff = {}".format(mdiff))
    return mdiff

def max_xdiff(sys, dt, xpart, g, tlims, bounds, samplef=lin_sample, n=50, log=True):
    mdiff = np.zeros((len(xpart) - 1,))
    if log:
        logger.debug("Starting max_xdiff")
    for i in range(n):
        if log and i % 10 == 0:
            logger.debug("Iteration: {}, mdiff = {}".format(i, mdiff))
        x0 = samplef(bounds, g, xpart)
        dx = s.sys_max_xdiff(sys, dt, xpart, x0, tlims)
        mdiff = np.max([mdiff, dx], axis=0)

    if log:
        logger.debug("mdiff = {}".format(mdiff))
    return mdiff

def max_tdiff(sys, dt, xpart, g, tlims, bounds, samplef=lin_sample, n=50, log=True):
    mdiff = np.zeros((len(xpart),))
    if log:
        logger.debug("Starting max_tdiff")
    for i in range(n):
        if log and i % 10 == 0:
            logger.debug("Iteration: {}, mdiff = {}".format(i, mdiff))
        x0 = samplef(bounds, g, xpart)
        dx = s.sys_max_tdiff(sys, dt, xpart, x0, tlims)
        mdiff = np.max([mdiff, dx], axis=0)

    if log:
        logger.debug("mdiff = {}".format(mdiff))
    return mdiff


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
