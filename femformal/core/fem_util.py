import logging
from bisect import bisect_left

import numpy as np

from femformal.core import system as sys, logic as logic


# from .. import femmilp.system_milp as sysmilp
logger = logging.getLogger(__name__)

def build_cs(system, d0, g, cregions, cspec, fdt_mult=1, bounds=None,
             pset=None, f=None, discretize_system=True, cstrue=None, error_bounds=None,
             eps=None, eta=None, nu=None, eps_xderiv=None, nu_xderiv=None):
    dt = system.dt
    xpart = system.xpart

    if discretize_system:
        dsystem = sys.cont_to_disc(system, dt)
        dsystem.dt = dt
        dsystem.xpart = xpart
    else:
        dsystem = None

    if cspec is not None:
        regions = {label: logic.ap_cont_to_disc(pred, xpart)
                for label, pred in cregions.items()}
        dspec = logic.subst_spec_labels_disc(cspec, regions)
        try:
            spec = logic.stl_parser(xpart, fdt_mult, bounds).parseString(dspec)[0]
        except Exception as e:
            logger.exception("Error while parsing specification:\n{}\n".format(dspec))
            raise e

        T = max(spec.horizon(), 0)
        # if discretize_system:
        logic.scale_time(spec, dt * fdt_mult)
        if error_bounds is not None:
            ((eps, eps_xderiv), (eta, eta_xderiv), (nu, nu_xderiv)) = error_bounds

        if eps is not None:
            eps_list = [eps, eps_xderiv]
            if isinstance(eps, list):
                kd = lambda i, isnode, dmu, uderivs: (
                    eps_list[uderivs][i] if not isnode else
                    (max(eps_list[uderivs][i], eps_list[uderivs][i+1])
                     if i > 0 and i < len(eps_list[uderivs]) - 1 else eps_list[uderivs][i]))
            else:
                kd = lambda i, isnode, dmu, uderivs: eps_list[uderivs]
        else:
            kd = lambda i, isnode, dmu, uderivs: 0.0
        if eta is not None:
            ke = lambda i, isnode, dmu, uderivs: (
                (eta[i] / 2.0 if uderivs == 0 else 0.0)
                + dmu * (xpart[i+1] - xpart[i]) / 2.0)
        else:
            ke = lambda i, isnode, dmu, uderivs: 0.0

        if nu is not None and nu_xderiv is not None:
            mn = [nu, nu_xderiv]
        else:
            mn = [[0.0 for i in range(len(xpart) - 1)] for i in range(2)]

        kn = lambda i, isnode, dmu, uderivs: (
            fdt_mult * (mn[uderivs][i] if isnode else
                        ((mn[uderivs][i] + mn[uderivs][i+1]) / 2.0)))
        logic.perturb(spec, kd)
        logic.perturb(spec, ke)
        logic.perturb(spec, kn)
    else:
        spec = None
        regions = None

    return CaseStudy({
        'system': system,
        'dsystem': dsystem,
        'xpart': xpart,
        'g': g,
        'dt': dt,
        'fdt_mult': fdt_mult,
        'd0': d0,
        'pset': pset,
        'f': f,
        'regions': regions,
        'spec': spec,
        'T': T
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


def max_diff(system, g, tlims, xlims, sys_true,
             bounds, sample=None, pw=False, xderiv=False, n=50, log=True):
    if sample is None:
        sample_ic = lin_sample
        sample_f = None
    else:
        sample_ic, sample_f = sample

    bounds_ic, bounds_f = bounds
    sys_x = system
    sys_y = sys_true
    mdiff = None
    if log:
        logger.debug("Starting max_diff")
    for i in range(n):
        if log and i % 10 == 0:
            logger.debug("Iteration: {}, mdiff = {}".format(i, mdiff))

        if sample_f is not None:
            f_nodal_x, f_nodal_y = sample_f(bounds_f, g, system.xpart, sys_true.xpart)
            sys_x, sys_y = system.copy(), sys_true.copy()
            try:
                sys_x.add_f_nodal(f_nodal_x)
                sys_y.add_f_nodal(f_nodal_y)
            except AttributeError:
                raise Exception("Can't sample f_nodal for this kind of system:"
                                "sysx: {}, sysy: {}".format(type(sys_x), type(sys_y)))
        x0, y0 = sample_ic(bounds_ic, g, sys_x.xpart, sys_y.xpart)

        diff = sys.sys_max_diff(sys_x, sys_y, x0, y0, tlims, xlims, pw=pw,
                              xderiv=xderiv, plot=False)
        if mdiff is None:
            mdiff = diff
        else:
            mdiff = np.amax([mdiff, diff], axis=0)

    if xderiv:
        xpart = (system.xpart[1:] + system.xpart[:-1] ) / 2.0
        xpart_true = (sys_true.xpart[1:] + sys_true.xpart[:-1] ) / 2.0
    else:
        xpart = system.xpart
        xpart_true = sys_true.xpart
    ratio = float(len(xpart_true)) / len(xpart)
    cover = int(np.ceil(ratio))
    mdiffgrouped = np.amax(
        [mdiff[int(np.floor(i * ratio)):int(np.floor(i * ratio) + cover + 1)]
         for i in range(len(xpart) - 1)], axis=1)
    if log:
        logger.debug("mdiff = {}".format(mdiffgrouped))
    return mdiffgrouped

def max_xdiff(system, g, tlims, bounds, sample=None, n=50, log=True):
    if sample is None:
        sample_ic = lin_sample
        sample_f = None
    else:
        sample_ic, sample_f = sample

    bounds_ic, bounds_f = bounds
    sys_x = system
    mdiff = np.zeros((len(sys.xpart) - 1,))
    if log:
        logger.debug("Starting max_xdiff")
    for i in range(n):
        if log and i % 10 == 0:
            logger.debug("Iteration: {}, mdiff = {}".format(i, mdiff))
        if sample_f is not None:
            f_nodal = sample_f(bounds_f, g, system.xpart)
            try:
                sys_x = system.copy()
                sys_x.add_f_nodal(f_nodal)
            except AttributeError:
                raise Exception("Can't sample f_nodal for this kind of system")
        x0 = sample_ic(bounds_ic, g, sys_x.xpart)
        dx = sys.sys_max_xdiff(sys_x, x0, tlims[0], tlims[1])
        mdiff = np.max([mdiff, dx], axis=0)

    if log:
        logger.debug("mdiff = {}".format(mdiff))
    return mdiff

def max_tdiff(system, dt, xpart, g, tlims, bounds, sample=None, n=50, log=True):
    if sample is None:
        sample_ic = lin_sample
        sample_f = None
    else:
        sample_ic, sample_f = sample

    bounds_ic, bounds_f = bounds
    sys_x = system
    mdiff = np.zeros((len(system.xpart),))
    if log:
        logger.debug("Starting max_tdiff")
    for i in range(n):
        if log and i % 10 == 0:
            logger.debug("Iteration: {}, mdiff = {}".format(i, mdiff))
        if sample_f is not None:
            f_nodal = sample_f(bounds_f, g, system.xpart)
            try:
                sys_x = system.copy()
                sys_x.add_f_nodal(f_nodal)
            except AttributeError:
                raise Exception("Can't sample f_nodal for this kind of system")
        x0 = sample_ic(bounds_ic, g, sys_x.xpart)
        dx = sys.sys_max_tdiff(sys_x, x0, tlims[0], tlims[1])
        mdiff = np.max([mdiff, dx], axis=0)

    if log:
        logger.debug("mdiff = {}".format(mdiff))
    return mdiff

def max_der_diff(system, g, tlims, bounds, sample, xderiv=False, n=50, log=True):
    if sample is None:
        sample_ic = lin_sample
        sample_f = None
    else:
        sample_ic, sample_f = sample

    bounds_ic, bounds_f = bounds
    sys_x = system
    xderiv_correct = -1 if xderiv else 0
    mdiff_x = np.zeros((len(system.xpart) - 1 + xderiv_correct,))
    mdiff_t = np.zeros((len(system.xpart) + xderiv_correct,))
    if log:
        logger.debug("Starting max_der_diff")
    for i in range(n):
        if log and i % 10 == 0:
            logger.debug("Iteration: {}, mdiff_x = {}".format(i, mdiff_x))
            logger.debug("Iteration: {}, mdiff_t = {}".format(i, mdiff_t))
        if sample_f is not None:
            f_nodal = sample_f(bounds_f, g, system.xpart)
            try:
                sys_x = system.copy()
                sys_x.add_f_nodal(f_nodal)
            except AttributeError:
                raise Exception("Can't sample f_nodal for this kind of system")
        x0 = sample_ic(bounds_ic, g, sys_x.xpart)
        dx, dtx = sys.sosys_max_der_diff(sys_x, x0, tlims, xderiv=xderiv)
        mdiff_x = np.max([mdiff_x, dx], axis=0)
        mdiff_t = np.max([mdiff_t, dtx], axis=0)

    if log:
        logger.debug("mdiff_x = {}".format(mdiff_x))
        logger.debug("mdiff_t = {}".format(mdiff_t))
    return mdiff_x, mdiff_t


def _perturb_profile_eps(p, eps, xpart, direction):
    def pp(x):
        i = bisect_left(xpart, x) - 1
        return p(x) + direction * eps[i]
    return pp

def _perturb_profile_eta(p, dp, eta, xpart, direction):
    def pp(x):
        i = bisect_left(xpart, x) - 1
        return p(x) + direction * (
            eta[i] / 2.0 + dp(x) * (xpart[i + 1] - xpart[i]) / 2.0)
    return pp

def _perturb_profile_nu(p, nu, xpart, fdt_mult, direction):
    def pp(x):
        i = bisect_left(xpart, x) - 1
        return p(x) + direction * (fdt_mult * (nu[i + 1] + nu[i]) / 2.0)
    return pp


def perturb_profile(apc, eps, eta, nu, xpart, fdt_mult):
    direction = -1 * apc.r
    eps_p = _perturb_profile_eps(apc.p, eps[apc.uderivs], xpart, direction)
    eta_p = _perturb_profile_eta(eps_p, apc.dp, eta[apc.uderivs], xpart, direction)
    nu_p = _perturb_profile_nu(eta_p, nu[apc.uderivs], xpart, fdt_mult, direction)

    return nu_p

class CaseStudy(object):
    def __init__(self, dic):
        copy = dic.copy()
        self.system = copy.pop('system', None)
        self.dsystem = copy.pop('dsystem', None)
        self.xpart = copy.pop('xpart', None)
        self.g = copy.pop('g', 0)
        self.dt = copy.pop('dt', 0)
        self.fdt_mult = copy.pop('fdt_mult', 1)
        self.d0 = copy.pop('d0', None)
        self.pset = copy.pop('pset', None)
        self.f = copy.pop('f', None)
        self.regions = copy.pop('regions', None)
        self.spec = copy.pop('spec', None)
        self.rh_N = copy.pop('rh_N', None)
        self.thunk = copy.pop('thunk', None)
        self.T = copy.pop('T', 0)

        if len(copy) > 0:
            raise Exception('Undefined parameters in CaseStudy: {}'.format(copy))

