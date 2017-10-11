import logging
from bisect import bisect_left

import numpy as np

from femformal.core import system as sys, logic as logic


# from .. import femmilp.system_milp as sysmilp
logger = logging.getLogger(__name__)

def build_cs(system, d0, g, cregions, cspec, fdt_mult=1, bounds=None,
             pset=None, f=None, discretize_system=True, cstrue=None, error_bounds=None,
             eps=None, eta=None, nu=None, eps_xderiv=None, nu_xderiv=None, T=1.0):
    dt = system.dt
    xpart = system.xpart

    if discretize_system:
        dsystem = sys.cont_to_disc(system, dt)
        dsystem.dt = dt
        dsystem.xpart = xpart
    else:
        dsystem = system

    if cspec is not None:
        dspec = logic.sstl_to_stl(cspec, cregions, xpart, system.mesh)
        try:
            if xpart is not None:
                spec = logic.stl_parser(xpart, fdt_mult, bounds).parseString(dspec)[0]
            else:
                spec = logic.stl_parser(
                        None, fdt_mult, bounds, system.mesh, system.build_elem
                    ).parseString(dspec)[0]
        except Exception as e:
            logger.exception("Error while parsing specification:\n{}\n".format(dspec))
            raise e

        T = max(spec.horizon(), 0)
        # if discretize_system:
        logic.scale_time(spec, dt * fdt_mult)

        if error_bounds is not None:
            ((eps, eps_xderiv), (eta, eta_xderiv), (nu, nu_xderiv)) = error_bounds
        eps_pert = EpsPerturbation(eps, eps_xderiv, xpart=xpart, mesh=system.mesh)
        eta_pert = EtaPerturbation(eta, xpart=xpart, mesh=system.mesh)
        nu_pert = NuPerturbation(nu, nu_xderiv, fdt_mult, xpart=xpart, mesh=system.mesh)

        logic.perturb(spec, eps_pert)
        logic.perturb(spec, eta_pert)
        logic.perturb(spec, nu_pert)
    else:
        spec = None
        # regions = None

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
        # 'regions': regions,
        'spec': spec,
        'T': T
    })


class Perturbation(object):
    def __init__(self, xpart=None, mesh=None):
        self.xpart = xpart
        self.mesh = mesh

    def perturb(self, stlpred):
        raise NotImplementedError()

    def __call__(self, stlpred):
        p = self.perturb(stlpred)
        if p is None:
            return 0.0
        else:
            return p


class EpsPerturbation(Perturbation):
    def __init__(self, eps, eps_xderiv, **kwargs):
        Perturbation.__init__(self, **kwargs)
        self.eps_list = np.array([eps, eps_xderiv])

    def perturb(self, stlpred):
        i = stlpred.index
        isnode = stlpred.isnode
        uderivs = stlpred.uderivs
        ucomp = stlpred.u_comp
        region_dim = stlpred.region_dim
        eps = self.eps_list[uderivs]
        if not hasattr(eps, 'shape'):
            return eps
        else:
            if len(eps.shape) == 1:
                eps = eps[None].T
            if self.xpart is not None:
                if isnode and i > 0 and i < len(eps):
                    return max(eps[i-1][ucomp], eps[i][ucomp]).tolist()
                elif isnode and i == len(eps):
                    return eps[i-1][ucomp].tolist()
                else:
                    return eps[i][ucomp].tolist()
            else:
                elems = self.mesh.find_2d_containing_elems(i, region_dim)
                ret = max([eps[e][ucomp] for e in elems.elems])
                return ret.tolist()


class EtaPerturbation(Perturbation):
    def __init__(self, eta, **kwargs):
        Perturbation.__init__(self, **kwargs)
        self.eta = eta

    def perturb(self, stlpred):
        i = stlpred.index
        dmu = stlpred.dp
        uderivs = stlpred.uderivs
        ucomp = stlpred.u_comp
        region_dim = stlpred.region_dim
        if self.eta is None or region_dim == 0:
            return 0.0
        else:
            if len(self.eta.shape) == 1:
                ret = ((self.eta[i] / 2.0 if uderivs == 0 else 0.0)
                        + dmu * (self.xpart[i+1] - self.xpart[i]) / 2.0)
                return ret.tolist()
            else:
                elem = self.mesh.get_elem(i, region_dim)
                cheby_radius = elem.chebyshev_radius()
                if region_dim == 0:
                    grad = 0
                elif region_dim == 1:
                    # FIXME
                    # grad = np.linalg.norm(self.eta[i, ucomp]) * np.sqrt(self.eta.shape[-1])
                    grad = 0
                elif region_dim == 2:
                    grad = np.linalg.norm(self.eta[i, ucomp]) * np.sqrt(self.eta.shape[-1])
                ret = cheby_radius * (dmu + grad)
                return ret.tolist()


class NuPerturbation(Perturbation):
    def __init__(self, nu, nu_xderiv, fdt_mult, **kwargs):
        Perturbation.__init__(self, **kwargs)
        self.nu_list = np.array([nu, nu_xderiv])
        self.fdt_mult = fdt_mult
        if self.mesh is not None:
            self.interpolations = [
                self.mesh.interpolate(x) for x in self.nu_list if x is not None]

    def perturb(self, stlpred):
        i = stlpred.index
        uderivs = stlpred.uderivs
        ucomp = stlpred.u_comp
        region_dim = stlpred.region_dim
        fdt_mult = self.fdt_mult
        nu = self.nu_list[uderivs]
        if nu is None:
            return 0.0
        else:
            if self.mesh is not None:
                x = self.mesh.get_elem(i, region_dim).chebyshev_center()
                ret = fdt_mult * self.interpolations[uderivs](*x)[ucomp]
                return ret.tolist()
            else:
                nu = self.nu_list[uderivs]
                if region_dim == 0:
                    ret = fdt_mult * nu[i]
                else:
                    ret = fdt_mult * (nu[i] + nu[i+1]) / 2.0
                return ret.tolist()


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

def id_sample_1dof(bounds, g, xpart_x, xpart_y=None):
    u0 = bounds
    x0 = [u0(x) for x in xpart_x]

    if xpart_y is not None:
        y0 = [u0(x) for x in xpart_y]
        return x0, y0
    else:
        return x0


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
            if system.mesh is None:
                f_nodal_x, f_nodal_y = sample_f(bounds_f, g, system.xpart, sys_true.xpart)
            else:
                f_nodal_x, f_nodal_y = sample_f(bounds_f, g, system, sys_true)
            sys_x, sys_y = system.copy(), sys_true.copy()
            try:
                sys_x.add_f_nodal(f_nodal_x)
                sys_y.add_f_nodal(f_nodal_y)
            except AttributeError:
                raise Exception("Can't sample f_nodal for this kind of system:"
                                "sysx: {}, sysy: {}".format(type(sys_x), type(sys_y)))
        if system.mesh is None:
            x0, y0 = sample_ic(bounds_ic, g, sys_x.xpart, sys_y.xpart)
        else:
            x0, y0 = sample_ic(bounds_ic, g, sys_x, sys_y)

        diff = sys.sys_max_diff(sys_x, sys_y, x0, y0, tlims, xlims, pw=pw,
                              xderiv=xderiv, plot=False)
        if mdiff is None:
            mdiff = diff
        else:
            mdiff = np.amax([mdiff, diff], axis=0)

    mdiffgrouped = _downsample_diffs(mdiff, system, sys_true, xderiv)
    if log:
        logger.debug("mdiff = {}".format(mdiffgrouped))
    return mdiffgrouped

def _downsample_diffs(mdiff, system, sys_true, xderiv):
    if system.xpart is not None:
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
    else:
        mdiffgrouped = np.zeros((system.mesh.nelems, mdiff.shape[-1]))
        for e in range(system.mesh.nelems):
            e_coords = system.mesh.elem_coords(e)
            covering = sys_true.mesh.find_elems_covering(e_coords[0], e_coords[2])
            nodes = set(
                np.array([sys_true.mesh.elems_nodes[cov_elem]
                          for cov_elem in covering.elems]).flatten().tolist())
            mdiffgrouped[e] = np.max(mdiff.take(list(nodes), axis=0), axis=0)

    return mdiffgrouped

def max_xdiff(system, g, tlims, bounds, sample=None, n=50, log=True):
    if sample is None:
        sample_ic = lin_sample
        sample_f = None
    else:
        sample_ic, sample_f = sample

    bounds_ic, bounds_f = bounds
    sys_x = system
    mdiff = np.zeros((len(system.xpart) - 1,))
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

def max_tdiff(system, g, tlims, bounds, sample=None, n=50, log=True):
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
    if system.xpart is not None:
        xderiv_correct = -1 if xderiv else 0
        mdiff_x = np.zeros((len(system.xpart) - 1 + xderiv_correct,))
        mdiff_t = np.zeros((len(system.xpart) + xderiv_correct,))
    else:
        mdiff_x = np.zeros((system.mesh.nelems, 2, 2))
        mdiff_t = np.zeros((2 * system.mesh.nnodes,))
    if log:
        logger.debug("Starting max_der_diff")
    for i in range(n):
        if log and i % 10 == 0:
            logger.debug("Iteration: {}, mdiff_x = {}".format(i, mdiff_x))
            logger.debug("Iteration: {}, mdiff_t = {}".format(i, mdiff_t))
        if sample_f is not None:
            if system.xpart is not None:
                f_nodal = sample_f(bounds_f, g, system.xpart)
            else:
                f_nodal = sample_f(bounds_f, g, system)
            try:
                sys_x = system.copy()
                sys_x.add_f_nodal(f_nodal)
            except AttributeError:
                raise Exception("Can't sample f_nodal for this kind of system")
        if system.xpart is not None:
            x0 = sample_ic(bounds_ic, g, sys_x.xpart)
        else:
            x0 = sample_ic(bounds_ic, g, sys_x)
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

def _perturb_profile_eps_2d(p, eps, mesh, direction):
    def pp(*x):
        i = mesh.find_containing_elem(x)
        return p(*x) + direction * eps[i]
    return pp

def _perturb_profile_eta_2d(p, dp, eta, mesh, direction):
    def pp(*x):
        i = mesh.find_containing_elem(x)
        elem = mesh.get_elem(i)
        h = elem.chebyshev_radius()
        return p(*x) + direction * h * (
            np.linalg.norm(eta[i]) * np.sqrt(eta.shape[-1]) + dp(*x))
    return pp

def _perturb_profile_nu_2d(p, nu, mesh, fdt_mult, direction):
    interp = mesh.interpolate(nu)
    def pp(*x):
        return (p(*x) + direction * fdt_mult * interp(*x))[0]
    return pp

def perturb_profile(apc, eps, eta, nu, xpart, fdt_mult, mesh):
    direction = -1 * apc.r
    if xpart is not None:
        eps_p = _perturb_profile_eps(apc.p, eps[apc.uderivs], xpart, direction)
        if np.isclose(apc.A[0], apc.A[1]):
            eta_p = eps_p
        else:
            eta_p = _perturb_profile_eta(eps_p, apc.dp, eta[apc.uderivs], xpart, direction)
        nu_p = _perturb_profile_nu(eta_p, nu[apc.uderivs], xpart, fdt_mult, direction)
    else:
        eps_p = _perturb_profile_eps_2d(apc.p, eps[apc.uderivs][:,apc.u_comp], mesh, direction)
        eta_p = _perturb_profile_eta_2d(eps_p, apc.dp, eta[apc.uderivs][:,apc.u_comp,:], mesh, direction)
        nu_p = _perturb_profile_nu_2d(eta_p, nu[apc.uderivs][apc.u_comp::2], mesh, fdt_mult, direction)

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

