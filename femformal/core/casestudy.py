"""
Top level functions for building femformal models
"""
from __future__ import division, absolute_import, print_function

import logging
from bisect import bisect_left

import numpy as np

from femformal.core import system as sys, logic as logic


# from .. import femmilp.system_milp as sysmilp
logger = logging.getLogger(__name__)

def build_cs(system, d0, g, cregions, cspec, fdt_mult=1, bounds=None,
             pset=None, f=None, discretize_system=False, cstrue=None, error_bounds=None,
             eps=None, eta=None, nu=None, eps_xderiv=None, nu_xderiv=None, T=1.0):
    """Builds a FEM model

    Builds a :class:`CaseStudy` object with all the information of the model,
    including the parsed, discretized and corrected S-STL specification.

    Parameters
    ----------
    system : any `System` class in :mod:`femformal.core.system`
        Base system
    d0 : array_like
        Initial value
    g : array_like
        PDE boundary conditions
    cregions : dict
        Dictionary mapping predicate labels to
        :class:`femformal.core.logic.APCont` for each predicate used in the
        specification
    cspec : str
        S-STL specification string, using labels in place of predicates
    fdt_mult : int, optional
        Multiplier of the time interval used in the system discretization. The
        resulting time interval is the one used to discretize in time the
        STL specification. Must be > 1
    bounds : array_like, optional
        [min, max] bounds for the secondary signal. Default is [-1000, 1000]
    pset : list of array_like
        Each element of the list is the H-representation of a polytope in
        which some parameters of the system is contained
    f : list of callable
        Each element is a function parameterized by the corresponding parameters
        defined by `pset`
    discretize_system : bool, optional
        Deprecated
    cstrue
        Deprecated
    error_bounds : tuple, optional
        Tuple ((eps, eps_xderiv), (eta, eta_xderiv), (nu, nu_xderiv)). Overrides
        the parameters of the same name
    T : float, optional
        Final time

    Returns
    -------
    :class:`CaseStudy`

    """
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

    return CaseStudy(**{
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
    """Base perturbation class

    Override :meth:`_perturb` in subclasses.

    Only `xpart` or `mesh` is needed to construct the object.

    Parameters
    ----------
    xpart : array_like, optional
        1D partition of the spatial domain, given as the list of nodes
    mesh : :class:`femformal.core.fem.mesh.Mesh`, optional
        2D mesh. Must have a `build_elem` attribute that constructs a
        :class:`femformal.core.fem.element.Element`

    """
    def __init__(self, xpart=None, mesh=None):
        self.xpart = xpart
        self.mesh = mesh

    def _perturb(self, stlpred):
        raise 0.0

    def __call__(self, stlpred):
        """Computes the perturbation for a predicate

        Parameters
        ----------
        stlpred : :class:`femformal.core.logic.STLPred`

        Returns
        -------
        float
            The perturbation

        """
        p = self._perturb(stlpred)
        if p is None:
            return 0.0
        else:
            return p


class EpsPerturbation(Perturbation):
    """Epsilon perturbation of a predicate

    The epsilon perturbation is defined as
    :math:`\\max_{x \in X_e} \\epsilon(x)`

    Parameters
    ----------
    eps, eps_xderiv : float or array_like, shape (nelems[, dofs])
        Approximation error in state or spatial derivative for each element
    **kwargs
        Arguments passed to :class:`Perturbation`

    """
    def __init__(self, eps, eps_xderiv, **kwargs):
        Perturbation.__init__(self, **kwargs)
        self.eps_list = np.array([eps, eps_xderiv])

    def _perturb(self, stlpred):
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
    """Eta perturbation of a predicate

    The eta perturbation is defined as
    :math:``.
    Note that `eta` is not the directional derivative but the directional
    difference.

    Parameters
    ----------
    eta : array_like, shape (nelems[, dofs, domain dimension])
        Approximation of the maximum directional differences.
    **kwargs
        Arguments passed to :class:`Perturbation`

    """
    def __init__(self, eta, **kwargs):
        Perturbation.__init__(self, **kwargs)
        self.eta = eta

    def _perturb(self, stlpred):
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
    """Nu perturbation of a predicate

    The nu perturbation is defined as
    :math:`\\max_{t} \\nu(t) \\Delta t_mult`. Note that `nu` is not the time
    derivative, but the time difference for a given time interval.

    Parameters
    ----------
    nu, nu_xderiv : array_like, shape (nnodes[, dofs])
        Approximation of the maximum temporal difference of state and its
        spatial derivatives
    fdt_mult : int
        STL time multiplier
    **kwargs
        Arguments passed to :class:`Perturbation`

    """
    def __init__(self, nu, nu_xderiv, fdt_mult, **kwargs):
        Perturbation.__init__(self, **kwargs)
        self.nu_list = np.array([nu, nu_xderiv])
        self.fdt_mult = fdt_mult
        if self.mesh is not None:
            self.interpolations = [
                self.mesh.interpolate(x) for x in self.nu_list if x is not None]

    def _perturb(self, stlpred):
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


class Sample(object):
    """Abstract class for sampling methods in the max diff functions

    Use :meth:`sample` to sample for one or two meshes (including x partitions).
    Override :meth:`_sample` to provide the implementation of the sampling
    method for a single mesh.

    """
    @staticmethod
    def _sample(bounds, g, mesh):
        raise NotImplementedError()

    @classmethod
    def sample(cls, bounds, g, mesh_x, mesh_y=None):
        """Samples for either one or two meshes

        Parameters
        ----------
        bounds
            Any object provided as bounds or parameters in general
        g : array_like
            PDE boundary conditions
        mesh_x : array_like or :class:`femformal.core.fem.mesh.Mesh`
        mesh_y : array_like or :class:`femformal.core.fem.mesh.Mesh`, optional

        Returns
        -------
        array_like
            The sampled object for `mesh_x`
        array_like, only returned if `mesh_y` is provided
            The sampled object for `mesh_y`

        """
        x = cls._sample(bounds, g, mesh_x)
        if mesh_y is not None:
            y = cls._sample(bounds, g, mesh_y)
            return x, y
        else:
            return x


class LinSample(Sample):
    """Linear sampling method for first order systems"""
    @staticmethod
    def _sample(bounds, g, mesh):
        a = (np.random.rand() * 4 - 2) * abs(bounds[1] - bounds[0]) / mesh[-1]
        b = np.random.rand() * abs(bounds[1] - bounds[0])
        x0 = [g[0]] + [min(max(a * x + b, bounds[0]), bounds[1])
                    for x in mesh[1:-1]] + [g[1]]
        return x0


lin_sample = LinSample.sample
"""Linear sampling method for first order systems"""

class SOLinSample(Sample):
    """Linear sampling method for second order systems"""
    @staticmethod
    def _sample(bounds, g, mesh):
        a = (np.random.rand()) * abs(bounds[1] - bounds[0]) / mesh[-1]
        x0 = [a*x for x in mesh]
        vx0 = [0.0 for x in mesh]
        if g[0] is not None:
            x0[0] = g[0]
        if g[-1] is not None:
            x0[-1] = g[-1]
        return [x0, vx0]


so_lin_sample = SOLinSample.sample
"""Linear sampling method for second order systems"""

class IDSample(Sample):
    """Identity sampling method for second order systems"""
    @staticmethod
    def _sample(bounds, g, mesh):
        u0, v0 = bounds
        x0 = [u0(x) for x in xpart_x]
        vx0 = [v0(x) for x in xpart_x]
        return [x0, vx0]


id_sample = IDSample.sample
"""Identity sampling method for second order systems"""

class IDSampleFO(Sample):
    """Identity sampling method for first order systems"""
    @staticmethod
    def _sample(bounds, g, mesh):
        u0 = bounds
        x0 = [u0(x) for x in xpart_x]
        return x0


id_sample_fo = IDSampleFO.sample
"""Identity sampling method for first order systems"""


def max_diff(system, g, tlims, xlims, sys_true,
             bounds, sample=None, pw=False, xderiv=False, n=50, log=True):
    """Estimates the max difference between the trajectories of two systems

    This function samples initial values and nodal forces for a number of
    iterations, computes the trajectories for each system and its absolute
    difference. Then, it computes the worst case error, either globally or for
    each element.

    Parameters
    ----------
    system : any `System` class in `femformal.core.system`
        The case study system
    g : array_like
        PDE boundary conditions
    tlims : tuple, (t0, T)
        Initial and final time of the trajectories
    xlims : tuple, (xl, xr)
        Left and right bounds of the rectangle in the spatial domain in which
        the difference is computed
    sys_true : any `System` class in `femformal.core.system`
        The base system, i.e., the one considered exact
    bounds : tuple (bounds_ic, bounds_f)
        Bounds or parameters passed to the sample functions
    sample : tuple of callable, (sample_ic, sample_f), optional
        Sampling functions `Type.sample` with `Type` a subclass of
        :class:`Sample`.  `sample_ic` samples the initial value. `sample_f`
        samples a nodal force. Defaults to (:meth:`lin_sample`, None)
    pw : bool, optional
        Whether the maximum should be computed pointwise or resumed
    xderiv : bool, optional
        If ``True``, compute the difference between the spatial derivatives
        instead
    n : int, optional
        Number of iterations
    log : bool, optional
        Log progress

    Returns
    -------
    float or array_like, shape (nelems of system[, dofs])
        If pw == False, the return value is the max diff for all elements

    """
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
    """Estimates the max directional difference in the trajectory of a system

    This function samples initial values and nodal forces for a number of
    iterations, computes the trajectory and its max directional difference.
    Then, it computes the worst case difference for each element.

    Parameters
    ----------
    system : any `System` class in `femformal.core.system`
        The case study system
    g : array_like
        PDE boundary conditions
    tlims : tuple, (t0, T)
        Initial and final time of the trajectories
    bounds : tuple (bounds_ic, bounds_f)
        Bounds or parameters passed to the sample functions
    sample : tuple of callable, (sample_ic, sample_f), optional
        Sampling functions `Type.sample` with `Type` a subclass of
        :class:`Sample`.  `sample_ic` samples the initial value. `sample_f`
        samples a nodal force. Defaults to (:meth:`lin_sample`, None)
    n : int, optional
        Number of iterations
    log : bool, optional
        Log progress

    Returns
    -------
    array_like, shape (nelems of system)

    """
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
    """Estimates the max temporal difference in the trajectory of a system

    This function samples initial values and nodal forces for a number of
    iterations, computes the trajectory and its max temporal difference.
    Then, it computes the worst case difference for each node.

    Parameters
    ----------
    system : any `System` class in `femformal.core.system`
        The case study system
    g : array_like
        PDE boundary conditions
    tlims : tuple, (t0, T)
        Initial and final time of the trajectories
    bounds : tuple (bounds_ic, bounds_f)
        Bounds or parameters passed to the sample functions
    sample : tuple of callable, (sample_ic, sample_f), optional
        Sampling functions `Type.sample` with `Type` a subclass of
        :class:`Sample`.  `sample_ic` samples the initial value. `sample_f`
        samples a nodal force. Defaults to (:meth:`lin_sample`, None)
    n : int, optional
        Number of iterations
    log : bool, optional
        Log progress

    Returns
    -------
    array_like, shape (nnodes of system)

    """
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
    """Estimates the max spatial and temporal difference of a system

    This function samples initial values and nodal forces for a number of
    iterations, computes the trajectory and its max spatial and temporal
    differences.
    Then, it computes the worst case difference for each node.

    Parameters
    ----------
    system : any `System` class in `femformal.core.system`
        The case study system
    g : array_like
        PDE boundary conditions
    tlims : tuple, (t0, T)
        Initial and final time of the trajectories
    bounds : tuple (bounds_ic, bounds_f)
        Bounds or parameters passed to the sample functions
    sample : tuple of callable, (sample_ic, sample_f)
        Sampling functions `Type.sample` with `Type` a subclass of
        :class:`Sample`.  `sample_ic` samples the initial value. `sample_f`
        samples a nodal force
    xderiv : bool, optional
        If ``True``, compute the difference between the spatial derivatives
        instead
    n : int, optional
        Number of iterations
    log : bool, optional
        Log progress

    Returns
    -------
    array_like, shape (nelems of system[, dofs, domain dimension])
    array_like, shape (nnodes of system[, dofs])

    """
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
    """Perturbs a continuous predicate

    The returned profile is accurate at the midpoints of the elements, i.e.,
    the values at the midpoints match what the perturbed STL predicate would
    be after discretization and correction.

    Parameters
    ----------
    apc : :class:`femformal.core.logic.APCont`
    eps : array_like, shape (2[, nelems[, dofs]])
        Approximation error in state or spatial derivative for each element
    eta : array_like, shape (2, nelems[, dofs, domain dimension])
        Approximation of the maximum directional differences of state and its
        derivatives.
    nu : array_like, shape (2, nnodes[, dofs])
        Approximation of the maximum temporal difference of state and its
        spatial derivatives
    xpart : array_like
        1D partition of the spatial domain, given as the list of nodes
    mesh : :class:`femformal.core.fem.mesh.Mesh`
        2D mesh. Must have a `build_elem` attribute that constructs a
        :class:`femformal.core.fem.element.Element`
    fdt_mult : int
        STL time multiplier

    Returns
    -------
    callable
        The perturbed profile

    """
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
    """Holds all information needed to run a femformal model

    Parameters
    ----------
    kwargs : dict
        Dictionary holding all attributes to set

        ==========  ===================================================
        key         meaning
        ==========  ===================================================
        'system'    FEM system
        'dsystem'   Discretized FEM system (deprecated)
        'xpart'     1D partition (deprecated)
        'g'         PDE boundary conditions
        'dt'        Time interval
        'fdt_mult'  STL time multiplier
        'd0'        Initial value
        'pset'      List of parameter polytopes
        'f'         List of parameterized functions
        'spec'      STL formula
        'T'         Final time
        ==========  ===================================================

    """
    def __init__(self, **kwargs):
        copy = kwargs
        self.system   = copy.pop('system', None)
        self.dsystem  = copy.pop('dsystem', None)
        self.xpart    = copy.pop('xpart', None)
        self.g        = copy.pop('g', 0)
        self.dt       = copy.pop('dt', 0)
        self.fdt_mult = copy.pop('fdt_mult', 1)
        self.d0       = copy.pop('d0', None)
        self.pset     = copy.pop('pset', None)
        self.f        = copy.pop('f', None)
        self.spec     = copy.pop('spec', None)
        self.T        = copy.pop('T', 0)

        if len(copy) > 0:
            raise Exception('Undefined parameters in CaseStudy: {}'.format(copy))
