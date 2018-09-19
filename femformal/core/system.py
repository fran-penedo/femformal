"""
ODE systems obtained from FEM discretization of PDEs and associated functions.
"""
from __future__ import division, absolute_import, print_function

import logging
from bisect import bisect_left, bisect_right

import numpy as np
import scipy
from scipy import linalg as la
from scipy.integrate import odeint
from scipy.optimize import linprog
from scipy.sparse import linalg as spla

from . import draw_util as draw


logger = logging.getLogger(__name__)

class FOSystem(object):
    """First order system: M dx + Kx = F

    Parameters
    ----------
    M, K, F : array_like
        System matrices
    xpart : array_like, optional
        1D partition of the spatial domain, given as the list of nodes
    mesh : :class:`femformal.core.fem.mesh.Mesh`, optional
        2D mesh. Must have a `build_elem` attribute that constructs a
        :class:`femformal.core.fem.element.Element`
    dt : float, optional
        Time interval to be used in integration

    """
    def __init__(self, M, K, F, xpart=None, dt=1.0, mesh=None):
        self.M = M
        self.K = K
        self.F = F
        xpart_given = xpart is not None
        mesh_given = mesh is not None
        if xpart_given and mesh_given:
            raise Exception("Expected either xpart or mesh")

        self.xpart = xpart
        self.mesh = mesh
        self.dt = dt

    def to_canon(self):
        M, K, F, n_e = _ns_sys_matrices(self)

        A = np.identity(self.n)
        A[np.ix_(n_e, n_e)] = np.linalg.solve(M, -K)
        b = np.zeros(self.n)
        b[n_e] = np.linalg.solve(M, F)
        C = np.empty(shape=(0,0))

        system = System(A, b, C, self.xpart, self.dt)

        return system

    @property
    def n(self):
        """State space dimension"""
        return len(self.M)

    def add_f_nodal(self, f_nodal):
        """Adds a nodal force to the system

        Parameters
        ----------
        f_nodal : array_like

        """
        self.F = self.F + f_nodal

    def copy(self):
        """Returns a copy of this system.

        Structural properties (mesh) are
        shallow copied, while system matrices are deep copied

        """
        return FOSystem(self.M.copy(), self.K.copy(), self.F.copy(),
                        self.xpart, self.dt, self.mesh)

    def __str__(self):
        return "M:\n{0}\nK:\n{1}\nF:\n{2}".format(self.M, self.K, self.F)


class SOSystem(object):
    """Second order system: M ddx + Kx = F

    Parameters
    ----------
    M, K, F : array_like
        System matrices
    xpart : array_like, optional
        1D partition of the spatial domain, given as the list of nodes
    mesh : :class:`femformal.core.fem.mesh.Mesh`, optional
        2D mesh. Must have a `build_elem` attribute that constructs a
        :class:`femformal.core.fem.element.Element`
    dt : float, optional
        Time interval to be used in integration

    """
    def __init__(self, M, K, F, xpart=None, dt=1.0, mesh=None):
        self.M = M
        self._K = K
        self.F = F
        xpart_given = xpart is not None
        mesh_given = mesh is not None
        if xpart_given and mesh_given:
            raise Exception("Expected either xpart or mesh")

        self.xpart = xpart
        self.mesh = mesh
        self.dt = dt

    def to_fosystem(self):
        """Transforms the SO system into a FO system by state augmentation"""
        n = self.n
        zeros = np.zeros((n, n))
        ident = np.identity(n)
        Maug = np.asarray(np.bmat([[ident, zeros],[zeros, self.M]]))
        Kaug = np.asarray(np.bmat([[zeros, -ident],[self.K, zeros]]))
        Faug = np.hstack([np.zeros(n), self.F])

        system = FOSystem(Maug, Kaug, Faug, self.xpart, self.dt)
        return system

    @property
    def n(self):
        """State space dimension"""
        return self.M.shape[0]

    @property
    def K(self):
        """K system matrix"""
        return self._K

    def add_f_nodal(self, f_nodal):
        """Adds a nodal force to the system

        Parameters
        ----------
        f_nodal : array_like

        """
        self.F = self.F + f_nodal

    def copy(self):
        """Returns a copy of this system.

        Structural properties (mesh) are
        shallow copied, while system matrices are deep copied

        """
        return SOSystem(self.M.copy(), self.K.copy(), self.F.copy(),
                        self.xpart, self.dt, self.mesh)

    def __str__(self):
        return "M:\n{0}\nK:\n{1}\nF:\n{2}".format(self.M, self.K, self.F)


class ControlSOSystem(SOSystem):
    """Second order controlled system: M ddx + Kx = F + u(t)

    Parameters
    ----------
    M, K, F : array_like
        System matrices
    f_nodal : callable
        The control input u(t)
    xpart : array_like, optional
        1D partition of the spatial domain, given as the list of nodes
    mesh : :class:`femformal.core.fem.mesh.Mesh`, optional
        2D mesh. Must have a `build_elem` attribute that constructs a
        :class:`femformal.core.fem.element.Element`
    dt : float, optional
        Time interval to be used in integration

    """
    def __init__(self, M, K, F, f_nodal, xpart=None, dt=1.0, mesh=None):
        SOSystem.__init__(self, M, K, F, xpart=xpart, dt=dt, mesh=mesh)
        self.f_nodal = f_nodal

    def add_f_nodal(self, f_nodal):
        """Adds a nodal force to the system (control input)

        Parameters
        ----------
        f_nodal : callable

        """
        self.f_nodal = f_nodal

    @staticmethod
    def from_sosys(sosys, f_nodal):
        """Constructs a :class:`ControlSOSystem` from a :class:`SOSystem`

        Parameters
        ----------
        sosys : :class:`SOSystem`
        f_nodal : callable

        Returns
        -------
        :class:`ControlSOSystem`

        """
        csosys = ControlSOSystem(
            sosys.M, sosys.K, sosys.F, f_nodal, sosys.xpart, sosys.dt, sosys.mesh)
        return csosys

    def copy(self):
        """Returns a copy of this system.

        Structural properties (mesh) are
        shallow copied, while system matrices are deep copied

        """
        return ControlSOSystem.from_sosys(
            super(ControlSOSystem, self).copy(), self.f_nodal)


class ControlFOSystem(FOSystem):
    """First order controlled system: M dx + Kx = F + u(t)

    Parameters
    ----------
    M, K, F : array_like
        System matrices
    f_nodal : callable
        The control input u(t)
    xpart : array_like, optional
        1D partition of the spatial domain, given as the list of nodes
    mesh : :class:`femformal.core.fem.mesh.Mesh`, optional
        2D mesh. Must have a `build_elem` attribute that constructs a
        :class:`femformal.core.fem.element.Element`
    dt : float, optional
        Time interval to be used in integration

    """
    def __init__(self, M, K, F, f_nodal, xpart=None, dt=1.0, mesh=None):
        FOSystem.__init__(self, M, K, F, xpart=xpart, dt=dt, mesh=mesh)
        self.f_nodal = f_nodal

    def add_f_nodal(self, f_nodal):
        """Adds a nodal force to the system (control input)

        Parameters
        ----------
        f_nodal : callable

        """
        self.f_nodal = f_nodal

    @staticmethod
    def from_fosys(fosys, f_nodal):
        """Constructs a :class:`ControlFOSystem` from a :class:`FOSystem`

        Parameters
        ----------
        fosys : :class:`FOSystem`
        f_nodal : callable

        Returns
        -------
        :class:`ControlFOSystem`

        """
        csosys = ControlFOSystem(
            fosys.M, fosys.K, fosys.F, f_nodal, fosys.xpart, fosys.dt, fosys.mesh)
        return csosys

    def copy(self):
        """Returns a copy of this system.

        Structural properties (mesh) are
        shallow copied, while system matrices are deep copied

        """
        return ControlFOSystem.from_fosys(
            super(ControlFOSystem, self).copy(), self.f_nodal)


def make_control_system(sys, f_nodal):
    """Constructs a controlled system from an ode system

    Parameters
    ----------
    sys : either :class:`FOSystem` or :class:`SOSystem`
    f_nodal : callable

    Returns
    -------
    Either :class:`ControlFOSystem` or :class:`ControlSOSystem`

    """
    if isinstance(sys, FOSystem):
        return ControlFOSystem.from_fosys(sys, f_nodal)
    else:
        return ControlSOSystem.from_sosys(sys, f_nodal)


class _MatrixFunction(object):
    def __init__(self, f):
        self.f = f
        self.keys = []

    def __getitem__(self, key):
        mf = _MatrixFunction(self.f)
        mf.keys = self.keys + [key]
        return mf

    def __call__(self, *args):
        val = self.f(*args)
        for k in self.keys:
            val = val[k]
        return val


class HybridParameter(object):
    def __init__(self, invariants, values, p=None):
        self._invariants = invariants
        self.values = values
        self.p = p

    @property
    def invariants(self):
        try:
            invs = [inv(self.p) for inv in self._invariants]
        except TypeError:
            return self._invariants
        def _fix_inv(inv):
            (a, b) = inv
            if len(a.shape) == 1:
                a = a[None]
            if not isinstance(b, np.ndarray):
                b = np.array([b])
            return (a, b)

        return [_fix_inv(inv) for inv in invs]


    @invariants.setter
    def invariants(self, value):
        self._invariants = value

    def invariant_representatives(self):
        return [la.lstsq(inv[0], inv[1] - np.abs(inv[1]) * 1e-3)[0]
                for inv in self.invariants]

    def _get_value(self, u):
        for (A, b), v in zip(self.invariants, self.values):
            if np.all(A.dot(u) <= b):
                return v
        raise Exception("Input not covered in any invariant")

    def __call__(self, args):
        return self._get_value(args)


class HybridSOSystem(SOSystem):
    def __init__(self, M, K, F, xpart=None, dt=1.0, mesh=None, bigN_deltas=1.0, bigN_int_force=1.0, bigN_acc=1.0):
        SOSystem.__init__(self, M, K, F, xpart, dt, mesh)
        self.bigN_deltas = bigN_deltas
        self.bigN_int_force = bigN_int_force
        self.bigN_acc = bigN_acc

    @property
    def K(self):
        return _MatrixFunction(self.K_global)

    def K_global(self, u):
        k_els = [self._K[i](u[i:i+2]) for i in range(len(self._K))]
        return self.K_build_global(k_els)

    def K_build_global(self, k_els):
        K = np.zeros(self.M.shape)
        for i in range(len(k_els)):
            K[i:i+2,i:i+2] = K[i:i+2,i:i+2] + k_els[i]
        return K

    def K_els(self):
        return self._K


class ControlHybridSOSystem(HybridSOSystem, ControlSOSystem):
    def __init__(self, M, K, F, f_nodal, xpart=None, dt=1.0):
        ControlSOSystem.__init__(self, M, K, F, f_nodal, xpart=xpart, dt=dt)

    @staticmethod
    def from_hysosys(sosys, f_nodal):
        csosys = ControlHybridSOSystem(
            sosys.M, sosys.K, sosys.F, f_nodal, sosys.xpart, sosys.dt)
        return csosys

    def copy(self):
        return HybridControlSOSystem.from_hysosys(
            super(HybridControlSOSystem, self).copy(), self.f_nodal)


class PWLFunction(object):
    """Right continuous piecewise linear function

    Parameters
    ----------
    ts : array_like
        Ordered list of inflection points, including the endpoints of the domain.
        A point may be repeated once in order to allow left discontinuity
    ys : array_like, optional
        List of values at the inflection points
    ybounds : array_like, optional
        Bounds on the values of the function. Used as synthesis information
        when `ys` is left unspecified
    x : float or array_like, optional
        Point of the spatial domain at which this function is nonzero. If not
        set, when evaluating the spatial point should be set to ``None``

    """
    def __init__(self, ts, ys=None, ybounds=None, x=None):
        self.ts = np.array(ts)
        if ys is not None:
            self.ys = np.array(ys)
        else:
            self.ys = None
        if ybounds is None:
            if len(self.ys.shape) == 1:
                self.ybounds = np.zeros(2)
            else:
                self.ybounds = np.zeros(self.ys.shape[0], 2)
        else:
            self.ybounds = np.array(ybounds)
        self.x = x

    def pset(self):
        """Creates an H-representation of the `ys` polytope

        If `ybounds` was specified, this function computes the polytope in which
        the `ys` are contained.

        Returns
        -------
        ndarray

        """
        ybounds = self.ybounds
        if len(ybounds.shape) == 1:
            ybounds = ybounds[None]

        matrix = []
        left = np.hstack([np.identity(len(self.ts)),
                          -np.identity(len(self.ts))])
        for yb in ybounds:
            right = np.r_[[yb[1] for x in self.ts],
                        [-yb[0] for x in self.ts]]
            matrix.append(np.vstack([left, right]).T)

        if ybounds.shape[0] == 1:
            return matrix[0]
        else:
            return np.array(matrix)

    def __call__(self, t, p=None, x=None):
        """Computes the value of the function at some point of the domain

        Computes the value at `t`, using `p` as the inflection point values and
        at the spatial point `x`.

        Parameters
        ----------
        t : float
        p : array_like, optional
        x : float or array_like, optional

        Returns
        -------
        float

        """
        if x != self.x:
            return 0.0
        ts = self.ts
        if self.ys is not None:
            ys = self.ys
            # if t < ts[0] or t > ts[-1]:
            #     raise Exception("Argument out of domain. t = {}".format(t))
        else:
            if p is None:
                raise Exception("y values not set")
            self.ys = ys = p
        if len(self.ybounds.shape) == 1:
            ys = [ys]
        # FIXME probable issue with t = ts[-1]
        if t >= ts[-1]:
            ret = [y[-1] for y in ys]
        elif t <= ts[0]:
            ret = [y[0] for y in ys]
        else:
            i = bisect_right(ts, t) - 1
            ret = [y[i] + (y[i+1] - y[i]) * (t - ts[i]) / (ts[i+1] - ts[i])
                    for y in ys]

        if len(self.ybounds.shape) == 1:
            return ret[0]
        else:
            return ret


# Abstraction based approach utilities

class System(object):
    """dx = Ax + b + Cw system. Currently deprecated"""

    def __init__(self, A, b, C=None, xpart=None, dt=1.0):
        """
        Creates the system $\dot{x} = A x + C w + b$

        Args:
            A: System matrix A
            b: Affine term b
            C: Perturbation matrix C
        """
        self.A = np.array(A)
        self.b = np.array(b)
        if C is not None:
            self.C = np.array(C)
        else:
            self.C = np.empty(shape=(0,0))
        self.xpart = xpart
        self.dt = dt

    def subsystem(self, indices):
        i = np.array(indices)
        A = self.A[np.ix_(i, i)]
        b = self.b[i]

        j = self._pert_indices(i)
        if len(j) > 0:
            C = self.A[np.ix_(i, j)]
        else:
            C = np.empty(shape=(0,0))

        return System(A, b, C)

    def pert_indices(self, indices):
        return self._pert_indices(np.array(indices)).tolist()

    def _pert_indices(self, i):
        j = np.setdiff1d(np.nonzero(self.A[i, :])[1], i)
        j = j[(j >= 0) & (j < self.n)]
        return j

    @property
    def n(self):
        return len(self.A)

    @property
    def m(self):
        return self.C.shape[1]

    def __str__(self):
        return "A:\n{0}\nb:\n{1}\nc:\n{2}".format(self.A, self.b, self.C)


def is_region_invariant(system, region, dist_bounds):
    for facet, normal, dim in facets(region):
        if not is_facet_separating(system, facet, normal, dim, dist_bounds):
            return False

    return True

def facets(region):
    for i in range(len(region)):
        facet = region.copy()
        facet[i][0] = facet[i][1]
        yield facet, 1, i
        facet = region.copy()
        facet[i][1] = facet[i][0]
        yield facet, -1, i

def is_facet_separating(system, facet, normal, dim, dist_bounds):
    return _is_facet_separating(system.A, system.b, system.C,
                                facet, normal, dim, dist_bounds)

def _is_facet_separating(A, b, C, facet, normal, dim, dist_bounds):
    A_j = np.delete(A[dim], dim) * normal
    if len(C) > 0:
        A_j = np.hstack([A_j, normal * C[dim]])
    b = (- b[dim] - A[dim, dim] * facet[dim][0]) * normal
    R = np.delete(facet, dim, 0)
    if len(C) > 0:
        R = np.vstack([R, dist_bounds])
    return rect_in_semispace(R, A_j, b)

def rect_in_semispace(R, a, b):
    if np.all(np.isclose(a, 0)):
        return b >= 0
    else:
        res = linprog(-a, a[None], b + 1, bounds=R)
        if res.status == 2:
            return False
        else:
            return - res.fun <= b

def cont_to_disc(system, dt=1.0):
    Adt = system.A * dt
    Atil = la.expm(Adt)
    btil = Atil.dot(- la.solve(
        system.A, (la.expm(-Adt) - np.identity(system.n)))).dot(system.b)
    return System(Atil, btil)

def cont_integrate(system, x0, t):
    return odeint(lambda x, t: (system.A.dot(x) + system.b.T).flatten(), x0, t)

def disc_integrate(system, x0, t):
    xs = [x0]
    x = x0
    for i in range(t - 1):
        x = (system.A.dot(x) + system.b.T).flatten()
        xs.append(x)
    return np.array(xs)


# Integration functions

def _ns_sys_matrices(system):
    nz_bool = system.M != 0
    try:
        nz_bool = nz_bool.toarray()
    except:
        pass
    n_e = np.nonzero(np.any(nz_bool, axis=1))[0]
    M = system.M[np.ix_(n_e, n_e)]
    try:
        M = M.tocsc()
    except:
        pass
    K = system.K[np.ix_(n_e, n_e)]
    F = system.F[n_e]

    return M, K, F, n_e

def _factorize(M):
    try:
        if scipy.sparse.issparse(M):
            return spla.factorized(M)
        else:
            lu = la.lu_factor(M)
            def solve(b):
                return la.lu_solve(lu, b)

            return solve
    except ValueError:
        logger.info("Integrating M = 0 system")
        return lambda x: np.zeros(len(x))

def trapez_integrate(fosys, d0, T, dt=.1, alpha=0.5, log=True):
    """Integrates a FO system using the trapezoidal rule

    Parameters
    ----------
    fosys : :class:`FOSystem`
    d0 : array_like
        Initial value
    T : float
        Final time
    dt : float, optional
        Time interval
    alpha : float, optional
        Trapezoidal rule parameter

    Returns
    -------
    array, shape (round(T / dt), len(d0))
        Array containing the value of `d` for each time until `T`, with `d0` in
        the first row

    """
    if log:
        logger.info(
            "Integrating FO system with trapezoidal rule: alpha = {}".format(alpha))
    M, K, F, n_e = _ns_sys_matrices(fosys)

    try:
        f_nodal = fosys.f_nodal
    except AttributeError:
        f_nodal = np.zeros(F.shape)

    its = int(round(T / dt))
    d = np.array(d0)
    if d.shape != (fosys.n,):
        raise ValueError("System and initial value shape do not agree: "
                         "system dimension = {}, d shape = {}".format(
                             fosys.n, d.shape))
    v = np.zeros(d.shape[0])
    td = np.zeros(d.shape[0])
    try:
        solve_m = _factorize(M + alpha * dt * K)
    except ValueError:
        logger.info("Integrating M = 0 system")
        solve_m = lambda x: np.zeros(len(n_e))
    try:
        f_nodal_c = f_nodal(0)[n_e]
    except TypeError:
        f_nodal_c = f_nodal
    try:
        K_cur = K(d)
    except TypeError:
        K_cur = K
    v[n_e] = la.solve(M, F + f_nodal_c - K_cur.dot(d[n_e]))
    ds = [d]
    vs = [v]
    for i in range(its):
        try:
            f_nodal_c = f_nodal((i+1) * dt)[n_e]
        except TypeError:
            f_nodal_c = f_nodal
        try:
            K_cur = K(d)
        except TypeError:
            K_cur = K
        if alpha > 0:
            td = d + (1 - alpha) * dt * v
            d = d.copy()
            d[n_e] = solve_m(alpha * dt * (F + f_nodal_c) + M.dot(td[n_e]))
            v = (d - td) / (alpha * dt)
        else:
            d = d + dt * v
            v[n_e] = solve_m(F + f_nodal_c - K_cur.dot(d[n_e]))
        ds.append(d)
        vs.append(v)
    return np.array(ds)

def central_diff_integrate(sosys, d0, v0, T, dt=.1):
    """Integrates a SO system using the central difference rule

    Parameters
    ----------
    sosys : :class:`SOSystem`
    d0 : array_like
        Initial value
    v0 : array_like
        Initial velocity
    T : float
        Final time
    dt : float, optional
        Time interval

    Returns
    -------
    array, shape (round(T / dt), len(d0))
        Array containing the value of `d` for each time until `T`, with `d0` in
        the first row
    array, shape (round(T / dt), len(v0))
        Array containing the value of `v` for each time until `T`, with `v0` in
        the first row

    """
    M, K, F, n_e = _ns_sys_matrices(sosys)

    try:
        f_nodal = sosys.f_nodal
    except AttributeError:
        f_nodal = np.zeros(F.shape)

    its = int(round(T / dt))
    d = np.array(d0)
    v = np.array(v0)
    if d.shape != v.shape or d.shape != (sosys.n,):
        raise ValueError("System and initial value shapes do not agree: "
                         "system dimension = {}, d shape = {}, v shape = {}".format(
                             sosys.n, d.shape, v.shape))
    a = np.zeros(d.shape[0])
    try:
        solve_m = _factorize(M)
    except ValueError:
        logger.info("Integrating M = 0 system")
        solve_m = lambda x: np.zeros(len(n_e))
    try:
        f_nodal_c = f_nodal(0)[n_e]
    except TypeError:
        f_nodal_c = f_nodal
    try:
        K_cur = K(d)
    except TypeError:
        K_cur = K
    a[n_e] = solve_m(F + f_nodal_c - K_cur.dot(d[n_e]))
    ds = [d]
    vs = [v]
    for i in range(its):
        tv = v + .5 * dt * a
        # tv[0] = tv[-1] = 0.0
        d = d + dt * tv
        try:
            f_nodal_c = f_nodal((i+1) * dt)[n_e]
        except TypeError:
            f_nodal_c = f_nodal
        try:
            K_cur = K(d)
        except TypeError:
            K_cur = K
        a[n_e] = solve_m(F + f_nodal_c - K_cur.dot(d[n_e]))
        v = tv + .5 * dt * a
        ds.append(d)
        vs.append(v)
    return np.array(ds), np.array(vs)

def newm_integrate(sosys, d0, v0, T, dt=.1, beta=0, gamma=.5):
    """Integrates a SO system using the newmark algorithm

    Only `gamma` == 0.5 has second order accuracy. Commonly used values for `beta`
    are the following:

    - `beta` == 0 : Central Difference. Explicit method, conditionally stable
    - `beta` == 0.25 : Trapezoidal rule (average acceleration). Implicit method,
        unconditionally stable
    - `beta` == 1/6 : Linear acceleration. Implicit method, conditionally stable
    - `beta` == 1/12 : Fox-Goodwin. Implicit method, conditionally stable

    Parameters
    ----------
    sosys : :class:`SOSystem`
    d0 : array_like
        Initial value
    v0 : array_like
        Initial velocity
    T : float
        Final time
    dt : float, optional
        Time interval
    beta : float, optional
    gamma : float, optional

    Returns
    -------
    array, shape (round(T / dt), len(d0))
        Array containing the value of `d` for each time until `T`, with `d0` in
        the first row
    array, shape (round(T / dt), len(v0))
        Array containing the value of `v` for each time until `T`, with `v0` in
        the first row

    """
    logger.debug("Integrating with newmark, parameters beta = {}, gamma = {}"
                 "".format(beta, gamma))
    M, K, F, n_e = _ns_sys_matrices(sosys)

    try:
        f_nodal = sosys.f_nodal
    except AttributeError:
        f_nodal = np.zeros(F.shape)

    its = int(round(T / dt))
    d = np.array(d0)
    v = np.array(v0)
    if d.shape != v.shape or d.shape != (sosys.n,):
        raise ValueError("System and initial value shapes do not agree: "
                         "system dimension = {}, d shape = {}, v shape = {}".format(
                             sosys.n, d.shape, v.shape))
    a = np.zeros(d.shape[0])
    try:
        K_cur = K(d)
        hybrid = True
    except TypeError:
        K_cur = K
        hybrid = False
    solve_m = _factorize(M + beta * dt * dt * K_cur)
    try:
        f_nodal_c = f_nodal(0)[n_e]
    except TypeError:
        f_nodal_c = f_nodal
    a[n_e] = _factorize(M)(F + f_nodal_c - K_cur.dot(d[n_e]))
    ds = [d]
    vs = [v]
    for i in range(its):
        # logger.debug(a)
        # tv[0] = tv[-1] = 0.0
        td = d + dt * v + 0.5 * dt * dt * (1 - 2 * beta) * a
        tv = v + (1 - gamma) * dt * a
        try:
            f_nodal_c = f_nodal((i+1) * dt)[n_e]
        except TypeError:
            f_nodal_c = f_nodal
        a[n_e] = solve_m(F + f_nodal_c - K_cur.dot(td[n_e]))
        d = td + beta * dt * dt * a
        v = tv + gamma * dt * a
        ds.append(d)
        vs.append(v)
        if hybrid:
            K_cur = K(d)
            solve_m = _factorize(M + beta * dt * dt * K_cur)
    return np.array(ds), np.array(vs)


# Functions computing differences between systems and time-space differences

def _linterx(d, xpart):
    def u(x):
        i = bisect_left(xpart, x)
        if i < 1:
            return d[:, 0]
        if i > len(xpart) - 1:
            return d[:, -1]
        else:
            return (d[:, i-1] * (xpart[i] - x) + d[:, i] * (x - xpart[i-1])) / \
                    (xpart[i] - xpart[i-1])
    return u

def _pwcx(d, xpart):
    def u(x):
        i = bisect_left(xpart, x) - 1
        return d[:, i]
    return u


def diff(x, y, dtx, dty, xpart, ypart, xl, xr, pwc=False):
    """Computes the difference between two trajectories of a 1D PDE FEM system

    Computes `x` - `y` if `dty` < `dtx`, otherwise assume all arguments are
    switched.

    Parameters
    ----------
    x : array, shape (integration points, len(xpart))
    y : array, shape (integration points, len(ypart))
    dtx, dty : float
    xpart, ypart : array_like
    xl, xr : float
        Left and right bounds of the rectangle in the spatial domain in which
        the difference is computed
    pwc : bool
        Whether the values in `x` and `y` should be understood as constant
        values at the elements. In that case, the length of the last axis for
        `x` and `y` should be one less

    Returns
    -------
    array, shape (integration points, nodes of ypart in [xl, xr])
        The difference between the two trajectories

    """
    if dty > dtx:
        x, y = y, x
        dtx, dty = dty, dtx
        xpart, ypart = ypart, xpart

    yy = y[::int(round(dtx / dty))]
    yl = bisect_left(ypart, xl)
    yr = bisect_right(ypart, xr)
    if pwc:
        xinter = _pwcx(x, xpart)
        yinter = _pwcx(yy, ypart)
        d = np.array([xinter(z) - yinter(z) for z in
                      (ypart[yl + 1:yr] + ypart[yl:yr - 1]) / 2.0]).T
    else:
        xinter = _linterx(x, xpart)
        yinter = _linterx(yy, ypart)
        d = np.array([xinter(z) - yinter(z) for z in ypart[yl:yr]]).T
    return d

def diff2d(x, y, dtx, dty, xmesh, ymesh, xl, xr):
    """Computes the difference between two trajectories of a 2D PDE FEM system

    Computes `x` - `y` if `dty` < `dtx`, otherwise assume all arguments are
    switched.

    Parameters
    ----------
    x : array, shape (integration points, len(xpart))
    y : array, shape (integration points, len(ypart))
    dtx, dty : float
    xmesh, ymesh : :class:`femformal.core.fem.mesh.Mesh`
    xl, xr : float
        Left and right bounds of the rectangle in the spatial domain in which
        the difference is computed

    Returns
    -------
    array, shape (integration points, ysys nodes in (xl, xr)[, dofs])
        The difference between the two trajectories

    """
    if dty > dtx:
        x, y = y, x
        dtx, dty = dty, dtx
        xmesh, ymesh = ymesh, xmesh

    yy = y[::int(round(dtx / dty))]
    xinter = xmesh.interpolate(x)
    yinter = ymesh.interpolate(yy)
    nodes = ymesh.find_nodes_between(xl, xr)
    d = np.array([xinter(*coords) - yinter(*coords) for _, coords in nodes])
    return d


def sys_diff(xsys, ysys, x0, y0, tlims, xlims, xderiv=False, plot=False):
    """Computes the absolute diff between the trajectories of two FO systems

    Parameters
    ----------
    xsys, ysys : :class:`FOSystem`
    x0, y0 : array_like
        Initial value
    tlims : tuple, (t0, T)
        Initial and final time of the trajectories
    xlims : tuple, (xl, xr)
        Left and right bounds of the rectangle in the spatial domain in which
        the difference is computed
    xderiv : bool, optional
        If ``True``, compute the difference between the spatial derivatives
        instead

    Returns
    -------
    array, shape (integration points, nodes of ysys.xpart in (xl, xr))
        The absolute difference between the trajectories of the two systems

    """
    dtx, dty = xsys.dt, ysys.dt
    xpart, ypart = xsys.xpart, ysys.xpart
    t0, T = tlims
    xl, xr = xlims
    x = trapez_integrate(xsys, x0, T, dtx, alpha=0.5)
    x = x[int(t0/dtx):]
    y = trapez_integrate(ysys, y0, T, dty, alpha=0.5)
    y = y[int(t0/dty):]
    absdif = np.abs(diff(x, y, dtx, dty, xpart, ypart, xl, xr))
    return absdif.T

def sosys_diff(xsys, ysys, x0, y0, tlims, xlims, xderiv=False, plot=False):
    """Computes the absolute diff between the trajectories of two SO systems

    Parameters
    ----------
    xsys, ysys : :class:`FOSystem`
    x0, y0 : array_like
        Initial value
    tlims : tuple, (t0, T)
        Initial and final time of the trajectories
    xlims : tuple, (xl, xr)
        Left and right bounds of the rectangle in the spatial domain in which
        the difference is computed
    xderiv : bool, optional
        If ``True``, compute the difference between the spatial derivatives
        instead

    Returns
    -------
    array, shape (integration points, ysys nodes in (xl, xr)[, dofs])
        The absolute difference between the trajectories of the two systems

    """
    dtx, dty = xsys.dt, ysys.dt
    t0, T = tlims
    x = newm_integrate(xsys, x0[0], x0[1], T, dtx, beta=0.25)[0]
    y = newm_integrate(ysys, y0[0], y0[1], T, dty, beta=0.25)[0]
    x = x[int(round(t0/dtx)):]
    y = y[int(round(t0/dty)):]
    # draw.draw_pde_trajectory(x, xpart, np.linspace(0, T, (int(round(T / dtx)))), animate=False)
    if xsys.xpart is not None:
        return _sosys_diff_1d(x, y, dtx, dty, xsys.xpart, ysys.xpart, xlims, xderiv, plot)
    else:
        err = diff2d(x, y, dtx, dty, xsys.mesh, ysys.mesh, xlims[0], xlims[1])
        if xderiv:
            return np.array([
                    [ysys.mesh.elements[elem].max_partial_derivs(
                        err_t[ysys.mesh.elem_nodes(elem, 2)])
                     for elem in range(ysys.mesh.nelems)]
                for err_t in err.transpose([2,0,1])]).transpose([1,2,3,0])
        else:
            return np.abs(err)

def _sosys_diff_1d(x, y, dtx, dty, xpart, ypart, xlims, xderiv, plot):
    # plot=True
    if xderiv:
        x = np.true_divide(np.diff(x), np.diff(xpart))
        y = np.true_divide(np.diff(y), np.diff(ypart))
        pwc = True
    else:
        pwc = False

    if any(isinstance(xlim, list) for xlim in xlims):
        absdif = [diff(x, y, dtx, dty, xpart, ypart, xlim[0], xlim[1], pwc)
               for xlim in xlims]
    else:
        xl, xr = xlims
        absdif = np.abs(diff(x, y, dtx, dty, xpart, ypart, xl, xr, pwc))
    if plot:
        # ttx = np.linspace(0, T, int(round(T / dtx)))
        # tty = np.linspace(0, T, int(round(T / dty)))
        ttx = np.arange(0, 1.0 + dtx / 2.0, dtx)
        tty = np.arange(0, 1.0 + dty / 2.0, dty)
        # draw.draw_pde_trajectory(x, xpart, ttx, hold=True)
        # draw.draw_pde_trajectory(y, ypart, tty, hold=True)
        draw.draw_pde_trajectories([x, y], [xpart, ypart], [ttx, tty], pwc=pwc)
        yl = bisect_left(ypart, xlims[0])
        yr = bisect_right(ypart, xlims[1]) - 1
        draw.draw_pde_trajectory(absdif, ypart[yl:yr], tty)
        # for xlim, dif in absdif:
        #     yl = bisect_left(ypart, xlim[0])
        #     yr = bisect_right(ypart, xlim[1]) - 1
        #     print yl, yr
        #     draw.draw_pde_trajectory(dif, ypart[yl:yr], ttx, hold=True)
        # draw.plt.show()
    return absdif.T

def sys_max_diff(xsys, ysys, x0, y0, tlims, xlims, xderiv=False, pw=False, plot=False):
    """Computes maximum absolute diff between two systems

    Parameters
    ----------
    xsys, ysys : :class:`FOSystem`
    x0, y0 : array_like
        Initial value
    tlims : tuple, (t0, T)
        Initial and final time of the trajectories
    xlims : tuple, (xl, xr)
        Left and right bounds of the rectangle in the spatial domain in which
        the difference is computed
    xderiv : bool, optional
        If ``True``, compute the difference between the spatial derivatives
        instead
    pw : bool, optional
        Whether the maximum should be computed pointwise or resumed

    Returns
    -------
    float or array, shape (integration points, ysys nodes in (xl, xr)[, dofs])
        The absolute difference between the trajectories of the two systems

    """
    if isinstance(xsys, SOSystem):
        diff_f = sosys_diff
    elif isinstance(xsys, FOSystem):
        diff_f = sys_diff
    else:
        diff_f = sys_diff

    absdif = diff_f(xsys, ysys, x0, y0, tlims, xlims, xderiv, plot)
    if isinstance(absdif, list):
        pwdiff = [np.amax(dif, axis=0) for dif in absdif]
        return pwdiff if pw else np.amax(pwdiff)
    else:
        return np.max(absdif, axis=-1 if pw else None)

def sys_max_xdiff(sys, x0, t0, T):
    """Maximum absolute spatial derivative of an FO system at each element

    Parameters
    ----------
    sys : :class:`FOSystem`
    x0 : array_like
        Initial value
    t0, T : float
        Initial and final time of the trajectory

    Returns
    -------
    array, shape (nodes of sys.xpart - 1, )

    """
    dt = sys.dt
    x = trapez_integrate(sys, x0, T, dt, alpha=0.5)
    x = x[int(round(t0/dt)):]

    dx = np.abs(np.diff(x))
    mdx = np.max(dx, axis=0)

    return mdx

def sys_max_tdiff(sys, x0, t0, T):
    """Maximum time spatial derivative of an FO system at each node

    Parameters
    ----------
    sys : :class:`FOSystem`
    x0 : array_like
        Initial value
    t0, T : float
        Initial and final time of the trajectory

    Returns
    -------
    array, shape (nodes of sys.xpart, )

    """
    dt = sys.dt
    x = trapez_integrate(sys, x0, T, dt, alpha=0.5)
    x = x[int(round(t0/dt)):]

    dtx = np.abs(np.diff(x, axis=0))
    mdtx = np.max(dtx, axis=0)

    return mdtx

def sosys_max_der_diff(sys, x0, tlims, xderiv=False, compute_derivative=False):
    """Maximum spatial and time spatial derivative of an SO system

    Parameters
    ----------
    sys : :class:`FOSystem`
    x0 : array_like
        Initial value
    tlims : tuple of float, (t0, T)
        Initial and final time of the trajectory
    xderiv : bool, optional
        If ``True``, compute the derivatives of the spatial derivatives instead

    Returns
    -------
    mdx : array, shape (nodes of system mesh - 1[, dofs[, domain dimension]])
        Maximum spatial derivative of the system at each element
    mdtx : array, shape (nodes of system mesh[, dofs])
        Maximum time derivative of the system at each node

    """
    x, vx = newm_integrate(sys, x0[0], x0[1], tlims[1], sys.dt, beta=0.25)
    x = x[int(round(tlims[0]/sys.dt)):]
    # draw.draw_pde_trajectory(x, sys.xpart, np.linspace(0, tlims[1], (int(round(tlims[1] / sys.dt)))), animate=False)
    if sys.xpart is not None:
        if xderiv:
            x = np.true_divide(np.diff(x), np.diff(sys.xpart))

        dx = np.abs(np.diff(x))
        dtx = np.abs(np.diff(x, axis=0))
        mdx = np.max(dx, axis=0)
        mdtx = np.max(dtx, axis=0)
    else:
        if xderiv:
            raise NotImplementedError("xderiv > 0 not implemented yet")

        # Compute max differences
        if not compute_derivative:
            interps = [sys.mesh.interpolate_derivatives(d) for d in x]
            dxs = []
            for interp in interps:
                dx = []
                for e in range(sys.mesh.nelems):
                    partials = np.abs(
                        np.array([interp(*coords) for coords in sys.mesh.elem_coords(e)]))
                    # logger.debug(partials)
                    dx.append(np.max(partials, axis=0))
                dxs.append(dx)
            mdx = np.max(dxs, axis=0)
        else:
            mdx = None
            for d in x:
                mpartials = np.array(
                    [sys.mesh.elements[e].max_partial_derivs(
                        sys.mesh.get_elem_values(d, e, 2))
                     for e in range(sys.mesh.nelems)])
                if mdx is None:
                    mdx = mpartials
                else:
                    mdx = np.max([mdx, mpartials], axis=0)

        dtx = np.abs(np.diff(x, axis=0))
        mdtx = np.max(dtx, axis=0)

    # logger.debug("dx = \n{}".format(dx[:,0]))
    # logger.debug("mdx = \n{}".format(mdx))

    return mdx, mdtx


def draw_system_disc(sys, x0, T, t0=0,
                     prefix=None, animate=True, allonly=False, hold=False,
                     ylabel='Temperature', xlabel='x'):
    dt = sys.dt
    xpart = sys.xpart

    tx = np.linspace(t0, T, int(round((T - t0)/dt)))
    x = disc_integrate(sys, x0[1:-1], int(round(T/dt)))
    x = np.c_[x0[0] * np.ones(x.shape[0]), x, x0[-1] * np.ones(x.shape[0])]
    x = x[int(round(t0/dt)):]
    draw.draw_pde_trajectory(x, xpart, tx, prefix=prefix,
                             animate=animate, hold=hold, allonly=allonly, ylabel=ylabel,
                             xlabel=xlabel)

def _draw_system_cont(sys, x0, T, t0=0, hold=False, **kargs):
    dt = sys.dt
    xpart = sys.xpart

    tx = np.linspace(t0, T, int(round((T - t0)/dt)))
    x = trapez_integrate(sys, x0, T, dt)
    # x = np.c_[x0[0] * np.ones(x.shape[0]), x, x0[-1] * np.ones(x.shape[0])]
    x = x[int(round(t0/dt)):]
    tx = tx[int(round(t0/dt)):]
    draw.draw_pde_trajectory(x, xpart, tx, hold=hold, **kargs)
    if hold:
        return draw.pop_holds()

def _draw_sosys(sosys, d0, v0, g, T, t0=0, hold=False, **kargs):
    dt = sosys.dt
    xpart = sosys.xpart

    tx = np.linspace(t0, T, int(round((T - t0)/dt)))
    d, v = newm_integrate(sosys, d0, v0, T, dt, beta=.25)
    d = d[int(round(t0/dt)):]
    draw.draw_pde_trajectory(d, xpart, tx, hold=hold, **kargs)
    if hold:
        return draw.pop_holds()

def draw_system(sys, d0, g, T, t0=0, **kargs):
    """Draws the trajectory of a FEM system from a 1D PDE

    Computes the trajectory and draws using the
    :func:`femformal.core.draw_util.draw_pde_trajectory`

    Parameters
    ----------
    sys : :class:`FOSystem` or :class:`SOSystem`
    d0 : array_like
        Initial value. Must have a row for each degree of freedom
    g : array_like
        Boundary conditions of the PDE
    T : float
        Final time of the trajectory
    t0 : float, optional
        Start time
    hold : bool
        Whether the figure should be held
    kwargs : dict, optional
        Extra arguments passed to the drawing function

    Returns
    -------
    list, only returned if hold == True
        List of `matplotlib` figures created

    """
    if isinstance(sys, FOSystem):
        return _draw_system_cont(sys, d0, T, t0, **kargs)
    elif isinstance(sys, SOSystem):
        return _draw_sosys(sys, d0[0], d0[1], g, T, t0, **kargs)
    else:
        raise NotImplementedError(
            "Not implemented for this class of system: {}".format(
                sys.__class__.__name__))


def draw_system_2d(sys, d0, g, T, t0=0, **kwargs):
    """Draws the trajectory of a FEM system from a 2D PDE

    Computes the trajectory and draws using the
    :func:`femformal.core.draw_util.draw_2d_pde_trajectory`

    Parameters
    ----------
    sys : :class:`SOSystem`
    d0 : array_like
        Initial value. Must have a row for each degree of freedom
    g : array_like
        Boundary conditions of the PDE
    T : float
        Final time of the trajectory
    t0 : float, optional
        Start time
    hold : bool
        Whether the figure should be held
    kwargs : dict, optional
        Extra arguments passed to the drawing function

    Returns
    -------
    list, only returned if hold == True
        List of `matplotlib` figures created

    """
    dt = sys.dt
    ts = np.linspace(t0, T, int(round((T - t0) / dt)))
    d, v = newm_integrate(sys, d0[0], d0[1], T, dt, beta=.25)
    # d = d[int(round(t0/dt)):]
    draw.draw_2d_pde_trajectory(
        d, sys.mesh.nodes_coords, sys.mesh.elems_nodes, ts, **kwargs)
    return draw.pop_holds()

def draw_system_deriv_2d(sys, d0, g, T, comp, t0=0, **kwargs):
    """Draws the trajectory of a FEM system from a 2D PDE

    Computes the trajectory and draws using the
    :func:`femformal.core.draw_util.draw_2d_pde_trajectory`

    Parameters
    ----------
    sys : :class:`SOSystem`
    d0 : array_like
        Initial value. Must have a row for each degree of freedom
    g : array_like
        Boundary conditions of the PDE
    T : float
        Final time of the trajectory
    comp : int
        Component of the stress vector to draw
    t0 : float, optional
        Start time
    hold : bool
        Whether the figure should be held
    kwargs : dict, optional
        Extra arguments passed to the drawing function

    Returns
    -------
    list, only returned if hold == True
        List of `matplotlib` figures created

    """
    dt = sys.dt
    ts = np.linspace(t0, T, int(round((T - t0) / dt)) + 1)
    d, v = newm_integrate(sys, d0[0], d0[1], T, dt, beta=.25)
    if 'system_t' in kwargs:
        sys_t = kwargs['system_t']
        d0_t = kwargs['d0_t']
        d_t, v_t = newm_integrate(sys_t, d0_t[0], d0_t[1], T, sys_t.dt, beta=.25)
        ts = np.linspace(t0, T, int(round((T - t0) / sys_t.dt)) + 1)
    draw.draw_derivative_2d(
        d, sys.mesh, ts, comp, ds_t=d_t, mesh_t=sys_t.mesh, **kwargs)
    return draw.pop_holds()

def draw_displacement_plot(sys, d0, g, T, t0=0, **kwargs):
    """Draws the trajectory of a FEM system from a 2D PDE

    Computes the trajectory and draws using the
    :func:`femformal.core.draw_util.draw_displacment_2d`

    Parameters
    ----------
    sys : :class:`SOSystem`
    d0 : array_like
        Initial value. Must have a row for each degree of freedom
    g : array_like
        Boundary conditions of the PDE
    T : float
        Final time of the trajectory
    t0 : float, optional
        Start time
    hold : bool
        Whether the figure should be held
    kwargs : dict, optional
        Extra arguments passed to the drawing function

    Returns
    -------
    list, only returned if hold == True
        List of `matplotlib` figures created

    """
    dt = sys.dt
    ts = np.linspace(t0, T, int(round((T - t0) / dt)) + 1)
    d, v = newm_integrate(sys, d0[0], d0[1], T, dt, beta=.25)
    if 'system_t' in kwargs:
        sys_t = kwargs['system_t']
        d0_t = kwargs['d0_t']
        d_t, v_t = newm_integrate(sys_t, d0_t[0], d0_t[1], T, sys_t.dt, beta=.25)
        ts = np.linspace(t0, T, int(round((T - t0) / sys_t.dt)) + 1)
    # d, v = central_diff_integrate(sys, d0[0], d0[1], T, dt)
    # d = d[int(round(t0/dt)):]
    draw.draw_displacement_2d(d, sys.mesh, ts, ds_t=d_t, mesh_t=sys_t.mesh, **kwargs)
    return draw.pop_holds()

def draw_displacement_snapshots(sys, d0, g, ts, **kwargs):
    t0, T = 0, max(ts)
    tx = np.linspace(t0, T, int(round((T - t0)/sys.dt)) + 1)
    d, v = newm_integrate(sys, d0[0], d0[0], T, sys.dt, beta=.25)
    if 'system_t' in kwargs:
        sys_t = kwargs['system_t']
        d0_t = kwargs['d0_t']
        d_t, v_t = newm_integrate(sys_t, d0_t[0], d0_t[1], T, sys_t.dt, beta=.25)
        tx = np.linspace(t0, T, int(round((T - t0) / sys_t.dt)) + 1)
    return draw.draw_displacement_2d_snapshot(d, sys.mesh, tx, ts, ds_t=d_t,
                                              mesh_t=sys_t.mesh, **kwargs)


def draw_pwlf(pwlf, ylabel='Force $u_L$', xlabel='Time t', axes=None):
    """Draws a piecewise linear function

    Parameters
    ----------
    pwlf : :class:`PWLFunction`
    ylabel : str, optional
    xlabel : str, optional
    axes : :class:`matplotlib.axes.Axes`
        Axes in which to draw the function

    Returns
    -------
    :class:`matplotlib.figure.Figure`, only returned if axes == None

    """
    return draw.draw_linear(pwlf.ys, pwlf.ts, ylabel, xlabel, axes=axes)

def _draw_sosys_snapshots(sosys, d0, v0, g, ts, hold=False, **kargs):
    dt = sosys.dt
    xpart = sosys.xpart

    t0, T = 0, max(ts)
    tx = np.linspace(t0, T, int(round((T - t0)/dt)))
    d, v = newm_integrate(sosys, d0, v0, T, dt, beta=.25)
    for t in ts:
        index = bisect_right(tx, t) -1
        draw.draw_pde_snapshot(
            xpart, d[index], np.true_divide(np.diff(d[index]), np.diff(xpart)),
            t, hold=hold, **kargs)

    if hold:
        return draw.pop_holds()

def _draw_fosys_snapshots(fosys, d0, g, ts, hold=False, **kargs):
    dt = fosys.dt
    xpart = fosys.xpart

    t0, T = 0, max(ts)
    tx = np.linspace(t0, T, int(round((T - t0)/dt)))
    d = trapez_integrate(fosys, d0, T, dt)
    for t in ts:
        index = bisect_right(tx, t) -1
        draw.draw_pde_snapshot(
            xpart, d[index], np.true_divide(np.diff(d[index]), np.diff(xpart)),
            t, hold=hold, **kargs)

    if hold:
        return draw.pop_holds()


def draw_system_snapshots(sys, d0, g, ts, **kargs):
    """Draws snapshots of the trajectory of a FEM system from a 1D PDE

    Uses the function :func:`femformal.core.draw_util.draw_pde_snapshot`.

    Parameters
    ----------
    sys : :class:`SOSystem`
    d0 : array_like
        Initial value. Must have a row for each degree of freedom
    g : array_like
        Boundary conditions of the PDE
    ts : array_like
        Times for each snapshot
    kwargs : dict, optional
        Extra arguments passed to the drawing function

    Returns
    -------
    list, only returned if hold == True
        List of `matplotlib` figures created

    """
    if isinstance(sys, SOSystem):
        return _draw_sosys_snapshots(sys, d0[0], d0[1], g, ts, **kargs)
    elif isinstance(sys, FOSystem):
        return _draw_fosys_snapshots(sys, d0, g, ts, **kargs)
    else:
        raise NotImplementedError(
            "Not implemented for this class of system: {}".format(
                sys.__class__.__name__))
