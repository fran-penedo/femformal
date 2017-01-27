import numpy as np
from scipy.optimize import linprog
from scipy.integrate import odeint
import scipy.linalg as la
from bisect import bisect_left, bisect_right
import draw_util as draw

import logging
logger = logging.getLogger('FEMFORMAL')

class System(object):

    def __init__(self, A, b, C=None):
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


class SOSystem(object):

    def __init__(self, M, K, F):
        self.M = M
        self.K = K
        self.F = F

    def to_fosystem(self):
        n = self.n
        zeros = np.zeros((n, n))
        ident = np.identity(n)
        Maug = np.asarray(np.bmat([[ident, zeros],[zeros, self.M]]))
        Kaug = np.asarray(np.bmat([[zeros, -ident],[self.K, zeros]]))
        Faug = np.hstack([np.zeros(n), self.F])

        A = np.linalg.solve(Maug, -Kaug)
        b = np.linalg.solve(Maug, Faug)
        C = np.empty(shape=(0,0))
        system = System(A, b, C)

        return system

    @property
    def n(self):
        return len(self.M)

    def __str__(self):
        return "M:\n{0}\nK:\n{1}\nF:\n{2}".format(self.M, self.K, self.F)

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

def newm_integrate(sosys, d0, v0, T, dt=.1):
    its = int(T / dt)
    d = np.array(d0)
    v = np.array(v0)
    a = np.linalg.solve(sosys.M, sosys.F - sosys.K.dot(d))
    ds = [d]
    vs = [v]
    for i in range(its):
        tv = v + .5 * dt * a
        # tv[0] = tv[-1] = 0.0
        d = d + dt * tv
        a = np.linalg.solve(sosys.M, sosys.F - sosys.K.dot(d))
        v = tv + .5 * dt * a
        ds.append(d)
        vs.append(v)
    return np.array(ds), np.array(vs)


def linterx(d, xpart):
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

def diff(x, y, dtx, dty, xpart, ypart, xl, xr):
    if dty > dtx:
        x, y = y, x
        dtx, dty = dty, dtx

    yy = y[::int(dtx / dty)]
    xinter = linterx(x, xpart)
    yinter = linterx(yy, ypart)
    yl = bisect_left(ypart, xl)
    yr = bisect_right(ypart, xr) - 1
    d = np.array([xinter(z) - yinter(z) for z in ypart[yl:yr]]).T
    return d

def sys_diff(xsys, ysys, dtx, dty, xpart, ypart, x0, y0, t0, T, xl, xr, plot=False):
    tx = int(T / dtx)
    ty = np.linspace(0, T, int(T / dty))
    x = disc_integrate(xsys, x0[1:-1], tx)
    x = np.c_[x0[0] * np.ones(x.shape[0]), x, x0[-1] * np.ones(x.shape[0])]
    x = x[int(t0/dtx):]
    y = cont_integrate(ysys, y0[1:-1], ty)
    y = np.c_[y0[0] * np.ones(y.shape[0]), y, y0[-1] * np.ones(y.shape[0])]
    y = y[int(t0/dty):]
    absdif = np.abs(diff(x, y, dtx, dty, xpart, ypart, xl, xr))
    if plot:
        yl = bisect_left(ypart, xl)
        yr = bisect_right(ypart, xr) - 1
        ttx = np.linspace(0, T, int(T / dtx))
        draw.draw_pde_trajectory(x, xpart, ttx, hold=True)
        draw.draw_pde_trajectory(y, ypart, ty, hold=True)
        draw.draw_pde_trajectory(absdif, ypart[yl:yr], ttx, hold=False)
    return absdif

def sys_max_diff(xsys, ysys, dtx, dty, xpart, ypart, x0, y0, t0, T, xl, xr):
    absdif = sys_diff(xsys, ysys, dtx, dty, xpart, ypart, x0, y0, t0, T, xl, xr)
    return np.max(absdif)


def sys_max_xdiff(sys, dt, xpart, x0, t0, T):
    # print x0
    t = np.linspace(0, T, int(T / dt))
    x = cont_integrate(sys, x0[1:-1], t)
    x = np.c_[x0[0] * np.ones(x.shape[0]), x, x0[-1] * np.ones(x.shape[0])]
    x = x[int(t0/dt):]

    # draw.draw_pde_trajectory(x, xpart, t)

    dx = np.abs(np.diff(x))
    mdx = np.max(dx, axis=0)

    return mdx

def sys_max_tdiff(sys, dt, xpart, x0, t0, T):
    t = int(T / dt)
    x = disc_integrate(sys, x0[1:-1], t)
    x = np.c_[x0[0] * np.ones(x.shape[0]), x, x0[-1] * np.ones(x.shape[0])]
    x = x[int(t0/dt):]

    dtx = np.abs(np.diff(x, axis=0))
    mdtx = np.max(dtx, axis=0)

    return mdtx



def draw_system_disc(sys, x0, dt, T, xpart, t0=0,
                     prefix=None, animate=True, allonly=False, hold=False):
    tx = np.linspace(t0, T, int((T - t0)/dt))
    x = disc_integrate(sys, x0[1:-1], int(T/dt))
    x = np.c_[x0[0] * np.ones(x.shape[0]), x, x0[-1] * np.ones(x.shape[0])]
    x = x[int(t0/dt):]
    draw.draw_pde_trajectory(x, xpart, tx, prefix=prefix,
                             animate=animate, hold=hold, allonly=allonly)

def draw_sosys(sosys, d0, v0, T, xpart, dt=0.1, t0=0,
               prefix=None, animate=True, allonly=False, hold=False):
    tx = np.linspace(t0, T, int((T - t0)/dt))
    d, v = newm_integrate(sosys, d0, v0, T, dt)



