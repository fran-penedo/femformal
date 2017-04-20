import numpy as np
from scipy.optimize import linprog
from scipy.integrate import odeint
import scipy.linalg as la
from bisect import bisect_left, bisect_right
import draw_util as draw

import logging
logger = logging.getLogger('FEMFORMAL')

class System(object):

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


class FOSystem(object):

    def __init__(self, M, K, F, xpart=None, dt=1.0):
        self.M = M
        self.K = K
        self.F = F
        self.xpart = xpart
        self.dt = dt

    def to_canon(self):
        A = np.linalg.solve(self.M, -self.K)
        b = np.linalg.solve(self.M, self.F)
        C = np.empty(shape=(0,0))
        system = System(A, b, C, self.xpart, self.dt)

        return system

    @property
    def n(self):
        return len(self.M)

    def __str__(self):
        return "M:\n{0}\nK:\n{1}\nF:\n{2}".format(self.M, self.K, self.F)


class SOSystem(object):

    def __init__(self, M, K, F, xpart=None, dt=1.0):
        self.M = M
        self.K = K
        self.F = F
        self.xpart = xpart
        self.dt = dt

    def to_fosystem(self):
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
        return len(self.M)

    def add_f_nodal(self, f_nodal):
        self.F = self.F + f_nodal

    def copy(self):
        return SOSystem(self.M.copy(), self.K.copy(), self.F.copy(),
                        self.xpart.copy(), self.dt)

    def __str__(self):
        return "M:\n{0}\nK:\n{1}\nF:\n{2}".format(self.M, self.K, self.F)



class ControlSOSystem(SOSystem):
    def __init__(self, M, K, F, f_nodal, xpart=None, dt=1.0):
        SOSystem.__init__(self, M, K, F, xpart=xpart, dt=dt)
        self.f_nodal = f_nodal

    def add_f_nodal(self, f_nodal):
        self.f_nodal = f_nodal

    @staticmethod
    def from_sosys(sosys, f_nodal):
        csosys = ControlSOSystem(
            sosys.M, sosys.K, sosys.F, f_nodal, sosys.xpart, sosys.dt)
        return csosys

    def copy(self):
        return ControlSOSystem.from_sosys(
            super(ControlSOSystem, self).copy(), self.f_nodal)



class PWLFunction(object):
    def __init__(self, ts, ys=None, ybounds=None, x=None):
        self.x = x
        self.ts = ts
        self.ys = ys
        self.ybounds = ybounds

    def pset(self):
        left = np.hstack([np.identity(len(self.ts)),
                             -np.identity(len(self.ts))])
        right = np.r_[[self.ybounds[1] for x in self.ts],
                      [-self.ybounds[0] for x in self.ts]]
        return np.vstack([left, right]).T

    def __call__(self, x, t, p=None):
        if x != self.x:
            return 0
        ts = self.ts
        if p is None:
            if self.ys is None:
                raise Exception("y values not set")
            ys = self.ys
            if t < ts[0] or t > ts[-1]:
                raise Exception("Argument out of domain. t = {}".format(t))
        else:
            self.ys = ys = p
        # FIXME probable issue with t = ts[-1]
        i = bisect_right(ts, t) - 1
        return ys[i] + (ys[i+1] - ys[i]) * (t - ts[i]) / (ts[i+1] - ts[i])


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
    n_e = np.nonzero(~np.all(sosys.M == 0, axis=1))[0]
    M = sosys.M[np.ix_(n_e, n_e)]
    K = sosys.K[np.ix_(n_e, n_e)]
    F = sosys.F[n_e]

    try:
        f_nodal = sosys.f_nodal
    except AttributeError:
        f_nodal = np.zeros(F.shape)

    its = int(round(T / dt))
    d = np.array(d0)
    v = np.array(v0)
    a = np.zeros(d.shape[0])
    M_LU = la.lu_factor(M)
    a[n_e] = la.lu_solve(M_LU, F - K.dot(d[n_e]))
    ds = [d]
    vs = [v]
    for i in range(its):
        tv = v + .5 * dt * a
        # tv[0] = tv[-1] = 0.0
        d = d + dt * tv
        try:
            f_nodal_c = f_nodal(i * dt)[n_e]
        except TypeError:
            f_nodal_c = f_nodal
        a[n_e] = la.lu_solve(M_LU, F + f_nodal_c - K.dot(d[n_e]))
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

def pwcx(d, xpart):
    def u(x):
        i = bisect_left(xpart, x) - 1
        return d[:, i]
    return u

def diff(x, y, dtx, dty, xpart, ypart, xl, xr, pwc=False):
    if dty > dtx:
        x, y = y, x
        dtx, dty = dty, dtx

    yy = y[::int(round(dtx / dty))]
    yl = bisect_left(ypart, xl)
    yr = bisect_right(ypart, xr)
    if pwc:
        xinter = pwcx(x, xpart)
        yinter = pwcx(yy, ypart)
        d = np.array([xinter(z) - yinter(z) for z in
                      (ypart[yl + 1:yr] + ypart[yl:yr - 1]) / 2.0]).T
    else:
        xinter = linterx(x, xpart)
        yinter = linterx(yy, ypart)
        d = np.array([xinter(z) - yinter(z) for z in ypart[yl:yr]]).T
    return d

def sys_diff(xsys, ysys, x0, y0, tlims, xlims, plot=False):
    dtx, dty = xsys.dt, ysys.dt
    xpart, ypart = xsys.xpart, ysys.xpart
    t0, T = tlims
    xl, xr = xlims
    tx = int(round(T / dtx))
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
        ttx = np.linspace(0, T, int(round(T / dtx)))
        draw.draw_pde_trajectory(x, xpart, ttx, hold=True)
        draw.draw_pde_trajectory(y, ypart, ty, hold=True)
        draw.draw_pde_trajectory(absdif, ypart[yl:yr], ttx, hold=False)
    return absdif

def sosys_diff(xsys, ysys, x0, y0, tlims, xlims, xderiv=False, plot=False):
    dtx, dty = xsys.dt, ysys.dt
    xpart, ypart = xsys.xpart, ysys.xpart
    t0, T = tlims
    x = newm_integrate(xsys, x0[0], x0[1], T, dtx)[0]
    y = newm_integrate(ysys, y0[0], y0[1], T, dty)[0]
    x = x[int(round(t0/dtx)):]
    y = y[int(round(t0/dty)):]
    # draw.draw_pde_trajectory(x, xpart, np.linspace(0, T, (int(round(T / dtx)))), animate=False)
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
    if True:
        ttx = np.linspace(0, T, int(round(T / dtx)))
        tty = np.linspace(0, T, int(round(T / dty)))
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
    return absdif

def sys_max_diff(xsys, ysys, x0, y0, tlims, xlims, xderiv=False, pw=False, plot=False):
    if isinstance(xsys, SOSystem):
        diff_f = sosys_diff
    elif isinstance(xsys, FOSystem):
        diff_f = sys_diff
        xsys = xsys.to_canon()
        ysys = ysys.to_canon()
    else:
        diff_f = sys_diff

    absdif = diff_f(xsys, ysys, x0, y0, tlims, xlims, xderiv, plot)
    if isinstance(absdif, list):
        pwdiff = [np.amax(dif, axis=0) for dif in absdif]
        return pwdiff if pw else np.amax(pwdiff)
    else:
        return np.max(absdif, axis=0 if pw else None)

def sys_max_xdiff(sys, x0, t0, T):
    dt = sys.dt
    xpart = sys.xpart
    # print x0
    t = np.linspace(0, T, int(round(T / dt)))
    x = cont_integrate(sys, x0[1:-1], t)
    x = np.c_[x0[0] * np.ones(x.shape[0]), x, x0[-1] * np.ones(x.shape[0])]
    x = x[int(round(t0/dt)):]

    # draw.draw_pde_trajectory(x, xpart, t)

    dx = np.abs(np.diff(x))
    mdx = np.max(dx, axis=0)

    return mdx

def sys_max_tdiff(sys, x0, t0, T):
    dt = sys.dt
    xpart = sys.xpart

    t = int(round(T / dt))
    x = disc_integrate(sys, x0[1:-1], t)
    x = np.c_[x0[0] * np.ones(x.shape[0]), x, x0[-1] * np.ones(x.shape[0])]
    x = x[int(round(t0/dt)):]

    dtx = np.abs(np.diff(x, axis=0))
    mdtx = np.max(dtx, axis=0)

    return mdtx

def sosys_max_der_diff(sys, x0, tlims, xderiv=False):
    x, vx = newm_integrate(sys, x0[0], x0[1], tlims[1], sys.dt)
    x = x[int(round(tlims[0]/sys.dt)):]
    # draw.draw_pde_trajectory(x, sys.xpart, np.linspace(0, tlims[1], (int(round(tlims[1] / sys.dt)))), animate=False)
    if xderiv:
        x = np.true_divide(np.diff(x), np.diff(sys.xpart))

    dx = np.abs(np.diff(x))
    dtx = np.abs(np.diff(x, axis=0))
    mdx = np.max(dx, axis=0)
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

def draw_system_cont(sys, x0, T, t0=0,
                     prefix=None, animate=True, allonly=False, hold=False,
                     ylabel='Temperature', xlabel='x'):
    dt = sys.dt
    xpart = sys.xpart

    tx = np.linspace(t0, T, int(round((T - t0)/dt)))
    x = cont_integrate(sys, x0[1:-1], tx)
    x = np.c_[x0[0] * np.ones(x.shape[0]), x, x0[-1] * np.ones(x.shape[0])]
    x = x[int(round(t0/dt)):]
    tx = tx[int(round(t0/dt)):]
    draw.draw_pde_trajectory(x, xpart, tx, prefix=prefix,
                             animate=animate, hold=hold, allonly=allonly, ylabel=ylabel,
                             xlabel=xlabel)

def draw_sosys(sosys, d0, v0, g, T, t0=0,
               prefix=None, animate=True, allonly=False, hold=False,
               ylabel='Displacement', xlabel='x'):
    dt = sosys.dt
    xpart = sosys.xpart

    tx = np.linspace(t0, T, int(round((T - t0)/dt)))
    d, v = newm_integrate(sosys, d0, v0, T, dt)
    d = d[int(round(t0/dt)):]
    draw.draw_pde_trajectory(d, xpart, tx, prefix=prefix,
                             animate=animate, hold=hold, allonly=allonly,
                             ylabel=ylabel, xlabel=xlabel)

def draw_pwlf(pwlf, ylabel='Force $u_L$', xlabel='Time t'):
    draw.draw_linear(pwlf.ys, pwlf.ts, ylabel, xlabel)