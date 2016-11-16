import numpy as np
from scipy.optimize import linprog
from scipy.integrate import odeint
import scipy.linalg as la

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

def cont_to_disc(system):
    A = la.expm(system.A)
    print A
    b = A.dot(- la.solve(
        system.A, (la.expm(-system.A) - np.identity(system.n)))).dot(system.b)
    return System(A, b)

def cont_integrate(system, x0, t):
    return odeint(lambda x, t: (system.A.dot(x) + system.b.T).flatten(), x0, t)

def disc_integrate(system, x0, t):
    xs = [x0]
    x = x0
    for i in range(t):
        x = (system.A.dot(x) + system.b.T).flatten()
        xs.append(x)
    return xs
