import numpy as np
from scipy.optimize import linprog

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

        j = np.setdiff1d(np.nonzero(self.A[i, :])[1], i)
        j = j[(j >= 0) & (j < self.n)]
        C = self.A[np.ix_(i, j)]

        return System(A, b, C)

    @property
    def n(self):
        return len(self.A)

    @property
    def m(self):
        return self.C.shape[1]


def is_facet_separating(A, b, C, facet, dim, dist_bounds):
    A_j = np.delete(A[dim], dim)
    if len(C) > 0:
        A_j = np.hstack([A_j, C[dim]])
    b = - b[dim] - A[dim, dim] * facet[dim][0]
    R = np.delete(facet, dim, 0)
    if len(C) > 0:
        R = np.vstack([R, dist_bounds])
    return rect_in_semispace(R, A_j, b)

def rect_in_semispace(R, a, b):
    if np.isclose(a, 0):
        return b >= 0
    else:
        res = linprog(-a, a[None], b + 1, bounds=R)
        return - res.fun <= b
