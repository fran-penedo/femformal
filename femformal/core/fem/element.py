"""
FEM element data structures and definitions
"""
from __future__ import division, absolute_import, print_function

import logging

import numpy as np

from .. import util


logger = logging.getLogger(__name__)

class Element2DOF(object):
    """Abstract 2D element with 2 DOFs

    Parameters
    ----------
    coords : array, shape (nnodes, 2)
        Coordinates of the nodes
    shape_order : int, optional
        Maximum order of the polynomials used as shape functions

    Attributes
    ----------
    coords : array
    dimension : int
        Dimension of the element
    shape_order : int

    """
    def __init__(self, coords, shape_order=0):
        self.coords = coords
        self.dimension = self.coords.shape[1]
        self.shape_order = shape_order

    def interpolate_phys(self, values, coords):
        return self.shapes(*self.normalize(coords)).dot(np.array(values))

    def interpolate_strain(self, values, coords):
        a, b = self.normalize(coords)
        x, y = zip(*self.coords)
        values = np.array(values)
        v = values.reshape(np.prod(values.shape))
        return self.strain_displacement(a, b, x, y).dot(v)

    def interpolate(self, values, coords):
        return self.shapes(*coords).dot(np.array(values))

    def interpolate_derivatives_phys(self, values, coords):
        """Designed for scalar fields or vector fields with same dimension as
        domain"""

        a, b = self.normalize(coords)
        x, y = zip(*self.coords)
        str_disp = self.strain_displacement(a, b, x, y)
        values = np.array(values)
        if np.prod(values.shape) != str_disp.shape[-1]:
            dof = 1 if len(values.shape) == 1 else values.shape[-1]
            values = np.c_[
                values,
                np.zeros((values.shape[0],
                          str_disp.shape[-1] / np.prod(values.shape) - dof))]

        ret = np.zeros((values.shape[-1], values.shape[-1]))
        for p in util.cycle(values.shape[-1]):
            p = list(p)
            v = values.T[np.ix_(p)].T
            v = v.reshape(np.prod(v.shape))
            derivs = str_disp.dot(v)[0:values.shape[-1]]
            ret[[p, range(values.shape[-1])]] = derivs

        return ret

    def normalize(self, coords):
        raise NotImplementedError()

    @staticmethod
    def shapes(*parameters):
        raise NotImplementedError()

    @staticmethod
    def shapes_derivatives(*parameters):
        raise NotImplementedError()

    @classmethod
    def strain_displacement(cls, *parameters):
        a, b, x, y = parameters
        jac_inv, jac_det = cls.jacobian(a, b, x, y)
        dshapes_phy = jac_inv.dot(cls.shapes_derivatives(a, b))
        n = 2 * dshapes_phy.shape[1]
        str_disp = np.zeros((3,n))
        str_disp[0, 0:n:2] = dshapes_phy[0]
        str_disp[1, 1:n:2] = dshapes_phy[1]
        str_disp[2, 0:n:2] = dshapes_phy[1]
        str_disp[2, 1:n:2] = dshapes_phy[0]

        return str_disp

    @classmethod
    def jacobian(cls, a, b, x, y):
        jac = cls.shapes_derivatives(a,b).dot(np.vstack([x,y]).T)
        jac_det = np.linalg.det(jac)
        jac_inv = np.array([[jac[1,1], -jac[0,1]], [-jac[1,0], jac[0,0]]]) / jac_det
        return jac_inv, jac_det

    def max_diff(self, values, axis):
        raise NotImplementedError()


class BLQuadQ4(Element2DOF):
    def __init__(self, coords):
        if coords.shape != (4, 2):
            raise ValueError("Q4 element coordinates must be 4x2")
        Element2DOF.__init__(self, coords)
        self.center = np.mean(coords, axis=0)
        if all([np.all([np.isclose(p, self.center) for p in coords])]):
            self.normalize = self._normalize_0d
        else:
            try:
                self.transf_matrix = self._transf_matrix(coords)
                self.normalize = self._normalize_full
            except ValueError:
                self.diff = self.coords[2] - self.coords[0]
                self.ndiff = np.linalg.norm(self.diff)
                self.normalize = self._normalize_1d

    @staticmethod
    def _transf_matrix(coords):
        x, y = coords.T
        det = 2 * ((-x[0] + x[2]) * (-y[1] + y[3]) + (x[1] - x[3]) * (-y[0] + y[2]))
        if not np.isclose(det, 0):
            return 4 / det * np.array([
                [-y[0] - y[1] + y[2] + y[3], x[0] + x[1] - x[2] - x[3]],
                [y[0] - y[1] - y[2] + y[3],  -x[0] + x[1] + x[2] - x[3]]
            ])
        else:
            raise ValueError("Cannot compute change of basis for degenerate element")

    @staticmethod
    def shapes(*parameters):
        a, b = parameters
        return 0.25 * np.array([(1 - a) * (1 - b), (1 + a) * (1 - b),
                                (1 + a) * (1 + b), (1 - a) * (1 + b)])

    @staticmethod
    def shapes_derivatives(*parameters):
        a, b = parameters
        return 0.25 * np.array([[- (1 - b), (1 - b), (1 + b), - (1 + b)],
                                [- (1 - a), - (1 + a), (1 + a), (1 - a)]])

    def _normalize_full(self, coords):
        return self.transf_matrix.dot(coords - self.center)

    def _normalize_0d(self, coords):
        return np.array([0,0])

    def _normalize_1d(self, coords):
        return 2 * (coords - self.coords[0]) / self.ndiff - 1

    def max_diff(self, values, axis):
        values = np.array(values)
        def take(i):
            return values.take(i, axis=-2)

        if axis == 0:
            return np.max(np.abs([take(0) - take(1), take(3) - take(2),
                                  take(0) - take(2), take(3) - take(1)]),
                          axis=0)
        elif axis == 1:
            return np.max(np.abs([take(0) - take(3), take(1) - take(2),
                                  take(0) - take(2), take(1) - take(3)]),
                          axis=0)
        else:
            raise ValueError("Axis must be 0 or 1. Given: {}".format(axis))

    def chebyshev_radius(self):
        return np.linalg.norm(self.coords[0] - self.coords[2]) / 2.0

    def chebyshev_center(self):
        return self.interpolate(self.coords, [0, 0])


class QuadQuadQ9(Element2DOF):
    """Quadratic quadrilateral element with 9 nodes

    Nodes are numbered in the following way, starting in index 0:

        3 - 6 - 2
        |   |   |
        7 - 8 - 5
        |   |   |
        0 - 4 - 1

    Parameters
    ----------
    coords : array, shape (9, 2)
        Coordinates of the nodes

    Attributes
    ----------
    center : array, shape (1, 2)
        Center of the element

    """
    def __init__(self, coords):
        if coords.shape != (9, 2):
            raise ValueError("Q9 element coordinates must be 9x2")
        Element2DOF.__init__(self, coords, shape_order=2)
        self.center = coords[-1]
        if all([np.all([np.isclose(p, self.center) for p in coords[:4]])]):
            self.normalize = self._normalize_0d
        else:
            try:
                self.transf_matrix = self._transf_matrix(coords)
                self.normalize = self._normalize_full
            except ValueError:
                self.diff = self.coords[2] - self.coords[0]
                self.ndiff = np.linalg.norm(self.diff)
                self.normalize = self._normalize_1d

    @staticmethod
    def _transf_matrix(coords):
        #FIXME I'm not sure this applies as is for Q9 from Q4
        x, y = coords.T
        det = 2 * ((-x[0] + x[2]) * (-y[1] + y[3]) + (x[1] - x[3]) * (-y[0] + y[2]))
        if not np.isclose(det, 0):
            return 4 / det * np.array([
                [-y[0] - y[1] + y[2] + y[3], x[0] + x[1] - x[2] - x[3]],
                [y[0] - y[1] - y[2] + y[3],  -x[0] + x[1] + x[2] - x[3]]
            ])
        else:
            raise ValueError("Cannot compute change of basis for degenerate element")

    @staticmethod
    def shapes(*parameters):
        a, b = parameters
        aa = a*a
        bb = b*b
        return 0.25 * np.array([(aa - a) * (bb - b),
                                (aa + a) * (bb - b),
                                (aa + a) * (bb + b),
                                (aa - a) * (bb + b),
                                2 * (1 - aa) * (bb - b),
                                2 * (aa + a) * (1 - bb),
                                2 * (1 - aa) * (bb + b),
                                2 * (aa - a) * (1 - bb),
                                4 * (1 - aa) * (1 - bb)])

    @staticmethod
    def shapes_derivatives(*parameters):
        a, b = parameters
        aa = a*a
        bb = b*b
        return 0.25 * np.array([
            [(2 * a - 1) * (bb - b),
             (2 * a + 1) * (bb - b),
             (2 * a + 1) * (bb + b),
             (2 * a - 1) * (bb + b),
             2 * (- 2 * a) * (bb - b),
             2 * (2 * a + 1) * (1 - bb),
             2 * (- 2 * a) * (bb + b),
             2 * (2 * a - 1) * (1 - bb),
             4 * (- 2 * a) * (1 - bb)],
            [(aa - a) * (2 * b - 1),
             (aa + a) * (2 * b - 1),
             (aa + a) * (2 * b + 1),
             (aa - a) * (2 * b + 1),
             2 * (1 - aa) * (2 * b - 1),
             2 * (aa + a) * (- 2 * b),
             2 * (1 - aa) * (2 * b + 1),
             2 * (aa - a) * (- 2 * b),
             4 * (1 - aa) * (- 2 * b)]])

    def _normalize_full(self, coords):
        return self.transf_matrix.dot(coords - self.center)

    def _normalize_0d(self, coords):
        return np.array([0,0])

    def _normalize_1d(self, coords):
        return 2 * (coords - self.coords[0]) / self.ndiff - 1

    def max_diff(self, values, axis):
        #FIXME implement me
        raise NotImplementedError()
        values = np.array(values)
        def take(i):
            return values.take(i, axis=-2)

        if axis == 0:
            return np.max(np.abs([take(0) - take(1), take(3) - take(2),
                                  take(0) - take(2), take(3) - take(1)]),
                          axis=0)
        elif axis == 1:
            return np.max(np.abs([take(0) - take(3), take(1) - take(2),
                                  take(0) - take(2), take(1) - take(3)]),
                          axis=0)
        else:
            raise ValueError("Axis must be 0 or 1. Given: {}".format(axis))

    def chebyshev_radius(self):
        return np.linalg.norm(self.coords[0] - self.coords[2]) / 2.0

    def chebyshev_center(self):
        return self.center
